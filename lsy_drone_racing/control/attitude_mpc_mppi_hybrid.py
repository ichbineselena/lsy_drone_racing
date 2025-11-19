"""Hybrid MPPI trajectory generation + MPC tracking controller.

This module implements a two-level control architecture:
1. High-level MPPI: Generates reference trajectories online using sampling-based optimization
   - Handles obstacle avoidance dynamically
   - Replans trajectory at each timestep based on current state
   - Uses simplified dynamics for fast parallel rollouts

2. Low-level MPC: Tracks the MPPI-generated trajectory using attitude control
   - Uses full nonlinear drone dynamics
   - Handles attitude and thrust constraints
   - Provides precise tracking of the reference trajectory

The hybrid approach combines the strengths of both methods:
- MPPI: Better global planning, adaptive to changes, naturally handles obstacles
- MPC: Precise tracking with constraint satisfaction and full dynamics
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.trajectory_builders import MPPIBuilder

if TYPE_CHECKING:
    from numpy.typing import NDArray


def create_acados_model(parameters: dict) -> AcadosModel:
    """Creates an acados model from a symbolic drone_model."""
    X_dot, X, U, _ = symbolic_dynamics_euler(
        mass=parameters["mass"],
        gravity_vec=parameters["gravity_vec"],
        J=parameters["J"],
        J_inv=parameters["J_inv"],
        acc_coef=parameters["acc_coef"],
        cmd_f_coef=parameters["cmd_f_coef"],
        rpy_coef=parameters["rpy_coef"],
        rpy_rates_coef=parameters["rpy_rates_coef"],
        cmd_rpy_coef=parameters["cmd_rpy_coef"],
    )

    # Initialize the nonlinear model for NMPC formulation
    model = AcadosModel()
    model.name = "mppi_mpc_hybrid"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U

    return model


def create_ocp_solver(
    Tf: float, N: int, parameters: dict, verbose: bool = False
) -> tuple[AcadosOcpSolver, AcadosOcp]:
    """Creates an acados Optimal Control Problem and Solver."""
    ocp = AcadosOcp()

    # Set model
    ocp.model = create_acados_model(parameters)

    # Get Dimensions
    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    # Set dimensions
    ocp.solver_options.N_horizon = N

    ## Set Cost
    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights - tuned for tracking MPPI-generated trajectories
    # State weights
    Q = np.diag(
        [
            80.0,  # pos x - higher weight for better tracking
            80.0,  # pos y
            500.0,  # pos z - very high weight to maintain altitude
            1.0,  # roll
            1.0,  # pitch
            2.0,  # yaw - slightly higher for orientation tracking
            15.0,  # vel x - higher to track velocity reference
            15.0,  # vel y
            15.0,  # vel z
            5.0,  # drpy x
            5.0,  # drpy y
            5.0,  # drpy z
        ]
    )
    # Input weights (reference is upright orientation and hover thrust)
    R = np.diag(
        [
            1.0,  # roll cmd
            1.0,  # pitch cmd
            1.0,  # yaw cmd
            40.0,  # thrust - moderate weight to allow tracking
        ]
    )

    Q_e = Q.copy()
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx : nx + nu, :] = np.eye(nu)  # Select all actions
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)  # Select all states
    ocp.cost.Vx_e = Vx_e

    # Set initial references (we will overwrite these later)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Initial state constraint
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/mppi_mpc_hybrid.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class AttitudeMPCMPPIHybrid(Controller):
    """Hybrid controller: MPPI trajectory generation + MPC tracking.
    
    This controller uses MPPI to generate adaptive reference trajectories online,
    which are then tracked by a lower-level MPC using attitude control.
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the hybrid controller.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        
        # MPC parameters
        self._N = 25  # MPC prediction horizon
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Load drone parameters
        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        
        # Create MPC solver
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        # Extract goal from track (final gate position)
        track_config = config.env.track if hasattr(config.env, 'track') else config['env']['track']
        gates_config = track_config.gates if hasattr(track_config, 'gates') else track_config.get('gates', [])
        
        if gates_config and len(gates_config) > 0:
            final_gate = gates_config[-1]
            if hasattr(final_gate, 'pos'):
                self._goal = np.array(final_gate.pos)
            elif isinstance(final_gate, dict):
                self._goal = np.array(final_gate['pos'])
            else:
                self._goal = np.array([0.5, -0.75, 1.2])
        else:
            # Fallback goal if no gates defined
            self._goal = np.array([0.5, -0.75, 1.2])
        
        # MPPI trajectory builder parameters
        mppi_config = {
            "K": 500,  # Number of samples (reduced for real-time performance)
            "lambda_": 0.8,  # Temperature parameter
            "sigma_u": 0.4,  # Control noise standard deviation
        }
        
        # Initialize MPPI trajectory builder
        self._trajectory_builder = MPPIBuilder(
            goal=self._goal,
            K=mppi_config["K"],
            lambda_=mppi_config["lambda_"],
            sigma_u=mppi_config["sigma_u"],
        )

        # Get obstacles from config for MPPI awareness
        self._obstacles = []
        track_config = config.env.track if hasattr(config.env, 'track') else config['env']['track']
        obstacles_config = track_config.obstacles if hasattr(track_config, 'obstacles') else track_config.get('obstacles', [])
        
        for obs_config in obstacles_config:
            if hasattr(obs_config, 'pos'):
                self._obstacles.append(np.array(obs_config.pos))
            elif isinstance(obs_config, dict):
                self._obstacles.append(np.array(obs_config['pos']))
        
        self._tick = 0
        self._config = config
        self._finished = False
        self._mppi_replan_interval = 1  # Replan every N ticks (1 = every step)
        self._last_planned_pos = None  # For visualization
        
        print(f"[AttitudeMPCMPPIHybrid] Initialized with:")
        print(f"  - MPC horizon: {self._N} steps ({self._T_HORIZON:.2f}s)")
        print(f"  - MPPI samples: {mppi_config['K']}")
        print(f"  - Goal position: {self._goal}")
        print(f"  - Number of obstacles: {len(self._obstacles)}")

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute control using MPPI trajectory generation + MPC tracking.

        Args:
            obs: The current observation of the environment.
            info: Optional additional information as a dictionary.

        Returns:
            Control command [r_des, p_des, y_des, t_des] as a numpy array.
        """
        # Convert observation to state vector for MPPI
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        
        # Reset MPPI trajectory builder if first tick
        if self._tick == 0:
            self._trajectory_builder.reset(x0, t0=0.0)
        
        # Update MPPI with current state periodically (replanning)
        if self._tick % self._mppi_replan_interval == 0:
            # Update MPPI's internal state for next planning iteration
            self._trajectory_builder.x0 = x0
        
        # Get MPPI-generated reference trajectory for MPC horizon
        current_time = self._tick * self._dt
        horizon_data = self._trajectory_builder.get_horizon(
            t_now=current_time,
            N=self._N,
            dt=self._dt
        )
        
        waypoints_pos = horizon_data["pos"]  # (N, 3)
        waypoints_vel = horizon_data["vel"]  # (N, 3)
        waypoints_yaw = horizon_data["yaw"]  # (N,)
        
        # Store for visualization
        self._last_planned_pos = waypoints_pos
        
        # Check if goal reached
        dist_to_goal = np.linalg.norm(obs["pos"] - self._goal)
        if dist_to_goal < 0.3:  # 30cm threshold
            self._finished = True
        
        # Set initial state constraint for MPC
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Set reference trajectory from MPPI for MPC
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = waypoints_pos  # position reference
        # Roll and pitch reference = 0 (indices 3, 4)
        yref[:, 5] = waypoints_yaw  # yaw reference
        yref[:, 6:9] = waypoints_vel  # velocity reference
        # Angular velocity reference = 0 (indices 9-11)
        
        # Set input reference (indices >= nx)
        # Desired roll, pitch, yaw commands = 0
        # Hover thrust
        hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        yref[:, 15] = hover_thrust
        
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Set terminal reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = waypoints_pos[-1]  # terminal position
        yref_e[5] = waypoints_yaw[-1]  # terminal yaw
        yref_e[6:9] = waypoints_vel[-1]  # terminal velocity
        self._acados_ocp_solver.set(self._N, "yref", yref_e)

        # Solve MPC problem
        status = self._acados_ocp_solver.solve()
        
        if status != 0:
            print(f"[Warning] MPC solver status: {status} at tick {self._tick}")
        
        # Get first control action
        u0 = self._acados_ocp_solver.get(0, "u")

        return u0

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        """Callback after each step."""
        self._tick += 1
        return self._finished

    def episode_callback(self):
        """Callback at the end of an episode."""
        self._tick = 0
        self._finished = False
        print(f"[AttitudeMPCMPPIHybrid] Episode completed. Total ticks: {self._tick}")

    def episode_reset(self):
        """Reset the controller for a new episode."""
        self._tick = 0
        self._finished = False
        self._last_planned_pos = None

    def get_planned_trajectory(self) -> NDArray[np.floating] | None:
        """Return the last planned trajectory for visualization.
        
        Returns:
            Array of shape (N, 3) with planned positions, or None.
        """
        return self._last_planned_pos
