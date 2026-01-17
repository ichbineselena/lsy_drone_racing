"""Advanced Hybrid MPPI + MPC controller with full gate and obstacle awareness.

This is the production-ready version that:
- Uses MPPIBuilderAdvanced for intelligent trajectory generation
- Handles gates sequentially
- Avoids obstacles dynamically
- Tracks generated trajectories with MPC
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
from lsy_drone_racing.control.trajectory_builders import MPPIBuilderAdvanced

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

    model = AcadosModel()
    model.name = "mppi_mpc_hybrid_advanced"
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
    ocp.model = create_acados_model(parameters)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N

    # Cost weights optimized for tracking MPPI trajectories
    Q = np.diag([
        80.0, 80.0, 500.0,  # position (high z-weight)
        1.0, 1.0, 2.0,      # orientation
        15.0, 15.0, 15.0,   # velocity
        5.0, 5.0, 5.0       # angular velocity
    ])
    R = np.diag([1.0, 1.0, 1.0, 40.0])  # control inputs

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q.copy()

    Vx = np.zeros((ny, nx))
    Vx[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx:nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    Vx_e = np.zeros((ny_e, nx))
    Vx_e[0:nx, 0:nx] = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    ocp.cost.yref = np.zeros((ny,))
    ocp.cost.yref_e = np.zeros((ny_e,))

    # Constraints
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    ocp.constraints.x0 = np.zeros((nx))

    # Solver options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1
    ocp.solver_options.qp_solver_iter_max = 50  # Increased from 20 for better convergence
    ocp.solver_options.nlp_solver_max_iter = 100  # Increased from 50 to handle complex trajectories
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/mppi_mpc_hybrid_advanced.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class AttitudeMPCMPPIHybridAdvanced(Controller):
    """Advanced hybrid controller with full gate/obstacle awareness."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the advanced hybrid controller."""
        super().__init__(obs, info, config)
        
        # MPC parameters
        self._N = 25
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

        # Extract gates from config
        gates = []
        track_config = config.env.track if hasattr(config.env, 'track') else config['env']['track']
        gates_config = track_config.gates if hasattr(track_config, 'gates') else track_config.get('gates', [])
        
        for gate_config in gates_config:
            if hasattr(gate_config, 'pos'):
                gates.append(np.array(gate_config.pos))
            elif isinstance(gate_config, dict):
                gates.append(np.array(gate_config['pos']))
        
        # Extract obstacles from config
        obstacles = []
        obstacles_config = track_config.obstacles if hasattr(track_config, 'obstacles') else track_config.get('obstacles', [])
        
        for obs_config in obstacles_config:
            if hasattr(obs_config, 'pos'):
                obstacles.append(np.array(obs_config.pos))
            elif isinstance(obs_config, dict):
                obstacles.append(np.array(obs_config['pos']))
        
        # MPPI parameters (can be tuned)
        mppi_K = 2000  # Number of samples
        mppi_lambda = 0.8  # Temperature
        mppi_sigma = 0.4  # Control noise
        
        # Initialize advanced MPPI trajectory builder
        self._trajectory_builder = MPPIBuilderAdvanced(
            gates=gates,
            obstacles=obstacles,
            K=mppi_K,
            lambda_=mppi_lambda,
            sigma_u=mppi_sigma,
            gate_radius=0.45,  # Tolerance for gate passage
            obstacle_radius=0.3,  # Safety margin around obstacles
        )

        self._tick = 0
        self._config = config
        self._finished = False
        self._last_planned_pos = None
        
        print(f"[AttitudeMPCMPPIHybridAdvanced] Initialized")
        print(f"  - MPC horizon: {self._N} steps ({self._T_HORIZON:.2f}s)")
        print(f"  - MPPI: K={mppi_K}, lambda={mppi_lambda}, sigma={mppi_sigma}")
        print(f"  - Gates: {len(gates)}, Obstacles: {len(obstacles)}")

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute control using advanced MPPI + MPC."""
        # Convert observation to state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        
        # Update target gate from environment observation
        target_gate_idx = int(obs.get("target_gate", 0))
        if target_gate_idx >= 0:  # -1 means all gates passed
            self._trajectory_builder.set_target_gate(target_gate_idx)
        
        # Initialize/update MPPI
        if self._tick == 0:
            self._trajectory_builder.reset(x0, t0=0.0)
        else:
            # Update MPPI's state for continuous replanning
            self._trajectory_builder.x0 = x0
        
        # Generate reference trajectory using MPPI
        current_time = self._tick * self._dt
        horizon_data = self._trajectory_builder.get_horizon(
            t_now=current_time,
            N=self._N,
            dt=self._dt
        )
        
        waypoints_pos = horizon_data["pos"]
        waypoints_vel = horizon_data["vel"]
        waypoints_yaw = horizon_data["yaw"]
        
        # Enforce minimum height on trajectory (safety constraint)
        min_safe_height = 0.2  # 20cm minimum
        waypoints_pos[:, 2] = np.maximum(waypoints_pos[:, 2], min_safe_height)
        
        self._last_planned_pos = waypoints_pos
        
        # Check completion (all gates passed)
        if self._trajectory_builder.current_gate_idx >= len(self._trajectory_builder.gates):
            # Check if close to final gate
            if len(self._trajectory_builder.gates) > 0:
                final_gate = self._trajectory_builder.gates[-1]
                dist = np.linalg.norm(obs["pos"] - final_gate)
                if dist < 0.5:
                    self._finished = True
        
        # Set MPC initial state
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Set MPC reference from MPPI trajectory
        hover_thrust = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = waypoints_pos
        yref[:, 5] = waypoints_yaw
        yref[:, 6:9] = waypoints_vel
        yref[:, 15] = hover_thrust
        
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Terminal reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = waypoints_pos[-1]
        yref_e[5] = waypoints_yaw[-1]
        yref_e[6:9] = waypoints_vel[-1]
        self._acados_ocp_solver.set(self._N, "yref", yref_e)

        # Solve MPC
        status = self._acados_ocp_solver.solve()
        if status != 0:
            print(f"[Warning] MPC solver status: {status}")
        
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
        """Callback at episode end."""
        print(f"[AttitudeMPCMPPIHybridAdvanced] Episode done. Ticks: {self._tick}")
        self._tick = 0
        self._finished = False

    def episode_reset(self):
        """Reset for new episode."""
        self._tick = 0
        self._finished = False
        self._last_planned_pos = None

    def get_planned_trajectory(self) -> NDArray[np.floating] | None:
        """Return planned trajectory for visualization."""
        return self._last_planned_pos
