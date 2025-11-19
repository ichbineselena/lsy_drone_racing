"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller

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
    model.name = "basic_example_mpc"
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
    # For more Information regarding Cost Function Definition in Acados:
    # https://github.com/acados/acados/blob/main/docs/problem_formulation/problem_formulation_ocp_mex.pdf
    #

    # Cost Type
    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"

    # Weights
    # State weights
    Q = np.diag(
        [
            50.0,  # pos
            50.0,  # pos
            400.0,  # pos
            3.0,  # rpy (increased to improve attitude tracking)
            3.0,  # rpy (increased to improve attitude tracking)
            3.0,  # rpy (increased to improve attitude tracking)
            10.0,  # vel
            10.0,  # vel
            10.0,  # vel
            5.0,  # drpy
            5.0,  # drpy
            5.0,  # drpy
        ]
    )
    # Input weights (reference is upright orientation and hover thrust)
    R = np.diag(
        [
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
            50.0,  # thrust
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

    # Set initial references (we will overwrite these later on to make the controller track the traj.)
    ocp.cost.yref, ocp.cost.yref_e = np.zeros((ny,)), np.zeros((ny_e,))

    # Set State Constraints (rpy < 30°)
    ocp.constraints.lbx = np.array([-0.5, -0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4, 5])

    # Set Input Constraints (rpy < 30°)
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, parameters["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, parameters["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # We have to set x0 even though we will overwrite it later on.
    ocp.constraints.x0 = np.zeros((nx))

    # Solver Options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"  # FULL_, PARTIAL_ ,_HPIPM, _QPOASES
    ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP"  # SQP, SQP_RTI
    ocp.solver_options.tol = 1e-6

    ocp.solver_options.qp_solver_cond_N = N
    ocp.solver_options.qp_solver_warm_start = 1

    ocp.solver_options.qp_solver_iter_max = 20
    ocp.solver_options.nlp_solver_max_iter = 50

    # set prediction horizon
    ocp.solver_options.tf = Tf

    acados_ocp_solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/lsy_example_mpc.json",
        verbose=verbose,
        build=True,
        generate=True,
    )

    return acados_ocp_solver, ocp


class AttitudeMPC(Controller):
    """Example of a MPC using the collective thrust and attitude interface."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 25
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Trajectory generation now happens online based on the next gate.
        # No pre-defined waypoints are stored.

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx

        self._tick = 0
        self._tick_max = None
        self._config = config
        self._finished = False
        # Obstacle avoidance parameters (reference shaping)
        self._avoid_influence_radius = 0.6
        self._avoid_safe_radius = 0.25
        self._avoid_step = 0.15
        # Gate approach shaping
        self._gate_d_before = 0.8
        self._gate_d_after = 0.8
        self._ref_max_speed = 0.85  # m/s cap for reference to avoid overshoot
        # Funnel corridor: start (wider) -> end (tighter) across last points
        self._gate_corridor_y = 0.30  # start lateral half-width (m)
        self._gate_corridor_z = 0.30  # start vertical half-height (m)
        self._gate_corridor_y_end = 0.12  # end lateral half-width (m)
        self._gate_corridor_z_end = 0.12  # end vertical half-height (m)
        self._gate_corridor_pts = 8   # number of last horizon points to clamp
        # Soft centering along approach (fractional shrink of lateral/vertical offsets)
        self._soft_center_gain = 0.6

    def _get_gate_pose_from_config(self, idx: int) -> tuple[np.ndarray, float]:
        gate = self._config.env.track.gates[idx]
        pos = np.array(gate.pos, dtype=float)
        yaw = float(gate.rpy[2]) if hasattr(gate, "rpy") else 0.0
        return pos, yaw

    def _compute_horizon_refs(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_pos = np.asarray(obs["pos"], dtype=float)
        tgt_idx = int(obs.get("target_gate", 0)) if isinstance(obs, dict) else 0

        if tgt_idx == -1:
            tgt_idx = len(self._config.env.track.gates) - 1

        gate_pos, gate_yaw = self._get_gate_pose_from_config(tgt_idx)

        n = np.array([np.cos(gate_yaw), np.sin(gate_yaw), 0.0])
        d_before = float(self._gate_d_before)
        d_after = float(self._gate_d_after)
        p1 = gate_pos - d_before * n
        p2 = gate_pos + d_after * n

        pos_ref = np.zeros((self._N, 3))
        N1 = max(1, self._N // 2)
        N2 = self._N - N1

        for k in range(self._N):
            if k < N1:
                s = k / max(1, N1 - 1)
                pos_ref[k] = x_pos + s * (p1 - x_pos)
            else:
                s = (k - N1) / max(1, N2 - 1)
                pos_ref[k] = p1 + s * (p2 - p1)

        # Apply obstacle avoidance shaping
        pos_ref = self._avoid_obstacles(pos_ref)

        # Limit reference step size to cap speed (helps avoid gate frame impacts)
        max_step = self._ref_max_speed * self._dt
        for k in range(1, self._N):
            dpos = pos_ref[k] - pos_ref[k - 1]
            norm = np.linalg.norm(dpos)
            if norm > max_step and norm > 1e-8:
                pos_ref[k] = pos_ref[k - 1] + dpos * (max_step / norm)

        # Softly pull the entire horizon toward the gate centerline (n-direction)
        t_lat = np.array([-np.sin(gate_yaw), np.cos(gate_yaw), 0.0])
        for k in range(self._N):
            if self._N <= 1:
                continue
            alpha = self._soft_center_gain * (k / (self._N - 1))
            if alpha <= 0:
                continue
            p = pos_ref[k]
            delta = p - gate_pos
            x_local = float(np.dot(delta, n))
            y_local = float(np.dot(delta, t_lat))
            z_local = float(delta[2])
            y_local *= max(0.0, 1.0 - alpha)
            z_local *= max(0.0, 1.0 - alpha)
            pos_ref[k] = gate_pos + x_local * n + y_local * t_lat + np.array([0.0, 0.0, z_local])

        # Clamp last few points to a safe corridor inside the gate opening (funnel)
        L = min(self._gate_corridor_pts, self._N)
        # Lateral unit vector in plane (rotate normal by +90deg about z)
        t_lat = np.array([-np.sin(gate_yaw), np.cos(gate_yaw), 0.0])
        for k in range(self._N - L, self._N):
            # 0 -> near start of clamped segment, 1 -> at very end
            frac = 0.0 if L <= 1 else (k - (self._N - L)) / (L - 1)
            w_y = (1.0 - frac) * self._gate_corridor_y + frac * self._gate_corridor_y_end
            w_z = (1.0 - frac) * self._gate_corridor_z + frac * self._gate_corridor_z_end
            p = pos_ref[k]
            # Components relative to gate center
            delta = p - gate_pos
            y_local = float(np.dot(delta, t_lat))
            z_local = float(delta[2])
            # Clamp inside corridor
            y_local = np.clip(y_local, -w_y, w_y)
            z_local = np.clip(z_local, -w_z, w_z)
            # Reconstruct point keeping longitudinal component along gate normal
            x_local = float(np.dot(delta, n))
            pos_ref[k] = gate_pos + x_local * n + y_local * t_lat + np.array([0.0, 0.0, z_local])

        vel_ref = np.zeros_like(pos_ref)
        if self._N > 1:
            vel_ref[:-1] = (pos_ref[1:] - pos_ref[:-1]) / self._dt
            vel_ref[-1] = vel_ref[-2]

        yaw_ref = np.zeros((self._N,))
        for k in range(self._N):
            v = vel_ref[k]
            if np.linalg.norm(v[:2]) > 1e-3:
                yaw_ref[k] = np.arctan2(v[1], v[0])
            else:
                yaw_ref[k] = 0.0
        # Lock yaw to gate orientation over last L points to stabilize entry
        for k in range(self._N - L, self._N):
            yaw_ref[k] = gate_yaw

        return pos_ref, vel_ref, yaw_ref

    def _avoid_obstacles(self, pos_ref: np.ndarray) -> np.ndarray:
        obstacles_cfg = getattr(self._config.env.track, "obstacles", [])
        if not obstacles_cfg:
            return pos_ref
        obstacles = [np.array(o.pos, dtype=float) for o in obstacles_cfg]
        for k in range(len(pos_ref)):
            p = pos_ref[k]
            grad = np.zeros(3)
            for c in obstacles:
                diff = p - c
                d = np.linalg.norm(diff)
                if d < 1e-6:
                    continue
                if d < self._avoid_influence_radius:
                    w = (1.0 / (d + 1e-3) - 1.0 / self._avoid_influence_radius)
                    if w > 0:
                        grad += w * diff / d
                if d < self._avoid_safe_radius:
                    grad += (self._avoid_safe_radius - d) * diff / d
            if np.linalg.norm(grad) > 0:
                pos_ref[k] = p + self._avoid_step * grad
        return pos_ref

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The orientation as roll, pitch, yaw angles, and the collective thrust [r_des, p_des, y_des, t_des] as a numpy array.
        """
        i = self._tick

        # Setting initial state
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])
        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)

        # Setting state reference (online gate-aware generation)
        yref = np.zeros((self._N, self._ny))
        pos_ref, vel_ref, yaw_ref = self._compute_horizon_refs(obs, info)
        yref[:, 0:3] = pos_ref
        yref[:, 5] = yaw_ref
        yref[:, 6:9] = vel_ref

        # Setting input reference (index > self._nx): zero rpy, hover thrust
        thrust_idx = self._nx + self._nu - 1
        yref[:, thrust_idx] = (
            self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        )
        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        yref_e[0:3] = pos_ref[-1]
        yref_e[5] = yaw_ref[-1]
        yref_e[6:9] = vel_ref[-1]
        self._acados_ocp_solver.set(self._N, "yref_e", yref_e)

        # Solving problem and getting first input
        status = self._acados_ocp_solver.solve()
        if status != 0:
            u0 = np.array(
                [
                    0.0,
                    0.0,
                    0.0,
                    self.drone_params["mass"]
                    * -self.drone_params["gravity_vec"][-1],
                ],
                dtype=float,
            )
        else:
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
        """Increment the tick counter."""
        self._tick += 1

        return self._finished

    def episode_callback(self):
        """Reset the integral error."""
        self._tick = 0
