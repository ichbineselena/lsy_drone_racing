"""
Attitude MPC (Low-Level) + MPPI (High-Level)
=============================================

This module implements a cascaded controller:

    High-Level MPPI -> desired acceleration + yaw rate
    Low-Level ACADOS MPC -> attitude + thrust commands

MPPI uses a 7D kinematic model and optimizes the drone's motion through
gates and obstacles online, enabling Level 3 racing (randomized layouts).
"""

from __future__ import annotations
from typing import TYPE_CHECKING, List, Tuple

import numpy as np
import scipy
import torch
import torch.nn.functional as F

from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from scipy.spatial.transform import Rotation as R

from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates

from lsy_drone_racing.control import Controller
#from lsy_drone_racing.control.highlevel_mppi import HighLevelMPPI

if TYPE_CHECKING:
    from numpy.typing import NDArray


# ================================================================
# ---------------  HIGH LEVEL MPPI CONTROLLER  -------------------
# ================================================================

class HighLevelMPPI:
    """
    High-level MPPI that outputs:
        - desired acceleration (3D)
        - desired yaw rate
    """

    def __init__(
        self,
        horizon: int = 20,
        dt: float = 0.05,
        num_samples: int = 1000,
        lambda_: float = 1.0,
        accel_noise: float = 3.0,
        yaw_rate_noise: float = 1.0,
    ):
        self.H = horizon
        self.dt = dt
        self.Ns = num_samples
        self.lambda_ = lambda_

        # Gaussian sampling covariance
        self.u_sigma = torch.tensor(
            [accel_noise, accel_noise, accel_noise, yaw_rate_noise],
            device="cpu",
            dtype=torch.float32,
        )

        # Last mean control sequence (warm start)
        self.u_mean = torch.zeros(self.H, 4, device="cpu")

        # Obstacles and gates (set from outside)
        self.obstacles: List[Tuple[np.ndarray, float]] = []
        self.gates: List[np.ndarray] = []

    # ------------------------------------------------------------
    #   Kinematic model
    # ------------------------------------------------------------
    def dynamics(self, x: torch.Tensor, u: torch.Tensor) -> torch.Tensor:
        """
        State x = [px, py, pz, vx, vy, vz, yaw]
        Control u = [ax, ay, az, yaw_rate]

        Forward Euler dynamics.
        """
        px, py, pz, vx, vy, vz, yaw = torch.chunk(x, 7, dim=-1)
        ax, ay, az, wyaw = torch.chunk(u, 4, dim=-1)

        nx = torch.cat(
            [
                px + vx * self.dt,
                py + vy * self.dt,
                pz + vz * self.dt,
                vx + ax * self.dt,
                vy + ay * self.dt,
                vz + az * self.dt,
                yaw + wyaw * self.dt,
            ],
            dim=-1,
        )
        return nx

    # ------------------------------------------------------------
    #   Cost function
    # ------------------------------------------------------------
    def cost(self, xs: torch.Tensor) -> torch.Tensor:
        """
        xs: shape [Ns, H+1, 7]
        Returns cost for each rollout.
        """
        cost = torch.zeros(xs.shape[0], device=xs.device)

        final_pos = xs[:, -1, 0:3]

        # ---- Gate costs (min distance to gate centerline) ----
        for gate in self.gates:
            gate_center = torch.tensor(gate[:3], device=xs.device)
            dist = torch.norm(xs[:, :, 0:3] - gate_center, dim=-1)
            cost += torch.sum(dist, dim=-1)

        # ---- Obstacle barrier cost ----
        for center_np, radius in self.obstacles:
            center = torch.tensor(center_np, device=xs.device)
            dist = torch.norm(xs[:, :, 0:3] - center, dim=-1)
            penalty = torch.exp(-(dist - radius)) * 50.0
            cost += torch.sum(penalty, dim=-1)

        # ---- Progress cost (encourage moving forward along x) ----
        cost += -5.0 * final_pos[:, 0]

        return cost
    # ------------------------------------------------------------
    #   MPPI update step
    # ------------------------------------------------------------
    def solve(self, x0: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Input:
            x0: initial state (numpy, shape [7])
        Output:
            (a_des(3), yaw_rate_des)
        """
        device = self.u_mean.device

        x0_t = torch.tensor(x0, device=device).float().unsqueeze(0)  # [1,7]

        # Sample controls: [Ns, H, 4]
        noise = torch.randn(self.Ns, self.H, 4, device=device) * self.u_sigma
        U = self.u_mean + noise

        # Roll out dynamics
        X = torch.zeros(self.Ns, self.H + 1, 7, device=device)
        X[:, 0] = x0_t

        for t in range(self.H):
            X[:, t + 1] = self.dynamics(X[:, t], U[:, t])

        # Compute costs
        S = self.cost(X)  # [Ns]

        # Softmax weights
        beta = torch.min(S)
        weights = torch.softmax(-(S - beta) / self.lambda_, dim=0)

        # Weighted averaging of controls
        U_mean_new = torch.sum(weights[:, None, None] * U, dim=0)
        self.u_mean = U_mean_new.detach()

        # Output only first action
        u0 = self.u_mean[0].cpu().numpy()

        a_des = u0[:3]
        yaw_rate_des = u0[3]
        return a_des, yaw_rate_des


# ================================================================
# --------------------  ACADOS MPC SETUP  ------------------------
# ================================================================

def create_acados_model(parameters: dict) -> AcadosModel:
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
    model.name = "mppi_lowlevel_mpc"
    model.f_expl_expr = X_dot
    model.f_impl_expr = None
    model.x = X
    model.u = U
    return model


def create_ocp_solver(Tf: float, N: int, params: dict) -> Tuple[AcadosOcpSolver, AcadosOcp]:
    ocp = AcadosOcp()
    ocp.model = create_acados_model(params)

    nx = ocp.model.x.rows()
    nu = ocp.model.u.rows()
    ny = nx + nu
    ny_e = nx

    ocp.solver_options.N_horizon = N

    # Weights
    Q = np.diag(
        [40, 40, 200, 1, 1, 1, 8, 8, 8, 4, 4, 4]
    )
    R = np.diag([1, 1, 1, 40])
    Q_e = Q.copy()

    ocp.cost.cost_type = "LINEAR_LS"
    ocp.cost.cost_type_e = "LINEAR_LS"
    ocp.cost.W = scipy.linalg.block_diag(Q, R)
    ocp.cost.W_e = Q_e

    Vx = np.zeros((ny, nx))
    Vx[:nx, :nx] = np.eye(nx)
    ocp.cost.Vx = Vx

    Vu = np.zeros((ny, nu))
    Vu[nx:nx + nu, :] = np.eye(nu)
    ocp.cost.Vu = Vu

    Vx_e = np.eye(nx)
    ocp.cost.Vx_e = Vx_e

    ocp.cost.yref = np.zeros(ny)
    ocp.cost.yref_e = np.zeros(ny_e)

    # State constraints (roll/pitch)
    ocp.constraints.lbx = np.array([-0.5, -0.5])
    ocp.constraints.ubx = np.array([0.5, 0.5])
    ocp.constraints.idxbx = np.array([3, 4])

    # Input constraints
    ocp.constraints.lbu = np.array([-0.5, -0.5, -0.5, params["thrust_min"] * 4])
    ocp.constraints.ubu = np.array([0.5, 0.5, 0.5, params["thrust_max"] * 4])
    ocp.constraints.idxbu = np.array([0, 1, 2, 3])

    # Solver options
    ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
    ocp.solver_options.integrator_type = "ERK"
    ocp.solver_options.nlp_solver_type = "SQP_RTI"
    ocp.solver_options.tol = 1e-6
    ocp.solver_options.tf = Tf

    solver = AcadosOcpSolver(
        ocp,
        json_file="c_generated_code/mppi_lowlevel_mpc.json",
        build=True,
        generate=True,
    )
    return solver, ocp


# ================================================================
# ---------------------  CONTROLLER CLASS  -----------------------
# ================================================================

class AttitudeMPC(Controller):
    """
    Combined MPPI + MPC controller.
    High-level MPPI produces desired accelerations.
    MPC tracks them with a full dynamic quadrotor model.
    """

    def init(self, obs, info, config):
        super().init(obs, info, config)

        # Timings
        self._N = 25
        self._dt = 1.0 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # Load drone parameters from model
        self.drone_params = load_params("so_rpy", config.sim.drone_model)

        # Create solver
        self._solver, self._ocp = create_ocp_solver(self._T_HORIZON, self._N, self.drone_params)
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()

        # ---- HIGH-LEVEL MPPI ----
        self.mppi = HighLevelMPPI(horizon=20, dt=self._dt)
        self.mppi.obstacles = config.env.obstacles  # [(center, radius), ...]
        self.mppi.gates = config.env.gates  # each = center (x,y,z)

        self._tick = 0

    # ------------------------------------------------------------
    #   Main control method
    # ------------------------------------------------------------
    def compute_control(self, obs, info=None):
        # ---- Construct full 7D MPPI state ----
        rpy = R.from_quat(obs["quat"]).as_euler("xyz")
        drpy = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])

        x_mppi = np.hstack([
            obs["pos"],
            obs["vel"],
            rpy[2],  # yaw only
        ])

        # ---- Run MPPI ----
        a_des, yaw_rate_des = self.mppi.solve(x_mppi)

        # ---- Convert MPPI output to tracking references for MPC ----

        # Position update from a_des integration (simple heuristic)
        pos_ref = obs["pos"] + obs["vel"] * self._dt + 0.5 * a_des * self._dt**2
        vel_ref = obs["vel"] + a_des * self._dt

        yaw_ref = rpy[2] + yaw_rate_des * self._dt

        # ---- Set initial state in ACADOS ----
        full_state = np.hstack([obs["pos"], rpy, obs["vel"], drpy])
        self._solver.set(0, "lbx", full_state)
        self._solver.set(0, "ubx", full_state)

        # ---- Fill references for MPC horizon ----
        for i in range(self._N):
            yref = np.zeros(self._nx + self._nu)

            # State references
            yref[0:3] = pos_ref
            yref[3:6] = [0, 0, yaw_ref]   # desired r,p,y
            yref[6:9] = vel_ref
            # yaw rates remain 0 except yaw
            # Input reference: hover thrust
            yref[self._nx + 3] = (
                self.drone_params["mass"] * -self.drone_params["gravity_vec"][2]
            )

            self._solver.set(i, "yref", yref)

        # Terminal reference
        yref_e = np.zeros(self._nx)
        yref_e[0:3] = pos_ref
        yref_e[5] = yaw_ref
        yref_e[6:9] = vel_ref
        self._solver.set(self._N, "y_ref", yref_e)

        # ---- Solve MPC ----
        self._solver.solve()
        u0 = self._solver.get(0, "u")
        return u0

    def step_callback(self, action, obs, reward, terminated, truncated, info):
        self._tick += 1
        return False

    def episode_callback(self):
        self._tick = 0

