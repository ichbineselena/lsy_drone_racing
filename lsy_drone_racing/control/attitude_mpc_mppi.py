"""This module implements an example MPC using attitude control for a quadrotor.

It utilizes the collective thrust interface for drone control to compute control commands based on
current state observations and desired waypoints.

The waypoints are generated using cubic spline interpolation from a set of predefined waypoints.
Note that the trajectory uses pre-defined waypoints instead of dynamically generating a good path.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING, Optional

import numpy as np
import scipy
from acados_template import AcadosModel, AcadosOcp, AcadosOcpSolver
from drone_models.core import load_params
from drone_models.so_rpy import symbolic_dynamics_euler
from drone_models.utils.rotation import ang_vel2rpy_rates
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
import torch
# from pytorch_mppi import MPPI, smooth_mppi as smppi
import pytorch_mppi
from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.trajectory_builders import SplineBuilder, MPPIBuilder, TrajectoryBuilder

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
            1.0,  # rpy
            1.0,  # rpy
            1.0,  # rpy
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
            1.0,  # thrust 50
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

    def __init__(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
        trajectory_builder: Optional[TrajectoryBuilder] = None,
    ):
        """Initialize the attitude controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)
        self._N = 25  # MPC horizon steps
        self._dt = 1 / config.env.freq
        self._T_HORIZON = self._N * self._dt

        # decide whether to use pytorch-mppi as a high-level planner
        tb_cfg_val = None
        try:
            tb_cfg = config.controller if hasattr(config, "controller") else None
            if tb_cfg is not None:
                try:
                    tb_cfg_val = tb_cfg.get("trajectory_builder", None)
                except Exception:
                    tb_cfg_val = getattr(tb_cfg, "trajectory_builder", None)
        except Exception:
            tb_cfg_val = None

        use_pytorch_mppi = False
        tb_choice = trajectory_builder if trajectory_builder is not None else tb_cfg_val
        for gate in config.env.track.gates:
            self.gate_pos = np.array(gate["pos"], dtype=float)
            self.gate_rpy = np.array(gate["rpy"], dtype=float)
            rot = R.from_euler("xyz", self.gate_rpy)
            self.gate_normal = rot.as_matrix()[:, 0]  # x-axis points through gate

        if isinstance(tb_choice, str) and tb_choice.lower() in ("pytorch_mppi", "pytorch-mppi", "pytorchmppi", "mppi"):
            use_pytorch_mppi = True
            self.prev_goal = obs["pos"]


        self._use_pytorch_mppi = use_pytorch_mppi
        if self._use_pytorch_mppi:
            # create a simple MPPI using a pos/vel kinematic model (6D state, 3D accel control)
            self.goal = obs["gates_pos"][obs["target_gate"]]
            self.gates = obs["gates_pos"]
            self.mppi_dt = 0.05 #self._dt
            self.target_gate_idx = int(obs["target_gate"])

            def gate_drag_cost(pos, gate_center, gate_normal,
                k_normal=1.0, k_side=1.0, gamma=2.0):
                # Vector from gate center to drone
                d = pos - gate_center[None, :]

                # Decompose
                d_normal = torch.sum(d * gate_normal, dim=-1, keepdim=True)
                d_side_vec = d - d_normal * gate_normal
                d_side = torch.norm(d_side_vec, dim=-1, keepdim=True)

                # Curved bowl-shaped attraction
                cost_normal = k_normal * (d_normal ** 2)
                cost_side   = k_side   * (d_side ** gamma)

                return cost_normal + cost_side


            def dynamics(x, u):
                px, py, pz, vx, vy, vz = torch.split(x, 1, dim=-1)
                ax, ay, az = torch.split(u, 1, dim=-1)

                px_next = px + vx * self.mppi_dt
                py_next = py + vy * self.mppi_dt
                pz_next = pz + vz * self.mppi_dt

                vx_next = vx + ax * self.mppi_dt
                vy_next = vy + ay * self.mppi_dt
                vz_next = vz + az * self.mppi_dt

                return torch.cat([px_next, py_next, pz_next, vx_next, vy_next, vz_next], dim=-1)

            # def running_cost(x, u):
            #     goal_t = torch.tensor(self.goal, device=x.device, dtype=x.dtype)
            #     pos = x[..., :3]
            #     vel = x[..., 3:6]
            #     # gate quaternion → rotation matrix → normal
            #     gate_quat = torch.tensor(obs["gates_quat"][self.target_gate_idx],
            #                             device=x.device, dtype=x.dtype)

            #     rot = R.from_quat(gate_quat.cpu().numpy())
            #     gate_normal_np = rot.as_matrix()[:, 0]     # x-axis is forward direction
            #     gate_normal = torch.tensor(gate_normal_np, device=x.device, dtype=x.dtype)

            #     gate_normal = gate_normal / torch.norm(gate_normal)

            #     # gate center
            #     gate_center = torch.tensor(obs["gates_pos"][self.target_gate_idx],
            #                             device=x.device, dtype=x.dtype)

            #     # Build orthonormal basis: n, t1, t2
            #     n = gate_normal
            #     # pick arbitrary vector not parallel to n
            #     tmp = torch.tensor([1.,0.,0.], device=x.device, dtype=x.dtype)
            #     if torch.abs(torch.dot(tmp,n)) > 0.8:
            #         tmp = torch.tensor([0.,1.,0.], device=x.device, dtype=x.dtype)

            #     t1 = torch.cross(n, tmp)
            #     t1 = t1 / torch.norm(t1)
            #     t2 = torch.cross(n, t1)
            #     t2 = t2 / torch.norm(t2)

            #     d = pos - gate_center

            #     # axis projections
            #     proj_n  = torch.sum(d * n, dim=-1)
            #     proj_t1 = torch.sum(d * t1, dim=-1)
            #     proj_t2 = torch.sum(d * t2, dim=-1)

            #     # ellipse radii
            #     a = torch.tensor(1.2, device=x.device)   # long axis → through the gate
            #     b = torch.tensor(0.15, device=x.device)   # narrow axis → the gate's plane

            #     # elliptical distance
            #     ellipse_dist = (proj_n / a)**2 + (proj_t1 / b)**2 + (proj_t2 / b)**2

            #     W_ellipse = 40.0   # tune weight

            #     c_ellipse = - W_ellipse / (ellipse_dist)

            #     #c_ellipse = - W_ellipse / ellipse_dist
            #     # --- 1. Position error ---
            #     # Weight z (altitude) more heavily to ensure takeoff
            #     W_pos = torch.tensor([10.0, 10.0, 50.0], device=x.device, dtype=x.dtype)
            #     c_pos = torch.sum(W_pos * (pos - goal_t) ** 2, dim=-1)

            #     # --- 2. Velocity penalty ---
            #     # Encourage reaching goal with moderate speed
            #     W_vel = torch.tensor([0.01, 0.01, 0.01], device=x.device, dtype=x.dtype)
            #     c_vel = torch.sum(W_vel * vel ** 2, dim=-1)

            #     # --- 3. Control effort penalty ---
            #     # Penalize very aggressive control inputs
            #     W_u = torch.tensor([0.01] * u.shape[-1], device=x.device, dtype=x.dtype)
            #     c_u = torch.sum(W_u * u ** 2, dim=-1)


            #     # --- Total cost ---
            #     total_cost = c_pos + c_vel + c_u + c_ellipse
            #     #print(f"[DEBUG] MPPI running cost components: pos {c_pos.mean().item():.3f}, ")

            #     return total_cost

            def running_cost(x, u):
                """
                MPPI running cost using gate progress:
                    J = ||p_{k+1} - p_gate||^2 - ||p_k - p_gate||^2
                x: [batch, state_dim]
                u: [batch, control_dim]
                dynamics_func: a function (x, u, dt) → x_next
                """

                goal_t = torch.tensor(self.goal, device=x.device, dtype=x.dtype)
                pos = x[..., :3]
                vel = x[..., 3:6]

                # Extract p_k
                p_k = x[..., :3]

                # Compute x_{k+1} using system dynamics
                x_k1 = self.mppi._dynamics(x, u, self.mppi_dt)

                # Extract p_{k+1}
                p_k1 = x_k1[..., :3]

                # Make gate_pos a proper tensor
                p_gate = goal_t.to(x.device).to(x.dtype)

                # ---- Gate-progress term (core) ----
                dist_k  = torch.sum((p_k  - p_gate)**2, dim=-1)
                dist_k1 = torch.sum((p_k1 - p_gate)**2, dim=-1)

                # Negative when moving closer → good
                c_gate = dist_k1 - dist_k

                # ---- Optional smoothing penalties ----

                # Velocity penalty
                vel = x[..., 3:6]
                W_vel = torch.tensor([0.01, 0.01, 0.01],
                                    device=x.device, dtype=x.dtype)
                c_vel = torch.sum(W_vel * vel**2, dim=-1)

                # Control effort penalty
                W_u = torch.tensor([0.001] * u.shape[-1],
                                device=x.device, dtype=x.dtype)
                c_u = torch.sum(W_u * u**2, dim=-1)

                # Weight z (altitude) more heavily to ensure takeoff
                W_pos = torch.tensor([10.0, 10.0, 20.0], device=x.device, dtype=x.dtype)
                c_pos = torch.sum(W_pos * (pos - goal_t) ** 2, dim=-1)

                 # gate quaternion → rotation matrix → normal
                gate_quat = torch.tensor(obs["gates_quat"][self.target_gate_idx],
                                        device=x.device, dtype=x.dtype)

                rot = R.from_quat(gate_quat.cpu().numpy())
                gate_normal_np = rot.as_matrix()[:, 0]     # x-axis is forward direction
                gate_normal = torch.tensor(gate_normal_np, device=x.device, dtype=x.dtype)

                gate_normal = gate_normal / torch.norm(gate_normal)

                # gate center
                gate_center = torch.tensor(obs["gates_pos"][self.target_gate_idx],
                                        device=x.device, dtype=x.dtype)

                # Build orthonormal basis: n, t1, t2
                n = gate_normal
                # pick arbitrary vector not parallel to n
                tmp = torch.tensor([1.,0.,0.], device=x.device, dtype=x.dtype)
                if torch.abs(torch.dot(tmp,n)) > 0.8:
                    tmp = torch.tensor([0.,1.,0.], device=x.device, dtype=x.dtype)

                t1 = torch.cross(n, tmp)
                t1 = t1 / torch.norm(t1)
                t2 = torch.cross(n, t1)
                t2 = t2 / torch.norm(t2)

                d = pos - gate_center

                # axis projections
                proj_n  = torch.sum(d * n, dim=-1)
                proj_t1 = torch.sum(d * t1, dim=-1)
                proj_t2 = torch.sum(d * t2, dim=-1)

                # ellipse radii
                a = torch.tensor(1.2, device=x.device)   # long axis → through the gate
                b = torch.tensor(0.15, device=x.device)   # narrow axis → the gate's plane

                # elliptical distance
                ellipse_dist = (proj_n / a)**2 + (proj_t1 / b)**2 + (proj_t2 / b)**2

                W_ellipse = 20.0   # tune weight

                c_ellipse = - W_ellipse / (ellipse_dist)
                print(f"[DEBUG] MPPI cost components: gate {15*c_gate.mean().item():.3f}, pos {c_pos.mean().item():.3f}, ellipse {c_ellipse.mean().item():.3f}")

                return 15*c_gate + c_vel + c_u + c_pos + c_ellipse


            H = 30 #self._N
            K = 5000# samples; tune for performance
            noise_sigma_matrix = 0.3 * torch.eye(3)

            # instantiate MPPI solver
            # try:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            noise_sigma = torch.tensor(
                noise_sigma_matrix,
                dtype=torch.double,
                device=device
            )
            
            self.mppi = pytorch_mppi.KMPPI(
                dynamics=dynamics,
                running_cost=running_cost,
                nx=6,
                noise_sigma=noise_sigma,
                num_samples=K,
                horizon=H,
                lambda_=1.0,
                u_min=torch.tensor([-2.0], dtype=torch.double, device=device), #[-0.5, -0.5, -0.5]
                u_max=torch.tensor([2.0], dtype=torch.double, device=device), #[0.5, 0.5, 0.5]
                device=device,
            )
            print("[AttitudeMPC] Using PyTorch MPPI as high-level trajectory planner.")

        MAX_STEPS_PER_SEGMENT = 1000  # to avoid issues if trajectory builder has no limit
        self._trajectory_builder = trajectory_builder

        self.drone_params = load_params("so_rpy", config.sim.drone_model)
        self._acados_ocp_solver, self._ocp = create_ocp_solver(
            self._T_HORIZON, self._N, self.drone_params
        )
        self._nx = self._ocp.model.x.rows()
        self._nu = self._ocp.model.u.rows()
        self._ny = self._nx + self._nu
        self._ny_e = self._nx
        self._tick = 0
        self._tick_max = MAX_STEPS_PER_SEGMENT - 1 - self._N
        self._config = config
        self._finished = False

    def _warmstart_cubic_spline(self, start_pos, start_vel, goal_pos, horizon, dt):
            """
            Build a cubic spline warm-start trajectory from current state to next gate.
            Returns pos_traj (H×3), vel_traj (H×3)
            """
            t = np.linspace(0, horizon * dt, horizon + 1)

            # Boundary conditions: match current position & velocity
            bc_pos = np.array(start_pos)
            bc_vel = np.array(start_vel)

            # Create simple straight-line intermediate points
            # (they only shape the spline; MPPI will refine)
            # mid1 = bc_pos + 0.33 * (goal_pos - bc_pos) 
            # mid2 = bc_pos + 0.5 * (goal_pos - bc_pos) 
            # mid3 = bc_pos + 0.66 * (goal_pos - bc_pos)

            xs = np.vstack([bc_pos, goal_pos]) #mid1, mid2, mid3,

            # Fit 3 independent cubic splines for x,y,z
            cs_x = CubicSpline([0, 1], xs[:, 0], bc_type=((1, bc_vel[0]), (1, 0.0)))
            cs_y = CubicSpline([0, 1], xs[:, 1], bc_type=((1, bc_vel[1]), (1, 0.0)))
            cs_z = CubicSpline([0, 1], xs[:, 2], bc_type=((1, bc_vel[2]), (1, 0.0)))

            # Evaluate over prediction horizon
            pos_traj = np.stack([cs_x(t), cs_y(t), cs_z(t)], axis=1)
            vel_traj = np.stack([cs_x(t, 1), cs_y(t, 1), cs_z(t, 1)], axis=1)

            return pos_traj, vel_traj

    def compute_control(
    self,
    obs: dict[str, NDArray[np.floating]],
    info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired collective thrust and roll/pitch/yaw of the drone."""

        i = min(self._tick, self._tick_max)
        if self._tick >= self._tick_max:
            self._finished = True

        # ------------------------
        # Setting initial state
        # ------------------------
        obs["rpy"] = R.from_quat(obs["quat"]).as_euler("xyz")
        obs["drpy"] = ang_vel2rpy_rates(obs["quat"], obs["ang_vel"])

        x0 = np.concatenate((obs["pos"], obs["rpy"], obs["vel"], obs["drpy"]))
        self._acados_ocp_solver.set(0, "lbx", x0)
        self._acados_ocp_solver.set(0, "ubx", x0)


        # ------------------------
        # Query trajectory builder
        # ------------------------
        t_now = self._tick * self._dt

        if getattr(self, "_use_pytorch_mppi", False):
            # ======================================================
            # 1. Determine goal from observation (gate detection)
            # ======================================================
            target_gate_idx = int(obs["target_gate"])
            self.goal = obs["gates_pos"][target_gate_idx]
            if self.goal[0] != self.prev_goal[0]:
                print(f"[DEBUG] MPPI goal updated to gate index {target_gate_idx} "
                    f"at position {self.goal}")
                
                # small forward offset through the gate
                gate_quat = obs["gates_quat"][target_gate_idx]
                rot = R.from_quat(gate_quat)
                forward = rot.as_matrix()[:, 0]
                goal_ws = self.goal + 0.1 * forward    # warm-start goal

                #self.goal = goal_ws

                # build warm-start from (p,v) → goal_ws
                start_pos = obs["pos"]
                start_vel = obs["vel"]
                pos_ws, vel_ws = self._warmstart_cubic_spline(
                    start_pos=start_pos,
                    start_vel=start_vel,
                    goal_pos=goal_ws,
                    horizon=15,#self.mppi.T,
                    dt=self.mppi_dt,
                )
                self._last_warmstart_pos = pos_ws 

                # feed warm-start to MPPI: force its initial means
                # self.mppi.reset()
                # self.mppi.U[:, :] = 0.0                           # zero action warm-start
                self.mppi.X = torch.tensor(
                    np.hstack([pos_ws, vel_ws]),
                    dtype=torch.double, device=self.mppi.d
                )
                self.prev_goal = self.goal

            # ======================================================
            # 2. Build MPPI input state: (px,py,pz,vx,vy,vz)
            # ======================================================
            x0 = torch.tensor(
                np.concatenate((obs["pos"], obs["vel"])),
                dtype=torch.double,
                device=self.mppi.d
            )

            # ======================================================
            # 3. Run MPPI once to update optimal control sequence
            # ======================================================
            with torch.no_grad():
                u0 = self.mppi.command(x0)  # first optimal action
                print("Opt u0: ", u0)

            # ======================================================
            # 4. Roll out predicted future states
            # ======================================================
            u_seq = self.mppi.U
            # print(f"[DEBUG] MPPI u_seq (first few): {u_seq[:3].cpu().numpy()}")

            x = x0.clone()
            xs = [x.cpu().numpy()]
            for k in range(min(self._N, self.mppi.T)):
                u_k = u_seq[k]
                x = self.mppi._dynamics(x, u_k, k)
                xs.append(x.cpu().numpy())

            x_traj_np = np.stack(xs, axis=0)  # (N+1, nx)

            # ======================================================
            # 5. Extract planned pos/vel/yaw for MPC reference
            # ======================================================
            pos_plan = x_traj_np[1:, 0:3]
            vel_plan = x_traj_np[1:, 3:6]
            yaw_plan = np.zeros((pos_plan.shape[0],))

            ref = {
                "pos": pos_plan,
                "vel": vel_plan,
                "yaw": yaw_plan,
            }

            self._last_planned_pos = pos_plan[:]
            print(f"[DEBUG] MPPI planned pos (first few): {pos_plan[:3]}")

        else:
            # ======================================================
            # No MPPI → use fallback trajectory builder
            # ======================================================
            ref = self._trajectory_builder.get_horizon(t_now, self._N, self._dt)

            try:
                self._last_planned_pos = np.asarray(ref.get("pos", ref["pos"]))
            except Exception:
                self._last_planned_pos = None

        # ==========================================================
        # Build yref for ACADOS MPC
        # ==========================================================
        yref = np.zeros((self._N, self._ny))
        yref[:, 0:3] = ref["pos"]      # position
        yref[:, 5] = ref["yaw"]       # yaw
        yref[:, 6:9] = ref["vel"]     # velocity


        # Map MPPI velocity plan -> per-step thrust reference using finite differences
        try:
            vel_plan = np.asarray(ref.get("vel", ref["vel"]))
            if vel_plan.ndim == 2 and vel_plan.shape[0] == self._N and vel_plan.shape[1] >= 3:
                a_z = np.zeros((self._N,))
                curr_vz = float(obs["vel"][2])
                a_z[0] = (float(vel_plan[0, 2]) - curr_vz) / max(self._dt, 1e-8)
                for k in range(1, self._N):
                    a_z[k] = (float(vel_plan[k, 2]) - float(vel_plan[k - 1, 2])) / max(self._dt, 1e-8)

                mass = float(self.drone_params.get("mass", 1.0))
                g_z = float(self.drone_params.get("gravity_vec", [0.0, 0.0, -9.81])[-1])
                thrusts = mass * (-g_z + a_z)

                tmin = self.drone_params.get("thrust_min", None)
                tmax = self.drone_params.get("thrust_max", None)
                if tmin is not None and tmax is not None:
                    thrusts = np.clip(thrusts, tmin * 4, tmax * 4)

                yref[:, 15] = thrusts
            else:
                yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
        except Exception:
            yref[:, 15] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]

        for j in range(self._N):
            self._acados_ocp_solver.set(j, "yref", yref[j])

        # Setting final state reference
        yref_e = np.zeros((self._ny_e))
        t_terminal = t_now + self._N * self._dt
        if getattr(self, "_use_pytorch_mppi", False) and 'x_traj_np' in locals():
            ref_e_pos = x_traj_np[min(self._N, x_traj_np.shape[0]-1), 0:3]
            ref_e_vel = x_traj_np[min(self._N, x_traj_np.shape[0]-1), 3:6]
            ref_e_yaw = 0.0
            yref_e[0:3] = ref_e_pos
            yref_e[5] = ref_e_yaw
            yref_e[6:9] = ref_e_vel
        else:
            ref_e = self._trajectory_builder.get_horizon(t_terminal, 1, self._dt)
            yref_e[0:3] = ref_e["pos"][0]
            yref_e[5] = ref_e["yaw"][0]
            yref_e[6:9] = ref_e["vel"][0]
        try:
            self._acados_ocp_solver.set(self._N, "yref_e", yref_e)
        except Exception:
            self._acados_ocp_solver.set(self._N, "yref", yref_e)

        # Solving problem
        status = self._acados_ocp_solver.solve()
        try:
            if hasattr(status, "value"):
                ok = int(status.value) == 0
            else:
                ok = int(status) == 0
        except Exception:
            ok = True

        if not ok:
            u0 = np.zeros((self._nu,))
            u0[3] = self.drone_params["mass"] * -self.drone_params["gravity_vec"][-1]
            print("[WARN] Solver failed, returning hover thrust")
        else:
            u0 = self._acados_ocp_solver.get(0, "u")
            print("[DEBUG] Control output (u0):", u0)

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