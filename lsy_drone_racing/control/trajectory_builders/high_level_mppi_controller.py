import numpy as np
import torch
from torch import tensor
from pytorch_mppi import MPPI

from lsy_drone_racing.control import Controller
from lsy_drone_racing.control.mpc.attitude_mpc import AttitudeMPC


class HighLevelMPPIController(Controller):
    """
    High-level MPPI trajectory optimizer using a pos/vel kinematic model.
    Output is sent to the AttitudeMPC low-level controller.

    State (6D): [px, py, pz, vx, vy, vz]
    Control (3D): velocity delta / acceleration command (interpreted as desired dv)
    """

    def __init__(self, obs, info, config):
        super().__init__(obs, info, config)

        #####################################################################
        # 1. Low-level controller (UNCHANGED AttitudeMPC)
        #####################################################################
        self.att_mpc = AttitudeMPC(obs, info, config)

        #####################################################################
        # 2. MPPI model setup (pos/vel kinematics)
        #####################################################################
        self.dt = config.sim.dt  # MPPI & MPC must share dt

        # --- MPPI dynamics model ---
        def dynamics(x, u):
            """
            Simple kinematic model:
            p_{t+1} = p + v * dt
            v_{t+1} = v + u * dt
            """
            px, py, pz, vx, vy, vz = torch.split(x, 1, dim=-1)
            ax, ay, az = torch.split(u, 1, dim=-1)

            px_next = px + vx * self.dt
            py_next = py + vy * self.dt
            pz_next = pz + vz * self.dt

            vx_next = vx + ax * self.dt
            vy_next = vy + ay * self.dt
            vz_next = vz + az * self.dt

            return torch.cat([px_next, py_next, pz_next,
                              vx_next, vy_next, vz_next], dim=-1)

        # --- Cost function ---
        def cost(x, u):
            """
            Quadratic tracking + smoothness.
            """
            # Want to reach the next gate or goal
            goal = torch.tensor(self.goal, device=x.device, dtype=x.dtype)

            pos = x[..., :3]
            vel = x[..., 3:6]

            # position cost
            c_pos = 4.0 * torch.sum((pos - goal)**2, dim=-1)

            # velocity moderation
            c_vel = 0.1 * torch.sum(vel**2, dim=-1)

            # control smoothness
            c_u = 0.01 * torch.sum(u**2, dim=-1)

            return c_pos + c_vel + c_u

        #####################################################################
        # 3. Create MPPI solver
        #####################################################################
        H = 20   # horizon steps
        K = 1000 # number of rollouts

        self.mppi = MPPI(
            dynamics=dynamics,
            running_cost=cost,
            nx=6,
            nu=3,
            num_samples=K,
            horizon=H,
            lambda_=1.0,
            noise_sigma=0.4 * torch.eye(3),
            u_min=-3.0,
            u_max=3.0
        )

        #####################################################################
        # 4. Selected target (default = first gate)
        #####################################################################
        self.goal = info["gates_pos"][0]  # update each step anyway


    #########################################################################
    # EXTRACT STATE FOR MPPI
    #########################################################################
    def _get_state(self, obs):
        pos = obs["pos"]
        vel = obs["vel"]
        return np.concatenate([pos, vel], axis=-1)


    #########################################################################
    # MAIN CONTROL LOOP
    #########################################################################
    def compute_control(self, obs):
        # update goal (track nearest gate)
        self.goal = obs["gates_pos"][0]

        # get MPPI initial state
        x0 = tensor(self._get_state(obs), dtype=torch.float32)

        # run MPPI optimization: gives optimal trajectory & controls
        with torch.no_grad():
            u_opt = self.mppi.command(x0)   # only returns first control
            x_traj = self.mppi.rollout_trajectory  # full trajectory (H+1, 6)

        x_next = x_traj[1].cpu().numpy()
        p_d = x_next[:3]
        v_d = x_next[3:6]

        # Attitude MPC expects:
        # - desired position
        # - desired velocity
        # - desired yaw
        yaw_d = 0.0  # (for now MPPI does not optimize yaw)

        # Use AttitudeMPC exactly as before
        u = self.att_mpc.compute_control(
            desired_position=p_d,
            desired_velocity=v_d,
            desired_yaw=yaw_d,
            obs=obs
        )

        return u
