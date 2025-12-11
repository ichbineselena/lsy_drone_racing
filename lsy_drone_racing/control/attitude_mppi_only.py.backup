"""Improved pure MPPI controller for quadrotor attitude control. 

This module implements a robust MPPI controller that directly outputs
roll-pitch-yaw and thrust commands with better cost shaping and dynamics. 
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Import PyTorch MPPI
import pytorch_mppi

# Import our PyTorch dynamics model
from drone_models.core import load_params
from lsy_drone_racing.utils.model_torch import DroneModelTorch

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class AttitudeMPPIController(Controller):
    """Improved pure MPPI controller for attitude control (RPY + Thrust)."""

    def __init__(
        self,
        obs:  dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
    ):
        """Initialize the MPPI attitude controller.

        Args:
            obs: The initial observation of the environment's state.
            info: Additional environment information from the reset.
            config: The configuration of the environment.
        """
        super().__init__(obs, info, config)

        # Simulation parameters
        self._dt = 1 / config.env.freq
        self._config = config

        # MPPI parameters - tuned for stability
        self.mppi_horizon = 20  # Longer horizon for smoother planning
        self.mppi_dt = self._dt * 2  # Coarser time discretization (0.04s instead of 0.02s)
        self.num_samples = 2000  # Reduced for faster computation
        self.lambda_weight = 10.0  # Higher temperature = smoother controls

        # Device selection
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.double

        print(f"[AttitudeMPPI] Initializing on device: {self.device}")

        # Load drone parameters
        self.drone_params = load_params("so_rpy", config.sim.drone_model)

        # Create PyTorch dynamics model
        self.drone_model = DroneModelTorch(
            parameters=self.drone_params,
            device=self.device,
            dtype=self.dtype,
            use_euler_state=True,
        )

        # Extract gate information
        self.gates_pos = []
        self.gates_quat = []
        self.gates_normal = []

        for gate in config.env.track.gates:
            pos = np.array(gate["pos"], dtype=float)
            rpy = np.array(gate["rpy"], dtype=float)
            rot = R.from_euler("xyz", rpy)

            self.gates_pos.append(pos)
            self.gates_quat.append(rot.as_quat())
            self.gates_normal.append(rot.as_matrix()[:, 0])

        self.gates_pos = np.array(self.gates_pos)
        self.gates_quat = np.array(self.gates_quat)
        self.gates_normal = np.array(self.gates_normal)

        # Current target
        self.target_gate_idx = int(obs["target_gate"])
        self.goal = self.gates_pos[self.target_gate_idx]
        self.prev_goal = self.goal.copy()

        # More conservative control limits
        self.rpy_max = 0.3  # ±17 degrees (more conservative)
        self.thrust_min = self.drone_params["thrust_min"] * 4
        self.thrust_max = self.drone_params["thrust_max"] * 4
        self.hover_thrust = self.drone_params["mass"] * abs(self.drone_params["gravity_vec"][-1])

        # Tuned cost weights for smoother behavior
        self.cost_weights = {
            "position": torch.tensor([10.0, 10.0, 20.0], device=self.device, dtype=self.dtype),
            "velocity": torch.tensor([0.5, 0.5, 0.5], device=self.device, dtype=self.dtype),
            "attitude": torch.tensor([5.0, 5.0, 1.0], device=self.device, dtype=self.dtype),
            "att_rate": torch.tensor([2.0, 2.0, 1.0], device=self.device, dtype=self.dtype),
            "control_effort": torch.tensor([1.0, 1.0, 0.5, 2.0], device=self.device, dtype=self.dtype),
            "control_rate": 5.0,  # Penalize control changes
            "gate_progress": 15.0,
        }

        # Store previous control for rate penalty
        self.prev_control = None

        # Define dynamics wrapper for MPPI
        def dynamics_fn(state, control):
            """Dynamics function for MPPI."""
            return self.drone_model.dynamics(state, control, self.mppi_dt)

        # Define running cost for MPPI
        def running_cost_fn(state, control):
            """Running cost for MPPI trajectory optimization."""
            return self.compute_running_cost(state, control)

        # Control bounds - more conservative
        u_min = torch.tensor(
            [-self.rpy_max, -self.rpy_max, -self.rpy_max, self.thrust_min],
            dtype=self.dtype,
            device=self.device
        )
        u_max = torch.tensor(
            [self.rpy_max, self.rpy_max, self.rpy_max, self.thrust_max],
            dtype=self.dtype,
            device=self.device
        )

        # Smaller control noise for smoother sampling
        noise_sigma = torch.diag(torch.tensor(
            [0.05, 0.05, 0.05, 0.2],  # Reduced noise
            dtype=self.dtype,
            device=self.device
        ))

        # Initialize MPPI solver
        self.mppi = pytorch_mppi.MPPI(
            dynamics=dynamics_fn,
            running_cost=running_cost_fn,
            nx=self.drone_model.nx,
            noise_sigma=noise_sigma,
            num_samples=self.num_samples,
            horizon=self.mppi_horizon,
            lambda_=self.lambda_weight,
            u_min=u_min,
            u_max=u_max,
            device=self.device,
        )

        # Initialize control sequence to hover
        initial_control = torch.zeros((self.mppi_horizon, 4), dtype=self.dtype, device=self.device)
        initial_control[: , 3] = self.hover_thrust
        self.mppi.U = initial_control

        self.step_count = 0

        print("[AttitudeMPPI] Initialization complete")
        print(f"  - Horizon: {self.mppi_horizon} steps @ {self.mppi_dt:.3f}s = {self.mppi_horizon * self.mppi_dt:.2f}s")
        print(f"  - Samples: {self.num_samples}")
        print(f"  - Lambda: {self.lambda_weight}")
        print(f"  - RPY limits: ±{np.rad2deg(self.rpy_max):.1f}°")

    def compute_running_cost(
        self,
        state: torch.Tensor,
        control: torch.Tensor
    ) -> torch.Tensor:
        """Compute running cost for MPPI optimization. 

        Args:
            state:  State tensor (... , 12): [pos(3), rpy(3), vel(3), drpy(3)]
            control: Control tensor (... , 4): [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]

        Returns:
            Cost tensor of shape (...)
        """
        # Extract state components
        pos = state[..., 0:3]
        rpy = state[..., 3:6]
        vel = state[..., 6:9]
        drpy = state[..., 9:12]

        # Current goal
        goal_pos = torch.tensor(self.goal, dtype=self.dtype, device=self.device)

        # 1. Position error cost (quadratic)
        pos_error = pos - goal_pos
        dist_sq = torch.sum(self.cost_weights["position"] * pos_error ** 2, dim=-1)
        c_pos = dist_sq

        # 2. Velocity cost - penalize excessive speed
        speed_sq = torch.sum(self.cost_weights["velocity"] * vel ** 2, dim=-1)
        c_vel = speed_sq

        # 3. Attitude cost - prefer level flight
        att_error_sq = torch.sum(self.cost_weights["attitude"] * rpy ** 2, dim=-1)
        c_att = att_error_sq

        # 4. Attitude rate cost - penalize fast rotations
        att_rate_sq = torch.sum(self.cost_weights["att_rate"] * drpy ** 2, dim=-1)
        c_att_rate = att_rate_sq

        # 5. Control effort - penalize deviation from hover
        control_ref = torch.tensor(
            [0.0, 0.0, 0.0, self.hover_thrust],
            dtype=self.dtype,
            device=self.device
        )
        control_error = control - control_ref
        c_control_effort = torch.sum(
            self.cost_weights["control_effort"] * control_error ** 2, 
            dim=-1
        )

        # 6. Control rate penalty - encourage smooth control changes
        c_control_rate = 0.0
        if self.prev_control is not None:
            prev_ctrl = self.prev_control.to(self.device)
            # Only penalize the first control in the sequence
            if control.dim() == 2:  # Batch of controls
                control_change = control[0] - prev_ctrl
            else:  # Single control
                control_change = control - prev_ctrl
            c_control_rate = self.cost_weights["control_rate"] * torch.sum(control_change ** 2)

        # 7. Progress-based cost (reward moving closer to gate)
        dist = torch.sqrt(dist_sq + 1e-6)
        c_progress = self.cost_weights["gate_progress"] * dist

        # Total cost
        total_cost = (
            c_pos +
            c_vel +
            c_att +
            c_att_rate +
            c_control_effort +
            c_control_rate +
            c_progress
        )

        return total_cost

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute control command using MPPI. 

        Args:
            obs:  Current observation
            info: Additional info (optional)

        Returns:
            Control command:  [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
        """
        self.step_count += 1

        # Update target gate if necessary
        new_target_idx = int(obs["target_gate"])
        if new_target_idx != self.target_gate_idx:
            self.target_gate_idx = new_target_idx
            self.prev_goal = self.goal.copy()
            self.goal = self.gates_pos[self.target_gate_idx]
            print(f"\n[AttitudeMPPI] Updated target to gate {self.target_gate_idx}")
            
            # Optionally reset control sequence on gate change
            initial_control = torch.zeros((self.mppi_horizon, 4), dtype=self.dtype, device=self.device)
            initial_control[:, 3] = self.hover_thrust
            self.mppi.U = initial_control

        # Convert observation to state tensor
        state = self.drone_model.obs_to_state(obs)

        # Run MPPI optimization
        with torch.no_grad():
            optimal_control = self.mppi.command(state)

        # Store for rate penalty
        self.prev_control = optimal_control.clone()

        # Convert to numpy
        control_np = optimal_control.cpu().numpy()

        # Apply additional safety clipping
        control_np[0: 3] = np.clip(control_np[0:3], -self.rpy_max, self.rpy_max)
        control_np[3] = np.clip(control_np[3], self.thrust_min, self.thrust_max)

        # Debug output (reduced frequency)
        if self.step_count % 10 == 0:
            pos = obs["pos"]
            vel = obs["vel"]
            dist_to_gate = np.linalg.norm(pos - self.goal)
            speed = np.linalg.norm(vel)
            print(f"[AttitudeMPPI] Step {self.step_count: 4d} | "
                  f"Gate {self.target_gate_idx} | "
                  f"Dist:  {dist_to_gate: 5.2f}m | "
                  f"Speed: {speed:4.2f}m/s | "
                  f"RPY: [{np.rad2deg(control_np[0]):5.1f}, {np.rad2deg(control_np[1]):5.1f}, {np.rad2deg(control_np[2]):5.1f}]° | "
                  f"T: {control_np[3]:.2f}N")

        return control_np

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info:  dict,
    ) -> bool:
        """Step callback."""
        return False

    def episode_callback(self):
        """Episode callback to reset state."""
        # Reset control sequence
        initial_control = torch.zeros((self.mppi_horizon, 4), dtype=self.dtype, device=self.device)
        initial_control[:, 3] = self.hover_thrust
        self.mppi.U = initial_control
        
        # Reset step counter and previous control
        self.step_count = 0
        self.prev_control = None
        
        print("[AttitudeMPPI] Episode reset")