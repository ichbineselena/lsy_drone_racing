"""Improved pure MPPI controller for quadrotor attitude control with visualization.  

This module implements a robust MPPI controller that directly outputs
roll-pitch-yaw and thrust commands with better cost shaping and dynamics.  
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Optional

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

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
        config:  dict,
        enable_plotting: bool = True,
        plot_frequency: int = 20,  # Plot every N steps
    ):
        """Initialize the MPPI attitude controller. 

        Args:
            obs: The initial observation of the environment's state. 
            info: Additional environment information from the reset.
            config: The configuration of the environment.
            enable_plotting: Whether to enable trajectory plotting
            plot_frequency: How often to update plots (in steps)
        """
        super().__init__(obs, info, config)

        # Simulation parameters
        self._dt = 1 / config.env.freq
        self._config = config

        # Plotting parameters
        self.enable_plotting = enable_plotting
        self.plot_frequency = plot_frequency
        
        # History tracking
        self.state_history = []
        self.control_history = []
        self.planned_trajectory_history = []
        self.cost_history = []
        self.goal_history = []

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
            "position":  torch.tensor([10.0, 10.0, 20.0], device=self.device, dtype=self.dtype),
            "velocity":  torch.tensor([0.5, 0.5, 0.5], device=self.device, dtype=self.dtype),
            "attitude": torch.tensor([5.0, 5.0, 1.0], device=self.device, dtype=self.dtype),
            "att_rate": torch.tensor([2.0, 2.0, 1.0], device=self.device, dtype=self.dtype),
            "control_effort":  torch.tensor([1.0, 1.0, 0.5, 2.0], device=self.device, dtype=self.dtype),
            "control_rate":  5.0,  # Penalize control changes
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
        initial_control[:, 3] = self.hover_thrust
        self.mppi.U = initial_control

        self.step_count = 0

        # Initialize plotting
        if self.enable_plotting:
            plt.ion()  # Interactive mode
            self.fig = plt.figure(figsize=(16, 10))
            self.setup_plots()

        print("[AttitudeMPPI] Initialization complete")
        print(f"  - Horizon: {self.mppi_horizon} steps @ {self.mppi_dt:.3f}s = {self.mppi_horizon * self.mppi_dt:.2f}s")
        print(f"  - Samples: {self.num_samples}")
        print(f"  - Lambda: {self.lambda_weight}")
        print(f"  - RPY limits: ±{np.rad2deg(self.rpy_max):.1f}°")
        print(f"  - Plotting:  {'Enabled' if self.enable_plotting else 'Disabled'}")
    def setup_plots(self):
        """Setup the plotting layout."""
        # 3D trajectory plot
        self.ax_3d = self.fig.add_subplot(2, 3, 1, projection='3d')
        self.ax_3d.set_title('3D Trajectory')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')

        # Position over time
        self.ax_pos = self.fig.add_subplot(2, 3, 2)
        self.ax_pos.set_title('Position vs Time')
        self.ax_pos.set_xlabel('Time (s)')
        self.ax_pos.set_ylabel('Position (m)')
        self.ax_pos.grid(True)

        # Velocity over time
        self.ax_vel = self.fig.add_subplot(2, 3, 3)
        self.ax_vel.set_title('Velocity vs Time')
        self.ax_vel.set_xlabel('Time (s)')
        self.ax_vel.set_ylabel('Velocity (m/s)')
        self.ax_vel.grid(True)

        # Attitude (RPY) over time
        self.ax_att = self.fig.add_subplot(2, 3, 4)
        self.ax_att.set_title('Attitude (RPY) vs Time')
        self.ax_att.set_xlabel('Time (s)')
        self.ax_att.set_ylabel('Angle (deg)')
        self.ax_att.grid(True)

        # Control inputs over time
        self.ax_ctrl = self.fig.add_subplot(2, 3, 5)
        self.ax_ctrl.set_title('Control Inputs vs Time')
        self.ax_ctrl.set_xlabel('Time (s)')
        self.ax_ctrl.set_ylabel('Control')
        self.ax_ctrl.grid(True)

        # Cost breakdown
        self.ax_cost = self.fig.add_subplot(2, 3, 6)
        self.ax_cost.set_title('Cost History')
        self.ax_cost.set_xlabel('Step')
        self.ax_cost.set_ylabel('Cost')
        self.ax_cost.grid(True)

        plt.tight_layout()

    def compute_running_cost(
        self,
        state: torch.Tensor,
        control: torch.Tensor
    ) -> torch.Tensor:
        """Compute running cost for MPPI optimization.  

        Args:
            state:   State tensor (..., 12): [pos(3), rpy(3), vel(3), drpy(3)]
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

    def rollout_best_trajectory(self, state: torch.Tensor) -> dict:
        """Rollout the best trajectory from MPPI."""
        # Get optimal control sequence
        u_seq = self.mppi.U.clone()
        
        # Rollout trajectory
        x = state.clone()
        trajectory = [x.cpu().numpy()]
        controls = []
        
        for k in range(self.mppi_horizon):
            u_k = u_seq[k]
            controls.append(u_k.cpu().numpy())
            x = self.drone_model.dynamics(x, u_k, self.mppi_dt)
            trajectory.append(x.cpu().numpy())
        
        trajectory = np.array(trajectory)  # Shape: (H+1, 12)
        controls = np.array(controls)      # Shape: (H, 4)
        
        return {
            'trajectory': trajectory,
            'controls': controls,
            'positions': trajectory[: , 0:3],
            'rpy': trajectory[:, 3:6],
            'velocities':  trajectory[:, 6:9],
            'drpy': trajectory[:, 9:12]
        }

    def update_plots(self, obs: dict, planned_traj: dict):
        """Update all plots with current data."""
        if not self.enable_plotting:
            return

        # Clear all axes
        self.ax_3d.cla()
        self.ax_pos.cla()
        self.ax_vel.cla()
        self.ax_att.cla()
        self.ax_ctrl.cla()
        self.ax_cost.cla()

        # Reset titles and labels
        self.ax_3d.set_title('3D Trajectory')
        self.ax_3d.set_xlabel('X (m)')
        self.ax_3d.set_ylabel('Y (m)')
        self.ax_3d.set_zlabel('Z (m)')

        # ===== 3D Trajectory Plot =====
        if len(self.state_history) > 0:
            hist_array = np.array(self.state_history)
            executed_pos = hist_array[:, 0:3]
            
            # Plot executed trajectory
            self.ax_3d.plot(executed_pos[: , 0], executed_pos[: , 1], executed_pos[: , 2],
                           'b-', linewidth=2, label='Executed')
            
            # Plot current position
            self.ax_3d.scatter(obs['pos'][0], obs['pos'][1], obs['pos'][2],
                             c='green', s=100, marker='o', label='Current')
            
            # Plot planned trajectory
            plan_pos = planned_traj['positions']
            self.ax_3d.plot(plan_pos[:, 0], plan_pos[:, 1], plan_pos[:, 2],
                           'r--', linewidth=1.5, alpha=0.7, label='Planned')

        # Plot all gates
        for i, gate_pos in enumerate(self.gates_pos):
            color = 'gold' if i == self.target_gate_idx else 'gray'
            alpha = 1.0 if i == self.target_gate_idx else 0.3
            self.ax_3d.scatter(gate_pos[0], gate_pos[1], gate_pos[2],
                             c=color, s=200, marker='*', alpha=alpha,
                             edgecolors='black', linewidths=1)
            
            # Draw gate normal
            if i == self.target_gate_idx:
                normal = self.gates_normal[i] * 0.5
                self.ax_3d.plot([gate_pos[0], gate_pos[0] + normal[0]],
                               [gate_pos[1], gate_pos[1] + normal[1]],
                               [gate_pos[2], gate_pos[2] + normal[2]],
                               'r-', linewidth=2, alpha=0.5)

        self.ax_3d.legend()
        self.ax_3d.set_box_aspect([1, 1, 1])

        # ===== Position vs Time =====
        if len(self.state_history) > 1:
            hist_array = np.array(self.state_history)
            time = np.arange(len(hist_array)) * self._dt
            
            self.ax_pos.plot(time, hist_array[:, 0], 'r-', label='X', linewidth=1.5)
            self.ax_pos.plot(time, hist_array[:, 1], 'g-', label='Y', linewidth=1.5)
            self.ax_pos.plot(time, hist_array[:, 2], 'b-', label='Z', linewidth=1.5)
            
            # Plot goal
            self.ax_pos.axhline(y=self.goal[0], color='r', linestyle='--', alpha=0.3, label='Goal X')
            self.ax_pos.axhline(y=self.goal[1], color='g', linestyle='--', alpha=0.3, label='Goal Y')
            self.ax_pos.axhline(y=self.goal[2], color='b', linestyle='--', alpha=0.3, label='Goal Z')
            
            self.ax_pos.set_title('Position vs Time')
            self.ax_pos.set_xlabel('Time (s)')
            self.ax_pos.set_ylabel('Position (m)')
            self.ax_pos.legend(loc='best', fontsize=8)
            self.ax_pos.grid(True)

        # ===== Velocity vs Time =====
        if len(self.state_history) > 1:
            hist_array = np.array(self.state_history)
            time = np.arange(len(hist_array)) * self._dt
            
            self.ax_vel.plot(time, hist_array[:, 6], 'r-', label='Vx', linewidth=1.5)
            self.ax_vel.plot(time, hist_array[:, 7], 'g-', label='Vy', linewidth=1.5)
            self.ax_vel.plot(time, hist_array[:, 8], 'b-', label='Vz', linewidth=1.5)
            
            # Plot total speed
            speed = np.linalg.norm(hist_array[:, 6:9], axis=1)
            self.ax_vel.plot(time, speed, 'k--', label='Speed', linewidth=2, alpha=0.7)
            
            self.ax_vel.set_title('Velocity vs Time')
            self.ax_vel.set_xlabel('Time (s)')
            self.ax_vel.set_ylabel('Velocity (m/s)')
            self.ax_vel.legend(loc='best', fontsize=8)
            self.ax_vel.grid(True)

        # ===== Attitude (RPY) vs Time =====
        if len(self.state_history) > 1:
            hist_array = np.array(self.state_history)
            time = np.arange(len(hist_array)) * self._dt
            
            # Convert to degrees
            rpy_deg = np.rad2deg(hist_array[:, 3:6])
            
            self.ax_att.plot(time, rpy_deg[: , 0], 'r-', label='Roll', linewidth=1.5)
            self.ax_att.plot(time, rpy_deg[:, 1], 'g-', label='Pitch', linewidth=1.5)
            self.ax_att.plot(time, rpy_deg[:, 2], 'b-', label='Yaw', linewidth=1.5)
            
            # Plot limits
            limit_deg = np.rad2deg(self.rpy_max)
            self.ax_att.axhline(y=limit_deg, color='k', linestyle='--', alpha=0.3)
            self.ax_att.axhline(y=-limit_deg, color='k', linestyle='--', alpha=0.3)
            
            self.ax_att.set_title('Attitude (RPY) vs Time')
            self.ax_att.set_xlabel('Time (s)')
            self.ax_att.set_ylabel('Angle (deg)')
            self.ax_att.legend(loc='best', fontsize=8)
            self.ax_att.grid(True)

        # ===== Control Inputs vs Time =====
        if len(self.control_history) > 1:
            ctrl_array = np.array(self.control_history)
            time = np.arange(len(ctrl_array)) * self._dt
            
            # Convert RPY commands to degrees
            rpy_cmd_deg = np.rad2deg(ctrl_array[:, 0:3])
            
            self.ax_ctrl.plot(time, rpy_cmd_deg[: , 0], 'r-', label='Roll cmd', linewidth=1.5, alpha=0.7)
            self.ax_ctrl.plot(time, rpy_cmd_deg[:, 1], 'g-', label='Pitch cmd', linewidth=1.5, alpha=0.7)
            self.ax_ctrl.plot(time, rpy_cmd_deg[: , 2], 'b-', label='Yaw cmd', linewidth=1.5, alpha=0.7)
            
            # Plot thrust on secondary y-axis
            ax_thrust = self.ax_ctrl.twinx()
            ax_thrust.plot(time, ctrl_array[:, 3], 'k--', label='Thrust', linewidth=2, alpha=0.7)
            ax_thrust.set_ylabel('Thrust (N)', color='k')
            ax_thrust.tick_params(axis='y', labelcolor='k')
            ax_thrust.axhline(y=self.hover_thrust, color='gray', linestyle=':', alpha=0.5, label='Hover')
            
            self.ax_ctrl.set_title('Control Inputs vs Time')
            self.ax_ctrl.set_xlabel('Time (s)')
            self.ax_ctrl.set_ylabel('Angle Command (deg)')
            
            # Combine legends
            lines1, labels1 = self.ax_ctrl.get_legend_handles_labels()
            lines2, labels2 = ax_thrust.get_legend_handles_labels()
            self.ax_ctrl.legend(lines1 + lines2, labels1 + labels2, loc='best', fontsize=8)
            self.ax_ctrl.grid(True)

        # ===== Cost History =====
        if len(self.cost_history) > 1:
            steps = np.arange(len(self.cost_history))
            cost_array = np.array(self.cost_history)
            
            if cost_array.ndim == 1:
                self.ax_cost.plot(steps, cost_array, 'b-', linewidth=2, label='Total Cost')
            else:
                # If we have cost breakdown
                self.ax_cost.plot(steps, cost_array, 'b-', linewidth=2, label='Total Cost')
            
            self.ax_cost.set_title('Cost History')
            self.ax_cost.set_xlabel('Step')
            self.ax_cost.set_ylabel('Cost')
            self.ax_cost.legend(loc='best', fontsize=8)
            self.ax_cost.grid(True)
            self.ax_cost.set_yscale('log')

        plt.tight_layout()
        plt.draw()
        plt.pause(0.001)

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
            Control command:   [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]
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
            initial_control[: , 3] = self.hover_thrust
            self.mppi.U = initial_control

        # Convert observation to state tensor
        state = self.drone_model.obs_to_state(obs)

        # Run MPPI optimization
        with torch.no_grad():
            optimal_control = self.mppi.command(state)

        # Rollout best trajectory for visualization
        planned_traj = self.rollout_best_trajectory(state)

        # Store for rate penalty
        self.prev_control = optimal_control.clone()

        # Convert to numpy
        control_np = optimal_control.cpu().numpy()

        # Apply additional safety clipping
        control_np[0: 3] = np.clip(control_np[0:3], -self.rpy_max, self.rpy_max)
        control_np[3] = np.clip(control_np[3], self.thrust_min, self.thrust_max)

        # Store history
        current_state = self.drone_model.obs_to_state(obs).cpu().numpy()
        self.state_history.append(current_state)
        self.control_history.append(control_np)
        self.goal_history.append(self.goal.copy())
        
        # Compute and store cost (approximate using first sample)
        with torch.no_grad():
            cost = self.compute_running_cost(state, optimal_control).cpu().item()
            self.cost_history.append(cost)

        # Update plots periodically
        if self.enable_plotting and (self.step_count % self.plot_frequency == 0):
            self.update_plots(obs, planned_traj)

        # Debug output (reduced frequency)
        if self.step_count % 10 == 0:
            pos = obs["pos"]
            vel = obs["vel"]
            dist_to_gate = np.linalg.norm(pos - self.goal)
            speed = np.linalg.norm(vel)
            print(f"[AttitudeMPPI] Step {self.step_count:4d} | "
                  f"Gate {self.target_gate_idx} | "
                  f"Dist:  {dist_to_gate: 5.2f}m | "
                  f"Speed: {speed:4.2f}m/s | "
                  f"RPY: [{np.rad2deg(control_np[0]):5.1f}, {np.rad2deg(control_np[1]):5.1f}, {np.rad2deg(control_np[2]):5.1f}]° | "
                  f"T: {control_np[3]:.2f}N | "
                  f"Cost:  {cost:.2f}")

        return control_np

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info:   dict,
    ) -> bool:
        """Step callback."""
        return False

    def episode_callback(self):
        """Episode callback to reset state."""
        # Save final plot
        if self.enable_plotting and len(self.state_history) > 0:
            print("[AttitudeMPPI] Saving final trajectory plot...")
            self.fig.savefig('mppi_trajectory_final.png', dpi=150, bbox_inches='tight')
            print("[AttitudeMPPI] Plot saved to mppi_trajectory_final.png")
        
        # Reset control sequence
        initial_control = torch.zeros((self.mppi_horizon, 4), dtype=self.dtype, device=self.device)
        initial_control[:, 3] = self.hover_thrust
        self.mppi.U = initial_control
        
        # Reset step counter and previous control
        self.step_count = 0
        self.prev_control = None
        
        # Clear history
        self.state_history = []
        self.control_history = []
        self.planned_trajectory_history = []
        self.cost_history = []
        self.goal_history = []
        
        print("[AttitudeMPPI] Episode reset")

    def __del__(self):
        """Cleanup when controller is destroyed."""
        if self.enable_plotting:
            plt.close(self.fig)