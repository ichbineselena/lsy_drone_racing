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
from lsy_drone_racing.dynamics.model_torch import DroneModelTorch

from lsy_drone_racing.control import Controller

#from time import time

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

        # MPPI hyperparameters - optimized for robust trajectory optimization
        self.mppi_horizon = 25  # Planning horizon steps (proven stable for 4-gate success)
        self.mppi_dt = self._dt * 2  # 0.04s time discretization
        self.num_samples = 3000  # Stable optimization quality
        self.lambda_weight = 9.5  # Temperature parameter tuned for improved transitions
        
        # Gate geometry constants (from physical measurement)
        # Total gate edge: 74cm, Opening: 40.5cm, Frame width: 17cm each side, Thickness: 2cm
        self.gate_opening = 0.35 #0.345 #0.405  # Opening width/height = 40.5cm
        self.gate_frame_width = 0.17 #0.2 #0.17  # Frame bar width = 17cm (on each side)
        self.gate_frame_thickness = 0.02  # Frame bar thickness/depth = 2cm
        # Frame bar centers at opening_half + frame_width/2 = 0.2025 + 0.085 = 0.2875m
        self.gate_frame_center_offset = 0.2875  # Frame bars centered at ±28.75cm from gate center
        
        # Obstacle geometry constants (from MuJoCo model)
        self.obstacle_radius = 0.015  # 1.5cm radius
        self.obstacle_half_length = 1.5  # half-length of capsule
        self.obstacle_length = 1.5  # total length
        self.obstacle_safety_margin = 0.05  # 5cm safety margin
        self.obstacle_avoidance_weight = 50.0  # Weight for obstacle cost

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
        self.old_gate_pos = self.goal.copy()
        
        self.current_gate_center = np.array(obs["gates_pos"][self.target_gate_idx], dtype=float)
        self.current_gate_quat = np.array(obs["gates_quat"][self.target_gate_idx], dtype=float)
        
        # Flag to track if goal has been shifted for current gate
        self.goal_shifted = False
        self.obstacles_pos = obs["obstacles_pos"].copy() if "obstacles_pos" in obs else np.array([])
        
        # Pause mechanism to safely pass through gate before switching targets
        self.pause_counter = 0
        self.pause_duration = 20  # ~0.8s at 25Hz to safely pass through gate opening

        # More conservative control limits
        self.rpy_max = 0.5 #0.3  # ±17 degrees (more conservative)
        self.thrust_min = self.drone_params["thrust_min"] * 4
        self.thrust_max = self.drone_params["thrust_max"] * 4
        self.hover_thrust = self.drone_params["mass"] * abs(self.drone_params["gravity_vec"][-1])

        # Simplified cost weights - balanced aggressive for 4-gate success
        self.cost_weights = {
              "position": torch.tensor([20.0, 20.0, 16.0], device=self.device, dtype=self.dtype),  # Breakthrough config
              "velocity": torch.tensor([0.040, 0.040, 0.125], device=self.device, dtype=self.dtype),  # Stabilized without losing too much speed
            "attitude": torch.tensor([1.5, 1.5, 0.2], device=self.device, dtype=self.dtype),  # Lower for agility
            "z_floor": 2000.0,  # Strong penalty for ground collision (Z < 0.03m)
            "obstacle": self.obstacle_avoidance_weight,  # Obstacle avoidance weight
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
        print(f"  - Gate: {self.gate_opening*100:.1f}cm opening, {self.gate_frame_width*100:.0f}cm frame width, {self.gate_frame_thickness*100:.0f}cm thickness")
        print(f"  - Frame centers at ±{self.gate_frame_center_offset*100:.1f}cm from gate center")
        print(f"  - Obstacles: {len(self.obstacles_pos)} obstacles with {self.obstacle_safety_margin*100:.1f}cm safety margin")

    def compute_obstacle_cost(
        self,
        pos: torch.Tensor
    ) -> torch.Tensor:
        """Compute obstacle avoidance cost for given positions.
        
        Args:
            pos: Position tensor of shape (..., 3)
            
        Returns:
            Obstacle cost tensor of shape (...)
        """
        if len(self.obstacles_pos) == 0:
            return torch.zeros(pos.shape[:-1], dtype=self.dtype, device=self.device)
        
        # Convert obstacles to tensor
        obstacles_t = torch.tensor(self.obstacles_pos, dtype=self.dtype, device=self.device)
        
        # Initialize total cost
        total_cost = torch.zeros(pos.shape[:-1], dtype=self.dtype, device=self.device)
        
        # For each obstacle (capsule oriented vertically)
        for i in range(len(self.obstacles_pos)):
            # Get obstacle center position
            obstacle_center = obstacles_t[i]
            
            # Capsule parameters
            radius = self.obstacle_radius
            half_length = self.obstacle_half_length
            safety_margin = self.obstacle_safety_margin
            effective_radius = radius + safety_margin
            
            # Vector from obstacle center to drone position
            rel_vec = pos - obstacle_center
            
            # For vertical capsule, we need to find closest point on the line segment
            # from (obstacle_center - [0,0,half_length]) to (obstacle_center + [0,0,half_length])
            
            # Project onto vertical axis
            z_proj = rel_vec[..., 2]  # Z component
            
            # Clamp projection to [-half_length, half_length]
            z_clamped = torch.clamp(z_proj, -half_length, half_length)
            
            # Closest point on the capsule segment
            closest_point_z = obstacle_center[2] + z_clamped
            closest_point = torch.stack([
                torch.full_like(z_clamped, obstacle_center[0]),
                torch.full_like(z_clamped, obstacle_center[1]),
                closest_point_z
            ], dim=-1)
            
            # Distance vector from closest point on capsule to drone
            dist_vec = pos - closest_point
            dist = torch.norm(dist_vec, dim=-1)
            
            # Inverse barrier cost: penalize being too close
            # Use a smooth penalty function that increases as distance decreases
            # Cost = weight / (distance^2 + epsilon) when distance < threshold
            
            # Threshold for penalty activation (safety margin + radius)
            penalty_threshold = effective_radius * 2.0  # Start penalty at 2x effective radius
            
            # Distance to capsule surface (subtract radius)
            surface_dist = dist - effective_radius
            
            # Apply penalty only when inside penalty threshold
            mask = surface_dist < penalty_threshold
            
            if torch.any(mask):
                # Smooth penalty: higher cost as distance decreases
                # Use exponential barrier for smoothness
                penalty = torch.exp(-surface_dist[mask] / (effective_radius * 0.5))
                
                # Scale by inverse of distance squared (becomes very large near obstacle)
                inv_dist_sq = 1.0 / (surface_dist[mask] ** 2 + 0.01)  # Add epsilon for numerical stability
                
                # Combine penalties
                obstacle_cost = self.cost_weights["obstacle"] * penalty * inv_dist_sq
                
                # Add to total cost at masked positions
                total_cost = total_cost.clone()
                total_cost[mask] = total_cost[mask] + obstacle_cost
        
        return total_cost

    def compute_all_gates_avoidance_cost(
        self,
        pos: torch.Tensor,
        obs: dict | None = None
    ) -> torch.Tensor:
        """Compute gate frame avoidance cost for ALL gates (not just current target).
        
        This prevents the drone from colliding with gates it has already passed
        or gates it will pass in the future.
        
        Args:
            pos: Position tensor of shape (..., 3)
            obs: Current observation dict (optional, for live gate poses)
            
        Returns:
            Gate avoidance cost tensor of shape (...)
        """
        # Initialize total gate avoidance cost
        total_gate_cost = torch.zeros(pos.shape[:-1], dtype=self.dtype, device=self.device)
        
        # Gate geometry (physical: 40.5cm opening, 17cm frame width, 2cm thickness)
        frame_center_offset = self.gate_frame_center_offset  # 0.2875m - where frame bar centers are
        frame_radius = self.gate_frame_width / 2.0  # 0.085m - half-width of frame bars (capsule radius)
        safety_r = 0.05  # 5cm safety margin beyond frame surface
        effective_avoid_dist = frame_radius + safety_r  # ~0.135m total distance to avoid from frame center
        
        # Opening limits for horizontal frame collision check
        opening_half = self.gate_opening / 2.0  # 0.2025m
        
        # Weights for gate frame avoidance (strong to prevent collision with passed gates)
        w_vert_all = 1000.0  # Vertical frame weight
        w_horiz_all = 800.0  # Horizontal frame weight
        
        # Get live gate poses if available
        if obs is not None and "gates_pos" in obs and "gates_quat" in obs:
            gates_pos_live = obs["gates_pos"]
            gates_quat_live = obs["gates_quat"]
        else:
            gates_pos_live = self.gates_pos
            gates_quat_live = self.gates_quat
        
        # Iterate over all gates
        for gate_idx in range(len(self.gates_pos)):
            # Get gate pose
            gate_center_np = np.array(gates_pos_live[gate_idx], dtype=float)
            gate_quat_np = np.array(gates_quat_live[gate_idx], dtype=float)
            gate_R_np = R.from_quat(gate_quat_np).as_matrix()
            
            gate_R = torch.tensor(gate_R_np, dtype=self.dtype, device=self.device)
            gate_center = torch.tensor(gate_center_np, dtype=self.dtype, device=self.device)
            
            # Transform position into gate-local coordinates
            rel = pos - gate_center
            x_n = torch.sum(rel * gate_R[:, 0], dim=-1)  # Normal direction
            y_p = torch.sum(rel * gate_R[:, 1], dim=-1)  # Lateral direction
            z_p = torch.sum(rel * gate_R[:, 2], dim=-1)  # Vertical direction
            
            # Only apply avoidance when drone is near the gate plane (within 0.8m)
            near_gate = torch.abs(x_n) < 1.5
            
            # Vertical frames: poles centered at y = ±frame_center_offset (±0.35m)
            dy_left = torch.abs(y_p + frame_center_offset)
            dy_right = torch.abs(y_p - frame_center_offset)
            
            # Penalty when closer than effective_avoid_dist to vertical frame centers
            c_vert_left = torch.clamp(effective_avoid_dist - dy_left, min=0.0) ** 2
            c_vert_right = torch.clamp(effective_avoid_dist - dy_right, min=0.0) ** 2
            c_vert = w_vert_all * (c_vert_left + c_vert_right)
            
            # Horizontal frames: bars centered at z = ±frame_center_offset (±0.35m)
            dz_top = torch.abs(z_p - frame_center_offset)
            dz_bottom = torch.abs(z_p + frame_center_offset)
            # Only apply when within lateral span (where horizontal bars exist)
            within_span = torch.abs(y_p) <= (frame_center_offset + 0.05)
            
            c_horiz_top = torch.clamp(effective_avoid_dist - dz_top, min=0.0) ** 2
            c_horiz_bottom = torch.clamp(effective_avoid_dist - dz_bottom, min=0.0) ** 2
            c_horiz = torch.where(
                within_span,
                w_horiz_all * (c_horiz_top + c_horiz_bottom),
                torch.zeros_like(dz_top)
            )
            
            # Apply cost only when near the gate plane
            gate_cost = torch.where(near_gate, c_vert + c_horiz, torch.zeros_like(c_vert))
            
            # Add to total
            total_gate_cost = total_gate_cost + gate_cost
        
        return total_gate_cost

    def compute_running_cost(
        self,
        state: torch.Tensor,
        control: torch.Tensor
    ) -> torch.Tensor:
        """Simplified MPPI running cost - removed redundant terms.

        Args:
            state: State tensor (..., 12): [pos(3), rpy(3), vel(3), drpy(3)]
            control: Control tensor (..., 4): [roll_cmd, pitch_cmd, yaw_cmd, thrust_cmd]

        Returns:
            Cost tensor of shape (...)
        """
        # Extract state components
        pos = state[..., 0:3]
        rpy = state[..., 3:6]
        vel = state[..., 6:9]

        # 1. Position tracking - MAIN OBJECTIVE with distance-adaptive scaling
        goal_t = torch.tensor(self.goal, dtype=self.dtype, device=self.device)
        pos_error = pos - goal_t
        dist_to_goal = torch.norm(pos_error, dim=-1)
        # 2.2x position cost when within 0.4m for tighter precision at gate
        proximity_scale = torch.where(dist_to_goal < 0.4, 2.2, 1.0)
        c_pos = proximity_scale * torch.sum(self.cost_weights["position"] * pos_error ** 2, dim=-1)

        # 1a. Attraction to gate opening and obstacle avoidance for gate frames
        # Transform position into gate-local coordinates: [x_normal, y_plane, z_plane]
        gate_center_np = self.current_gate_center
        gate_quat_np = self.current_gate_quat
        gate_R_np = R.from_quat(gate_quat_np).as_matrix()
        gate_R = torch.tensor(gate_R_np, dtype=self.dtype, device=self.device)
        gate_center = torch.tensor(gate_center_np, dtype=self.dtype, device=self.device)
        # local coordinates
        rel = pos - gate_center
        x_n = torch.sum(rel * gate_R[:, 0], dim=-1)
        y_p = torch.sum(rel * gate_R[:, 1], dim=-1)
        z_p = torch.sum(rel * gate_R[:, 2], dim=-1)

        # Opening and frame geometry (physical measurement)
        # 40.5cm opening, 17cm frame width, 2cm thickness
        opening_half = self.gate_opening / 2.0  # 0.2025m inner opening edge
        frame_center_offset = self.gate_frame_center_offset  # 0.2875m frame bar center position
        frame_radius = self.gate_frame_width / 2.0  # 0.085m frame bar half-width (capsule radius)
        safety_r = 0.05  # 5cm safety margin beyond frame surface
        effective_avoid_dist = frame_radius + safety_r  # ~0.135m total avoid distance from frame center

        # Attraction: encourage being inside opening box in gate plane
        # Hinge loss if outside opening in y or z
        y_excess = torch.clamp(torch.abs(y_p) - opening_half, min=0.0)
        z_excess = torch.clamp(torch.abs(z_p) - opening_half, min=0.0)
        w_open = 40.0  # Strong opening attraction
        c_open_hinge = w_open * (y_excess ** 2 + z_excess ** 2)

        # Soft attraction toward the 2D opening center (y=0, z=0) when near the gate plane
        near_plane = torch.abs(x_n) < 0.8
        w_center = 15.0  # Strong centering for small opening
        c_center = torch.where(near_plane, w_center * (y_p ** 2 + z_p ** 2), torch.zeros_like(x_n))

        # Obstacle avoidance: vertical frames at y=±frame_center_offset (±0.2875m)
        dy_left = torch.abs(y_p + frame_center_offset)
        dy_right = torch.abs(y_p - frame_center_offset)
        w_vert = 600.0  # Strong vertical frame avoidance
        c_vert = w_vert * (torch.clamp(effective_avoid_dist - dy_left, min=0.0) ** 2 + torch.clamp(effective_avoid_dist - dy_right, min=0.0) ** 2)

        # Obstacle avoidance: horizontal frames at z=±frame_center_offset (±0.2875m)
        dz_top = torch.abs(z_p - frame_center_offset)
        dz_bottom = torch.abs(z_p + frame_center_offset)
        # Active only when within lateral span where horizontal bars exist
        within_span = torch.abs(y_p) <= (frame_center_offset + frame_radius + 0.05)
        w_horiz = 550.0  # Strong horizontal frame avoidance
        c_horiz = torch.where(within_span, w_horiz * (torch.clamp(effective_avoid_dist - dz_top, min=0.0) ** 2 + torch.clamp(effective_avoid_dist - dz_bottom, min=0.0) ** 2), torch.zeros_like(dz_top))

        c_gate_struct = c_open_hinge + c_center + c_vert + c_horiz

        # 2. Velocity penalty - light damping
        c_vel = torch.sum(self.cost_weights["velocity"] * vel ** 2, dim=-1)

        # 2a. Risk-aware slowdown: if too close to any frame, add speed penalty
        min_frame_dist = torch.minimum(
            torch.minimum(dy_left, dy_right),
            torch.minimum(dz_top, dz_bottom)
        )
        high_risk = min_frame_dist < effective_avoid_dist
        w_slow = 0.5
        c_slow = torch.where(high_risk, w_slow * torch.sum(vel ** 2, dim=-1), torch.zeros_like(min_frame_dist))

        # 3. Attitude - prefer level flight
        c_att = torch.sum(self.cost_weights["attitude"] * rpy ** 2, dim=-1)

        # 4. Ground collision penalty (critical safety constraint)
        z_violation = torch.clamp(0.03 - pos[..., 2], min=0.0)
        c_z_floor = self.cost_weights["z_floor"] * z_violation ** 2
        
        # 5. Obstacle avoidance cost
        c_obstacle = self.compute_obstacle_cost(pos)
        
        # 6. All gates avoidance cost (prevents collision with passed/future gates)
        c_all_gates = self.compute_all_gates_avoidance_cost(pos)

        # Total cost (simplified) + gate structure shaping and risk-aware slowdown + all gates avoidance
        total_cost = c_pos + c_vel + c_att + c_z_floor + c_gate_struct + c_slow + c_obstacle + c_all_gates
        
        # Debug: Print obstacle cost if significant
        if torch.any(c_obstacle > 1.0):
            max_obstacle_cost = torch.max(c_obstacle).item()
            if self.step_count % 20 == 0:
                print(f"[AttitudeMPPI] Max obstacle cost: {max_obstacle_cost:.2f}")

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
        # curr_time = time()
        self.step_count += 1
        
        # Update obstacles information
        if "obstacles_pos" in obs:
            self.obstacles_pos = obs["obstacles_pos"].copy()
            if self.step_count % 50 == 0 and len(self.obstacles_pos) > 0:
                print(f"[AttitudeMPPI] Tracking {len(self.obstacles_pos)} obstacles")
                for i, obs_pos in enumerate(self.obstacles_pos):
                    print(f"  Obstacle {i}: [{obs_pos[0]:.2f}, {obs_pos[1]:.2f}, {obs_pos[2]:.2f}]")

        # Update live gate pose for current target gate index (will be refined below if index changes)
        try:
            self.current_gate_center = np.array(obs["gates_pos"][int(obs["target_gate"])], dtype=float)
            self.current_gate_quat = np.array(obs["gates_quat"][int(obs["target_gate"])], dtype=float)
        except Exception:
            # Fallback to previously cached values
            pass

        # Decrement pause counter each step
        if self.pause_counter > 0:
            self.pause_counter -= 1
        
        # Goal update logic with 3 cases
        new_target_idx = int(obs["target_gate"])
        drone_pos = np.array(obs["pos"], dtype=float)
        
        # CASE 1: Target index changed (deferred if pause is active)
        if new_target_idx != self.target_gate_idx and self.pause_counter == 0:
            self.old_gate_pos = self.gates_pos[self.target_gate_idx]
            self.target_gate_idx = new_target_idx
            self.prev_goal = self.goal.copy()
            self.goal = self.gates_pos[self.target_gate_idx].copy()
            self.goal_shifted = False  # Reset flag for new gate
            print(f"\n[AttitudeMPPI] Case 1: Updated target to gate {self.target_gate_idx}")
            
            # Reset control sequence on gate change
            initial_control = torch.zeros((self.mppi_horizon, 4), dtype=self.dtype, device=self.device)
            initial_control[:, 3] = self.hover_thrust
            self.mppi.U = initial_control
            
            # Update live gate pose after index change
            try:
                self.current_gate_center = np.array(obs["gates_pos"][self.target_gate_idx], dtype=float)
                self.current_gate_quat = np.array(obs["gates_quat"][self.target_gate_idx], dtype=float)
            except Exception:
                pass
        elif self.pause_counter > 0 and new_target_idx != self.target_gate_idx:
            # During pause, maintain current goal and don't switch targets yet
            if self.step_count % 10 == 0:
                print(f"[AttitudeMPPI] Pause active ({self.pause_counter} steps remaining) - deferring gate {self.target_gate_idx} -> {new_target_idx}")
        else:
            # CASE 2: Same index, but gate position given more precisely
            new_gate_pos = np.array(obs["gates_pos"][self.target_gate_idx], dtype=float)
            if np.any(new_gate_pos != self.old_gate_pos):
                self.old_gate_pos = new_gate_pos.copy()
                self.prev_goal = self.goal.copy()
                self.goal = new_gate_pos.copy()
                self.goal_shifted = False  # Reset flag on gate position update
                print(f"\n[AttitudeMPPI] Case 2: Gate position updated (gate {self.target_gate_idx})")
                
                # Reset control sequence on goal position change
                initial_control = torch.zeros((self.mppi_horizon, 4), dtype=self.dtype, device=self.device)
                initial_control[:, 3] = self.hover_thrust
                self.mppi.U = initial_control
                
                # Update live gate pose after goal update
                try:
                    self.current_gate_center = np.array(obs["gates_pos"][self.target_gate_idx], dtype=float)
                    self.current_gate_quat = np.array(obs["gates_quat"][self.target_gate_idx], dtype=float)
                except Exception:
                    pass
            
            # CASE 3: Drone is within 5cm before the gate opening plane, move goal forward (only once)
            # Calculate signed distance from drone to gate plane (along gate normal)
            gate_rot = R.from_quat(self.current_gate_quat)
            gate_normal = gate_rot.as_matrix()[:, 0]  # X-axis is the normal to the opening plane
            
            rel_pos = drone_pos - self.current_gate_center
            dist_to_plane = np.dot(rel_pos, gate_normal)  # Signed distance along normal (negative = before the plane)
            
            if not self.goal_shifted and -0.15 < dist_to_plane < 0:  # Within 5cm before gate, and not yet shifted
                self.goal_shifted = True  # Mark that we've shifted for this gate
                self.prev_goal = self.goal.copy()
                # Get forward direction from gate rotation
                gate_quat = np.array(obs["gates_quat"][self.target_gate_idx], dtype=float)
                rot = R.from_quat(gate_quat)
                forward = rot.as_matrix()[:, 0]
                # Move goal 8cm forward
                self.goal = self.goal + 0.2 * forward
                # Activate pause to maintain shifted goal and avoid immediate gate switching
                self.pause_counter = self.pause_duration
                print(f"\n[AttitudeMPPI] Case 3: Drone within 5cm before gate (dist={dist_to_plane:.3f}m), "
                      f"shifting goal forward and activating {self.pause_duration}-step pause to: {self.goal}")
                
                # Reset control sequence on goal progression
                initial_control = torch.zeros((self.mppi_horizon, 4), dtype=self.dtype, device=self.device)
                initial_control[:, 3] = self.hover_thrust
                self.mppi.U = initial_control

        # Ensure live pose is aligned with the final target index for this step
        try:
            self.current_gate_center = np.array(obs["gates_pos"][self.target_gate_idx], dtype=float)
            self.current_gate_quat = np.array(obs["gates_quat"][self.target_gate_idx], dtype=float)
        except Exception:
            pass

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

        # Enhanced debug output
        if self.step_count % 10 == 0:
            pos = obs["pos"]
            vel = obs["vel"]
            gate_pos = self.gates_pos[self.target_gate_idx]
            dist_to_gate = np.linalg.norm(pos - gate_pos)
            z_offset = pos[2] - gate_pos[2]
            xy_dist = np.linalg.norm(pos[:2] - gate_pos[:2])
            speed = np.linalg.norm(vel)
            
            # Check proximity to obstacles
            obstacle_proximity = float('inf')
            if len(self.obstacles_pos) > 0:
                # Calculate distance to nearest obstacle
                distances = np.linalg.norm(pos - self.obstacles_pos, axis=1)
                obstacle_proximity = np.min(distances) - self.obstacle_radius
                
            print(f"[AttitudeMPPI] Step {self.step_count:4d} | "
                  f"G{self.target_gate_idx} | "
                  f"Pos:[{pos[0]:5.2f},{pos[1]:5.2f},{pos[2]:5.2f}] | "
                  f"3D:{dist_to_gate:4.2f}m XY:{xy_dist:4.2f}m ΔZ:{z_offset:+.2f}m | "
                  f"V:{speed:.1f}m/s | "
                  f"T:{control_np[3]:.2f}N | "
                  f"Obs:{obstacle_proximity:.2f}m")
        # print(f"[AttitudeMPPI] Computation time: {time() - curr_time:.4f}s")
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
        self.obstacles_pos = np.array([])
        self.pause_counter = 0
        
        print("[AttitudeMPPI] Episode reset")

    def get_optimal_trajectory(self, obs: dict) -> np.ndarray:
        """Get the optimal MPPI trajectory from current observation.
        
        This simulates forward using the optimal control sequence (mppi.U),
        which represents the TRUE optimal trajectory (weighted combination of all samples).
        
        Args:
            obs: Current observation dictionary
            
        Returns:
            Array of shape (T, 3) containing the planned position trajectory
        """
        try:
            # Convert observation to state tensor
            current_state = self.drone_model.obs_to_state(obs)
            
            # Ensure state is batched (shape [1, nx])
            if current_state.ndim == 1:
                current_state = current_state.unsqueeze(0)
            
            # Get optimal control sequence (weighted average from MPPI)
            optimal_controls = self.mppi.U  # Shape: [T, 4]
            
            # Simulate trajectory forward (keep everything batched)
            state_tensor = current_state.clone()  # Shape: [1, nx]
            trajectory_positions = [state_tensor[0, 0:3].cpu().numpy()]  # Extract position only
            
            for t in range(min(self.mppi_horizon, 25)):  # Limit to 25 steps for visibility
                control_t = optimal_controls[t].unsqueeze(0)  # Shape: [1, 4]
                state_tensor = self.drone_model.dynamics(
                    state_tensor, 
                    control_t, 
                    self.mppi_dt
                )
                # Ensure result is batched
                if state_tensor.ndim == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                # Extract position (first 3 elements) and store
                trajectory_positions.append(state_tensor[0, 0:3].cpu().numpy())
            
            # Convert list to numpy array
            pos_traj = np.array(trajectory_positions)  # Shape: [T, 3]
            return pos_traj
        except Exception as e:
            print(f"[AttitudeMPPI] Error computing optimal trajectory: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])

    def get_planned_trajectory(self) -> np.ndarray:
        """Get the optimal planned trajectory from the last MPPI optimization.
        
        Returns:
            Array of shape (T, 3) containing the planned position trajectory
        """
        try:
            # Simulate forward using the optimal control sequence (mppi.U)
            # This represents the TRUE optimal trajectory (weighted combination of all samples)
            current_state = self.mppi.U.new_zeros(1, self.drone_model.nx)
            # Note: This returns trajectory from zero state. For actual state, need to pass obs.
            # Better to get from sim.py where we have access to current obs.
            
            # Get optimal control sequence
            optimal_controls = self.mppi.U  # Shape: [T, 4]
            
            # Simulate trajectory forward
            trajectory = []
            state_tensor = current_state.clone()
            
            for t in range(min(self.mppi_horizon, 25)):
                trajectory.append(state_tensor.clone())
                control_t = optimal_controls[t].unsqueeze(0)
                state_tensor = self.drone_model.dynamics(
                    state_tensor, 
                    control_t, 
                    self.mppi_dt
                )
            
            # Convert to numpy array
            trajectory_np = torch.cat(trajectory, dim=0).cpu().numpy()
            pos_traj = trajectory_np[:, 0:3]
            return pos_traj
        except Exception:
            return np.array([])