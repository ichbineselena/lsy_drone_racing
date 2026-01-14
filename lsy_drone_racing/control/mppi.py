"""Improved pure MPPI controller for quadrotor attitude control.  

This module implements a robust MPPI controller that directly outputs
roll-pitch-yaw and thrust commands with better cost shaping and dynamics. 
Enhanced with structured gate traversal and dynamic corridor constraints.
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Optional, List, Tuple

import numpy as np
import torch
from scipy.spatial.transform import Rotation as R

# Import PyTorch MPPI
import pytorch_mppi

# Import our PyTorch dynamics model
from drone_models.core import load_params
from lsy_drone_racing.dynamics.model_torch import DroneModelTorch

from lsy_drone_racing.control import Controller

if TYPE_CHECKING: 
    from numpy.typing import NDArray


class GatePhase(Enum):
    """Gate traversal phase state machine."""
    APPROACH = 0
    ALIGN = 1
    TRAVERSE = 2
    EXIT = 3


class AttitudeMPPIController(Controller):
    """Improved pure MPPI controller for attitude control (RPY + Thrust)."""

    def __init__(
        self,
        obs:  dict[str, NDArray[np.floating]],
        info: dict,
        config: dict,
    ):
        """Initialize the MPPI attitude controller."""
        super().__init__(obs, info, config)

        # Simulation parameters
        self._dt = 1 / config.env.freq
        self._config = config

        # MPPI hyperparameters
        self.mppi_horizon = 25
        self.mppi_dt = self._dt * 2
        self.num_samples = 6000
        self.lambda_weight = 9.5

        # Gate geometry constants
        self.gate_opening = 0.405
        self.gate_frame_width = 0.17
        self.gate_frame_thickness = 0.02
        self.gate_frame_center_offset = 0.2875

        # Obstacle geometry constants
        self.obstacle_radius = 0.015
        self.obstacle_half_length = 1.5
        self.obstacle_safety_margin = 0.05
        self.obstacle_avoidance_weight = 50.0

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
        self.num_gates = len(self.gates_pos)

        # Current target
        self.target_gate_idx = int(obs["target_gate"])

        # Gate traversal configuration - RELAXED for less hesitation
        self.traversal_config = {
            "approach_distance": 0.35,       # Reduced from 0.45
            "exit_distance": 0.45,           # Increased from 0.30 - more clearance before turning
            "align_threshold": 0.12,         # Reduced from 0.20
            "traverse_threshold": 0.03,      # Reduced from 0.05
            "exit_threshold": 0.20,          # Reduced from 0.30
            "max_gate_speed": 2.5,           # Increased from 1.8
            "corridor_far_radius": 0.8,      # Increased from 0.6
            "corridor_activation_dist": 0.4, # Reduced from 0.8 - only activate close to gate
        }

        # Initialize waypoint system
        self._init_waypoint_system(obs)

        # Gate phase tracking
        self.gate_phase = GatePhase.APPROACH
        self.phase_entry_time = 0

        # Live gate pose tracking
        self.current_gate_center = np.array(obs["gates_pos"][self.target_gate_idx], dtype=float)
        self.current_gate_quat = np.array(obs["gates_quat"][self.target_gate_idx], dtype=float)
        self.current_gate_normal = self._compute_gate_normal(self.current_gate_quat)

        # Obstacle tracking
        self.obstacles_pos = obs.get("obstacles_pos", np.array([])).copy()

        # Pause mechanism
        self.pause_counter = 0
        self.pause_duration = 10

        # Recently passed gate tracking for enhanced avoidance
        self.recently_passed_gate = -1
        self.recently_passed_decay = 0  # Steps since passing

        # Control limits
        self.rpy_max = 0.5
        self.thrust_min = self.drone_params["thrust_min"] * 4
        self.thrust_max = self.drone_params["thrust_max"] * 4
        self.hover_thrust = self.drone_params["mass"] * abs(self.drone_params["gravity_vec"][-1])

        # Cost weights - REBALANCED:  reduced new costs, kept original working values
        self.cost_weights = {
            # Position tracking (original values that worked)
            "position":  torch.tensor([20.0, 20.0, 16.0], device=self.device, dtype=self.dtype),
            "position_proximity_scale": 2.2,
            "proximity_threshold": 0.4,

            # Velocity (original values)
            "velocity": torch.tensor([0.040, 0.040, 0.125], device=self.device, dtype=self.dtype),
            
            # NEW costs - significantly reduced to avoid over-constraining
            "velocity_alignment": 3.0,        # Reduced from 15.0
            "speed_gate_penalty": 1.0,        # Reduced from 5.0

            # Attitude (original)
            "attitude": torch.tensor([1.5, 1.5, 0.2], device=self.device, dtype=self.dtype),

            # Safety (original)
            "z_floor": 2000.0,
            "obstacle":  self.obstacle_avoidance_weight,

            # Gate structure - REDUCED to be less aggressive
            "gate_opening": 40.0,             # Keep original
            "gate_center":  15.0,              # Keep original
            "gate_frame_vertical":  400.0,     # Reduced from 600.0
            "gate_frame_horizontal": 350.0,   # Reduced from 550.0
            "corridor":  50.0,                 # Significantly reduced from 200.0
            "risk_slowdown": 0.3,             # Reduced from 0.5

            # All gates avoidance - INCREASED for safety after passing
            "all_gates_vertical": 900.0,      # Increased from 600.0
            "all_gates_horizontal": 750.0,    # Increased from 500.0
            
            # Recently passed gate gets extra avoidance
            "recently_passed_multiplier": 2.5,  # Extra weight for just-passed gates
            "recently_passed_decay_steps": 50,  # Steps to decay the extra weight
        }

        # Store previous control
        self.prev_control = None

        # Define dynamics wrapper for MPPI
        def dynamics_fn(state, control):
            return self.drone_model.dynamics(state, control, self.mppi_dt)

        def running_cost_fn(state, control):
            return self.compute_running_cost(state, control)

        # Control bounds
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

        # Control noise
        noise_sigma = torch.diag(torch.tensor(
            [0.05, 0.05, 0.05, 0.2],
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
        self._reset_control_sequence()

        self.step_count = 0

        self._print_init_info()

    def _print_init_info(self):
        """Print initialization information."""
        print("[AttitudeMPPI] Initialization complete")
        print(f"  - Horizon: {self.mppi_horizon} steps @ {self.mppi_dt:.3f}s = {self.mppi_horizon * self.mppi_dt:.2f}s")
        print(f"  - Samples: {self.num_samples}")
        print(f"  - Lambda: {self.lambda_weight}")
        print(f"  - RPY limits: ±{np.rad2deg(self.rpy_max):.1f}°")
        print(f"  - Gate:  {self.gate_opening*100:.1f}cm opening, {self.gate_frame_width*100:.0f}cm frame width")
        print(f"  - Number of gates: {self.num_gates}")

    def _reset_control_sequence(self):
        """Reset MPPI control sequence to hover."""
        initial_control = torch.zeros((self.mppi_horizon, 4), dtype=self.dtype, device=self.device)
        initial_control[: , 3] = self.hover_thrust
        self.mppi.U = initial_control

    def _compute_gate_normal(self, gate_quat: np.ndarray) -> np.ndarray:
        """Compute gate normal from quaternion."""
        rot = R.from_quat(gate_quat)
        return rot.as_matrix()[:, 0]

    def _compute_gate_axes(self, gate_quat:  np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute all gate axes from quaternion."""
        rot = R.from_quat(gate_quat)
        mat = rot.as_matrix()
        return mat[:, 0], mat[: , 1], mat[:, 2]

    # =========================================================================
    # Waypoint System - Simplified
    # =========================================================================

    def _init_waypoint_system(self, obs: dict):
        """Initialize the waypoint system."""
        self.waypoints = []
        self.waypoint_index = 0
        self._generate_gate_waypoints(obs)

        if len(self.waypoints) > 0:
            self.goal = self.waypoints[self.waypoint_index].copy()
        else:
            self.goal = self.gates_pos[self.target_gate_idx].copy()

        self.prev_goal = self.goal.copy()
        self.old_gate_pos = self.goal.copy()

    def _generate_gate_waypoints(self, obs: dict):
        """Generate waypoints for current target gate."""
        if self.target_gate_idx < 0 or self.target_gate_idx >= self.num_gates:
            return

        gate_center = np.array(obs["gates_pos"][self.target_gate_idx], dtype=float)
        gate_quat = np.array(obs["gates_quat"][self.target_gate_idx], dtype=float)
        gate_normal = self._compute_gate_normal(gate_quat)

        approach_dist = self.traversal_config["approach_distance"]
        exit_dist = self.traversal_config["exit_distance"]

        # Simple waypoints:  approach -> center -> exit
        approach_point = gate_center - approach_dist * gate_normal
        exit_point = gate_center + exit_dist * gate_normal

        self.waypoints = [approach_point, gate_center, exit_point]
        self.waypoint_index = 0

        # Check for detour only if coming from a previous gate at sharp angle
        if self.target_gate_idx > 0:
            self._check_and_add_detour(obs, gate_center, gate_quat)

    def _check_and_add_detour(self, obs: dict, gate_center: np.ndarray, gate_quat: np.ndarray):
        """Check if detour waypoint is needed for sharp turns."""
        prev_gate_idx = self.target_gate_idx - 1
        prev_gate_center = np.array(obs["gates_pos"][prev_gate_idx], dtype=float)

        gate_to_gate = gate_center - prev_gate_center
        gate_to_gate_norm = np.linalg.norm(gate_to_gate)

        if gate_to_gate_norm < 1e-6:
            return

        gate_to_gate_unit = gate_to_gate / gate_to_gate_norm
        gate_normal, gate_y, gate_z = self._compute_gate_axes(gate_quat)

        cos_angle = np.clip(np.dot(gate_to_gate_unit, gate_normal), -1.0, 1.0)
        angle_deg = np.degrees(np.arccos(cos_angle))

        # Only add detour for very sharp angles (>130°)
        angle_threshold = 130.0
        if angle_deg > angle_threshold: 
            detour_distance = 0.55

            v_proj = gate_to_gate_unit - np.dot(gate_to_gate_unit, gate_normal) * gate_normal
            v_proj_norm = np.linalg.norm(v_proj)

            if v_proj_norm < 1e-6:
                detour_direction = gate_y
            else:
                v_proj_y = np.dot(v_proj, gate_y)
                v_proj_z = np.dot(v_proj, gate_z)
                proj_angle = np.degrees(np.arctan2(v_proj_z, v_proj_y))

                if -90 <= proj_angle < 45:
                    detour_direction = gate_y
                elif 45 <= proj_angle < 135:
                    detour_direction = gate_z
                else: 
                    detour_direction = -gate_y

            detour_waypoint = gate_center + detour_distance * detour_direction
            self.waypoints.insert(0, detour_waypoint)

            print(f"[AttitudeMPPI] Added detour waypoint for gate {self.target_gate_idx} "
                  f"(angle={angle_deg:.1f}°)")

    def _update_waypoint_progress(self, drone_pos: np.ndarray):
        """Update waypoint index based on drone position."""
        if len(self.waypoints) == 0:
            return

        current_waypoint = self.waypoints[self.waypoint_index]
        dist_to_waypoint = np.linalg.norm(drone_pos - current_waypoint)

        # Use consistent threshold for all waypoints
        threshold = 0.18

        if dist_to_waypoint < threshold and self.waypoint_index < len(self.waypoints) - 1:
            self.waypoint_index += 1
            self.prev_goal = self.goal.copy()
            self.goal = self.waypoints[self.waypoint_index].copy()
            print(f"[AttitudeMPPI] Advanced to waypoint {self.waypoint_index}/{len(self.waypoints)-1}")

    # =========================================================================
    # Gate Phase Management - Simplified
    # =========================================================================

    def _update_gate_phase(self, drone_pos: np.ndarray):
        """Update gate traversal phase based on drone position."""
        rel_pos = drone_pos - self.current_gate_center
        dist_to_plane = np.dot(rel_pos, self.current_gate_normal)

        config = self.traversal_config

        if self.gate_phase == GatePhase.APPROACH:
            if dist_to_plane > -config["align_threshold"]:
                self.gate_phase = GatePhase.ALIGN
                self.phase_entry_time = self.step_count

        elif self.gate_phase == GatePhase.ALIGN: 
            if dist_to_plane > config["traverse_threshold"]:
                self.gate_phase = GatePhase.TRAVERSE
                self.phase_entry_time = self.step_count
                self.pause_counter = self.pause_duration

        elif self.gate_phase == GatePhase.TRAVERSE:
            if dist_to_plane > config["exit_threshold"]:
                self.gate_phase = GatePhase.EXIT
                self.phase_entry_time = self.step_count

    # =========================================================================
    # Cost Functions
    # =========================================================================

    def compute_obstacle_cost(self, pos: torch.Tensor) -> torch.Tensor:
        """Compute obstacle avoidance cost."""
        if len(self.obstacles_pos) == 0:
            return torch.zeros(pos.shape[:-1], dtype=self.dtype, device=self.device)

        obstacles_t = torch.tensor(self.obstacles_pos, dtype=self.dtype, device=self.device)
        total_cost = torch.zeros(pos.shape[:-1], dtype=self.dtype, device=self.device)

        for i in range(len(self.obstacles_pos)):
            obstacle_center = obstacles_t[i]

            radius = self.obstacle_radius
            half_length = self.obstacle_half_length
            safety_margin = self.obstacle_safety_margin
            effective_radius = radius + safety_margin

            rel_vec = pos - obstacle_center
            z_proj = rel_vec[..., 2]
            z_clamped = torch.clamp(z_proj, -half_length, half_length)

            closest_point_z = obstacle_center[2] + z_clamped
            closest_point = torch.stack([
                torch.full_like(z_clamped, obstacle_center[0]),
                torch.full_like(z_clamped, obstacle_center[1]),
                closest_point_z
            ], dim=-1)

            dist_vec = pos - closest_point
            dist = torch.norm(dist_vec, dim=-1)

            penalty_threshold = effective_radius * 2.0
            surface_dist = dist - effective_radius

            mask = surface_dist < penalty_threshold

            if torch.any(mask):
                penalty = torch.exp(-surface_dist[mask] / (effective_radius * 0.5))
                inv_dist_sq = 1.0 / (surface_dist[mask] ** 2 + 0.01)
                obstacle_cost = self.cost_weights["obstacle"] * penalty * inv_dist_sq

                total_cost = total_cost.clone()
                total_cost[mask] = total_cost[mask] + obstacle_cost

        return total_cost

    def compute_corridor_cost(
        self,
        pos: torch.Tensor,
        x_n: torch.Tensor,
        y_p: torch.Tensor,
        z_p: torch.Tensor
    ) -> torch.Tensor:
        """Compute corridor constraint cost - ONLY active very close to gate."""
        opening_half = self.gate_opening / 2.0
        config = self.traversal_config

        dist_to_plane = torch.abs(x_n)

        # Only activate corridor when very close to gate plane
        activation_dist = config["corridor_activation_dist"]
        
        # No cost if far from gate
        far_from_gate = dist_to_plane > activation_dist
        
        max_corridor = config["corridor_far_radius"]

        # Linear interpolation only when close
        t = torch.clamp(dist_to_plane / activation_dist, 0.0, 1.0)
        corridor_radius = opening_half + (max_corridor - opening_half) * t

        lateral_dist = torch.sqrt(y_p**2 + z_p**2)
        corridor_violation = torch.clamp(lateral_dist - corridor_radius, min=0.0)

        # Zero cost when far from gate
        cost = self.cost_weights["corridor"] * corridor_violation ** 2
        return torch.where(far_from_gate, torch.zeros_like(cost), cost)

    def compute_velocity_alignment_cost(
        self,
        vel: torch.Tensor,
        x_n: torch.Tensor
    ) -> torch.Tensor:
        """Compute velocity alignment cost - ONLY when very close to gate plane."""
        gate_normal_t = torch.tensor(
            self.current_gate_normal,
            dtype=self.dtype,
            device=self.device
        )

        vel_along_normal = torch.sum(vel * gate_normal_t, dim=-1, keepdim=True)
        vel_perpendicular = vel - vel_along_normal * gate_normal_t

        # Only apply when VERY close to gate plane (within 15cm)
        near_gate_plane = torch.abs(x_n) < 0.15

        perp_vel_sq = torch.sum(vel_perpendicular ** 2, dim=-1)

        return torch.where(
            near_gate_plane,
            self.cost_weights["velocity_alignment"] * perp_vel_sq,
            torch.zeros_like(x_n)
        )

    def compute_speed_gate_cost(
        self,
        vel: torch.Tensor,
        dist_to_goal: torch.Tensor
    ) -> torch.Tensor:
        """Compute speed penalty near gates - ONLY when very close."""
        speed = torch.norm(vel, dim=-1)
        
        # Only penalize when very close to goal (within 25cm)
        very_close = dist_to_goal < 0.25

        max_gate_speed = self.traversal_config["max_gate_speed"]
        speed_excess = torch.clamp(speed - max_gate_speed, min=0.0)

        return torch.where(
            very_close,
            self.cost_weights["speed_gate_penalty"] * speed_excess ** 2,
            torch.zeros_like(speed)
        )

    def compute_gate_structure_cost(
        self,
        x_n: torch.Tensor,
        y_p: torch.Tensor,
        z_p: torch.Tensor
    ) -> torch.Tensor:
        """Compute gate structure costs (opening attraction + frame avoidance)."""
        opening_half = self.gate_opening / 2.0
        frame_center_offset = self.gate_frame_center_offset
        frame_radius = self.gate_frame_width / 2.0
        safety_r = 0.06  # Slightly reduced from 0.08
        effective_avoid_dist = frame_radius + safety_r

        # Opening attraction:  hinge loss if outside opening
        y_excess = torch.clamp(torch.abs(y_p) - opening_half, min=0.0)
        z_excess = torch.clamp(torch.abs(z_p) - opening_half, min=0.0)
        c_open_hinge = self.cost_weights["gate_opening"] * (y_excess ** 2 + z_excess ** 2)

        # Center attraction when near gate plane
        near_plane = torch.abs(x_n) < 0.6  # Reduced from 0.8
        c_center = torch.where(
            near_plane,
            self.cost_weights["gate_center"] * (y_p ** 2 + z_p ** 2),
            torch.zeros_like(x_n)
        )

        # Frame avoidance - only when close to gate plane
        very_near_plane = torch.abs(x_n) < 0.3
        
        # Vertical frame avoidance
        dy_left = torch.abs(y_p + frame_center_offset)
        dy_right = torch.abs(y_p - frame_center_offset)
        c_vert_raw = self.cost_weights["gate_frame_vertical"] * (
            torch.clamp(effective_avoid_dist - dy_left, min=0.0) ** 2 +
            torch.clamp(effective_avoid_dist - dy_right, min=0.0) ** 2
        )
        c_vert = torch.where(very_near_plane, c_vert_raw, torch.zeros_like(c_vert_raw))

        # Horizontal frame avoidance
        dz_top = torch.abs(z_p - frame_center_offset)
        dz_bottom = torch.abs(z_p + frame_center_offset)
        within_span = torch.abs(y_p) <= (frame_center_offset + frame_radius + 0.05)
        c_horiz_raw = self.cost_weights["gate_frame_horizontal"] * (
            torch.clamp(effective_avoid_dist - dz_top, min=0.0) ** 2 +
            torch.clamp(effective_avoid_dist - dz_bottom, min=0.0) ** 2
        )
        c_horiz = torch.where(
            very_near_plane & within_span,
            c_horiz_raw,
            torch.zeros_like(c_horiz_raw)
        )

        return c_open_hinge + c_center + c_vert + c_horiz

    def compute_all_gates_avoidance_cost(
        self,
        pos: torch.Tensor,
        obs: dict = None
    ) -> torch.Tensor:
        """Compute gate frame avoidance cost for ALL gates - with distance gating."""
        total_gate_cost = torch.zeros(pos.shape[:-1], dtype=self.dtype, device=self.device)

        frame_center_offset = self.gate_frame_center_offset
        frame_radius = self.gate_frame_width / 2.0
        safety_r = 0.07  # Increased from 0.05 for more margin
        effective_avoid_dist = frame_radius + safety_r

        if obs is not None and "gates_pos" in obs and "gates_quat" in obs:
            gates_pos_live = obs["gates_pos"]
            gates_quat_live = obs["gates_quat"]
        else:
            gates_pos_live = self.gates_pos
            gates_quat_live = self.gates_quat

        for gate_idx in range(len(self.gates_pos)):
            # Skip current target gate (handled by compute_gate_structure_cost)
            if gate_idx == self.target_gate_idx:
                continue
            
            # Compute weight multiplier for recently passed gates
            weight_multiplier = 1.0
            if gate_idx == self.recently_passed_gate and self.recently_passed_decay > 0:
                decay_steps = self.cost_weights["recently_passed_decay_steps"]
                decay_factor = self.recently_passed_decay / decay_steps
                weight_multiplier = 1.0 + (self.cost_weights["recently_passed_multiplier"] - 1.0) * decay_factor
                
            gate_center_np = np.array(gates_pos_live[gate_idx], dtype=float)
            gate_quat_np = np.array(gates_quat_live[gate_idx], dtype=float)
            gate_R_np = R.from_quat(gate_quat_np).as_matrix()

            gate_R = torch.tensor(gate_R_np, dtype=self.dtype, device=self.device)
            gate_center = torch.tensor(gate_center_np, dtype=self.dtype, device=self.device)

            rel = pos - gate_center
            x_n = torch.sum(rel * gate_R[: , 0], dim=-1)
            y_p = torch.sum(rel * gate_R[: , 1], dim=-1)
            z_p = torch.sum(rel * gate_R[: , 2], dim=-1)

            # Increased activation distance for recently passed gates
            activation_dist = 0.7 if gate_idx == self.recently_passed_gate else 0.5
            near_this_gate = torch.abs(x_n) < activation_dist

            # Vertical frames
            dy_left = torch.abs(y_p + frame_center_offset)
            dy_right = torch.abs(y_p - frame_center_offset)
            c_vert = weight_multiplier * self.cost_weights["all_gates_vertical"] * (
                torch.clamp(effective_avoid_dist - dy_left, min=0.0) ** 2 +
                torch.clamp(effective_avoid_dist - dy_right, min=0.0) ** 2
            )

            # Horizontal frames
            dz_top = torch.abs(z_p - frame_center_offset)
            dz_bottom = torch.abs(z_p + frame_center_offset)
            within_span = torch.abs(y_p) <= (frame_center_offset + 0.08)  # Increased from 0.05
            c_horiz = torch.where(
                within_span,
                weight_multiplier * self.cost_weights["all_gates_horizontal"] * (
                    torch.clamp(effective_avoid_dist - dz_top, min=0.0) ** 2 +
                    torch.clamp(effective_avoid_dist - dz_bottom, min=0.0) ** 2
                ),
                torch.zeros_like(dz_top)
            )

            # Only add cost when near this gate
            gate_cost = torch.where(near_this_gate, c_vert + c_horiz, torch.zeros_like(c_vert))
            total_gate_cost = total_gate_cost + gate_cost

        return total_gate_cost

    def compute_risk_slowdown_cost(
        self,
        vel: torch.Tensor,
        x_n: torch.Tensor,
        y_p: torch.Tensor,
        z_p: torch.Tensor
    ) -> torch.Tensor:
        """Compute speed penalty when too close to gate frames."""
        frame_center_offset = self.gate_frame_center_offset
        frame_radius = self.gate_frame_width / 2.0
        safety_r = 0.06
        effective_avoid_dist = frame_radius + safety_r

        dy_left = torch.abs(y_p + frame_center_offset)
        dy_right = torch.abs(y_p - frame_center_offset)
        dz_top = torch.abs(z_p - frame_center_offset)
        dz_bottom = torch.abs(z_p + frame_center_offset)

        min_frame_dist = torch.minimum(
            torch.minimum(dy_left, dy_right),
            torch.minimum(dz_top, dz_bottom)
        )

        # Only apply when near gate plane AND close to frame
        near_plane = torch.abs(x_n) < 0.25
        high_risk = (min_frame_dist < effective_avoid_dist) & near_plane

        return torch.where(
            high_risk,
            self.cost_weights["risk_slowdown"] * torch.sum(vel ** 2, dim=-1),
            torch.zeros_like(min_frame_dist)
        )

    def compute_running_cost(
        self,
        state: torch.Tensor,
        control: torch.Tensor
    ) -> torch.Tensor:
        """Compute MPPI running cost with all components."""
        # Extract state components
        pos = state[..., 0:3]
        rpy = state[..., 3:6]
        vel = state[..., 6:9]

        # 1.Position tracking with distance-adaptive scaling (MAIN DRIVER)
        goal_t = torch.tensor(self.goal, dtype=self.dtype, device=self.device)
        pos_error = pos - goal_t
        dist_to_goal = torch.norm(pos_error, dim=-1)

        proximity_scale = torch.where(
            dist_to_goal < self.cost_weights["proximity_threshold"],
            self.cost_weights["position_proximity_scale"],
            1.0
        )
        c_pos = proximity_scale * torch.sum(self.cost_weights["position"] * pos_error ** 2, dim=-1)

        # 2.Transform to gate-local coordinates
        gate_center_np = self.current_gate_center
        gate_quat_np = self.current_gate_quat
        gate_R_np = R.from_quat(gate_quat_np).as_matrix()
        gate_R = torch.tensor(gate_R_np, dtype=self.dtype, device=self.device)
        gate_center = torch.tensor(gate_center_np, dtype=self.dtype, device=self.device)

        rel = pos - gate_center
        x_n = torch.sum(rel * gate_R[:, 0], dim=-1)
        y_p = torch.sum(rel * gate_R[:, 1], dim=-1)
        z_p = torch.sum(rel * gate_R[:, 2], dim=-1)

        # 3.Gate structure cost (opening + frames) - primary gate shaping
        c_gate_struct = self.compute_gate_structure_cost(x_n, y_p, z_p)

        # 4.Corridor constraint - only close to gate
        c_corridor = self.compute_corridor_cost(pos, x_n, y_p, z_p)

        # 5.Velocity costs
        c_vel = torch.sum(self.cost_weights["velocity"] * vel ** 2, dim=-1)
        c_vel_align = self.compute_velocity_alignment_cost(vel, x_n)
        c_speed_gate = self.compute_speed_gate_cost(vel, dist_to_goal)

        # 6.Risk-aware slowdown
        c_slow = self.compute_risk_slowdown_cost(vel, x_n, y_p, z_p)

        # 7.Attitude regularization
        c_att = torch.sum(self.cost_weights["attitude"] * rpy ** 2, dim=-1)

        # 8.Ground collision penalty
        z_violation = torch.clamp(0.03 - pos[..., 2], min=0.0)
        c_z_floor = self.cost_weights["z_floor"] * z_violation ** 2

        # 9.Obstacle avoidance
        c_obstacle = self.compute_obstacle_cost(pos)

        # 10.All gates avoidance (non-target gates only)
        c_all_gates = self.compute_all_gates_avoidance_cost(pos)

        # Total cost
        total_cost = (
            c_pos +
            c_gate_struct +
            c_corridor +
            c_vel +
            c_vel_align +
            c_speed_gate +
            c_slow +
            c_att +
            c_z_floor +
            c_obstacle +
            c_all_gates
        )

        return total_cost

    # =========================================================================
    # Main Control Loop
    # =========================================================================

    def compute_control(
        self,
        obs: dict[str, NDArray[np.floating]],
        info: dict = None
    ) -> NDArray[np.floating]:
        """Compute control command using MPPI."""
        self.step_count += 1
        drone_pos = np.array(obs["pos"], dtype=float)

        # Update obstacles
        if "obstacles_pos" in obs:
            self.obstacles_pos = obs["obstacles_pos"].copy()

        # Update live gate pose
        self._update_current_gate_pose(obs)

        # Decrement pause counter
        if self.pause_counter > 0:
            self.pause_counter -= 1

        # Decay recently passed gate tracking
        if self.recently_passed_decay > 0:
            self.recently_passed_decay -= 1

        # Handle target gate changes
        new_target_idx = int(obs["target_gate"])
        if new_target_idx != self.target_gate_idx and self.pause_counter == 0:
            self._handle_gate_change(obs, new_target_idx)
        elif self.pause_counter > 0 and new_target_idx != self.target_gate_idx:
            if self.step_count % 10 == 0:
                print(f"[AttitudeMPPI] Pause active ({self.pause_counter} steps) - "
                      f"deferring gate {self.target_gate_idx} -> {new_target_idx}")
        else:
            # Check for gate position updates
            self._check_gate_position_update(obs)

        # Update waypoint progress
        self._update_waypoint_progress(drone_pos)

        # Update gate phase
        self._update_gate_phase(drone_pos)

        # Set goal from waypoints
        if len(self.waypoints) > 0:
            self.goal = self.waypoints[min(self.waypoint_index, len(self.waypoints) - 1)].copy()

        # Convert observation to state tensor
        state = self.drone_model.obs_to_state(obs)

        # Run MPPI optimization
        with torch.no_grad():
            optimal_control = self.mppi.command(state)

        self.prev_control = optimal_control.clone()

        # Convert to numpy and apply safety clipping
        control_np = optimal_control.cpu().numpy()
        control_np[0: 3] = np.clip(control_np[0:3], -self.rpy_max, self.rpy_max)
        control_np[3] = np.clip(control_np[3], self.thrust_min, self.thrust_max)

        # Debug output
        self._print_debug_info(obs, control_np)

        return control_np

    def _update_current_gate_pose(self, obs: dict):
        """Update current gate pose from live observations."""
        try:
            self.current_gate_center = np.array(
                obs["gates_pos"][self.target_gate_idx], dtype=float
            )
            self.current_gate_quat = np.array(
                obs["gates_quat"][self.target_gate_idx], dtype=float
            )
            self.current_gate_normal = self._compute_gate_normal(self.current_gate_quat)
        except (IndexError, KeyError):
            pass

    def _handle_gate_change(self, obs: dict, new_target_idx: int):
        """Handle transition to new target gate."""
        print(f"\n[AttitudeMPPI] Target gate changed:  {self.target_gate_idx} -> {new_target_idx}")

        # Track recently passed gate for enhanced avoidance
        self.recently_passed_gate = self.target_gate_idx
        self.recently_passed_decay = int(self.cost_weights["recently_passed_decay_steps"])
        print(f"[AttitudeMPPI] Marking gate {self.recently_passed_gate} as recently passed")

        self.target_gate_idx = new_target_idx
        self.gate_phase = GatePhase.APPROACH

        # Regenerate waypoints for new gate
        self._generate_gate_waypoints(obs)

        if len(self.waypoints) > 0:
            self.prev_goal = self.goal.copy()
            self.goal = self.waypoints[0].copy()

        # Update gate pose
        self._update_current_gate_pose(obs)

        # Reset control sequence
        self._reset_control_sequence()

    def _check_gate_position_update(self, obs: dict):
        """Check if gate position has been updated."""
        try:
            new_gate_pos = np.array(obs["gates_pos"][self.target_gate_idx], dtype=float)
            if not np.allclose(new_gate_pos, self.current_gate_center, atol=0.01):
                print(f"[AttitudeMPPI] Gate {self.target_gate_idx} position updated")
                self._generate_gate_waypoints(obs)
                self._update_current_gate_pose(obs)
        except (IndexError, KeyError):
            pass

    def _print_debug_info(self, obs: dict, control_np: np.ndarray):
        """Print debug information periodically."""
        if self.step_count % 10 != 0:
            return

        pos = obs["pos"]
        vel = obs["vel"]
        gate_pos = self.gates_pos[self.target_gate_idx]
        dist_to_gate = np.linalg.norm(pos - gate_pos)
        z_offset = pos[2] - gate_pos[2]
        xy_dist = np.linalg.norm(pos[: 2] - gate_pos[:2])
        speed = np.linalg.norm(vel)

        obstacle_proximity = float('inf')
        if len(self.obstacles_pos) > 0:
            distances = np.linalg.norm(pos - self.obstacles_pos, axis=1)
            obstacle_proximity = np.min(distances) - self.obstacle_radius

        print(f"[AttitudeMPPI] Step {self.step_count:4d} | "
              f"G{self.target_gate_idx} ({self.gate_phase.name[: 3]}) | "
              f"WP {self.waypoint_index}/{len(self.waypoints)-1} | "
              f"Pos:[{pos[0]: 5.2f},{pos[1]: 5.2f},{pos[2]:5.2f}] | "
              f"3D:{dist_to_gate:4.2f}m XY:{xy_dist:4.2f}m ΔZ:{z_offset:+.2f}m | "
              f"V:{speed:.1f}m/s | "
              f"T:{control_np[3]:.2f}N")

    # =========================================================================
    # Callbacks
    # =========================================================================

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated:  bool,
        info: dict,
    ) -> bool:
        """Step callback."""
        return False

    def episode_callback(self):
        """Episode callback to reset state."""
        self._reset_control_sequence()

        self.step_count = 0
        self.prev_control = None
        self.obstacles_pos = np.array([])
        self.pause_counter = 0
        self.gate_phase = GatePhase.APPROACH
        self.waypoint_index = 0
        self.recently_passed_gate = -1
        self.recently_passed_decay = 0

        print("[AttitudeMPPI] Episode reset")

    # =========================================================================
    # Trajectory Visualization
    # =========================================================================

    def get_optimal_trajectory(self, obs: dict) -> np.ndarray:
        """Get the optimal MPPI trajectory from current observation."""
        try:
            current_state = self.drone_model.obs_to_state(obs)

            if current_state.ndim == 1:
                current_state = current_state.unsqueeze(0)

            optimal_controls = self.mppi.U

            state_tensor = current_state.clone()
            trajectory_positions = [state_tensor[0, 0:3].cpu().numpy()]

            for t in range(min(self.mppi_horizon, 25)):
                control_t = optimal_controls[t].unsqueeze(0)
                state_tensor = self.drone_model.dynamics(
                    state_tensor,
                    control_t,
                    self.mppi_dt
                )
                if state_tensor.ndim == 1:
                    state_tensor = state_tensor.unsqueeze(0)
                trajectory_positions.append(state_tensor[0, 0:3].cpu().numpy())

            return np.array(trajectory_positions)
        except Exception as e: 
            print(f"[AttitudeMPPI] Error computing optimal trajectory: {e}")
            import traceback
            traceback.print_exc()
            return np.array([])

    def get_planned_trajectory(self) -> np.ndarray:
        """Get the optimal planned trajectory from the last MPPI optimization."""
        try:
            current_state = self.mppi.U.new_zeros(1, self.drone_model.nx)
            optimal_controls = self.mppi.U

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

            trajectory_np = torch.cat(trajectory, dim=0).cpu().numpy()
            return trajectory_np[: , 0:3]
        except Exception: 
            return np.array([])

    def get_waypoints(self) -> np.ndarray:
        """Get current waypoints for visualization."""
        if len(self.waypoints) > 0:
            return np.array(self.waypoints)
        return np.array([])

    def get_current_waypoint_index(self) -> int:
        """Get current waypoint index."""
        return self.waypoint_index