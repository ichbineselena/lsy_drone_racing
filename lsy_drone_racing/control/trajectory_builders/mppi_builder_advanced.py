"""Advanced MPPI trajectory builder with obstacle and gate awareness.

This builder extends the basic MPPI with:
- Gate awareness and sequencing
- Obstacle avoidance costs
- Dynamic replanning based on current progress
- More sophisticated cost function
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

import numpy as np

from .base import TrajectoryBuilder

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MPPIBuilderAdvanced:
    """Advanced MPPI trajectory builder with gate and obstacle awareness.
    
    This builder uses sampling-based Model Predictive Path Integral control to
    generate reference trajectories that:
    - Navigate through gates in sequence
    - Avoid obstacles
    - Maintain smooth, dynamically feasible motion
    """

    def __init__(
        self,
        gates: Sequence[NDArray] | None = None,
        obstacles: Sequence[NDArray] | None = None,
        K: int = 500,
        lambda_: float = 0.8,
        sigma_u: float = 0.4,
        gate_radius: float = 0.45,
        obstacle_radius: float = 0.3,
    ):
        """Initialize the advanced MPPI builder.
        
        Args:
            gates: List of gate positions (each is [x, y, z])
            obstacles: List of obstacle positions (each is [x, y, z])
            K: Number of sample trajectories
            lambda_: Temperature parameter for MPPI
            sigma_u: Control noise standard deviation
            gate_radius: Radius to consider "passing through" a gate
            obstacle_radius: Safety radius around obstacles
        """
        self.gates = [np.asarray(g).reshape(3,) for g in gates] if gates else []
        self.obstacles = [np.asarray(o).reshape(3,) for o in obstacles] if obstacles else []
        self.K = int(K)
        self.lambda_ = float(lambda_)
        self.sigma_u = float(sigma_u)
        self.gate_radius = gate_radius
        self.obstacle_radius = obstacle_radius
        
        # State tracking
        self.current_gate_idx = 0  # Which gate to target next
        self.U_nom = None
        self.x0 = None
        self.T = None
        
        print(f"[MPPIBuilderAdvanced] Initialized:")
        print(f"  - Gates: {len(self.gates)}")
        print(f"  - Obstacles: {len(self.obstacles)}")
        print(f"  - K={self.K}, lambda={self.lambda_}, sigma_u={self.sigma_u}")

    def reset(self, initial_state: NDArray, t0: float = 0.0) -> None:
        """Reset the builder with a new initial state.
        
        Args:
            initial_state: State vector [pos(3), rpy(3), vel(3), drpy(3)]
            t0: Initial time
        """
        self.x0 = np.asarray(initial_state).copy()
        self.t0 = float(t0)
        self.U_nom = None
        self.current_gate_idx = 0
    
    def set_target_gate(self, gate_idx: int) -> None:
        """Set the current target gate index.
        
        Args:
            gate_idx: Index of the gate to target (0-indexed)
        """
        self.current_gate_idx = max(0, min(gate_idx, len(self.gates) - 1))

    def _double_integrator_rollout(self, x0: NDArray, U: NDArray, dt: float) -> NDArray:
        """Vectorized rollout using double integrator dynamics.
        
        Args:
            x0: Initial state [pos(3), rpy(3), vel(3), drpy(3)]
            U: Control sequences (K, T, 3) - accelerations in world frame
            dt: Time step
            
        Returns:
            States (K, T+1, 6) where each state is [pos(3), vel(3)]
        """
        K, T, _ = U.shape
        states = np.zeros((K, T + 1, 6))
        
        # Extract initial position and velocity
        pos0 = x0[0:3]
        vel0 = x0[6:9]
        states[:, 0, 0:3] = pos0[None, :]
        states[:, 0, 3:6] = vel0[None, :]
        
        # Integrate forward
        for t in range(T):
            a = U[:, t, :]  # (K, 3)
            # Euler integration
            states[:, t + 1, 3:6] = states[:, t, 3:6] + a * dt  # velocity
            states[:, t + 1, 0:3] = states[:, t, 0:3] + states[:, t, 3:6] * dt + 0.5 * a * dt**2  # position
        
        return states

    def _cost(self, states: NDArray, U: NDArray) -> NDArray:
        """Compute costs for all sampled trajectories.
        
        Args:
            states: State trajectories (K, T+1, 6)
            U: Control sequences (K, T, 3)
            
        Returns:
            Costs (K,) for each trajectory
        """
        K, T_plus_1, _ = states.shape
        T = T_plus_1 - 1
        pos_seq = states[:, 1:, 0:3]  # (K, T, 3)
        vel_seq = states[:, 1:, 3:6]  # (K, T, 3)
        
        cost = np.zeros(K)
        
        # ===== Gate costs =====
        if len(self.gates) > 0:
            # Target the next gate
            target_gate = self.gates[self.current_gate_idx]
            
            # Terminal cost: end near target gate
            terminal_pos = pos_seq[:, -1, :]  # (K, 3)
            dist_to_gate_terminal = np.linalg.norm(terminal_pos - target_gate[None, :], axis=1)  # (K,)
            cost += 300.0 * dist_to_gate_terminal**2  # Quadratic for stronger pull when far
            
            # Find minimum distance to gate across entire trajectory
            all_dist = np.linalg.norm(pos_seq - target_gate[None, None, :], axis=2)  # (K, T)
            min_dist_to_gate = np.min(all_dist, axis=1)  # (K,)
            
            # DOMINANT COST: Heavily penalize trajectories that don't get close to gate
            # This is the primary objective - pass through the gate
            cost += 2000.0 * min_dist_to_gate  # Huge penalty for min distance
            
            # HUGE REWARD: Bonus for actually passing within gate radius  
            close_bonus = np.maximum(0, self.gate_radius - min_dist_to_gate)  # How much inside gate
            cost -= 5000.0 * close_bonus  # Massive reward for gate passage
            
            # CRITICAL: Slow down when near gate for precise passage
            # Penalize high speed when very close to gate
            for t in range(T):
                pos_t = pos_seq[:, t, :]  # (K, 3)
                vel_t = vel_seq[:, t, :]  # (K, 3)
                dist_t = np.linalg.norm(pos_t - target_gate[None, :], axis=1)  # (K,)
                speed_t = np.linalg.norm(vel_t, axis=1)  # (K,)
                
                # When within 0.6m of gate (just before passage), penalize high speeds
                proximity_factor = np.maximum(0, 0.6 - dist_t) / 0.6  # 1.0 at gate, 0.0 at 0.6m+ away
                cost += 20.0 * proximity_factor * speed_t**2  # Moderate speed penalty very near gate
        
        # ===== Obstacle avoidance =====
        if len(self.obstacles) > 0:
            for obs_pos in self.obstacles:
                # Check distance to obstacle at each timestep
                for t in range(T):
                    pos_t = pos_seq[:, t, :]  # (K, 3)
                    dist_to_obs = np.linalg.norm(pos_t - obs_pos[None, :], axis=1)  # (K,)
                    
                    # Penalty for being too close (exponential cost)
                    safety_margin = self.obstacle_radius + 0.2  # Add 20cm safety
                    penetration = np.maximum(0, safety_margin - dist_to_obs)
                    cost += 500.0 * penetration**2  # Quadratic penalty
        
        # ===== Velocity regularization =====
        # Prefer moderate speeds (not too fast, not too slow)
        speed = np.linalg.norm(vel_seq, axis=2)  # (K, T)
        preferred_speed = 1.2  # m/s - balanced between speed and control
        speed_error = (speed - preferred_speed)**2
        cost += 0.3 * np.sum(speed_error, axis=1)  # Light penalty - don't overly constrain
        
        # ===== Control effort =====
        # Penalize large accelerations
        cost += 0.05 * np.sum(U**2, axis=(1, 2))  # Very light - allow dynamic maneuvers
        
        # ===== Control smoothness =====
        # Penalize rapid changes in control - important but not dominant
        if T > 1:
            dU = U[:, 1:, :] - U[:, :-1, :]  # (K, T-1, 3)
            cost += 1.5 * np.sum(dU**2, axis=(1, 2))  # Moderate smoothness requirement
        
        # ===== Height safety =====
        # Penalize getting too low - CRITICAL for safety
        min_height = 0.25  # meters - 25cm minimum safe height
        height_violation = np.maximum(0, min_height - pos_seq[:, :, 2])  # (K, T)
        cost += 2000.0 * np.sum(height_violation**2, axis=1)  # High penalty to prevent ground contact
        
        # ===== Boundary safety =====
        # Soft penalties for getting close to workspace boundaries
        # Safety limits: x: [-2.5, 2.5], y: [-1.5, 1.5], z: [0, 2.0]
        x_low_violation = np.maximum(0, -2.3 - pos_seq[:, :, 0])  # 20cm margin from x=-2.5
        x_high_violation = np.maximum(0, pos_seq[:, :, 0] - 2.3)  # 20cm margin from x=2.5
        y_low_violation = np.maximum(0, -1.3 - pos_seq[:, :, 1])  # 20cm margin from y=-1.5
        y_high_violation = np.maximum(0, pos_seq[:, :, 1] - 1.3)  # 20cm margin from y=1.5
        z_high_violation = np.maximum(0, pos_seq[:, :, 2] - 1.8)  # 20cm margin from z=2.0
        
        boundary_penalty = (x_low_violation + x_high_violation + 
                           y_low_violation + y_high_violation + z_high_violation)
        cost += 100.0 * np.sum(boundary_penalty**2, axis=1)  # Moderate penalty to stay away from edges
        
        return cost

    def _check_gate_advancement(self, current_pos: NDArray) -> None:
        """Deprecated: Gate advancement now handled by environment.
        
        This method is kept for compatibility but does nothing.
        Use set_target_gate() to update the target gate from environment feedback.
        
        Args:
            current_pos: Current position [x, y, z]
        """
        pass  # Gate advancement handled externally

    def get_horizon(self, t_now: float, N: int, dt: float) -> dict:
        """Generate reference trajectory for the next N steps using MPPI.
        
        Args:
            t_now: Current time
            N: Horizon length
            dt: Time step
            
        Returns:
            Dictionary with 'pos', 'vel', 'yaw' arrays
        """
        T = int(N)
        self.T = T
        m = 3  # Control dimension (ax, ay, az)
        
        # Initialize nominal control if needed
        if self.U_nom is None or self.U_nom.shape[0] != T:
            # Initialize with upward bias to counteract gravity and enable takeoff
            self.U_nom = np.zeros((T, m))
            self.U_nom[:, 2] = 2.0  # Strong upward acceleration bias (2.0 m/s²) for reliable takeoff
        
        # Gate progress is now managed externally via set_target_gate()
        # No internal gate advancement needed
        
        # Sample control perturbations
        dU = np.random.normal(scale=self.sigma_u, size=(self.K, T, m))
        U_samples = self.U_nom[None, :, :] + dU  # (K, T, m)
        
        # Clip to reasonable acceleration limits (±5 m/s²)
        U_samples = np.clip(U_samples, -5.0, 5.0)
        
        # Rollout all samples
        states = self._double_integrator_rollout(self.x0, U_samples, dt)  # (K, T+1, 6)
        
        # Compute costs
        costs = self._cost(states, U_samples)  # (K,)
        
        # MPPI weight computation
        S_min = costs.min()
        exp_arg = -(costs - S_min) / max(self.lambda_, 1e-8)
        w = np.exp(exp_arg)
        w = w / (np.sum(w) + 1e-12)
        
        # Update nominal control
        weighted_dU = (w[:, None, None] * dU).sum(axis=0)  # (T, m)
        self.U_nom = self.U_nom + weighted_dU
        
        # Generate nominal trajectory
        states_nom = self._double_integrator_rollout(self.x0, self.U_nom[None, :, :], dt)[0]
        
        # Extract position and velocity references
        pos = states_nom[1:, 0:3]  # (T, 3)
        vel = states_nom[1:, 3:6]  # (T, 3)
        
        # Compute yaw from velocity direction (face direction of motion)
        yaw = np.zeros(T)
        for i in range(T):
            if np.linalg.norm(vel[i, :2]) > 0.1:  # Only update if moving
                yaw[i] = np.arctan2(vel[i, 1], vel[i, 0])
        
        # Print diagnostics occasionally
        if int(t_now * 10) % 10 == 0:  # Every ~1 second
            current_pos = self.x0[0:3]
            print(f"[MPPIBuilderAdvanced] t={t_now:.1f}s, gate={self.current_gate_idx + 1}/{len(self.gates)}, "
                  f"cost_min={S_min:.1f}, pos={current_pos}")
        
        return {"pos": pos, "vel": vel, "yaw": yaw}

    def get_total_steps(self) -> int:
        """Return a large number since MPPI generates trajectories online."""
        return 100000
