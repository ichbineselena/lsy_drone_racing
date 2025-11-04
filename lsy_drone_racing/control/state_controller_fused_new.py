from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StateController(Controller):
    """
    Fused controller with improved obstacle avoidance:
    - Dynamic, gate-informed waypoint generation
    - Potential field-based obstacle avoidance
    - Trajectory safety checking
    - Smooth cubic-spline reference with derivatives
    - PD tracking with feedforward acceleration
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = float(config.env.freq)
        self._tick = 0
        self._finished = False

        # limits
        self._acc_limit = 6.0  # m/s^2
        # allow passing short gates (opening center at 0.5m) with a small margin
        self._min_z = 0.45

        # PD gains
        self._Kp = np.diag([6.0, 6.0, 15.0])
        self._Kd = np.diag([4.5, 4.5, 7.5])

        # Modes: which component supplies shaping vs tracking
        self._shaping_mode = "v2"   # {"v2", "E"}
        self._tracking_mode = "E"   # {"E", "v2"}

        # Build initial waypoints and spline
        waypoints = self._build_waypoints(obs)
        self._t_total = 20.0
        t = np.linspace(0.0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._pos_spline = self._des_pos_spline  # alias used by sim drawer

        self._last_waypoints = waypoints

    def _get_gate_frame_points(self, gate_pos: NDArray[np.floating], gate_rpy: NDArray[np.floating], 
                              gate_opening_size: float | None = None,
                              frame_thickness: float = 0.045) -> NDArray[np.floating]:
        """Calculate sampled points along the inner opening edges and outer frame edges.

        The gates are square-shaped. By default we use opening size 0.30 m and
        frame thickness 0.045 m (4.5 cm). The method returns world-frame points
        that represent both the inner opening edges and the outer frame edges so
        they can be treated as obstacles by the potential field.

        Args:
            gate_pos: Position of gate center (world frame)
            gate_rpy: Roll, pitch, yaw angles of gate
            gate_opening_size: side length of the square opening (meters). If None,
                defaults to 0.30 m.
            frame_thickness: thickness of the frame (meters) measured from opening
                edge outward (total added thickness on each side).
        """
        # Defaults: square opening 0.30 m if not provided
        if gate_opening_size is None:
            gate_opening_size = 0.30

        # Inner (opening) half sizes
        inner_half_w = gate_opening_size / 2.0
        inner_half_h = gate_opening_size / 2.0

        # Outer (frame) half sizes = inner_half + frame_thickness
        outer_half_w = inner_half_w + frame_thickness
        outer_half_h = inner_half_h + frame_thickness

        n_points = 8  # samples per edge
        local_points = []

        # Inner edges (opening)
        for y in [-inner_half_h, inner_half_h]:
            for x in np.linspace(-inner_half_w, inner_half_w, n_points):
                local_points.append([x, y, 0.0])
        for x in [-inner_half_w, inner_half_w]:
            for y in np.linspace(-inner_half_h, inner_half_h, n_points):
                local_points.append([x, y, 0.0])

        # Outer edges (frame outer boundary)
        for y in [-outer_half_h, outer_half_h]:
            for x in np.linspace(-outer_half_w, outer_half_w, n_points):
                local_points.append([x, y, 0.0])
        for x in [-outer_half_w, outer_half_w]:
            for y in np.linspace(-outer_half_h, outer_half_h, n_points):
                local_points.append([x, y, 0.0])

        local_points = np.array(local_points)

        # Create rotation matrix from RPY angles
        roll, pitch, yaw = gate_rpy
        cos_y, sin_y = np.cos(yaw), np.sin(yaw)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        cos_r, sin_r = np.cos(roll), np.sin(roll)

        Rx = np.array([
            [1, 0, 0],
            [0, cos_r, -sin_r],
            [0, sin_r, cos_r],
        ])
        Ry = np.array([
            [cos_p, 0, sin_p],
            [0, 1, 0],
            [-sin_p, 0, cos_p],
        ])
        Rz = np.array([
            [cos_y, -sin_y, 0],
            [sin_y, cos_y, 0],
            [0, 0, 1],
        ])

        R = Rz @ Ry @ Rx

        # Transform sampled local points to world frame
        world_points = np.array([R @ p + gate_pos for p in local_points])
        return world_points

    def _compute_potential_field(self, pos: NDArray[np.floating], target: NDArray[np.floating], 
                               obstacles: NDArray[np.floating], gate_frames: NDArray[np.floating],
                               gate_influence: float = 1.0) -> NDArray[np.floating]:
        """Compute potential field forces at a given position.

        Notes:
        - Repulsive forces are applied when the point is within `max_influence_dist`.
        - Magnitudes are clamped to avoid numerical singularities.
        """
        # Gains and distances tuned for narrow passages
        att_gain = 0.85  # slight reduction to let repulsion dominate near frames
        rep_gain = 2.5  # repulsion from generic obstacles
        gate_frame_rep_gain = 12.0  # much stronger repulsion from gate frames
        safe_dist = 0.2  # geometry-based soft margin for obstacles (m)
        gate_safe_dist = 0.08  # extra soft margin around gate frames (m)
        max_influence_dist = 1.0  # start avoiding within 1m

        # Attractive force (toward gate center)
        d_target = target - pos
        dist_target = np.linalg.norm(d_target)
        if dist_target > 1e-6:
            f_att = att_gain * gate_influence * d_target / dist_target
            # small downward bias to avoid grazing the top frame
            f_att[2] -= 0.02
        else:
            f_att = np.zeros(3)

        # Repulsive forces from obstacles and gate frames
        f_rep = np.zeros(3)
        eps = 1e-3

        # Regular obstacles
        if obstacles is not None and obstacles.size:
            for obs_pos in obstacles:
                d_obs = pos - obs_pos
                dist_obs = np.linalg.norm(d_obs)
                if dist_obs < max_influence_dist:
                    d = max(dist_obs, eps)
                    # standard potential field repulsive term (scaled)
                    rep_magnitude = rep_gain * max((1.0 / d - 1.0 / max_influence_dist), 0.0) * (1.0 / (d**2))
                    # cap magnitude to avoid explosions
                    rep_magnitude = min(rep_magnitude, 20.0)
                    f_rep += rep_magnitude * (d_obs / d)

        # Gate frame points (treated as higher-priority obstacles)
        if gate_frames is not None and gate_frames.size:
            for frame_point in gate_frames:
                d_frame = pos - frame_point
                dist_frame = np.linalg.norm(d_frame)
                if dist_frame < max_influence_dist:
                    d = max(dist_frame, eps)
                    # stronger repulsion close to frame; amplify inside a tighter band
                    rep_magnitude = gate_frame_rep_gain * max((1.0 / d - 1.0 / max_influence_dist), 0.0) * (1.0 / (d**2))
                    # boost strongly when inside a small danger zone
                    # amplify repulsion inside a tight danger band
                    if d < (gate_safe_dist + 0.03):
                        rep_magnitude *= 4.0
                    # additional multiplier for upper-frame points (push drone down)
                    target_z = None
                    try:
                        target_z = float(target[2])
                    except Exception:
                        target_z = None
                    if target_z is not None:
                        dz = frame_point[2] - target_z
                        if dz > 0.0:
                            # frame point is above opening center: push stronger
                            rep_magnitude *= 2.0
                        # lateral proximity: if frame point is lateral relative to approach direction,
                        # amplify repulsion to avoid grazing vertical edges.
                        appr = target - pos
                        appr_xy = np.array([appr[0], appr[1], 0.0])
                        na = np.linalg.norm(appr_xy)
                        if na > 1e-6:
                            appr_unit = appr_xy / na
                            lateral = np.array([-appr_unit[1], appr_unit[0], 0.0])
                            lateral_dist = abs(np.dot(frame_point - target, lateral))
                            if lateral_dist > 0.10:
                                rep_magnitude *= 1.6
                    rep_magnitude = min(rep_magnitude, 200.0)
                    f_rep += rep_magnitude * (d_frame / d)

        return f_att + f_rep

    def _check_trajectory_safety(self, start: NDArray[np.floating], end: NDArray[np.floating], 
                               obstacles: NDArray[np.floating], gate_frames: NDArray[np.floating] | None = None,
                               n_points: int = 10) -> bool:
        """Check if trajectory between two points is safe."""
        safe_dist = 0.15  # Minimum safe distance for trajectory (m)
        
        # Check points along the trajectory
        for t in np.linspace(0, 1, n_points):
            point = start + t * (end - start)
            # Check generic obstacles
            for obs_pos in obstacles:
                if np.linalg.norm(point - obs_pos) < safe_dist:
                    return False
            # Check gate frames if provided (a smaller margin)
            if gate_frames is not None and gate_frames.size:
                for frame_pos in gate_frames:
                    if np.linalg.norm(point - frame_pos) < (safe_dist + 0.07):
                        return False
        return True

    def _build_waypoints(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Construct waypoints using potential fields for obstacle avoidance."""
        if self._shaping_mode == "v2":
            # Static/tuned waypoints from state_controller2
            wp = np.array([
                [-1.5, 0.75, 0.05],
                [-1.0, 0.55, 0.4],
                [0.3, 0.35, 0.7],
                [1.3, -0.15, 0.9],
                [0.85, 0.85, 1.2],
                [-0.5, -0.05, 0.7],
                [-1.2, -0.2, 0.8],
                [-1.2, -0.2, 1.2],
                [-0.0, -0.75, 1.2],
                [0.5, -0.75, 1.2],
            ], dtype=float)
            wp[:, 2] = np.maximum(wp[:, 2], self._min_z)
            return wp

        pos0 = np.asarray(obs.get("pos", [0.0, 0.0, 0.6]), dtype=float)
        gates = np.asarray(obs.get("gates_pos", []), dtype=float)
        obstacles = np.asarray(obs.get("obstacles_pos", []), dtype=float)
        
        if gates.ndim != 2 or gates.shape[1] < 3 or gates.shape[0] < 4:
            base = np.array([
                pos0,
                [0.5, 0.25, 0.7],
                [1.05, 0.75, 1.2],
                [-1.0, -0.25, 0.7],
                [0.0, -0.75, 1.2],
            ], dtype=float)
            base[:, 2] = np.maximum(base[:, 2], self._min_z)
            return base

        # Initialize waypoints list with start position
        waypoints = [pos0]
        current_pos = pos0

        # Get gate orientations if available
        gates_rpy = np.asarray(obs.get("gates_rpy", np.zeros((len(gates), 3))), dtype=float)
        
        # Calculate gate frame points for all gates using opening center (lowered z by 0.195)
        all_gate_frames = []
        for gate_pos, gate_rpy in zip(gates, gates_rpy):
            opening_center = gate_pos.copy()
            opening_center[2] = opening_center[2] - 0.195
            frame_points = self._get_gate_frame_points(opening_center, gate_rpy)
            all_gate_frames.extend(frame_points)
        all_gate_frames = np.array(all_gate_frames)
        
        # Generate waypoints for each gate using potential fields
        for i, gate in enumerate(gates):
            # Higher influence for next immediate gate, lower for future gates
            gate_influence = 1.0 / (i + 1)
            
            # Generate intermediate points using potential field
            n_steps = 8  # Number of intermediate points (smoother approach)
            for _ in range(n_steps):
                # Use opening center (lowered z by 0.195) as target so spline goes through opening
                target = gate.copy()
                target[2] = target[2] - 0.195
                # Compute next position based on potential field with gate frame avoidance
                force = self._compute_potential_field(current_pos, target, obstacles, all_gate_frames, gate_influence)
                step_size = 0.18  # smaller step size for finer trajectory updates
                next_pos = current_pos + step_size * force
                
                # Ensure minimum height
                next_pos[2] = max(next_pos[2], self._min_z)
                
                # Only add point if the trajectory to it is safe
                if obstacles.size > 0 and not self._check_trajectory_safety(current_pos, next_pos, obstacles, all_gate_frames):
                    # If unsafe, try a higher path
                    next_pos[2] += 0.2
                
                waypoints.append(next_pos)
                current_pos = next_pos
            
            # Add the gate opening center as final waypoint for this segment
            opening_center = gate.copy()
            opening_center[2] = opening_center[2] - 0.195
            waypoints.append(opening_center)
            current_pos = opening_center

        return np.array(waypoints, dtype=float)

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True

        # Desired from spline
        p_d = self._des_pos_spline(t)
        v_d = self._des_pos_spline(t, 1)
        a_d = self._des_pos_spline(t, 2)
        
        # Debug printing for gate passages
        gates_pos = obs.get("gates_pos", [])
        if len(gates_pos) > 0:
            pos = np.asarray(obs["pos"], dtype=float)
            # Check if we're near any gate (within 0.5m)
            for i, gate in enumerate(gates_pos):
                gate_center = gate.copy()
                gate_center[2] -= 0.195  # Adjust to opening center
                dist_to_gate = np.linalg.norm(pos - gate_center)
                if dist_to_gate < 0.5:
                    print(f"\nGate {i} Passage:")
                    print(f"  Gate center: {gate_center}")
                    print(f"  Desired position: {p_d}")
                    print(f"  Actual drone position: {pos}")
                    print(f"  Distance to gate center: {dist_to_gate:.3f}m")

        if self._tracking_mode == "E":
            # E-style: rely on firmware to track position; fill the rest with zeros
            # keep a helpful yaw along the tangent
            if np.linalg.norm(v_d[:2]) > 1e-3:
                yaw_d = float(np.arctan2(v_d[1], v_d[0]))
            else:
                yaw_d = 0.0
            action = np.array([
                *p_d,
                0.0, 0.0, 0.0,  # vx, vy, vz
                0.0, 0.0, 0.0,  # ax, ay, az
                yaw_d,
                0.0, 0.0, 0.0,
            ], dtype=np.float32)
            return action

        # v2-style: PD + feedforward acceleration (no gravity)
        p = np.asarray(obs["pos"], dtype=float)
        v = np.asarray(obs["vel"], dtype=float)
        a_fb = self._Kp @ (p_d - p) + self._Kd @ (v_d - v)
        a_cmd = a_d + a_fb
        na = np.linalg.norm(a_cmd)
        if na > self._acc_limit:
            a_cmd = a_cmd * (self._acc_limit / (na + 1e-6))

        if np.linalg.norm(v_d[:2]) > 1e-3:
            yaw_d = float(np.arctan2(v_d[1], v_d[0]))
        elif np.linalg.norm(a_cmd[:2]) > 1e-3:
            yaw_d = float(np.arctan2(a_cmd[1], a_cmd[0]))
        else:
            yaw_d = 0.0

        action = np.array([
            *p_d,
            *v_d,
            *a_cmd,
            yaw_d,
            0.0, 0.0, 0.0,
        ], dtype=np.float32)
        return action

    def step_callback(
        self,
        action: NDArray[np.floating],
        obs: dict[str, NDArray[np.floating]],
        reward: float,
        terminated: bool,
        truncated: bool,
        info: dict,
    ) -> bool:
        self._tick += 1

        # Rebuild waypoints occasionally or when gates move significantly
        if (self._tick % int(self._freq * 0.5)) == 0:  # every 0.5s
            new_wp = self._build_waypoints(obs)
            if not np.allclose(new_wp, self._last_waypoints, atol=0.03):
                self._last_waypoints = new_wp
                t = np.linspace(0.0, self._t_total, len(new_wp))
                self._des_pos_spline = CubicSpline(t, new_wp)
                self._pos_spline = self._des_pos_spline

        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._finished = False