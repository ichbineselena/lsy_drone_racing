"""
Advanced State-MPC controller with:
 - obstacle-aware reference shaping (potential-field like repulsion)
 - jerk-input dynamics (states = [pos, vel, acc], inputs = jerk)
 - time-varying feedforward LQR tracking using spline derivatives (jerk as feedforward)
 - smooth blending of gate estimate updates
 - safe-altitude enforcement and yaw pointing toward look-ahead

Outputs the state command expected by the env:
 [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
"""

from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StateMPCAdvanced(Controller):
    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        # ---------- basic config ----------
        self._freq = float(config.env.freq)
        self._dt = 1.0 / self._freq
        self._T_total = 15.0
        # MPC horizon in seconds (preview); horizon steps N = horizon_seconds * freq
        self._horizon_s = 1.0  # 1s preview as a baseline — increase if needed
        self._N = max(2, int(self._horizon_s * self._freq))
        self._tick = 0
        self._finished = False

        # ---------- safety ----------
        self.SAFE_Z = 1.45  # slightly above obstacle top (1.52m). Tune if necessary.
        self.ACC_MAX = 6.0  # acceleration clip (m/s^2)
        self.JERK_MAX = 20.0  # jerk clip (m/s^3)

        # ---------- LQR costs (state is 9-d: p(3), v(3), a(3)) ----------
        q_pos = 50.0
        q_z = 300.0
        q_vel = 6.0
        q_acc = 1.0
        # ordering: [px,py,pz,vx,vy,vz,ax,ay,az]
        self.Q = np.diag([q_pos, q_pos, q_z, q_vel, q_vel, q_vel, q_acc, q_acc, q_acc])
        self.R = np.eye(3) * 0.1  # penalize jerk magnitude (smoothness)

        # ---------- dynamics matrices for jerk-input model ----------
        dt = self._dt
        # state x = [p, v, a] (9)
        A = np.eye(9)
        # p integrates v and a and jerk contribution
        # p_{k+1} = p_k + v_k*dt + 0.5 a_k dt^2 + (1/6) u_k dt^3
        A[0, 3] = dt
        A[1, 4] = dt
        A[2, 5] = dt
        A[0, 6] = 0.5 * dt * dt
        A[1, 7] = 0.5 * dt * dt
        A[2, 8] = 0.5 * dt * dt
        # v integrates a and jerk
        # v_{k+1} = v_k + a_k dt + 0.5 u_k dt^2
        A[3, 3] = 1.0
        A[4, 4] = 1.0
        A[5, 5] = 1.0
        A[3, 6] = dt
        A[4, 7] = dt
        A[5, 8] = dt
        # a_{k+1} = a_k + u_k dt
        A[6, 6] = 1.0
        A[7, 7] = 1.0
        A[8, 8] = 1.0

        B = np.zeros((9, 3))
        # contributions of u (jerk) to p, v, a
        coef_p = (dt ** 3) / 6.0
        coef_v = 0.5 * (dt ** 2)
        coef_a = dt
        B[0, 0] = coef_p
        B[1, 1] = coef_p
        B[2, 2] = coef_p
        B[3, 0] = coef_v
        B[4, 1] = coef_v
        B[5, 2] = coef_v
        B[6, 0] = coef_a
        B[7, 1] = coef_a
        B[8, 2] = coef_a

        self.A = A
        self.B = B

        # ---------- build spline from initial gates ----------
        gates = self._extract_gates(obs, info)
        gates[:, 2] = np.maximum(gates[:, 2], self.SAFE_Z)
        # apply obstacle-aware shaping immediately if obstacles available
        obstacles = self._extract_obstacles(obs, info)
        if obstacles.shape[0] > 0:
            gates = self._shape_waypoints_around_obstacles(gates, obstacles)

        self._waypoints = gates
        times = np.linspace(0.0, self._T_total, len(gates))
        self._spline = CubicSpline(times, gates, axis=0)
        self._spline_vel = self._spline.derivative(1)
        self._spline_acc = self._spline.derivative(2)
        self._spline_jerk = self._spline.derivative(3)

        self._last_gate_obs = gates.copy()

    # ----------------------------- utilities --------------------------------
    def _extract_gates(self, obs: dict, info: dict | None) -> NDArray[np.floating]:
        """Robustly extract gate positions (Nx3). Fallback to config nominal if necessary."""
        # prefer obs["gates_pos"] if present
        if obs is not None:
            for k in ("gates_pos", "gates", "gate_positions", "gate_poses"):
                if k in obs and obs[k] is not None:
                    arr = np.asarray(obs[k], dtype=np.float64)
                    if arr.ndim == 2 and arr.shape[1] >= 3:
                        return arr[:, :3].copy()
                    if arr.ndim == 1 and arr.size >= 3:
                        return arr.reshape(-1, 3)[:, :3].copy()
        # try info
        if info:
            for k in ("track", "env_track", "track_info"):
                if k in info and isinstance(info[k], dict):
                    t = info[k]
                    if "gates" in t:
                        arr = np.asarray(t["gates"], dtype=np.float64)
                        if arr.ndim == 2 and arr.shape[1] >= 3:
                            return arr[:, :3].copy()
        # fallback: try config stored in self._config attribute (Controller may have stored)
        cfg = getattr(self, "_config", None) or getattr(self, "config", None)
        if isinstance(cfg, dict):
            t = cfg.get("env", {}).get("track", {})
            if "gates" in t:
                arr = np.array([g.get("pos", g) for g in t["gates"]], dtype=np.float64)
                if arr.size:
                    return arr[:, :3].copy()
        # last resort: a small default
        return np.array([[0.5, 0.25, 1.0], [1.0, 0.75, 1.2], [-1.0, -0.25, 0.7], [0.0, -0.75, 1.2]], dtype=np.float64)

    def _extract_obstacles(self, obs: dict, info: dict | None) -> NDArray[np.floating]:
        """Try several keys for obstacle positions; return Nx3 or empty array."""
        if obs is not None:
            for k in ("obstacles_pos", "obstacles", "obstacle_positions", "obstacle_poses"):
                if k in obs and obs[k] is not None:
                    arr = np.asarray(obs[k], dtype=np.float64)
                    if arr.ndim == 2 and arr.shape[1] >= 3:
                        return arr[:, :3].copy()
        if info:
            if "obstacles" in info:
                arr = np.asarray(info["obstacles"], dtype=np.float64)
                if arr.ndim == 2 and arr.shape[1] >= 3:
                    return arr[:, :3].copy()
        return np.zeros((0, 3), dtype=np.float64)

    # ----------------------- obstacle-aware shaping --------------------------
    def _shape_waypoints_around_obstacles(self, waypoints: NDArray[np.floating], obstacles: NDArray[np.floating]) -> NDArray[np.floating]:
        """
        Simple potential-field style repulsion: for each waypoint, push it away from nearby obstacles.
        After applying repulsion, run a smoothing pass (averaging neighbors) to avoid sharp kinks.
        """
        wp = waypoints.copy()
        avoid_radius = 0.6  # meters
        max_push = 0.5  # max displacement allowed per waypoint
        gain = 0.8  # repulsion strength

        for i in range(wp.shape[0]):
            p = wp[i]
            total_push = np.zeros(3, dtype=np.float64)
            for o in obstacles:
                dvec = p - o
                dist = np.linalg.norm(dvec)
                if dist < 1e-6:
                    # degenerate: move upward
                    push = np.array([0.0, 0.0, max_push])
                elif dist < avoid_radius:
                    # repulsive magnitude scales with how close we are
                    mag = gain * (avoid_radius - dist) / avoid_radius
                    push_dir = dvec / dist
                    push = push_dir * mag
                else:
                    push = np.zeros(3)
                total_push += push
            # cap push
            if np.linalg.norm(total_push) > max_push:
                total_push = total_push / np.linalg.norm(total_push) * max_push
            wp[i] = p + total_push

        # smoothing: simple 1D convolution along waypoint index
        if wp.shape[0] >= 3:
            wp_smooth = wp.copy()
            for i in range(1, wp.shape[0] - 1):
                wp_smooth[i] = 0.25 * wp[i - 1] + 0.5 * wp[i] + 0.25 * wp[i + 1]
            wp = wp_smooth

        # enforce minimum altitude
        wp[:, 2] = np.maximum(wp[:, 2], self.SAFE_Z)
        return wp

    # ---------------- Gate and Obstacle Cost Helpers ---------------- #

    def gate_half_sphere_cost(pos, c_i, n_i, r_entry, r_exit, w_entry, w_exit):
        """
        Half-sphere cost encouraging the drone to pass through the gate center.
        pos: [x,y,z] current position
        c_i: gate center
        n_i: gate normal (unit vector, facing forward)
        """
        d = pos - c_i
        dist = np.linalg.norm(d)
        proj = np.dot(d, n_i)

        # entry hemisphere (before gate)
        entry_pen = max(0.0, proj + r_entry)
        # exit hemisphere (after gate)
        exit_pen = max(0.0, -proj + r_exit)

        return w_entry * entry_pen**2 + w_exit * exit_pen**2


    def obstacle_cylinder_cost(pos, c_o, r_o, h, w_obs):
        """
        Cylindrical penalty around obstacles (e.g., poles).
        pos: [x,y,z]
        c_o: obstacle base center [x,y,z]
        r_o: radius
        h: height
        """
        d_xy = np.linalg.norm(pos[:2] - c_o[:2])
        within_height = 0.0 <= pos[2] - c_o[2] <= h
        if within_height and d_xy < r_o:
            return w_obs * (r_o - d_xy)**2
        elif within_height:
            # Soft exponential decay for near misses
            return w_obs * np.exp(-((d_xy - r_o)/r_o)**2)
        return 0.0

    # ---------------------- build reference horizon -------------------------
    def _build_reference_horizon(self, t_now: float):
        """
        For k = 0..N-1 build x_ref[k] (9-d) and u_ref[k] (3-d jerk).
        Use spline derivatives: pos, vel, acc, jerk.
        """
        times = t_now + (np.arange(1, self._N + 1) * self._dt)
        times = np.clip(times, 0.0, self._T_total)
        pos = self._spline(times)                # (N,3)
        vel = self._spline_vel(times)            # (N,3)
        acc = self._spline_acc(times)            # (N,3)
        # cubic spline derivative(3) is piecewise constant jerk
        try:
            jerk = self._spline_jerk(times)      # (N,3)
        except Exception:
            # fallback: finite difference of acceleration
            jerk = np.zeros_like(acc)
            jerk[:-1] = (acc[1:] - acc[:-1]) / self._dt
            jerk[-1] = jerk[-2] if self._N > 1 else np.zeros(3)

        # enforce safe altitude
        pos[:, 2] = np.maximum(pos[:, 2], self.SAFE_Z)

        xr = np.zeros((self._N, 9), dtype=np.float64)
        xr[:, 0:3] = pos
        xr[:, 3:6] = vel
        xr[:, 6:9] = acc

        ur = np.clip(jerk, -self.JERK_MAX, self.JERK_MAX)
        return xr, ur

    # ----------------------- time-varying LQR backward pass ------------------
    def _compute_time_varying_lqr(self, A: NDArray[np.floating], B: NDArray[np.floating], Q: NDArray[np.floating], R: NDArray[np.floating], N: int):
        """
        Compute time-invariant Riccati (we will also compute K_k for each step using same A,B).
        For a true time-varying LQR with different A_k/B_k you'd do a backward Riccati pass;
        here A/B are constant so we do a DARE-like iteration and then form K matrix used at each step.
        """
        P = Q.copy()
        max_iters = 500
        tol = 1e-8
        for _ in range(max_iters):
            BtPB = B.T @ P @ B
            S = R + BtPB
            K = np.linalg.solve(S, B.T @ P @ A)
            P_next = A.T @ P @ A - A.T @ P @ B @ K + Q
            if np.max(np.abs(P_next - P)) < tol:
                P = P_next
                break
            P = P_next

        S = R + B.T @ P @ B
        K = np.linalg.solve(S, B.T @ P @ A)  # K shape (m, n)
        return K, P

    # --------------------------- main compute_control -----------------------
    def compute_control(self, obs: dict[str, NDArray[np.floating]], info: dict | None = None) -> NDArray[np.floating]:
        """
        Returns desired state vector:
         [x,y,z, vx,vy,vz, ax,ay,az, yaw, rrate, prate, yrate]
        """
        t_now = min(self._tick / self._freq, self._T_total)
        if t_now >= self._T_total:
            self._finished = True

        # current measured state
        pos = np.asarray(obs["pos"], dtype=np.float64).reshape(3)
        vel = np.asarray(obs["vel"], dtype=np.float64).reshape(3)
        # approximate acceleration: if obs provides accel use it; otherwise zero
        acc = np.asarray(obs.get("acc", np.zeros(3)), dtype=np.float64).reshape(3)

        x_meas = np.hstack((pos, vel, acc))  # 9

        # build reference horizon (x_ref: N x 9, u_ref: N x 3)
        x_ref_h, u_ref_h = self._build_reference_horizon(t_now)

        # compute time-invariant LQR K (we reuse for each step here)
        K, P = self._compute_time_varying_lqr(self.A, self.B, self.Q, self.R, self._N)
        # use first reference
        x_ref_next = x_ref_h[0]
        u_ref_next = u_ref_h[0]

        # control law: u = u_ref - K (x - x_ref)
        u = u_ref_next - K @ (x_meas - x_ref_next)
        # clip jerk
        u = np.clip(u, -self.JERK_MAX, self.JERK_MAX)

        # produce predicted next-state using discrete dynamics
        x_pred = self.A @ x_meas + self.B @ u
        des_pos = x_pred[0:3]
        des_vel = x_pred[3:6]
        des_acc = x_pred[6:9]

        # clip accelerations
        des_acc = np.clip(des_acc, -self.ACC_MAX, self.ACC_MAX)

        # yaw: look-ahead point on spline
        lookahead = min(t_now + 0.5, self._T_total)
        future = self._spline(lookahead)
        dir_vec = future - des_pos
        des_yaw = float(np.arctan2(dir_vec[1], dir_vec[0]))

        # Build action vector (13-d)
        action = np.zeros(13, dtype=np.float32)
        action[0:3] = des_pos.astype(np.float32)
        action[3:6] = des_vel.astype(np.float32)
        action[6:9] = des_acc.astype(np.float32)
        action[9] = des_yaw
        action[10:] = 0.0
        return action

    # --------------------------- step_callback: handle replanning -----------
    def step_callback(self, action: NDArray[np.floating], obs: dict[str, NDArray[np.floating]],
                      reward: float, terminated: bool, truncated: bool, info: dict) -> bool:
        self._tick += 1

        # blend new gate observations
        current_gates = self._extract_gates(obs, info)
        # quick check for shape change or significant motion
        if current_gates.shape != self._waypoints.shape or not np.allclose(current_gates, self._last_gate_obs, atol=0.02):
            alpha = 0.25
            blended = (1 - alpha) * self._waypoints + alpha * current_gates
            # apply obstacle shaping if obstacles present
            obstacles = self._extract_obstacles(obs, info)
            if obstacles.shape[0] > 0:
                blended = self._shape_waypoints_around_obstacles(blended, obstacles)
            blended[:, 2] = np.maximum(blended[:, 2], self.SAFE_Z)
            self._waypoints = blended
            # rebuild spline with remaining time
            t0 = min(self._tick / self._freq, self._T_total)
            times = np.linspace(t0, self._T_total, len(self._waypoints))
            self._spline = CubicSpline(times, self._waypoints, axis=0)
            self._spline_vel = self._spline.derivative(1)
            self._spline_acc = self._spline.derivative(2)
            try:
                self._spline_jerk = self._spline.derivative(3)
            except Exception:
                # the cubic spline supports deriv(3) but keep a fallback
                self._spline_jerk = None
            self._last_gate_obs = current_gates.copy()

        return self._finished

    def episode_callback(self):
        self._tick = 0
        self._finished = False