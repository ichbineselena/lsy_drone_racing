from __future__ import annotations

from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline

from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StateController(Controller):
    """
    Fused controller:
    - Dynamic, gate-informed waypoint generation (based on state_controller_E)
    - Smooth cubic-spline reference with derivatives
    - PD tracking with feedforward acceleration (based on state_controller2)

    Outputs 13-d state command:
    [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate]
    """

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)
        self._freq = float(config.env.freq)
        self._tick = 0
        self._finished = False

        # limits
        self._acc_limit = 6.0  # m/s^2
        self._min_z = 0.6

        # PD gains
        self._Kp = np.diag([3.0, 3.0, 5.0])
        self._Kd = np.diag([2.5, 2.5, 3.5])

        # Modes: which component supplies shaping vs tracking
        # Defaults per your note: 2 has better shaping; E has better tracking
        self._shaping_mode = "v2"   # {"v2", "E"}
        self._tracking_mode = "E"   # {"E", "v2"}

        # Build initial waypoints and spline
        waypoints = self._build_waypoints(obs)
        self._t_total = 20.0
        t = np.linspace(0.0, self._t_total, len(waypoints))
        self._des_pos_spline = CubicSpline(t, waypoints)
        self._pos_spline = self._des_pos_spline  # alias used by sim drawer

        self._last_waypoints = waypoints

    # ---------------- waypoint logic -----------------
    def _build_waypoints(self, obs: dict[str, NDArray[np.floating]]) -> NDArray[np.floating]:
        """Construct waypoints using either v2 (static, tuned list) or E (gate-aware offsets)."""
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
            self.waypoints = wp
            return wp

        # Else: E-style shaping (gate-aware with offsets)
        pos0 = np.asarray(obs.get("pos", [0.0, 0.0, 0.6]), dtype=float)
        gates = np.asarray(obs.get("gates_pos", []), dtype=float)
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

        wp = [
            pos0,
            gates[0] + np.array([-1.5, 0.3, -0.3]),
            gates[0],
            gates[0] + np.array([-0.2, -0.1, 0.0]),
            gates[1] + np.array([-0.74, -0.4, -0.5]),
            gates[1],
            gates[1] + np.array([-0.2, 0.1, 0.0]),
            gates[1] + np.array([-1.0, -0.7, -0.5]),
            np.array([-0.5, -0.05, 0.7]),
            gates[2],
            np.array([-1.2, -0.2, 0.8]),
            np.array([-1.2, -0.2, 1.2]),
            np.array([0.0, -0.7, 1.2]),
            gates[3],
            np.array([0.5, -0.75, 1.2]),
        ]
        waypoints = np.array(wp, dtype=float)
        waypoints[:, 2] = np.maximum(waypoints[:, 2], self._min_z)

        obstacles = np.asarray(obs.get("obstacles_pos", []), dtype=float)
        if obstacles.ndim == 2 and obstacles.shape[1] >= 3 and obstacles.shape[0] > 0:
            safe_dist = 0.3
            for i, p in enumerate(waypoints):
                for o in obstacles:
                    d = p - o
                    dist = np.linalg.norm(d)
                    if 1e-6 < dist < safe_dist:
                        waypoints[i] = o + d / dist * safe_dist

        return waypoints

    # --------------- main control --------------------
    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:

        # If obstacles are present, slightly repel waypoints away from them
        if "obstacles_pos" in obs and len(obs["obstacles_pos"]) > 0:
            obstacles = np.array(obs["obstacles_pos"])
            safe_dist = 0.3  # meters (minimum desired distance)


            for i, wp in enumerate(self.waypoints):
                for obs_pos in obstacles:
                    diff = wp - obs_pos
                    dist = np.linalg.norm(diff)
                    if dist < safe_dist and dist > 1e-6:
                        direction = diff / dist
                        self.waypoints[i] = obs_pos + direction * safe_dist
                        print(f"Adjusted waypoint {i} to avoid obstacle at {obs_pos}")

        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:
            self._finished = True

        # Desired from spline
        p_d = self._des_pos_spline(t)
        v_d = self._des_pos_spline(t, 1)
        a_d = self._des_pos_spline(t, 2)

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

        # Rebuild waypoints occasionally or when gates move significantly (only for E-shaping)
        if self._shaping_mode == "E" and (self._tick % int(self._freq * 0.5)) == 0:  # every 0.5s
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
