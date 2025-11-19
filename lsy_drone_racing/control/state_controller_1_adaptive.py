from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R
from lsy_drone_racing.control import Controller

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StateController(Controller):
    """Controller following a spline-based trajectory through gates with optional obstacle avoidance."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        super().__init__(obs, info, config)

        # Environment parameters
        self.env = config.env
        self.freq = config.env.freq
        self.sensor_range = config.env.sensor_range
        self.dt = 1.0 / self.freq
        self._t_total = 25.0

        # Gate traversal configuration
        self.approach_d = 0.35
        self.exit_d = 0.55

        # Obstacle avoidance
        self.avoid_dist = 0.6
        self.obs_radius = 0.15
        self.margin = 0.25

        # PD control gains
        self.Kp = np.diag([6.0, 6.0, 10.0])
        self.Kd = np.diag([3.0, 3.0, 5.0])

        # Internal state
        self.tick = 0
        self.finished = False
        self.last_update_tick = -100

        # Register gates
        self.gates = []
        for g in config.env.track.gates:
            pos = np.array(g["pos"], float)
            rot = R.from_euler("xyz", g["rpy"])
            normal = rot.as_matrix()[:, 0]
            self.gates.append({
                "nominal_pos": pos,
                "nominal_normal": normal,
                "detected_pos": None,
                "detected_normal": None,
            })
        self.n_gates = len(self.gates)

        # Build initial trajectory
        self._rebuild_trajectory()

    # ===============================================================
    # Main control logic
    # ===============================================================
    def compute_control(self, obs: dict[str, NDArray[np.floating]], info=None) -> NDArray[np.floating]:
        """Compute desired state (pos, vel, acc, yaw, rates)."""
        # 1️⃣ Update gate detections occasionally
        early_phase = self.tick < int(5.0 * self.freq)
        update_period = 5 if early_phase else 20
        if self.tick - self.last_update_tick > update_period:
            self._detect_gates(obs)

        # 2️⃣ Evaluate spline trajectory
        t = min(self.tick / self.freq, self._t_total)
        if t >= self._t_total:
            self.finished = True

        des_pos = self._des_pos_spline(t)
        des_vel = self._des_pos_spline(t, 1)
        des_acc = self._des_pos_spline(t, 2)

        cur_pos = np.asarray(obs["pos"], float)
        cur_vel = np.asarray(obs["vel"], float)

        # 3️⃣ Local obstacle avoidance (merged inline)
        if "obstacles_pos" in obs and len(obs["obstacles_pos"]) > 0:
            des_pos = self._avoid_obstacles(des_pos, cur_pos, des_vel, np.asarray(obs["obstacles_pos"]))

        # 4️⃣ PD feedback control
        pos_err = des_pos - cur_pos
        vel_err = des_vel - cur_vel
        acc_cmd = des_acc + self.Kp @ pos_err + self.Kd @ vel_err

        # 5️⃣ Desired yaw from motion direction
        if np.linalg.norm(des_vel[:2]) > 1e-3:
            yaw = np.arctan2(des_vel[1], des_vel[0])
        elif np.linalg.norm(acc_cmd[:2]) > 1e-3:
            yaw = np.arctan2(acc_cmd[1], acc_cmd[0])
        else:
            yaw = 0.0

        # 6️⃣ Build action
        self.tick += 1
        return np.array([*des_pos, *des_vel, *acc_cmd, yaw, 0, 0, 0], dtype=np.float32)

    # ===============================================================
    # Helpers
    # ===============================================================
    def _rebuild_trajectory(self):
        """Build or rebuild a cubic spline trajectory through gates."""
        wps = [np.array([-1.5, 0.75, 0.05], float)]
        for i in range(self.n_gates):
            wps += self._gate_points(i)
            if i < self.n_gates - 1:
                next_approach = self._gate_points(i + 1)[0]
                wps.append((wps[-1] + next_approach) / 2)
        wps.append(wps[-1] + np.array([0.5, 0.5, 0.0]))
        self.waypoints = np.array(wps)
        t = np.linspace(0, self._t_total, len(wps))
        self._des_pos_spline = CubicSpline(t, wps)

    def _gate_points(self, i: int) -> list[np.ndarray]:
        """Return [approach, center, exit, (optional steering)] for a gate."""
        g = self.gates[i]
        pos = g["detected_pos"] if g["detected_pos"] is not None else g["nominal_pos"]
        normal = g["detected_normal"] if g["detected_normal"] is not None else g["nominal_normal"]

        approach = pos - self.approach_d * normal
        center = pos
        exit_pt = pos + self.exit_d * normal

        # Optional steering for gate 3 (index 2)
        if i == 2:
            steer = exit_pt + np.array([-0.2, 0.0, 0.5])
            return [[-0.5, -0.05, 0.7], approach, center, exit_pt, steer]
        return [approach, center, exit_pt]

    def _detect_gates(self, obs: dict[str, NDArray[np.floating]]):
        """Detect nearby gates and rebuild trajectory if one has shifted."""
        if "gates_pos" not in obs or "gates_quat" not in obs:
            return
        cur_pos = obs["pos"]
        updated = False

        for i, gpos in enumerate(obs["gates_pos"]):
            if np.linalg.norm(gpos - cur_pos) > self.sensor_range:
                continue
            if self.gates[i]["detected_pos"] is None:
                rot = R.from_quat(obs["gates_quat"][i])
                normal = rot.as_matrix()[:, 0]
                diff = np.linalg.norm(gpos - self.gates[i]["nominal_pos"])
                self.gates[i]["detected_pos"] = gpos.copy()
                self.gates[i]["detected_normal"] = normal.copy()
                if diff > 0.05:
                    print(f"[Tick {self.tick}] Gate {i} shifted by {diff:.2f}m → replanning.")
                    updated = True
        if updated:
            self._rebuild_trajectory()
            self.last_update_tick = self.tick

    def _avoid_obstacles(
        self,
        des_pos: NDArray[np.floating],
        cur_pos: NDArray[np.floating],
        des_vel: NDArray[np.floating],
        obstacles: NDArray[np.floating],
    ) -> NDArray[np.floating]:
        """Lateral obstacle avoidance in XY plane."""
        traj_dir = des_vel[:2] if np.linalg.norm(des_vel[:2]) > 1e-2 else (des_pos[:2] - cur_pos[:2])
        if np.linalg.norm(traj_dir) < 1e-3:
            return des_pos
        traj_dir /= np.linalg.norm(traj_dir)
        perp = np.array([-traj_dir[1], traj_dir[0]])

        total_offset = np.zeros(2)
        clearance = self.obs_radius + self.margin

        for obs in obstacles:
            to_obs = obs[:2] - cur_pos[:2]
            if np.linalg.norm(to_obs) > self.avoid_dist:
                continue
            proj = np.dot(to_obs, traj_dir)
            if proj < -0.2:  # behind us
                continue
            lateral = abs(np.dot(to_obs, perp))
            if lateral < clearance:
                side = np.sign(np.dot(to_obs, perp))
                deficit = clearance - lateral
                strength = 1.5 * deficit / max(clearance, 1e-6)
                total_offset += (-side) * perp * strength * 0.2

        if np.any(total_offset):
            des_pos = des_pos.copy()
            des_pos[:2] += total_offset
        return des_pos

    # ===============================================================
    # Callbacks
    # ===============================================================
    def step_callback(self, *args, **kwargs) -> bool:
        return self.finished

    def episode_callback(self):
        """Reset internal state for new episode."""
        self.tick = 0
        self.last_update_tick = -100
        for g in self.gates:
            g["detected_pos"] = None
            g["detected_normal"] = None
        self._rebuild_trajectory()
