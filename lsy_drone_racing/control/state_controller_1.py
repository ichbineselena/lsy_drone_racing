"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

from __future__ import annotations  # Python 3.10 type hints

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control import Controller
from lsy_drone_racing.utils import draw_line

if TYPE_CHECKING:
    from numpy.typing import NDArray


class StateController(Controller):
    """State controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        self.env = config.env
        self._freq = config.env.freq
        self._sensor_range = config.env.sensor_range
        
        # Gate traversal parameters
        self._enter_dist = 0.35   # meters before gate center
        self._exit_dist = 0.55       # meters after gate center

        # Obstacle avoidance parameters
        self._avoidance_distance = 0.5   # detection range
        self._obstacle_dim = 0.15     # how far the obstacle must be avoided from the centre
        
        # PD controller gains for trajectory tracking
        self._Kp = np.diag([3.0, 3.0, 5.0])  # Position proportional gains [x, y, z]
        self._Kd = np.diag([3.0, 3.0, 5.0])  # Velocity derivative gains [x, y, z]
        
        # Gate detection tracking
        self._last_update_tick = -1000  # Prevent frequent trajectory updates
            
        # Store pre gate information from config
        self._gates = []
        for gate in config.env.track.gates:
            gate_pos = np.array(gate["pos"], dtype=float)
            gate_rpy = np.array(gate["rpy"], dtype=float)
            rot = R.from_euler("xyz", gate_rpy)
            gate_rpy = rot.as_matrix()[:, 0]  # x-axis points through gate
            
            self._gates.append({
                "pre_pos": gate_pos.copy(),
                "pre_rpy": gate_rpy.copy(),
                "obs_pos": None,
                "obs_rpy": None,
            })
            
        self._num_gates = len(self._gates)

#         # Same waypoints as in the attitude controller. Determined by trial and error.
#         waypoints = np.array(
#             [
#                 [-1.5, 0.75, 0.05],
#                 [-1.0, 0.55, 0.4],
#                 [0.3, 0.35, 0.7],
#                 [1.3, -0.15, 0.9],

#                 [0.85, 0.85, 1.2],
#                 [-0.5, -0.05, 0.7],
#                 [-1.2, -0.2, 0.8],

#                 [-1.2, -0.2, 1.2],
#                 [-0.0, -0.7, 1.2],
#                 [0.5, -0.75, 1.2],
#             ]
#         )
#         # Tall gates: 1.195m height. Short gates: 0.695m height. Height is measured from the ground to the center of the gate.
# [[env.track.gates]]
# pos = [0.5 , 0.25, 0.7]
# rpy = [0.0, 0.0, -0.78]
# [[env.track.gates]]
# pos = [1.05, 0.75, 1.2]
# rpy = [0.0, 0.0, 2.35]
# [[env.track.gates]]

# pos = [-1.0, -0.25, 0.7]
# rpy = [0.0, 0.0, 3.14]
# [[env.track.gates]]

# pos = [0.0, -0.75, 1.2]
# rpy = [0.0, 0.0, 0.0]


        # waypoints = np.expand_dims(obs["pos"], axis=0)
        # waypoints = np.insert(waypoints, len(waypoints), obs["gates_pos"][0]+[-2.0, 0.5, -0.65], axis=0)
        # waypoints = np.insert(waypoints, len(waypoints), obs["gates_pos"][0]+[-1.5, 0.3, -0.3], axis=0)
        # waypoints = np.insert(waypoints, len(waypoints), obs["gates_pos"][1]+[-0.75, -0.4, -0.5], axis=0)
        # waypoints = np.insert(waypoints, len(waypoints), obs["gates_pos"][1]+[0.25, -0.8, -0.3], axis=0)
        # waypoints = np.insert(waypoints, len(waypoints), obs["gates_pos"][2]+[1.85, 1, 0], axis=0)
        # waypoints = np.insert(waypoints, len(waypoints), obs["gates_pos"][2]+[0.5, 0.2, 0], axis=0)
        # waypoints = np.insert(waypoints, len(waypoints), obs["gates_pos"][2]+[-0.2, 0.05, 0.1], axis=0)
        # waypoints = np.insert(waypoints, len(waypoints), obs["gates_pos"][2]+[-0.2, 0.1, 0.5], axis=0)
        # waypoints = np.insert(waypoints, len(waypoints), obs["gates_pos"][3]+[0.0, -0.7, 0], axis=0)
        # waypoints = np.insert(waypoints, len(waypoints), [0.5, -0.75, 1.2], axis=0)
        # Build initial trajectory using state_simple-style generation
        self._t_total = 25.0  # s
        self._build_trajectory()

        self._tick = 0
        self._finished = False

    def compute_control(
        self, obs: dict[str, NDArray[np.floating]], info: dict | None = None
    ) -> NDArray[np.floating]:
        """Compute the next desired state of the drone.

        Args:
            obs: The current observation of the environment. See the environment's observation space
                for details.
            info: Optional additional information as a dictionary.

        Returns:
            The drone state [x, y, z, vx, vy, vz, ax, ay, az, yaw, rrate, prate, yrate] as a numpy
                array.
        """
        # Waypoints are generated once in __init__ via _build_trajectory.
        # compute_control now only evaluates the prebuilt spline.
        
        # Update gate detections periodically (but not too frequently to avoid trajectory jumps)
        early_phase = self._tick < int(5.0 * self._freq)
        update_gap = 5 if early_phase else 20
        if self._tick - self._last_update_tick > update_gap:
            self.detect_gates(obs)

        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        des_pos = self._des_pos_spline(t)
        # Get desired velocity and acceleration from spline derivatives
        des_vel = self._des_pos_spline(t, 1)
        des_acc = self._des_pos_spline(t, 2)

        # Apply lateral obstacle avoidance (horizontal plane) if obstacles are present
        if "obstacles_pos" in obs and len(obs["obstacles_pos"]) > 0:
            des_pos = self.avoid_obstacles(
                des_pos,
                obs["pos"],
                des_vel,
                np.asarray(obs["obstacles_pos"], dtype=float)
            )
        
        # PD control: compute feedback acceleration based on tracking errors
        current_pos = np.asarray(obs["pos"], dtype=float)
        current_vel = np.asarray(obs["vel"], dtype=float)
        
        pos_error = des_pos - current_pos
        vel_error = des_vel - current_vel
        
        # PD control law: acc = feedforward + P*pos_error + D*vel_error
        acc_feedback = self._Kp @ pos_error + self._Kd @ vel_error
        acc_cmd = des_acc + acc_feedback
        
        # Compute yaw from velocity direction
        if np.linalg.norm(des_vel[:2]) > 1e-3:
            yaw_d = float(np.arctan2(des_vel[1], des_vel[0]))
        elif np.linalg.norm(acc_cmd[:2]) > 1e-3:
            yaw_d = float(np.arctan2(acc_cmd[1], acc_cmd[0]))
        else:
            yaw_d = 0.0
        
        # Build state action: [pos(3), vel(3), acc(3), yaw(1), rates(3)]
        action = np.array([
            *des_pos,
            *des_vel,
            *acc_cmd,
            yaw_d,
            0.0, 0.0, 0.0
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
        """Increment the time step counter.

        Returns:
            True if the controller is finished, False otherwise.
        """
        self._tick += 1
        return self._finished

    def _build_trajectory(self):
        """Build spline trajectory through all gates (state_simple-style)."""
        waypoints = []

        # Starting position (fixed like in state_simple)
        waypoints.append(np.array([-1.5, 0.75, 0.05], dtype=float))
        waypoints.append(np.array([-1.0, 0.55, 0.4], dtype=float))

        # Smooth transition to first gate
        first_gate_wps = self.observe_gates(0)
        # mid_to_first = (waypoints[0] + first_gate_wps[0]) / 2
        # waypoints.append(mid_to_first)

        # Waypoints for each gate with mid transitions
        for i in range(self._num_gates):
            gate_wps = self.observe_gates(i)
            waypoints.extend(gate_wps)

            if i < self._num_gates - 1:
                next_gate_wps = self.observe_gates(i + 1)
                mid_point = (gate_wps[-1] + next_gate_wps[0]) / 2
                waypoints.append(mid_point)

        # Final hover after last gate
        last_exit = waypoints[-1]
        waypoints.append(last_exit + np.array([0.5, 0.5, 0.0], dtype=float))

        self.waypoints = np.array(waypoints, dtype=float)

        # Create spline through waypoints
        t = np.linspace(0, self._t_total, len(self.waypoints))
        self._des_pos_spline = CubicSpline(t, self.waypoints)

    def detect_gates(self, obs: dict[str, NDArray[np.floating]]):
        """Detect gate positions and rpys when detected within sensor range.
        
        Rebuilds trajectory if any gate position changes significantly.
        """
        if "gates_pos" not in obs or "gates_quat" not in obs:
            return
        
        current_pos = obs["pos"]
        trajectory_updated = False
        
        for i, gate_pos in enumerate(obs["gates_pos"]):
            # Check if gate is within sensor range
            dist = np.linalg.norm(gate_pos - current_pos)
            if dist > self._sensor_range:
                continue
            
            # First time detecting this gate
            if self._gates[i]["obs_pos"] is None:
                # Get gate orientation from quaternion
                rot = R.from_quat(obs["gates_quat"][i])
                gate_rpy = rot.as_matrix()[:, 0]  # x-axis points through gate
                
                # Check if position changed significantly from nominal
                pos_change = np.linalg.norm(gate_pos - self._gates[i]["pre_pos"])
                
                # Store detected position and rpy
                self._gates[i]["obs_pos"] = gate_pos.copy()
                self._gates[i]["obs_rpy"] = gate_rpy.copy()
                
                if pos_change > 0.05:  # Significant change (>5cm)
                    print(f"[Tick {self._tick}] Gate {i} detected with {pos_change:.3f}m offset")
                    trajectory_updated = True
                else:
                    print(f"[Tick {self._tick}] Gate {i} detected (nominal position)")
        
        # Rebuild trajectory if any gate changed significantly
        if trajectory_updated:
            print(f"[Tick {self._tick}] Rebuilding trajectory with updated gates...")
            self._build_trajectory()
            self._last_update_tick = self._tick

    def avoid_obstacles(
        self,
        des_pos: NDArray[np.floating],
        current_pos: NDArray[np.floating],
        des_vel: NDArray[np.floating],
        obstacles_pos: NDArray[np.floating]
    ) -> NDArray[np.floating]:
        """Shift the drone away from the obstacles.

        - Only adjusts x/y (keeps altitude).
        - Shifts perpendicular to the trajectory direction to avoid cutting corners into obstacles.
        """
        if obstacles_pos is None or len(obstacles_pos) == 0:
            return des_pos

        # Determine trajectory direction in the horizontal plane
        traj_dir = des_vel[:2].copy()
        if np.linalg.norm(traj_dir) < 1e-2:
            traj_dir = (des_pos[:2] - current_pos[:2])
        if np.linalg.norm(traj_dir) < 1e-3:
            return des_pos
        traj_dir = traj_dir / (np.linalg.norm(traj_dir) + 1e-12)

        # Perpendicular to trajectory in XY
        perp = np.array([-traj_dir[1], traj_dir[0]], dtype=float)

        push_away = np.zeros(2, dtype=float)
        safe_dist = 2*self._obstacle_dim

        for o in obstacles_pos:
            # distance checks in XY
            to_obs_from_drone = o[:2] - current_pos[:2]
            to_obs_from_des   = o[:2] - des_pos[:2]

            # Skip far obstacles
            if min(np.linalg.norm(to_obs_from_drone), np.linalg.norm(to_obs_from_des)) > self._avoidance_distance:
                continue

            # Check if obstacle is roughly along the forward direction
            forward_proj = float(np.dot(to_obs_from_drone, traj_dir))
            if forward_proj < -0.2:
                continue

            # Lateral distance to trajectory line
            lateral_dist = abs(float(np.dot(to_obs_from_drone, perp)))

            if lateral_dist < safe_dist:
                # Determine side of obstacle relative to trajectory
                side = np.sign(float(np.dot(to_obs_from_drone, perp)))
                # Compute minimal required lateral push
                deficit = (safe_dist - lateral_dist)
                # Scale gently so we don't overreact
                factor = 1.5 * deficit / max(safe_dist, 1e-6)
                push_away += (-side) * perp * factor * 0.20

        if np.allclose(push_away, 0.0):
            return des_pos

        mod = des_pos.copy()
        mod[:2] += push_away
        return mod

    def observe_gates(self, gate_idx: int) -> list:
        """Get waypoints for smooth gate traversal including entrance and exit."""
        gate = self._gates[gate_idx]
        # Use detected position if available, else nominal
        pos = gate["obs_pos"] if gate["obs_pos"] is not None else gate["pre_pos"]
        rpy = gate["obs_rpy"] if gate["obs_rpy"] is not None else gate["pre_rpy"]

        # Create enter, center, and exit points
        enter_point = pos - self._enter_dist * rpy
        center = pos.copy()
        exit_point = pos + self._exit_dist * rpy

        if gate_idx == 2:  # Gate 3 (0-indexed)
            pre_enter = center.copy()
            pre_enter[0] += 0.5  # move right (positive x)
            pre_enter[1] -= 0.65  # move upward (positive
            steering_point = exit_point.copy()
            steering_point[0] -= 0.2  # move right (positive x)
            steering_point[2] += 0.5  # move upward (positive z)
            return [[-0.5, -0.05, 0.7], enter_point, center, exit_point, steering_point]
        
        return [enter_point, center, exit_point]

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0
        self._last_update_tick = -1000
        # Reset gate detections
        for gate in self._gates:
            gate["obs_pos"] = None
            gate["obs_rpy"] = None
        # Rebuild trajectory from nominal positions
        self._build_trajectory()
