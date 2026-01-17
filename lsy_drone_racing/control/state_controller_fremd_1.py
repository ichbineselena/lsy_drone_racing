"""Controller that follows a pre-defined trajectory.

It uses a cubic spline interpolation to generate a smooth trajectory through a series of waypoints.
At each time step, the controller computes the next desired position by evaluating the spline.

.. note::
    The waypoints are hard-coded in the controller for demonstration purposes. In practice, you
    would need to generate the splines adaptively based on the track layout, and recompute the
    trajectory if you receive updated gate and obstacle poses.
"""

# Future imports (always placed at the very top)
from __future__ import annotations

# Standard library imports
from typing import TYPE_CHECKING, List, Tuple, Dict, Any

# Third-party imports
import numpy as np
from scipy.interpolate import interp1d
from scipy.spatial.transform import Rotation as R

# Local application imports
from lsy_drone_racing.control import Controller
# from lsy_drone_racing.utils import utils  # Uncomment when needed

# Type checking
if TYPE_CHECKING:
    from numpy.typing import NDArray

def length(v: np.ndarray) -> float:
    """Computes the Euclidean length of a vector.

    Args:
        v (np.ndarray): Input vector.

    Returns:
        float: The norm (length) of the vector.
    """
    return np.linalg.norm(v)


def normalize(v: np.ndarray) -> np.ndarray:
    """Normalizes a vector to unit length.

    Args:
        v (np.ndarray): Input vector.

    Returns:
        np.ndarray: Normalized vector. If the input has zero length, returns the original vector.
    """
    length_v = length(v)
    if length_v != 0:
        return v / length_v
    return v


def angle_between(v1: np.ndarray, v2: np.ndarray) -> float:
    """Calculates the angle in degrees between two vectors.

    Args:
        v1 (np.ndarray): First vector.
        v2 (np.ndarray): Second vector.

    Returns:
        float: Angle between the vectors in degrees.
    """
    v1 = normalize(v1)
    v2 = normalize(v2)
    dot = np.clip(np.dot(v1, v2), -1.0, 1.0)
    angle_rad = np.arccos(dot)
    return np.degrees(angle_rad)


def compute_evasion_angles(V1: np.ndarray, V2: np.ndarray, evas: List[np.ndarray]) -> Tuple[float, float]:
    """Computes the angular deviation between the original direction and two evasive directions.

    Args:
        V1 (np.ndarray): Starting point.
        V2 (np.ndarray): Target point.
        evas (List[np.ndarray]): List containing two evasive points.

    Returns:
        Tuple[float, float]: Absolute angles between original direction and each evasive direction.
    """
    original_dir = V2 - V1
    evas_dir_0 = V2 - evas[0]
    evas_dir_1 = V2 - evas[1]

    angle_0 = angle_between(original_dir, evas_dir_0)
    angle_1 = angle_between(original_dir, evas_dir_1)

    return abs(angle_0), abs(angle_1)


def sort_by_distance(points: List[np.ndarray], reference_point: np.ndarray) -> List[np.ndarray]:
    """Sorts a list of points by their distance to a reference point.

    Args:
        points (List[np.ndarray]): List of points to sort.
        reference_point (np.ndarray): Point to measure distances from.

    Returns:
        List[np.ndarray]: Points sorted by ascending distance to the reference point.
    """
    distances = [np.linalg.norm(p - reference_point) for p in points]
    paired = list(zip(points, distances))
    paired_sorted = sorted(paired, key=lambda x: x[1])
    sorted_points = [p for p, _ in paired_sorted]
    return sorted_points


class Pipe:
    """Represents a cylindrical pipe segment in 3D space, used for collision detection and evasion logic."""

    def __init__(
        self,
        center_pos: List[float],
        direction: List[float],
        ri: float,
        ra: float,
        h: float
    ) -> None:
        """Initializes a Pipe object.

        Args:
            center_pos (List[float]): Center position of the pipe.
            direction (List[float]): Direction vector of the pipe's axis.
            ri (float): Inner radius of the pipe.
            ra (float): Outer radius of the pipe.
            h (float): Height of the pipe.
        """
        self.center_pos = np.array(center_pos, dtype=float)
        self.direction = np.array(direction, dtype=float)
        self.half_h = 0.5 * h
        self.ri = ri
        self.ra = ra

        self.axis_start = self.center_pos - self.direction * self.half_h
        self.axis_end = self.center_pos + self.direction * self.half_h

        self._compute_bounds()
        self.fix_evasion_pos: List[np.ndarray] = []

        if h < 1:
            fix_pos = self.center_pos + np.array([0, 0, 1.1*self.ra]) - self.direction * 0.15
            self.fix_evasion_pos = [fix_pos, fix_pos]

    def _compute_bounds(self) -> None:
        """Computes the bounding box of the pipe based on its radius and axis."""
        points = [
            self.axis_start + np.array([dx, dy, dz])
            for dx in [-self.ra, self.ra]
            for dy in [-self.ra, self.ra]
            for dz in [-self.ra, self.ra]
        ] + [
            self.axis_end + np.array([dx, dy, dz])
            for dx in [-self.ra, self.ra]
            for dy in [-self.ra, self.ra]
            for dz in [-self.ra, self.ra]
        ]
        points = np.array(points)
        self.bbox_min = np.min(points, axis=0)
        self.bbox_max = np.max(points, axis=0)

    def contains_point(self, point: np.ndarray) -> bool:
        """Checks whether a given point lies within the pipe's volume.

        Args:
            point (np.ndarray): The point to check.

        Returns:
            bool: True if the point is inside the pipe, False otherwise.
        """
        v = point - self.axis_start
        proj_len = np.dot(v, self.direction)

        if proj_len < 0 or proj_len > 2 * self.half_h:
            return False

        self.proj_point = self.axis_start + proj_len * self.direction
        radial_dist = np.linalg.norm(point - self.proj_point)

        return self.ri <= radial_dist <= self.ra

    def is_colliding(self, V1: np.ndarray, V2: np.ndarray) -> bool:
        """Checks whether a line segment between V1 and V2 collides with the pipe.

        Args:
            V1 (np.ndarray): Start point of the segment.
            V2 (np.ndarray): End point of the segment.

        Returns:
            bool: True if the segment collides with the pipe, False otherwise.
        """
        seg_min = np.minimum(V1, V2)
        seg_max = np.maximum(V1, V2)

        if np.any(seg_max < self.bbox_min) or np.any(seg_min > self.bbox_max):
            return False

        seg_dir = V2 - V1
        seg_len = np.linalg.norm(seg_dir)

        if seg_len == 0:
            return self.contains_point(V1)

        steps = int(seg_len / 0.05) + 1
        for i in range(steps + 1):
            alpha = i / steps
            point = V1 + alpha * seg_dir
            if self.contains_point(point):
                if self.fix_evasion_pos:
                    self.evasion_pos = self.fix_evasion_pos
                else:
                    self.evasion_pos = []
                    cross = normalize(np.cross(seg_dir, self.direction))
                    self.evasion_pos.append(self.proj_point + self.ra * cross)
                    self.evasion_pos.append(self.proj_point - self.ra * cross)
                return True

        return False

    def set_up_evasion(self, obstacles: List["Pipe"]) -> None:
        """Computes an evasion direction based on surrounding obstacles.

        Args:
            obstacles (List[Pipe]): List of other pipe objects to avoid.
        """
        total_vec = np.zeros(3)

        for other in obstacles:
            if other is not self:
                vec = self.center_pos - other.center_pos
                if np.linalg.norm(vec) > 0:
                    total_vec += vec / np.linalg.norm(vec)

        if np.linalg.norm(total_vec) == 0:
            d = self.direction / np.linalg.norm(self.direction)
            total_vec = np.cross(d, [1, 0, 0])
            if np.linalg.norm(total_vec) == 0:
                total_vec = np.cross(d, [0, 1, 0])

        evas_dir = total_vec / np.linalg.norm(total_vec) * self.ra
        self.evas_dir = evas_dir


class Pathfinder:
    """Computes and manages a navigable path through gates and obstacles using evasion logic."""

    def __init__(self, obs: Dict[str, Any]) -> None:
        """Initializes the Pathfinder with environment data.

        Args:
            obs (Dict[str, Any]): Observation dictionary containing position, gates, and obstacles.
        """
        self.gate_pos_offset = 0.2
        self.gate_ri = 0.1
        self.gate_ra = 0.6
        self.gate_h = 0.2
        self.stab_ra = 0.3
        self.fly_offset = 0.15
        self.fly_speed = 1.5  # points per second

        self.start_pos = obs['pos']
        self.current_pos = self.start_pos
        self.fly_end = self.start_pos
        self.path_free_i = 0
        self.is_rrt = False
        self.last_t = -0.001

    def update(self, obs: Dict[str, Any]) -> None:
        """Updates the current position and recalculates the path.

        Args:
            obs (Dict[str, Any]): Updated observation data.
        """
        self.current_pos = obs['pos']
        self.set_obs(obs)
        self.check_path()
        self.interpolate_path()

    def set_obs(self, obs: Dict[str, Any]) -> None:
        """Sets up obstacles and gates based on the observation data.

        Args:
            obs (Dict[str, Any]): Observation data containing gates and obstacles.
        """
        self.obstacles: List[Pipe] = []
        self.current_pos = obs['pos']
        self.path_free = [self.start_pos]

        for gate_i, gate_pos in enumerate(obs['gates_pos']):
            gate_before, gate_after, gate_dir = self.get_gate_pos_and_dir(
                gate_pos, obs['gates_quat'][gate_i]
            )
            self.path_free.extend([gate_before, gate_pos, gate_after])
            self.obstacles.append(Pipe(gate_pos, gate_dir, self.gate_ri, self.gate_ra, self.gate_h))

        for stab_pos in obs['obstacles_pos']:
            self.obstacles.append(Pipe(stab_pos, [0, 0, 1], 0, self.stab_ra, 4))

    def check_path(self) -> None:
        """Constructs the path with evasion points between gates and obstacles."""
        self.path_eva = [self.start_pos]
        i = 1
        while i + 2 < len(self.path_free):
            pos_before_before = self.path_free[i - 1]
            pos_before = self.path_free[i]
            self.add_evasion_pos(pos_before_before, pos_before)
            self.path_eva.append(pos_before)
            i += 1
            self.path_eva.append(self.path_free[i])
            i += 1
            self.path_eva.append(self.path_free[i])
            i += 1

    def add_evasion_pos(self, V1: np.ndarray, V2: np.ndarray) -> None:
        """Adds evasion points between two path segments if a collision is detected.

        Args:
            V1 (np.ndarray): Start point of the segment.
            V2 (np.ndarray): End point of the segment.
        """
        new_evasion_pos: List[np.ndarray] = []

        for obstacle in self.obstacles:
            if obstacle.is_colliding(V1, V2):
                evas_pos = obstacle.evasion_pos
                da0, da1 = compute_evasion_angles(V1, V2, evas_pos)
                i_c = 0 if da0 < da1 else 1

                for obst in self.obstacles:
                    if obst is not obstacle and obst.contains_point(evas_pos[i_c]):
                        i_c = 1 - i_c
                        break

                new_evasion_pos.append(evas_pos[i_c])

        new_evasion_pos = sort_by_distance(new_evasion_pos, V1)
        self.path_eva += new_evasion_pos

    def get_gate_pos_and_dir(self, pos: np.ndarray, quat: List[float]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Computes gate direction and offset positions based on quaternion orientation.

        Args:
            pos (np.ndarray): Gate position.
            quat (List[float]): Quaternion representing gate orientation.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]: Offset positions before and after the gate, and direction vector.
        """
        rot = R.from_quat(quat)
        gate_dir = rot.apply([-1, 0, 0])
        shift = gate_dir * self.gate_pos_offset
        return pos + shift, pos - shift, gate_dir

    def is_path_free(self, path: List[np.ndarray]) -> bool:
        """Checks whether a given path is free of collisions.

        Args:
            path (List[np.ndarray]): List of path points.

        Returns:
            bool: True if the path is collision-free, False otherwise.
        """
        for i in range(1, len(path)):
            V1 = path[i - 1]
            V2 = path[i]
            for obstacle in self.obstacles:
                if obstacle.is_colliding(V1, V2):
                    return False
        return True

    def interpolate_path(self) -> None:
        """Interpolates the path using linear interpolation for smooth navigation."""
        self.path = np.asarray(self.path_eva)
        self.N = len(self.path)
        t_values = np.linspace(0, self.N / self.fly_speed, self.N)
        self.spline = interp1d(t_values, self.path, axis=0)

    def des_pos(self, t: float) -> np.ndarray:
        """Returns the desired position at time t along the interpolated path.

        Args:
            t (float): Time value.

        Returns:
            np.ndarray: Interpolated position at time t.
        """
        if t >= self.N / self.fly_speed:
            return self.current_pos
        return self.spline(t) - [0,0,0.05]


class StateController(Controller):
    """State controller following a pre-defined trajectory."""

    def __init__(self, obs: dict[str, NDArray[np.floating]], info: dict, config: dict):
        """Initialization of the controller.

        Args:
            obs: The initial observation of the environment's state. See the environment's
                observation space for details.
            info: The initial environment information from the reset.
            config: The race configuration. See the config files for details. Contains additional
                information such as disturbance configurations, randomizations, etc.
        """
        super().__init__(obs, info, config)
        self._freq = config.env.freq

        self._t_total = 30 

        self.pf = Pathfinder(obs)

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
        t = min(self._tick / self._freq, self._t_total)
        if t >= self._t_total:  # Maximum duration reached
            self._finished = True

        self.pf.update(obs)

        #utils.draw_line(info['env'],np.asarray(self.pf.path_eva))
        des_pos = self.pf.des_pos(t)
        
        action = np.concatenate((des_pos, np.zeros(10)), dtype=np.float32)
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

    def episode_callback(self):
        """Reset the internal state."""
        self._tick = 0
