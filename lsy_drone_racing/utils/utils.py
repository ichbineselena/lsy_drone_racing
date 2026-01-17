"""Utility module."""

from __future__ import annotations

import importlib.util
import inspect
import logging
import sys
from typing import TYPE_CHECKING, Type

import mujoco
import numpy as np
import toml
from ml_collections import ConfigDict
from scipy.spatial.transform import Rotation as R

from lsy_drone_racing.control.controller import Controller

if TYPE_CHECKING:
    from pathlib import Path
    from typing import Any

    from numpy.typing import NDArray

    from lsy_drone_racing.envs.race_core import RaceCoreEnv


logger = logging.getLogger(__name__)


def load_controller(path: Path) -> Type[Controller]:
    """Load the controller module from the given path and return the Controller class.

    Args:
        path: Path to the controller module.
    """
    assert path.exists(), f"Controller file not found: {path}"
    assert path.is_file(), f"Controller path is not a file: {path}"
    spec = importlib.util.spec_from_file_location("controller", path)
    controller_module = importlib.util.module_from_spec(spec)
    sys.modules["controller"] = controller_module
    spec.loader.exec_module(controller_module)

    def filter(mod: Any) -> bool:
        """Filter function to identify valid controller classes.

        Args:
            mod: Any attribute of the controller module to be checked.
        """
        subcls = inspect.isclass(mod) and issubclass(mod, Controller)
        return subcls and mod.__module__ == controller_module.__name__

    controllers = inspect.getmembers(controller_module, filter)
    controllers = [c for _, c in controllers if issubclass(c, Controller)]
    assert len(controllers) > 0, f"No controller found in {path}. Have you subclassed Controller?"
    assert len(controllers) == 1, f"Multiple controllers found in {path}. Only one is allowed."
    controller_module.Controller = controllers[0]
    assert issubclass(controller_module.Controller, Controller)

    try:
        return controller_module.Controller
    except ImportError as e:
        raise e


def load_config(path: Path) -> ConfigDict:
    """Load the race config file.

    Args:
        path: Path to the config file.

    Returns:
        The configuration.
    """
    assert path.exists(), f"Configuration file not found: {path}"
    assert path.suffix == ".toml", f"Configuration file has to be a TOML file: {path}"

    with open(path, "r") as f:
        return ConfigDict(toml.load(f))


def draw_line(
    env: RaceCoreEnv,
    points: NDArray,
    rgba: NDArray | None = None,
    min_size: float = 3.0,
    max_size: float = 3.0,
):
    """Draw a line into the simulation.

    Args:
        env: The drone racing environment.
        points: An array of [N, 3] points that make up the line.
        rgba: The color of the line.
        min_size: The minimum line size. We linearly interpolate the size from min_size to max_size.
        max_size: The maximum line size.

    Note:
        This function has to be called every time before the env.render() step.
    """
    assert points.ndim == 2, f"Expected array of [N, 3] points, got Array of shape {points.shape}"
    assert points.shape[-1] == 3, f"Points must be 3D, are {points.shape[-1]}"
    sim = env.unwrapped.sim
    if sim.viewer is None:  # Do not attempt to add markers if viewer is still None
        return
    if sim.max_visual_geom < points.shape[0]:
        raise RuntimeError("Attempted to draw too many lines. Try to increase Sim.max_visual_geom")
    viewer = sim.viewer.viewer
    sizes = np.zeros_like(points)[:-1, :]
    sizes[:, 2] = np.linalg.norm(points[1:] - points[:-1], axis=-1)
    sizes[:, :2] = np.linspace(min_size, max_size, len(sizes))[..., None]
    if rgba is None:
        rgba = np.array([1.0, 0, 0, 1])
    mats = _rotation_matrix_from_points(points[:-1], points[1:]).as_matrix().reshape(-1, 9)
    for i in range(len(points) - 1):
        viewer.add_marker(
            type=mujoco.mjtGeom.mjGEOM_LINE, size=sizes[i], pos=points[i], mat=mats[i], rgba=rgba
        )


def draw_point(
    env: RaceCoreEnv,
    position: NDArray,
    color: tuple = (1.0, 0.0, 0.0, 1.0),
    size: float = 0.02,
    sphere_segments: int = 8,
):
    """Draw a point as a sphere into the simulation.
    
    Args:
        env: The drone racing environment.
        position: 3D position of the point.
        color: RGBA color tuple.
        size: Radius of the sphere.
        sphere_segments: Number of segments to approximate the sphere.
    """
    sim = env.unwrapped.sim
    if sim.viewer is None:
        return
        
    viewer = sim.viewer.viewer
    rgba = np.array(color)
    
    # Draw sphere using a set of lines to approximate it
    angles = np.linspace(0, 2*np.pi, sphere_segments)
    
    # Draw circles in XY plane
    for angle in angles:
        start = position.copy()
        end = position.copy()
        start[0] += size * np.cos(angle - np.pi/sphere_segments)
        start[1] += size * np.sin(angle - np.pi/sphere_segments)
        end[0] += size * np.cos(angle + np.pi/sphere_segments)
        end[1] += size * np.sin(angle + np.pi/sphere_segments)
        draw_line(env, np.array([start, end]), rgba=rgba, min_size=size*100, max_size=size*100)
    
    # Draw circles in XZ plane
    for angle in angles:
        start = position.copy()
        end = position.copy()
        start[0] += size * np.cos(angle - np.pi/sphere_segments)
        start[2] += size * np.sin(angle - np.pi/sphere_segments)
        end[0] += size * np.cos(angle + np.pi/sphere_segments)
        end[2] += size * np.sin(angle + np.pi/sphere_segments)
        draw_line(env, np.array([start, end]), rgba=rgba, min_size=size*100, max_size=size*100)
    
    # Draw circles in YZ plane
    for angle in angles:
        start = position.copy()
        end = position.copy()
        start[1] += size * np.cos(angle - np.pi/sphere_segments)
        start[2] += size * np.sin(angle - np.pi/sphere_segments)
        end[1] += size * np.cos(angle + np.pi/sphere_segments)
        end[2] += size * np.sin(angle + np.pi/sphere_segments)
        draw_line(env, np.array([start, end]), rgba=rgba, min_size=size*100, max_size=size*100)


def draw_capsule(
    env: RaceCoreEnv,
    start: NDArray,
    end: NDArray,
    radius: float,
    color: tuple = (0.0, 1.0, 1.0, 0.5),
    segments: int = 8,
):
    """Draw a capsule into the simulation.
    
    Args:
        env: The drone racing environment.
        start: Start position of the capsule.
        end: End position of the capsule.
        radius: Radius of the capsule.
        color: RGBA color tuple.
        segments: Number of segments to approximate the capsule.
    """
    # Calculate direction vector and length
    direction = end - start
    length = np.linalg.norm(direction)
    if length < 1e-6:
        # If start and end are the same, draw a sphere
        draw_point(env, start, color=color, size=radius)
        return
        
    direction = direction / length
    
    # Find a perpendicular vector
    if np.abs(direction[0]) < 0.9:
        perp = np.cross(direction, np.array([1.0, 0.0, 0.0]))
    else:
        perp = np.cross(direction, np.array([0.0, 1.0, 0.0]))
    perp = perp / np.linalg.norm(perp)
    
    # Find another perpendicular vector
    perp2 = np.cross(direction, perp)
    perp2 = perp2 / np.linalg.norm(perp2)
    
    rgba = np.array(color)
    
    # Draw cylindrical body using lines
    angles = np.linspace(0, 2*np.pi, segments)
    for angle in angles:
        # Calculate circle point
        circle_vec = radius * (np.cos(angle) * perp + np.sin(angle) * perp2)
        
        # Draw line along the cylinder
        line_start = start + circle_vec
        line_end = end + circle_vec
        draw_line(env, np.array([line_start, line_end]), rgba=rgba, 
                 min_size=radius*50, max_size=radius*50)
    
    # Draw end caps (hemispheres)
    cap_segments = segments // 2
    for i in range(cap_segments):
        phi = i * np.pi / cap_segments
        for j in range(segments):
            theta = j * 2*np.pi / segments
            
            # Start cap
            cap_offset = radius * np.array([
                np.sin(phi) * np.cos(theta),
                np.sin(phi) * np.sin(theta),
                np.cos(phi)
            ])
            # Rotate cap_offset to align with direction
            if phi <= np.pi/2:  # Start hemisphere
                cap_pos = start + cap_offset
            else:  # End hemisphere
                cap_pos = end - cap_offset
                
            # Draw small line at cap position
            draw_line(env, np.array([cap_pos, cap_pos + 0.001*direction]), 
                     rgba=rgba, min_size=radius*30, max_size=radius*30)


def draw_box(
    env: RaceCoreEnv,
    center: NDArray,
    quat: NDArray,
    half_sizes: NDArray,
    color: tuple = (1.0, 1.0, 0.0, 0.5),
):
    """Draw a box into the simulation.
    
    Args:
        env: The drone racing environment.
        center: Center position of the box.
        quat: Quaternion orientation of the box.
        half_sizes: Half sizes of the box in x, y, z directions.
        color: RGBA color tuple.
    """
    rot = R.from_quat(quat)
    R_matrix = rot.as_matrix()
    
    rgba = np.array(color)
    
    # Define the 8 corners of the box in local coordinates
    corners_local = np.array([
        [-half_sizes[0], -half_sizes[1], -half_sizes[2]],
        [ half_sizes[0], -half_sizes[1], -half_sizes[2]],
        [ half_sizes[0],  half_sizes[1], -half_sizes[2]],
        [-half_sizes[0],  half_sizes[1], -half_sizes[2]],
        [-half_sizes[0], -half_sizes[1],  half_sizes[2]],
        [ half_sizes[0], -half_sizes[1],  half_sizes[2]],
        [ half_sizes[0],  half_sizes[1],  half_sizes[2]],
        [-half_sizes[0],  half_sizes[1],  half_sizes[2]],
    ])
    
    # Transform to world coordinates
    corners_world = np.array([center + R_matrix @ corner for corner in corners_local])
    
    # Draw edges of the box
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Bottom face
        (4, 5), (5, 6), (6, 7), (7, 4),  # Top face
        (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
    ]
    
    for edge in edges:
        draw_line(env, np.array([corners_world[edge[0]], corners_world[edge[1]]]), 
                 rgba=rgba, min_size=1.0, max_size=1.0)


def _rotation_matrix_from_points(p1: NDArray, p2: NDArray) -> R:
    """Generate rotation matrices that align their z-axis to p2-p1."""
    z_axis = (v := p2 - p1) / np.linalg.norm(v, axis=-1, keepdims=True)
    random_vector = np.random.rand(*z_axis.shape)
    x_axis = (v := np.cross(random_vector, z_axis)) / np.linalg.norm(v, axis=-1, keepdims=True)
    y_axis = np.cross(z_axis, x_axis)
    return R.from_matrix(np.stack((x_axis, y_axis, z_axis), axis=-1))