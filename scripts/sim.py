"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

# Suppress DLPack buffer alignment warnings
logging.getLogger("tensorflow").setLevel(logging.ERROR)
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller, draw_line, draw_point, draw_capsule, draw_box

# Import after lsy_drone_racing to avoid SCIPY_ARRAY_API issues
from scipy.spatial.transform import Rotation as R

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level2.toml",
    controller: str | None = None,
    n_runs: int = 1,
    render: bool | None = None,
) -> list[float]:
    """Evaluate the drone controller over multiple episodes.

    Args:
        config: The path to the configuration file. Assumes the file is in `config/`.
        controller: The name of the controller file in `lsy_drone_racing/control/` or None. If None,
            the controller specified in the config file is used.
        n_runs: The number of episodes.
        render: Enable/disable rendering the simulation.

    Returns:
        A list of episode times.
    """
    # Load configuration and check if firmare should be used.
    config = load_config(Path(__file__).parents[1] / "config" / config)
    if render is None:
        render = config.sim.render
    else:
        # Convert string to bool if necessary (fire.Fire passes CLI args as strings)
        if isinstance(render, str):
            render = render.lower() in ('true', '1', 'yes', 'on')
        config.sim.render = render
    # Load the controller module
    control_path = Path(__file__).parents[1] / "lsy_drone_racing/control"
    controller_path = control_path / (controller or config.controller.file)
    controller_cls = load_controller(controller_path)  # This returns a class, not an instance
    # Create the racing environment
    env: DroneRaceEnv = gymnasium.make(
        config.env.id,
        freq=config.env.freq,
        sim_config=config.sim,
        sensor_range=config.env.sensor_range,
        control_mode=config.env.control_mode,
        track=config.env.track,
        disturbances=config.env.get("disturbances"),
        randomizations=config.env.get("randomizations"),
        seed=config.env.seed,
    )
    env = JaxToNumpy(env)

    ep_times = []
    for _ in range(n_runs):  # Run n_runs episodes with the controller
        obs, info = env.reset()
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq

            action = controller.compute_control(obs, info)
            
            # --- Draw visualizations ---
            if config.sim.render and config.sim.visualize:
                try:
                    # 1. Draw current goal point
                    if hasattr(controller, 'goal'):
                        goal_pos = controller.goal
                        draw_point(env, goal_pos, color=(1.0, 0.0, 0.0, 1.0), size=0.03)
                    
                    # 2. Draw obstacle modeling
                    if hasattr(controller, 'obstacles_pos') and len(controller.obstacles_pos) > 0:
                        if hasattr(controller, 'obstacle_radius') and hasattr(controller, 'obstacle_half_length'):
                            for obs_pos in controller.obstacles_pos:
                                # Draw capsule with safety margin (cyan)
                                bottom = obs_pos.copy()
                                bottom[2] -= controller.obstacle_half_length
                                top = obs_pos.copy()
                                top[2] += controller.obstacle_half_length
                                
                                draw_capsule(
                                    env,
                                    start=bottom,
                                    end=top,
                                    radius=controller.obstacle_radius + controller.obstacle_safety_margin,
                                    color=(0.0, 1.0, 1.0, 0.4),  # Cyan with transparency
                                    segments=6
                                )
                                
                                # Draw pole line for obstacle (red)
                                draw_line(
                                    env,
                                    np.array([bottom, top]),
                                    rgba=np.array([1.0, 0.0, 0.0, 1.0]),  # Red for obstacle axis
                                    min_size=2.0,
                                    max_size=2.0
                                )
                    
                    # 3. Draw ALL gate frames modeling (not just current target)
                    if hasattr(controller, 'gates_pos') and hasattr(controller, 'gates_quat'):
                        if hasattr(controller, 'gate_opening'):
                            half_opening = controller.gate_opening / 2.0  # 0.2025m
                            # Use gate_frame_center_offset if available (0.2875m)
                            if hasattr(controller, 'gate_frame_center_offset'):
                                frame_center_offset = controller.gate_frame_center_offset  # 0.2875m
                            else:
                                frame_center_offset = 0.2875  # Default
                            # Use gate_frame_width for capsule radius (17cm / 2 = 8.5cm)
                            if hasattr(controller, 'gate_frame_width'):
                                frame_radius = controller.gate_frame_width / 2.0  # 0.085m
                            else:
                                frame_radius = 0.085  # Default
                            frame_safety_margin = 0.05  # 5cm safety margin matching controller
                            frame_radius_with_margin = frame_radius + frame_safety_margin
                            
                            # Get live gate poses if available
                            if "gates_pos" in obs and "gates_quat" in obs:
                                gates_pos_live = obs["gates_pos"]
                                gates_quat_live = obs["gates_quat"]
                            else:
                                gates_pos_live = controller.gates_pos
                                gates_quat_live = controller.gates_quat
                            
                            # Iterate over all gates
                            for gate_idx in range(len(controller.gates_pos)):
                                gate_center = np.array(gates_pos_live[gate_idx], dtype=float)
                                gate_quat = np.array(gates_quat_live[gate_idx], dtype=float)
                                rot = R.from_quat(gate_quat)
                                R_matrix = rot.as_matrix()
                                
                                # Use different colors for current target vs other gates
                                is_target = hasattr(controller, 'target_gate_idx') and gate_idx == controller.target_gate_idx
                                if is_target:
                                    vert_color = (0.0, 1.0, 1.0, 0.4)  # Cyan for target gate
                                    horiz_color = (1.0, 0.5, 0.0, 0.5)  # Orange for target gate
                                    line_color = np.array([1.0, 0.0, 1.0, 1.0])  # Magenta for target
                                else:
                                    vert_color = (0.5, 0.5, 1.0, 0.25)  # Light blue for other gates
                                    horiz_color = (1.0, 0.7, 0.3, 0.25)  # Light orange for other gates
                                    line_color = np.array([0.6, 0.3, 0.6, 0.7])  # Dim magenta for others
                                
                                # Draw gate opening box (only for target)
                                if is_target:
                                    half_sizes = np.array([0.01, half_opening, half_opening])
                                    draw_box(
                                        env,
                                        center=gate_center,
                                        quat=gate_quat,
                                        half_sizes=half_sizes,
                                        color=(0.0, 1.0, 0.0, 0.3)  # Green for opening
                                    )
                                
                                # Left vertical frame (pole) - at y = -0.35m in gate frame
                                left_center_local = np.array([0, -frame_center_offset, 0])
                                left_center_world = gate_center + R_matrix @ left_center_local
                                left_top = left_center_world + R_matrix @ np.array([0, 0, frame_center_offset])
                                left_bottom = left_center_world - R_matrix @ np.array([0, 0, frame_center_offset])
                                
                                draw_capsule(
                                    env,
                                    start=left_top,
                                    end=left_bottom,
                                    radius=frame_radius_with_margin,
                                    color=vert_color,
                                    segments=8#4
                                )
                                draw_line(
                                    env,
                                    np.array([left_bottom, left_top]),
                                    rgba=line_color,
                                    min_size=2.0,
                                    max_size=2.0
                                )
                                
                                # Right vertical frame (pole) - at y = +0.35m in gate frame
                                right_center_local = np.array([0, frame_center_offset, 0])
                                right_center_world = gate_center + R_matrix @ right_center_local
                                right_top = right_center_world + R_matrix @ np.array([0, 0, frame_center_offset])
                                right_bottom = right_center_world - R_matrix @ np.array([0, 0, frame_center_offset])
                                
                                draw_capsule(
                                    env,
                                    start=right_top,
                                    end=right_bottom,
                                    radius=frame_radius_with_margin,
                                    color=vert_color,
                                    segments=8#4
                                )
                                draw_line(
                                    env,
                                    np.array([right_bottom, right_top]),
                                    rgba=line_color,
                                    min_size=2.0,
                                    max_size=2.0
                                )
                                
                                # Top horizontal frame (capsule) - at z = +0.35m in gate frame
                                top_center_local = np.array([0, 0, frame_center_offset])
                                top_center_world = gate_center + R_matrix @ top_center_local
                                top_left = top_center_world - R_matrix @ np.array([0, frame_center_offset, 0])
                                top_right = top_center_world + R_matrix @ np.array([0, frame_center_offset, 0])
                                
                                draw_capsule(
                                    env,
                                    start=top_left,
                                    end=top_right,
                                    radius=frame_radius_with_margin,
                                    color=horiz_color,
                                    segments=8#4
                                )
                                
                                # Bottom horizontal frame (capsule) - at z = -0.35m in gate frame
                                bottom_center_local = np.array([0, 0, -frame_center_offset])
                                bottom_center_world = gate_center + R_matrix @ bottom_center_local
                                bottom_left = bottom_center_world - R_matrix @ np.array([0, frame_center_offset, 0])
                                bottom_right = bottom_center_world + R_matrix @ np.array([0, frame_center_offset, 0])
                                
                                draw_capsule(
                                    env,
                                    start=bottom_left,
                                    end=bottom_right,
                                    radius=frame_radius_with_margin,
                                    color=horiz_color,
                                    segments=8#4
                                )
                                
                                # Draw gate normal vector (only for target)
                                if is_target:
                                    normal_end = gate_center + R_matrix[:, 0] * 0.3
                                    draw_line(
                                        env,
                                        np.array([gate_center, normal_end]),
                                        rgba=np.array([0.0, 0.0, 1.0, 1.0]),
                                        min_size=1.0,
                                        max_size=1.0
                                    )
                    
                except Exception as e:
                    # Silently fail - visualization shouldn't break the simulation
                    pass
            
            # --- Draw optimal MPPI trajectory ---
            if config.sim.visualize:
                if hasattr(controller, 'get_optimal_trajectory'):
                    # Get the optimal trajectory from the controller
                    # This is the TRUE optimal trajectory (weighted combination of all samples)
                    pos_traj = controller.get_optimal_trajectory(obs)
                    
                    # Draw the simulated optimal trajectory
                    if len(pos_traj) > 1:
                        # Filter out consecutive duplicate points to avoid numerical issues
                        filtered_traj = [pos_traj[0]]
                        for i in range(1, len(pos_traj)):
                            # Only add point if it's sufficiently different from the last
                            if np.linalg.norm(pos_traj[i] - filtered_traj[-1]) > 0.001:  # 1mm threshold
                                filtered_traj.append(pos_traj[i])
                        
                        if len(filtered_traj) > 1:
                            try:
                                draw_line(
                                    env,
                                    np.array(filtered_traj),
                                    rgba=np.array([0.0, 1.0, 1.0, 0.8]),  # Semi-transparent cyan
                                    min_size=2.0,
                                    max_size=4.0,
                                )
                            except (np.linalg.LinAlgError, RuntimeWarning):
                                # Skip drawing if numerical issues occur
                                pass

                # Draw planned trajectory from controller (if provided).
                # Controllers may expose a `get_planned_trajectory()` method returning an (N,3) array
                # or a `_last_planned_pos` attribute.
                try:
                    traj_points = None
                    if hasattr(controller, "get_planned_trajectory"):
                        traj_points = controller.get_planned_trajectory()
                    elif hasattr(controller, "_last_planned_pos"):
                        traj_points = getattr(controller, "_last_planned_pos")
                    if traj_points is not None:
                        # ensure an ndarray and correct shape
                        import numpy as _np

                        traj_points = _np.asarray(traj_points)
                        if traj_points.ndim == 2 and traj_points.shape[1] == 3:
                            draw_line(env, traj_points)
                except Exception:
                    # drawing must never break the sim loop
                    pass
                
                # --- Draw warm-start spline if available ---
                try:
                    warm = getattr(controller, "_last_warmstart_pos", None)
                    if warm is not None:
                        warm = np.asarray(warm)
                        if warm.ndim == 2 and warm.shape[1] == 3:
                            draw_line(
                                env,
                                warm,
                                rgba=np.array([0.0, 1.0, 0.0, 1.0]),  # green
                                min_size=2.0,
                                max_size=4.0,
                            )
                except Exception:
                    pass

            # Convert to a buffer that meets XLA's alginment restrictions to prevent warnings. See
            # https://github.com/jax-ml/jax/discussions/6055
            # traj_points = controller._des_pos_spline(np.linspace(0, controller._t_total, 200))
            # draw_line(env, traj_points)
            # Tracking issue:
            # https://github.com/jax-ml/jax/issues/29810
            action = np.asarray(jp.asarray(action), copy=True)

            obs, reward, terminated, truncated, info = env.step(action)
            # Update the controller internal state and models.
            controller_finished = controller.step_callback(
                action, obs, reward, terminated, truncated, info
            )
            # Add up reward, collisions
            if terminated or truncated or controller_finished:
                break
            if config.sim.render:  # Render the sim if selected.
                if ((i * fps) % config.env.freq) < fps:
                    env.render()
            i += 1

        controller.episode_callback()  # Update the controller internal state and models.
        log_episode_stats(obs, info, config, curr_time)
        controller.episode_reset()
        ep_times.append(curr_time if obs["target_gate"] == -1 else None)

    # Close the environment
    env.close()
    return ep_times


def log_episode_stats(obs: dict, info: dict, config: ConfigDict, curr_time: float):
    """Log the statistics of a single episode."""
    gates_passed = obs["target_gate"]
    if gates_passed == -1:  # The drone has passed the final gate
        gates_passed = len(config.env.track.gates)
    finished = gates_passed == len(config.env.track.gates)
    logger.info(
        f"Flight time (s): {curr_time}\nFinished: {finished}\nGates passed: {gates_passed}\n"
    )


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)