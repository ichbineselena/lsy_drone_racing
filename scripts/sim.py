"""Simulate the competition as in the IROS 2022 Safe Robot Learning competition.

Run as:

    $ python scripts/sim.py --config level0.toml

Look for instructions in `README.md` and in the official documentation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

import fire
import gymnasium
import jax.numpy as jp
import numpy as np
from gymnasium.wrappers.jax_to_numpy import JaxToNumpy

from lsy_drone_racing.utils import load_config, load_controller, draw_line, draw_point, draw_capsule, draw_box

if TYPE_CHECKING:
    from ml_collections import ConfigDict

    from lsy_drone_racing.control.controller import Controller
    from lsy_drone_racing.envs.drone_race import DroneRaceEnv


logger = logging.getLogger(__name__)


def simulate(
    config: str = "level3.toml",
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
                                    segments=12
                                )
                                
                                # Draw actual obstacle size (gray)
                                draw_capsule(
                                    env,
                                    start=bottom,
                                    end=top,
                                    radius=controller.obstacle_radius,
                                    color=(0.5, 0.5, 0.5, 0.7),  # Gray for actual size
                                    segments=12
                                )
                    
                    # 3. Draw current gate frame modeling
                    if hasattr(controller, 'current_gate_center') and hasattr(controller, 'current_gate_quat'):
                        if hasattr(controller, 'gate_opening') and hasattr(controller, 'gate_frame_thickness'):
                            # Draw gate opening box
                            half_opening = controller.gate_opening / 2.0
                            half_sizes = np.array([0.01, half_opening, half_opening])  # Thin in x direction
                            draw_box(
                                env,
                                center=controller.current_gate_center,
                                quat=controller.current_gate_quat,
                                half_sizes=half_sizes,
                                color=(0.0, 1.0, 0.0, 0.3)  # Green for opening
                            )
                            
                            # Draw gate frames
                            rot = R.from_quat(controller.current_gate_quat)
                            R_matrix = rot.as_matrix()
                            total_half = half_opening + controller.gate_frame_thickness / 2.0
                            
                            # Define frame positions
                            # Left vertical frame
                            left_center_local = np.array([0, -total_half, 0])
                            left_center_world = controller.current_gate_center + R_matrix @ left_center_local
                            left_top = left_center_world + R_matrix @ np.array([0, 0, half_opening])
                            left_bottom = left_center_world - R_matrix @ np.array([0, 0, half_opening])
                            
                            draw_capsule(
                                env,
                                start=left_top,
                                end=left_bottom,
                                radius=controller.gate_frame_thickness / 2.0,
                                color=(1.0, 1.0, 0.0, 0.6),  # Yellow for frame
                                segments=8
                            )
                            
                            # Right vertical frame
                            right_center_local = np.array([0, total_half, 0])
                            right_center_world = controller.current_gate_center + R_matrix @ right_center_local
                            right_top = right_center_world + R_matrix @ np.array([0, 0, half_opening])
                            right_bottom = right_center_world - R_matrix @ np.array([0, 0, half_opening])
                            
                            draw_capsule(
                                env,
                                start=right_top,
                                end=right_bottom,
                                radius=controller.gate_frame_thickness / 2.0,
                                color=(1.0, 1.0, 0.0, 0.6),
                                segments=8
                            )
                            
                            # Top horizontal frame
                            top_center_local = np.array([0, 0, total_half])
                            top_center_world = controller.current_gate_center + R_matrix @ top_center_local
                            top_left = top_center_world - R_matrix @ np.array([0, half_opening, 0])
                            top_right = top_center_world + R_matrix @ np.array([0, half_opening, 0])
                            
                            draw_capsule(
                                env,
                                start=top_left,
                                end=top_right,
                                radius=controller.gate_frame_thickness / 2.0,
                                color=(1.0, 1.0, 0.0, 0.6),
                                segments=8
                            )
                            
                            # Bottom horizontal frame
                            bottom_center_local = np.array([0, 0, -total_half])
                            bottom_center_world = controller.current_gate_center + R_matrix @ bottom_center_local
                            bottom_left = bottom_center_world - R_matrix @ np.array([0, half_opening, 0])
                            bottom_right = bottom_center_world + R_matrix @ np.array([0, half_opening, 0])
                            
                            draw_capsule(
                                env,
                                start=bottom_left,
                                end=bottom_right,
                                radius=controller.gate_frame_thickness / 2.0,
                                color=(1.0, 1.0, 0.0, 0.6),
                                segments=8
                            )
                            
                            # Draw gate normal vector
                            normal_end = controller.current_gate_center + R_matrix[:, 0] * 0.3
                            draw_line(
                                env,
                                np.array([controller.current_gate_center, normal_end]),
                                rgba=np.array([0.0, 0.0, 1.0, 1.0]),
                                min_size=1.0,
                                max_size=1.0
                            )
                    
                except Exception as e:
                    # Silently fail - visualization shouldn't break the simulation
                    pass
            
            # --- Draw optimal MPPI trajectory ---
            if config.sim.visualize:
                try:
                    if hasattr(controller, 'mppi'):
                        # Get the optimal control sequence and simulate forward
                        mppi = controller.mppi
                        
                        # The optimal trajectory is stored in mppi.U (control sequence)
                        # We need to simulate the dynamics forward to get the state trajectory
                        if hasattr(mppi, 'states') and mppi.states is not None:
                            # Get the optimal trajectory from the last MPPI iteration
                            # states shape: [1, K, T, nx] where nx=12
                            states = mppi.states
                            if states is not None:
                                # Get the optimal trajectory (first sample is the nominal)
                                states_np = states[0, 0].cpu().numpy()  # Shape: [T, nx]
                                
                                # Extract position coordinates (first 3 columns)
                                pos_traj = states_np[:, 0:3]
                                
                                # Draw the optimal trajectory as a cyan line
                                if len(pos_traj) > 1:
                                    draw_line(
                                        env,
                                        pos_traj,
                                        rgba=np.array([0.0, 1.0, 1.0, 1.0]),  # Cyan color
                                        min_size=2.0,
                                        max_size=4.0,
                                    )
                        else:
                            # Fallback: Simulate forward using current state and optimal controls
                            import torch
                            current_state = controller.drone_model.obs_to_state(obs)
                            
                            # Get optimal control sequence
                            optimal_controls = mppi.U  # Shape: [T, 4]
                            
                            # Simulate trajectory forward
                            state_tensor = current_state.clone()
                            trajectory = [state_tensor.clone()]
                            
                            for t in range(min(controller.mppi_horizon, 25)):  # Limit to 25 steps for visibility
                                control_t = optimal_controls[t].unsqueeze(0)
                                state_tensor = controller.drone_model.dynamics(
                                    state_tensor, 
                                    control_t, 
                                    controller.mppi_dt
                                )
                                trajectory.append(state_tensor.clone())
                            
                            # Convert to numpy array
                            trajectory_np = torch.cat(trajectory, dim=0).cpu().numpy()
                            pos_traj = trajectory_np[:, 0:3]
                            
                            # Draw the simulated optimal trajectory
                            if len(pos_traj) > 1:
                                draw_line(
                                    env,
                                    pos_traj,
                                    rgba=np.array([0.0, 1.0, 1.0, 0.8]),  # Semi-transparent cyan
                                    min_size=2.0,
                                    max_size=4.0,
                                )
                except Exception as e:
                    # Silently fail - visualization shouldn't break the simulation
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