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

from lsy_drone_racing.utils import load_config, load_controller, draw_line

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
                print("Could not draw warm-start trajectory.")
                pass


            # # --- Draw 20 lowest-cost MPPI trajectories as white lines ---
            # try:
            #     if hasattr(controller, "mppi"):
            #         mppi = controller.mppi
            #         states = getattr(mppi, "states", None)
            #         costs  = getattr(mppi, "cost_total", None)

            #         if states is not None and costs is not None:
            #             import numpy as _np
            #             costs = _np.asarray(costs)

            #             # states shape: [1, K, T, nx]
            #             rollouts = states[0].cpu().numpy()   # shape [K, T, nx]

            #             K, T, nx = rollouts.shape
            #             T_display = T  # number of steps to draw

            #             # select best 20 trajectories
            #             k = min(20, K)
            #             best_idx = _np.argsort(costs)[:k]

            #             for idx in best_idx:
            #                 traj = rollouts[idx, :, :3]  # xyz only

            #                 # draw each segment of this sampled trajectory
            #                 for a, b in zip(traj[:-1], traj[1:]):
            #                     env._env.draw_line(
            #                         start=a,
            #                         end=b,
            #                         color=(1.0, 1.0, 1.0, 0.15),  # slightly transparent white
            #                         width=1.0
            #                     )

            # except Exception as e:
            #     print("Could not draw MPPI sample trajectories:", e)




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
