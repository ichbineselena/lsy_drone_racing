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
        # Log gate & obstacle poses for this run
        _log_run_objects(obs)
        controller: Controller = controller_cls(obs, info, config)
        i = 0
        fps = 60

        while True:
            curr_time = i / config.env.freq

            action = controller.compute_control(obs, info)
            # Convert to a buffer that meets XLA's alginment restrictions to prevent warnings. See
            # https://github.com/jax-ml/jax/discussions/6055
            traj_points = controller._des_pos_spline(np.linspace(0, controller._t_total, 200))
            draw_line(env, traj_points)
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
                # Print collision details if available (env.info includes last_contact)
                _log_episode_end_details(obs, info, config, terminated, truncated)
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


def _np(a):
    try:
        return np.asarray(a)
    except Exception:
        return a


def _squeeze(a):
    return np.asarray(a).squeeze()


def _log_run_objects(obs: dict):
    """Log the gates and obstacles positions at the start of each run."""
    try:
        gates_pos = _squeeze(obs.get("gates_pos"))
        obstacles_pos = _squeeze(obs.get("obstacles_pos"))
        # Expect shapes: (n_gates,3) and (n_obstacles,3)
        if gates_pos.ndim == 1:
            gates_pos = gates_pos[None, :]
        if obstacles_pos.ndim == 1:
            obstacles_pos = obstacles_pos[None, :]
        gp_str = ", ".join([f"g{i}={gates_pos[i].tolist()}" for i in range(len(gates_pos))])
        op_str = ", ".join(
            [f"o{i}={obstacles_pos[i].tolist()}" for i in range(len(obstacles_pos))]
        )
        logger.info(f"Run objects -> Gates: [{gp_str}] | Obstacles: [{op_str}]")
    except Exception as e:
        logger.warning(f"Failed to log run objects: {e}")


def _log_episode_end_details(obs: dict, info: dict, config: ConfigDict, terminated: bool, truncated: bool):
    """On episode end, print likely collision details if not finished.

    Heuristics:
    - If target_gate == -1: finished; nothing to report.
    - Else if terminated: likely contact or bounds; report drone pos and nearest gate/obstacle.
    - Else if truncated: time limit; report last pos and nearest objects for context.
    """
    try:
        tgt = int(_squeeze(obs.get("target_gate", -2)))
        pos = _squeeze(obs.get("pos"))
        gates_pos = _squeeze(obs.get("gates_pos", np.zeros((0, 3))))
        obstacles_pos = _squeeze(obs.get("obstacles_pos", np.zeros((0, 3))))
        if pos.ndim != 1 or pos.shape[-1] != 3:
            return
        # Normalize shapes
        if gates_pos.size == 0:
            gates_pos = np.zeros((0, 3))
        if obstacles_pos.size == 0:
            obstacles_pos = np.zeros((0, 3))
        if gates_pos.ndim == 1:
            gates_pos = gates_pos[None, :]
        if obstacles_pos.ndim == 1:
            obstacles_pos = obstacles_pos[None, :]

        # Finished -> skip unless a contact was recorded
        if tgt == -1 and not info.get("last_contact"):
            return

        # Distances
        def _min_dist(mat):
            if mat.shape[0] == 0:
                return (np.inf, -1)
            d = np.linalg.norm(mat - pos[None, :], axis=1)
            idx = int(np.argmin(d))
            return (float(d[idx]), idx)

        d_gate, gi = _min_dist(gates_pos)
        d_obs, oi = _min_dist(obstacles_pos)

        # Bounds check vs env defaults (RaceCoreEnv uses fixed bounds)
        bounds_low = np.array([-3.0, -3.0, -1e-3])
        bounds_high = np.array([3.0, 3.0, 2.5])
        out_of_bounds = np.any(pos < bounds_low) or np.any(pos > bounds_high)

        reason = "terminated" if terminated else ("time-limit" if truncated else "ended")
        msg = [f"End ({reason}) @ drone_pos={pos.tolist()}"]

        # Prefer exact contact from env if present
        lc = info.get("last_contact") or {}
        if lc:
            ctype = lc.get("type")
            cpos = lc.get("pos")
            cidx = lc.get("index")
            if ctype == "gate" and cidx is not None:
                msg.append(f"contact gate g{cidx} @ {cpos}")
                logger.info(" | ".join(msg))
                return
            if ctype == "obstacle" and cidx is not None:
                msg.append(f"contact obstacle o{cidx} @ {cpos}")
                logger.info(" | ".join(msg))
                return
            if ctype in ("world", "bounds", "contact"):
                msg.append(f"contact {ctype} @ {cpos}")
                logger.info(" | ".join(msg))
                return

        if out_of_bounds:
            msg.append("likely out-of-bounds")

        if d_obs < d_gate - 0.05:
            near = f"nearest obstacle o{oi} @ {obstacles_pos[oi].tolist()} (d={d_obs:.3f}m)"
            msg.append(near)
        elif gi >= 0:
            near = f"nearest gate g{gi} @ {gates_pos[gi].tolist()} (d={d_gate:.3f}m)"
            msg.append(near)

        logger.info(" | ".join(msg))
    except Exception as e:
        logger.warning(f"Failed to log end details: {e}")


if __name__ == "__main__":
    logging.basicConfig()
    logging.getLogger("lsy_drone_racing").setLevel(logging.INFO)
    logger.setLevel(logging.INFO)
    fire.Fire(simulate, serialize=lambda _: None)
