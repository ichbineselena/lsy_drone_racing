import numpy as np

from lsy_drone_racing.control.trajectory_builders import SplineBuilder, MPPIBuilder


def make_obs():
    obs = {}
    obs["pos"] = np.zeros(3)
    obs["quat"] = np.array([0.0, 0.0, 0.0, 1.0])
    obs["vel"] = np.zeros(3)
    obs["ang_vel"] = np.zeros(3)
    return obs


def test_spline_builder_shapes():
    waypoints = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    freq = 10.0
    sb = SplineBuilder(waypoints, t_total=1.0, freq=freq)
    obs = make_obs()
    # initial_state format expected by builder.reset
    initial_state = np.concatenate((obs["pos"], np.zeros(3), obs["vel"], np.zeros(3)))
    sb.reset(initial_state)
    ref = sb.get_horizon(0.0, N=5, dt=1.0 / freq)
    assert ref["pos"].shape == (5, 3)
    assert ref["vel"].shape == (5, 3)
    assert ref["yaw"].shape == (5,)


def test_mppi_builder_shapes():
    goal = np.array([1.0, 0.0, 0.0])
    mb = MPPIBuilder(goal, K=20, lambda_=1.0, sigma_u=0.1)
    obs = make_obs()
    initial_state = np.concatenate((obs["pos"], np.zeros(3), obs["vel"], np.zeros(3)))
    mb.reset(initial_state)
    ref = mb.get_horizon(0.0, N=8, dt=0.02)
    assert ref["pos"].shape == (8, 3)
    assert ref["vel"].shape == (8, 3)
    assert ref["yaw"].shape == (8,)
