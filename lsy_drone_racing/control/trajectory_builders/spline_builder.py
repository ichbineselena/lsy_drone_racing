from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from scipy.interpolate import CubicSpline

from .base import TrajectoryBuilder

if TYPE_CHECKING:
    from numpy.typing import NDArray


class SplineBuilder:
    """Wrap the existing cubic-spline waypoint logic into a reusable builder.

    This mirrors the behavior previously hard-coded into AttitudeMPC.
    """

    def __init__(self, waypoints: NDArray, t_total: float, freq: float):
        self.waypoints = np.asarray(waypoints)
        self.t_total = float(t_total)
        self.freq = float(freq)

        t = np.linspace(0, self.t_total, len(self.waypoints))
        self._spline = CubicSpline(t, self.waypoints)
        self._vel_spline = self._spline.derivative()

        # sample the full trajectory used to compute total steps
        self._ts_full = np.linspace(0, self.t_total, int(self.freq * self.t_total))
        self._pos_full = self._spline(self._ts_full)
        self._vel_full = self._vel_spline(self._ts_full)
        self._yaw_full = np.zeros((len(self._ts_full),))

        self._t0 = 0.0

    def reset(self, initial_state: np.ndarray, t0: float = 0.0) -> None:
        self._t0 = float(t0)

    def get_horizon(self, t_now: float, N: int, dt: float) -> dict:
        # Build time stamps and clip to the spline bounds
        ts = t_now + np.arange(N) * dt
        ts = np.clip(ts, 0.0, self.t_total)
        pos = self._spline(ts)
        vel = self._vel_spline(ts)
        yaw = np.zeros((N,))
        return {"pos": pos, "vel": vel, "yaw": yaw}

    def get_total_steps(self) -> int:
        return len(self._ts_full)
