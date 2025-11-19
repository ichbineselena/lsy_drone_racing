from __future__ import annotations

from typing import Protocol

import numpy as np


class TrajectoryBuilder(Protocol):
    """Minimal TrajectoryBuilder protocol used by AttitudeMPC.

    Implementations should provide the methods below.
    """

    def reset(self, initial_state: np.ndarray, t0: float = 0.0) -> None:
        ...

    def get_horizon(self, t_now: float, N: int, dt: float) -> dict:
        """Return a dict with keys:
            - 'pos': np.ndarray shape (N, 3)
            - 'vel': np.ndarray shape (N, 3)
            - 'yaw': np.ndarray shape (N,)
        """

    def get_total_steps(self) -> int:
        ...
