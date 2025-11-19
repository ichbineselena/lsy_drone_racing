from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from .base import TrajectoryBuilder

if TYPE_CHECKING:
    from numpy.typing import NDArray


class MPPIBuilder:
    """A small, self-contained MPPI-like builder for generating position/velocity references.

    This is a lightweight, easy-to-use implementation intended for testing and
    experimentation. It uses a simple double-integrator dynamics and a quadratic
    cost to drive to a goal position. It's not intended as a production-grade
    MPPI implementation but should be enough to integrate with the AttitudeMPC and
    validate the interface.
    """

    def __init__(self, goal: NDArray, K: int = 200, lambda_: float = 1.0, sigma_u: float = 0.5):
        self.goal = np.asarray(goal).reshape(3,)
        self.K = int(K)
        self.lambda_ = float(lambda_)
        self.sigma_u = float(sigma_u)
        print(f"[MPPIBuilder] Initialized with goal={self.goal}, K={self.K}, lambda={self.lambda_}, sigma_u={self.sigma_u}")

        # nominal control sequence (acceleration commands) will be created on reset
        self.U_nom = None
        self.x0 = None
        self.T = None

    def reset(self, initial_state: NDArray, t0: float = 0.0) -> None:
        # initial_state expected [pos(3), rpy(3), vel(3), drpy(3)]
        self.x0 = np.asarray(initial_state).copy()
        self.t0 = float(t0)
        self.U_nom = None

    def _double_integrator_rollout(self, x0: NDArray, U: NDArray, dt: float) -> NDArray:
        """Vectorized rollout for K samples. U shape: (K, T, 3) accelerations in world frame.

        Returns states shape (K, T+1, 6) with state = [pos(3), vel(3)].
        """
        K, T, _ = U.shape
        states = np.zeros((K, T + 1, 6))
        # initialize
        pos0 = x0[0:3]
        vel0 = x0[6:9]
        states[:, 0, 0:3] = pos0[None, :]
        states[:, 0, 3:6] = vel0[None, :]
        for t in range(T):
            a = U[:, t, :]
            # integrate
            states[:, t + 1, 3:6] = states[:, t, 3:6] + a * dt
            states[:, t + 1, 0:3] = states[:, t, 0:3] + states[:, t, 3:6] * dt + 0.5 * a * dt * dt
        return states

    def _cost(self, states: NDArray, U: NDArray) -> NDArray:
        # states: (K, T+1, 6) U: (K, T, 3)
        pos_seq = states[:, 1:, 0:3]
        # terminal cost (distance to goal)
        terminal = np.sum((pos_seq[:, -1, :] - self.goal[None, :]) ** 2, axis=1)
        # running cost (sum of squared distances)
        running = np.sum(np.sum((pos_seq - self.goal[None, None, :]) ** 2, axis=2), axis=1)
        control = np.sum(U ** 2, axis=(1, 2))
        return running + 10.0 * terminal + 0.1 * control

    def get_horizon(self, t_now: float, N: int, dt: float) -> dict:
        # Create nominal if needed
        T = int(N)
        self.T = T
        m = 3  # control dim (ax, ay, az)
        if self.U_nom is None:
            self.U_nom = np.zeros((T, m))

        # sample K perturbations
        dU = np.random.normal(scale=self.sigma_u, size=(self.K, T, m))
        U_samples = self.U_nom[None, :, :] + dU

        # vectorized rollouts
        states = self._double_integrator_rollout(self.x0, U_samples, dt)

        # costs and weights
        costs = self._cost(states, U_samples)
        Smin = costs.min()
        w = np.exp(-(costs - Smin) / max(self.lambda_, 1e-8))
        w = w / (np.sum(w) + 1e-12)

        # update nominal
        weighted_dU = (w[:, None, None] * dU).sum(axis=0)
        self.U_nom = self.U_nom + weighted_dU

        # produce single rollout from U_nom
        states_nom = self._double_integrator_rollout(self.x0, self.U_nom[None, :, :], dt)[0]
        pos = states_nom[1:, 0:3]
        vel = states_nom[1:, 3:6]
        yaw = np.zeros((T,))
        return {"pos": pos, "vel": vel, "yaw": yaw}

    def get_total_steps(self) -> int:
        # MPPI does not have a pre-sampled trajectory; return a large number to avoid limiting
        return 100000
