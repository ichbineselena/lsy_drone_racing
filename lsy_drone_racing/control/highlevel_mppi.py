# highlevel_mppi.py
import torch
import numpy as np
from typing import Sequence, Tuple, Dict, Any

# Device helper
def get_device(prefer_cuda: bool = True):
    if prefer_cuda and torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class HighLevelMPPI:
    """
    Vectorized GPU/CPU MPPI for a reduced kinematic drone model.
    State: x = [px, py, pz, vx, vy, vz, yaw]  (7)
    Control: u = [ax, ay, az, yaw_rate]       (4)
    """

    def __init__(self,
                 dt: float = 0.05,
                 H: int = 40,
                 K: int = 2048,
                 sigma: Sequence[float] = (0.6, 0.6, 0.4, 0.3),
                 lam: float = 1.0,
                 device: torch.device = None,
                 params: Dict[str, float] = None):
        """
        Args:
            dt: timestep for internal propagation
            H: planning horizon (timesteps)
            K: number of rollouts (samples)
            sigma: std dev for Gaussian sampling for each control dimension
            lam: temperature lambda for weight computation
            device: torch device
            params: cost weights and geometry defaults as dict
        """
        self.dt = dt
        self.H = H
        self.K = K
        self.dim_x = 7
        self.dim_u = 4
        self.device = device if device is not None else get_device()
        self.lam = lam
        self.sigma = torch.tensor(sigma, dtype=torch.float32, device=self.device).view(1,1,4)

        # nominal control (warm-start) shape (H, dim_u)
        self.u_bar = torch.zeros((H, self.dim_u), dtype=torch.float32, device=self.device)

        # default cost params (can be overridden by params arg)
        self.params = {
            "w_gate": 200.0,
            "w_gate_border": 600.0,
            "w_obs": 1000.0,
            "w_u": 1.0,
            "w_du": 5.0,
            "w_terminal": 50.0,
            "gate_thickness": 0.2,
            "gate_radius_default": 1.0,
            "obs_radius_default": 1.0,
            "obs_safety_margin": 0.1,  # extra margin for obstacles
            "v_pref": 5.0,  # preferred forward speed (m/s)
            "w_vel": 1.0,
        }
        if params:
            self.params.update(params)

        # small epsilon
        self.eps = 1e-12

    # ---- dynamics: batch vectorized kinematic model ----
    def _batch_rollout(self, x0: torch.Tensor, U: torch.Tensor) -> torch.Tensor:
        """
        Vectorized rollout.
        Inputs:
            x0: (dim_x,) or (1,dim_x) tensor (float32) current state
            U: (K, H, dim_u) control sequences
        Returns:
            X: (K, H+1, dim_x) trajectories
        """
        K, H, dim_u = U.shape
        # prepare
        x = x0.view(1, self.dim_x).repeat(K, 1).to(self.device)  # (K, dim_x)
        X = torch.zeros((K, H+1, self.dim_x), device=self.device, dtype=torch.float32)
        X[:, 0, :] = x

        # vectorized propagation (no autograd needed)
        with torch.no_grad():
            for t in range(H):
                u = U[:, t, :]  # (K, dim_u) [ax,ay,az,yaw_rate]
                # state split
                pos = x[:, 0:3]       # (K,3)
                vel = x[:, 3:6]       # (K,3)
                yaw = x[:, 6:7]       # (K,1)
                a = u[:, 0:3]
                yaw_rate = u[:, 3:4]
                # integrate
                pos = pos + vel * self.dt + 0.5 * a * (self.dt**2)
                vel = vel + a * self.dt
                yaw = yaw + yaw_rate * self.dt
                x = torch.cat([pos, vel, yaw], dim=1)
                X[:, t+1, :] = x
        return X

    # ---- cost evaluation ----
    def _compute_costs(self,
                       X: torch.Tensor,
                       U: torch.Tensor,
                       gates: Sequence[Tuple[float, float, float]],
                       gate_times: Sequence[int],
                       gate_radii: Sequence[float],
                       obstacles: Sequence[Tuple[float, float, float]],
                       obstacle_radii: Sequence[float]) -> torch.Tensor:
        """
        Compute scalar cost for each trajectory (vectorized).
        X: (K, H+1, dim_x)
        U: (K, H, dim_u)
        returns S: (K,) costs
        """
        K, H1, _ = X.shape
        H = H1 - 1
        device = self.device
        params = self.params

        cost = torch.zeros((K,), device=device)

        # Gate costs at gate_times: encourage being within radius and penalize border crossing
        for i, g in enumerate(gates):
            t_idx = int(gate_times[i])
            if t_idx > H: t_idx = H
            g_c = torch.tensor(g, device=device, dtype=torch.float32).view(1,3)
            pos_t = X[:, t_idx, 0:3]   # (K,3)
            dist = torch.norm(pos_t - g_c, dim=1)  # (K,)
            r_g = gate_radii[i] if gate_radii is not None else params["gate_radius_default"]
            # outside penalty: linear outside radius
            cost = cost + params["w_gate"] * torch.relu(dist - r_g)
            # border penalty: penalize being too close to ring edge (|dist - r| > thickness/2)
            border_pen = torch.relu(torch.abs(dist - r_g) - (params["gate_thickness"]/2.0))
            cost = cost + params["w_gate_border"] * border_pen

        # obstacle penalty over whole trajectory
        if len(obstacles) > 0:
            # make tensors
            obs_centers = torch.tensor(obstacles, device=device, dtype=torch.float32).view(1,1,len(obstacles),3)  # (1,1,M,3)
            obs_r = torch.tensor(obstacle_radii, device=device, dtype=torch.float32).view(1,1,len(obstacles))  # (1,1,M)

            pos = X[:, :, 0:3].unsqueeze(2)  # (K,H+1,1,3)
            d = torch.norm(pos - obs_centers, dim=3)  # (K,H+1,M)
            # safe distance: obstacle radius + margin
            safe = obs_r + params["obs_safety_margin"]
            penetration = torch.relu(safe.unsqueeze(1) - d)  # (K,H+1,M)
            # square penetration to push strong penalty for penetration
            cost = cost + params["w_obs"] * (penetration.pow(2).sum(dim=(1,2)))  # sum over time and obstacles

        # control magnitude and smoothness
        cost = cost + params["w_u"] * (U.pow(2).sum(dim=(1,2)))
        du = U[:,1:,:] - U[:,:-1,:]  # (K, H-1, dim_u)
        if du.shape[1] > 0:
            cost = cost + params["w_du"] * (du.pow(2).sum(dim=(1,2)))

        # optional velocity penalty toward a preferred speed (reduce stalling)
        # apply at all times to forward speed magnitude
        v = X[:, :, 3:6]  # (K,H+1,3)
        speed = torch.norm(v, dim=2)  # (K,H+1)
        speed_err = (speed - params["v_pref"]).pow(2).sum(dim=1)
        cost = cost + params["w_vel"] * speed_err

        # terminal penalty (distance to last gate center)
        if len(gates) > 0:
            final_gate_center = torch.tensor(gates[-1], device=device, dtype=torch.float32).view(1,3)
            pos_final = X[:, -1, 0:3]
            cost = cost + params["w_terminal"] * torch.norm(pos_final - final_gate_center, dim=1)

        return cost

    # ---- MPPI plan call ----
    def plan(self,
             x0: np.ndarray,
             gates: Sequence[Tuple[float, float, float]],
             gate_times: Sequence[int],
             gate_radii: Sequence[float],
             obstacles: Sequence[Tuple[float, float, float]],
             obstacle_radii: Sequence[float],
             timeout_ms: float = None) -> Tuple[np.ndarray, float, Dict[str, Any]]:
        """
        Run MPPI and return first control, yaw_rate, and a planned short trajectory.

        Args:
            x0: numpy array (dim_x,) current state
            gates: list of gate centers
            gate_times: list of desired crossing timesteps (ints)
            gate_radii: list of radii for gates
            obstacles: list of obstacle centers
            obstacle_radii: list of obstacle radii
            timeout_ms: not used but kept for API compatibility

        Returns:
            acc_cmd: numpy (3,) acceleration to apply
            yaw_rate_cmd: float yaw_rate
            diagnostics: dict with 'traj' (H+1, dim_x), 'S_mean', 'S_min', 'ESS', 'u_bar' (updated)
        """
        # cast tensors
        x0_t = torch.tensor(x0, dtype=torch.float32, device=self.device)
        K = self.K
        H = self.H

        # sample noises: (K, H, dim_u)
        noise = torch.randn((K, H, self.dim_u), device=self.device) * self.sigma  # broadcast sigma (1,1,4)
        # candidate controls (K,H,4)
        U = self.u_bar.unsqueeze(0) + noise  # (1->K applied)

        # rollout vectorized
        X = self._batch_rollout(x0_t, U)  # (K, H+1, dim_x)

        # compute costs
        S = self._compute_costs(X, U, gates, gate_times, gate_radii, obstacles, obstacle_radii)  # (K,)

        # weights: numerically stable
        S_min = torch.min(S)
        exp_arg = - (S - S_min) / (self.lam + self.eps)
        w = torch.exp(exp_arg)
        w = w / (torch.sum(w) + self.eps)

        # effective sample size
        ESS = 1.0 / torch.sum(w.pow(2)).item()

        # weighted update of u_bar using noise
        delta = (w.view(K, 1, 1) * noise).sum(dim=0)  # (H, dim_u)
        u_opt = self.u_bar + delta      # (H, dim_u)

        # pick first action to return
        first_u = u_opt[0, :].detach().cpu().numpy()
        # convert to outputs
        acc_cmd = first_u[0:3].copy()   # ax,ay,az
        yaw_rate_cmd = float(first_u[3])

        # warm start: shift left and pad
        u_next_bar = torch.cat([u_opt[1:], u_opt[-1:].unsqueeze(0)], dim=0).detach()
        self.u_bar = u_next_bar

        diag = {
            "S_mean": S.mean().item(),
            "S_min": S_min.item(),
            "ESS": ESS,
            "weights_max": float(torch.max(w).item()),
            "u_bar": self.u_bar.detach().cpu().numpy(),
            "traj_best": X[torch.argmin(S)].detach().cpu().numpy(),   # best trajectory (H+1, dim_x)
        }

        return acc_cmd, yaw_rate_cmd, diag