"""Visualize the MPPI cost function for gate navigation."""
import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R


def plot_cost_slice_through_gate(gate_pos, gate_quat, grid_range=2.0, N=200):
    """Plot the cost map in a slice along the gate normal (approach direction).
    
    Args:
        gate_pos: Gate position [x, y, z]
        gate_quat: Gate quaternion [x, y, z, w]
        grid_range: Range in meters for the grid
        N: Resolution of the grid
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract gate normal from quaternion
    rot = R.from_quat(gate_quat)
    gate_normal_np = rot.as_matrix()[:, 0]  # x-axis is forward direction
    gate_normal = torch.tensor(gate_normal_np, device=device, dtype=torch.float32)
    gate_normal = gate_normal / torch.norm(gate_normal)
    
    gate_center = torch.tensor(gate_pos, device=device, dtype=torch.float32)
    goal_t = gate_center
    
    # Build orthonormal basis: n (forward), t1 (vertical), t2 (horizontal)
    n = gate_normal
    
    # t1 should be vertical (aligned with z-axis as much as possible)
    z_axis = torch.tensor([0., 0., 1.], device=device, dtype=torch.float32)
    t2 = torch.linalg.cross(n, z_axis)
    if torch.norm(t2) < 0.1:  # Gate normal is aligned with z-axis
        t2 = torch.tensor([1., 0., 0.], device=device, dtype=torch.float32)
    t2 = t2 / torch.norm(t2)
    
    t1 = torch.linalg.cross(t2, n)
    t1 = t1 / torch.norm(t1)
    
    # Create sampling grid: n (forward/backward), t1 (up/down)
    # This gives us a vertical slice through the gate along the approach direction
    tn = torch.linspace(-grid_range, grid_range, N, device=device)
    tt1 = torch.linspace(-grid_range, grid_range, N, device=device)
    
    TN, TT1 = torch.meshgrid(tn, tt1, indexing="ij")
    
    # Points in R^3: gate_center + n * TN + t1 * TT1
    points = gate_center + TN[..., None] * n + TT1[..., None] * t1
    
    # Build x state vectors: [pos(3), vel(3)] → vel = zero for visualization
    x = torch.zeros((N, N, 6), device=device)
    x[..., :3] = points
    x = x.reshape(N * N, 6)
    
    # Zero control for visualization (3D control: ax, ay, az)
    u = torch.zeros((N * N, 3), device=device)
    
    # Compute cost (same as in running_cost function)
    pos = x[..., :3]
    vel = x[..., 3:6]
    
    # Compute distance to gate center
    d = pos - gate_center
    
    # Axis projections
    proj_n = torch.sum(d * n, dim=-1)
    proj_t1 = torch.sum(d * t1, dim=-1)
    proj_t2 = torch.sum(d * t2, dim=-1)
    
    # Ellipse radii
    a = torch.tensor(1.2, device=device)  # long axis → through the gate
    b = torch.tensor(0.3, device=device)  # narrow axis → the gate's plane
    
    # Elliptical distance
    ellipse_dist = (proj_n / a)**2 + (proj_t1 / b)**2 + (proj_t2 / b)**2
    
    W_ellipse = 50.0  # tune weight
    c_ellipse = -W_ellipse / (ellipse_dist + 1e-6)
    
    # Position error
    W_pos = torch.tensor([10.0, 10.0, 50.0], device=device, dtype=torch.float32)
    c_pos = torch.sum(W_pos * (pos - goal_t) ** 2, dim=-1)
    
    # Velocity penalty
    W_vel = torch.tensor([0.1, 0.1, 0.1], device=device, dtype=torch.float32)
    c_vel = torch.sum(W_vel * vel ** 2, dim=-1)
    
    # Control effort penalty
    W_u = torch.tensor([0.01] * u.shape[-1], device=device, dtype=torch.float32)
    c_u = torch.sum(W_u * u ** 2, dim=-1)
    
    total_cost = c_pos + c_vel + c_u + c_ellipse
    cost = total_cost.reshape(N, N).detach().cpu().numpy()
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    contour = ax.contourf(TN.cpu(), TT1.cpu(), cost, levels=50, cmap='viridis')
    plt.colorbar(contour, label="Cost", ax=ax)
    ax.set_xlabel("Distance along gate normal (meters)\n← Behind gate | Through gate | Ahead of gate →")
    ax.set_ylabel("Vertical distance (meters)")
    ax.set_title(f"MPPI Cost Slice Through Gate\nGate at [{gate_pos[0]:.2f}, {gate_pos[1]:.2f}, {gate_pos[2]:.2f}]")
    ax.grid(True, alpha=0.3)
    
    # Mark the gate center
    ax.plot(0, 0, 'r*', markersize=20, label='Gate Center', markeredgecolor='white', markeredgewidth=1)
    
    # Draw gate boundaries (vertical line at n=0, spanning ±0.3m vertically)
    gate_half_height = 0.3
    ax.plot([0, 0], [-gate_half_height, gate_half_height], 'r-', linewidth=3, label='Gate Boundary')
    
    # Draw ellipse representing the reward zone
    theta = np.linspace(0, 2*np.pi, 100)
    ellipse_n = 1.2 * np.cos(theta)
    ellipse_t1 = 0.3 * np.sin(theta)
    ax.plot(ellipse_n, ellipse_t1, 'r--', linewidth=2, alpha=0.7, label='Ellipse Reward Zone')
    
    ax.legend()
    ax.set_aspect('equal')
    
    return fig


def plot_cost_map_for_gate(gate_pos, gate_quat, grid_range=2.0, N=200):
    """Plot the cost map in the gate plane.
    
    Args:
        gate_pos: Gate position [x, y, z]
        gate_quat: Gate quaternion [x, y, z, w]
        grid_range: Range in meters for the grid
        N: Resolution of the grid
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Extract gate normal from quaternion
    rot = R.from_quat(gate_quat)
    gate_normal_np = rot.as_matrix()[:, 0]  # x-axis is forward direction
    gate_normal = torch.tensor(gate_normal_np, device=device, dtype=torch.float32)
    gate_normal = gate_normal / torch.norm(gate_normal)
    
    gate_center = torch.tensor(gate_pos, device=device, dtype=torch.float32)
    goal_t = gate_center
    
    # Build orthonormal basis: n, t1, t2
    n = gate_normal
    tmp = torch.tensor([1., 0., 0.], device=device, dtype=torch.float32)
    if torch.abs(torch.dot(tmp, n)) > 0.8:
        tmp = torch.tensor([0., 1., 0.], device=device, dtype=torch.float32)
    
    t1 = torch.linalg.cross(n, tmp)
    t1 = t1 / torch.norm(t1)
    t2 = torch.linalg.cross(n, t1)
    t2 = t2 / torch.norm(t2)
    
    # Create sampling grid in the gate plane
    tt1 = torch.linspace(-grid_range, grid_range, N, device=device)
    tt2 = torch.linspace(-grid_range, grid_range, N, device=device)
    
    T1, T2 = torch.meshgrid(tt1, tt2, indexing="ij")
    
    # Points in R^3: gate_center + t1 * T1 + t2 * T2
    points = gate_center + T1[..., None] * t1 + T2[..., None] * t2
    
    # Build x state vectors: [pos(3), vel(3)] → vel = zero for visualization
    x = torch.zeros((N, N, 6), device=device)
    x[..., :3] = points
    x = x.reshape(N * N, 6)
    
    # Zero control for visualization (3D control: ax, ay, az)
    u = torch.zeros((N * N, 3), device=device)
    
    # Compute cost (same as in running_cost function)
    pos = x[..., :3]
    vel = x[..., 3:6]
    
    # Compute distance to gate center
    d = pos - gate_center
    
    # Axis projections
    proj_n = torch.sum(d * n, dim=-1)
    proj_t1 = torch.sum(d * t1, dim=-1)
    proj_t2 = torch.sum(d * t2, dim=-1)
    
    # Ellipse radii
    a = torch.tensor(1.2, device=device)  # long axis → through the gate
    b = torch.tensor(0.3, device=device)  # narrow axis → the gate's plane
    
    # Elliptical distance
    ellipse_dist = (proj_n / a)**2 + (proj_t1 / b)**2 + (proj_t2 / b)**2
    
    W_ellipse = 50.0  # tune weight
    c_ellipse = -W_ellipse / (ellipse_dist + 1e-6)
    
    # Position error
    W_pos = torch.tensor([100.0, 100.0, 500.0], device=device, dtype=torch.float32)
    c_pos = torch.sum(W_pos * (pos - goal_t) ** 2, dim=-1)
    
    # Velocity penalty
    W_vel = torch.tensor([0.1, 0.1, 0.1], device=device, dtype=torch.float32)
    c_vel = torch.sum(W_vel * vel ** 2, dim=-1)
    
    # Control effort penalty
    W_u = torch.tensor([0.01] * u.shape[-1], device=device, dtype=torch.float32)
    c_u = torch.sum(W_u * u ** 2, dim=-1)
    
    total_cost = c_pos + c_vel + c_u + c_ellipse
    cost = total_cost.reshape(N, N).detach().cpu().numpy()
    
    # Plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(T1.cpu(), T2.cpu(), cost, levels=50, cmap='viridis')
    plt.colorbar(contour, label="Cost")
    plt.xlabel("t1 axis (meters) - orthogonal to gate normal")
    plt.ylabel("t2 axis (meters) - orthogonal to gate normal")
    plt.title(f"MPPI Cost Map (Gate Plane)\nGate at [{gate_pos[0]:.2f}, {gate_pos[1]:.2f}, {gate_pos[2]:.2f}]")
    plt.axis("equal")
    plt.grid(True, alpha=0.3)
    
    # Mark the gate center
    plt.plot(0, 0, 'r*', markersize=15, label='Gate Center')
    plt.legend()
    
    return plt


if __name__ == "__main__":
    # Example gate configuration (modify as needed)
    gate_pos = np.array([0.5, 0.25, 0.7])
    
    # Gate orientation as quaternion [x, y, z, w]
    # Default orientation (no rotation)
    gate_rpy = np.array([0, 0, 0])  # Roll, pitch, yaw in radians
    rot = R.from_euler('xyz', gate_rpy)
    gate_quat = rot.as_quat()  # Returns [x, y, z, w]
    
    print(f"Plotting cost maps for gate at position: {gate_pos}")
    print(f"Gate orientation (RPY): {gate_rpy}")
    print(f"Gate quaternion (xyzw): {gate_quat}")
    
    # Plot 1: Slice through gate (vertical cross-section along approach)
    print("\n1. Creating slice through gate (approach direction)...")
    fig1 = plot_cost_slice_through_gate(gate_pos, gate_quat, grid_range=1.5, N=150)
    fig1.savefig('mppi_cost_slice.png', dpi=150, bbox_inches='tight')
    print("   Saved to: mppi_cost_slice.png")
    
    # Plot 2: Plane of the gate (perpendicular to approach)
    print("\n2. Creating gate plane view...")
    plot_cost_map_for_gate(gate_pos, gate_quat, grid_range=0.5, N=150)
    plt.savefig('mppi_cost_map.png', dpi=150, bbox_inches='tight')
    print("   Saved to: mppi_cost_map.png")
    
    plt.show()
