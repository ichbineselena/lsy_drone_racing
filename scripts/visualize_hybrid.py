"""Visualization tool for MPPI+MPC hybrid controller.

This script helps visualize:
- MPPI sampled trajectories
- Best trajectory selection
- MPC tracking performance
- Gate and obstacle positions
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))

from lsy_drone_racing.control.trajectory_builders import MPPIBuilderAdvanced


def visualize_mppi_sampling():
    """Visualize MPPI sampling process."""
    # Setup
    gates = [
        np.array([0.5, 0.25, 0.7]),
        np.array([1.05, 0.75, 1.2]),
        np.array([-1.0, -0.25, 0.7]),
        np.array([0.0, -0.75, 1.2]),
    ]
    
    obstacles = [
        np.array([0.0, 0.75, 1.55]),
        np.array([1.0, 0.25, 1.55]),
        np.array([-1.5, -0.25, 1.55]),
        np.array([-0.5, -0.75, 1.55]),
    ]
    
    # Create MPPI builder
    mppi = MPPIBuilderAdvanced(
        gates=gates,
        obstacles=obstacles,
        K=100,  # Reduced for visualization
        lambda_=0.8,
        sigma_u=0.4,
    )
    
    # Initial state
    initial_state = np.array([
        -1.5, 0.75, 0.01,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
        0.0, 0.0, 0.0,
    ])
    
    mppi.reset(initial_state, t0=0.0)
    
    # Sample trajectories manually to visualize
    T = 25
    dt = 0.02
    
    # Create figure
    fig = plt.figure(figsize=(15, 10))
    ax1 = fig.add_subplot(221, projection='3d')
    ax2 = fig.add_subplot(222)
    ax3 = fig.add_subplot(223)
    ax4 = fig.add_subplot(224)
    
    # Sample some trajectories
    dU = np.random.normal(scale=mppi.sigma_u, size=(mppi.K, T, 3))
    U_samples = dU  # Start from zero nominal
    U_samples = np.clip(U_samples, -5.0, 5.0)
    
    # Rollout
    states = mppi._double_integrator_rollout(initial_state, U_samples, dt)
    
    # Compute costs
    costs = mppi._cost(states, U_samples)
    
    # Plot 1: 3D trajectories
    ax1.set_title('MPPI Sampled Trajectories (3D)')
    ax1.set_xlabel('X (m)')
    ax1.set_ylabel('Y (m)')
    ax1.set_zlabel('Z (m)')
    
    # Plot sample trajectories (color by cost)
    costs_norm = (costs - costs.min()) / (costs.max() - costs.min() + 1e-8)
    for k in range(min(50, mppi.K)):  # Plot first 50
        color = plt.cm.RdYlGn_r(costs_norm[k])
        ax1.plot(states[k, :, 0], states[k, :, 1], states[k, :, 2],
                alpha=0.3, linewidth=0.5, color=color)
    
    # Best trajectory
    best_k = np.argmin(costs)
    ax1.plot(states[best_k, :, 0], states[best_k, :, 1], states[best_k, :, 2],
            'b-', linewidth=2, label='Best trajectory')
    
    # Gates
    for i, gate in enumerate(gates):
        ax1.scatter(gate[0], gate[1], gate[2], s=200, c='green',
                   marker='o', alpha=0.7, label=f'Gate {i+1}' if i == 0 else '')
    
    # Obstacles
    for i, obs in enumerate(obstacles):
        ax1.scatter(obs[0], obs[1], obs[2], s=200, c='red',
                   marker='x', alpha=0.7, label=f'Obstacle' if i == 0 else '')
    
    # Start position
    ax1.scatter(initial_state[0], initial_state[1], initial_state[2],
               s=300, c='blue', marker='*', label='Start')
    
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Top-down view
    ax2.set_title('Top-Down View (X-Y)')
    ax2.set_xlabel('X (m)')
    ax2.set_ylabel('Y (m)')
    ax2.set_aspect('equal')
    
    for k in range(min(50, mppi.K)):
        color = plt.cm.RdYlGn_r(costs_norm[k])
        ax2.plot(states[k, :, 0], states[k, :, 1],
                alpha=0.3, linewidth=0.5, color=color)
    
    ax2.plot(states[best_k, :, 0], states[best_k, :, 1],
            'b-', linewidth=2, label='Best')
    
    for gate in gates:
        ax2.scatter(gate[0], gate[1], s=200, c='green', marker='o', alpha=0.7)
        circle = plt.Circle((gate[0], gate[1]), mppi.gate_radius,
                           color='green', fill=False, linestyle='--', alpha=0.5)
        ax2.add_patch(circle)
    
    for obs in obstacles:
        ax2.scatter(obs[0], obs[1], s=200, c='red', marker='x', alpha=0.7)
        circle = plt.Circle((obs[0], obs[1]), mppi.obstacle_radius + 0.2,
                           color='red', fill=False, linestyle='--', alpha=0.5)
        ax2.add_patch(circle)
    
    ax2.scatter(initial_state[0], initial_state[1], s=300, c='blue', marker='*')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    # Plot 3: Cost distribution
    ax3.set_title('Cost Distribution')
    ax3.hist(costs, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
    ax3.axvline(costs[best_k], color='blue', linestyle='--',
               linewidth=2, label=f'Best: {costs[best_k]:.1f}')
    ax3.axvline(costs.mean(), color='red', linestyle='--',
               linewidth=2, label=f'Mean: {costs.mean():.1f}')
    ax3.set_xlabel('Cost')
    ax3.set_ylabel('Count')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: MPPI weights
    S_min = costs.min()
    exp_arg = -(costs - S_min) / max(mppi.lambda_, 1e-8)
    w = np.exp(exp_arg)
    w = w / (np.sum(w) + 1e-12)
    
    ax4.set_title('MPPI Weights')
    ax4.scatter(costs, w, alpha=0.5, s=20)
    ax4.scatter(costs[best_k], w[best_k], color='blue', s=100,
               marker='*', label='Best trajectory')
    ax4.set_xlabel('Cost')
    ax4.set_ylabel('Weight')
    ax4.set_yscale('log')
    ax4.grid(True, alpha=0.3)
    ax4.legend()
    
    # Info text
    ESS = 1.0 / np.sum(w**2)
    info_text = (f'K = {mppi.K} samples\n'
                f'λ = {mppi.lambda_}\n'
                f'ESS = {ESS:.1f}\n'
                f'Best cost = {costs[best_k]:.1f}\n'
                f'Mean cost = {costs.mean():.1f}')
    ax4.text(0.02, 0.98, info_text, transform=ax4.transAxes,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    
    # Save figure
    output_path = Path(__file__).parent.parent / "mppi_visualization.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {output_path}")
    
    plt.show()


def plot_controller_comparison():
    """Plot comparison between MPC and MPPI+MPC."""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Dummy data for illustration (replace with actual benchmark results)
    controllers = ['Original\nMPC', 'MPPI+MPC\nHybrid']
    success_rates = [85, 95]
    mean_times = [16.5, 14.8]
    std_times = [2.3, 1.8]
    
    # Success rate
    axes[0, 0].bar(controllers, success_rates, color=['skyblue', 'lightgreen'],
                  alpha=0.7, edgecolor='black')
    axes[0, 0].set_ylabel('Success Rate (%)')
    axes[0, 0].set_title('Success Rate')
    axes[0, 0].set_ylim([0, 100])
    axes[0, 0].grid(True, alpha=0.3, axis='y')
    for i, v in enumerate(success_rates):
        axes[0, 0].text(i, v + 2, f'{v}%', ha='center', fontweight='bold')
    
    # Completion time
    axes[0, 1].bar(controllers, mean_times, yerr=std_times,
                  color=['skyblue', 'lightgreen'], alpha=0.7,
                  edgecolor='black', capsize=5)
    axes[0, 1].set_ylabel('Time (seconds)')
    axes[0, 1].set_title('Mean Completion Time')
    axes[0, 1].grid(True, alpha=0.3, axis='y')
    for i, (m, s) in enumerate(zip(mean_times, std_times)):
        axes[0, 1].text(i, m + s + 0.5, f'{m:.1f}s', ha='center', fontweight='bold')
    
    # Features comparison
    features = ['Adaptive\nPlanning', 'Obstacle\nAvoidance', 'Constraint\nSatisfaction',
               'Real-time\nCapable']
    mpc_scores = [2, 3, 5, 5]  # Out of 5
    hybrid_scores = [5, 5, 5, 4]
    
    x = np.arange(len(features))
    width = 0.35
    
    axes[1, 0].bar(x - width/2, mpc_scores, width, label='Original MPC',
                  color='skyblue', alpha=0.7, edgecolor='black')
    axes[1, 0].bar(x + width/2, hybrid_scores, width, label='MPPI+MPC',
                  color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 0].set_ylabel('Score (out of 5)')
    axes[1, 0].set_title('Feature Comparison')
    axes[1, 0].set_xticks(x)
    axes[1, 0].set_xticklabels(features)
    axes[1, 0].set_ylim([0, 5.5])
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3, axis='y')
    
    # Computation time breakdown
    categories = ['MPPI\nPlanning', 'MPC\nTracking', 'Total']
    times_original = [0, 5, 5]
    times_hybrid = [7, 5, 12]
    
    x = np.arange(len(categories))
    axes[1, 1].bar(x - width/2, times_original, width, label='Original MPC',
                  color='skyblue', alpha=0.7, edgecolor='black')
    axes[1, 1].bar(x + width/2, times_hybrid, width, label='MPPI+MPC',
                  color='lightgreen', alpha=0.7, edgecolor='black')
    axes[1, 1].set_ylabel('Time (ms/step)')
    axes[1, 1].set_title('Computational Cost')
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(categories)
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3, axis='y')
    axes[1, 1].axhline(y=20, color='red', linestyle='--', alpha=0.5,
                      label='Real-time limit (50Hz)')
    
    plt.tight_layout()
    
    output_path = Path(__file__).parent.parent / "controller_comparison.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Comparison saved to: {output_path}")
    
    plt.show()


def main():
    """Main visualization function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Visualize MPPI+MPC hybrid")
    parser.add_argument(
        "--type",
        choices=["mppi", "comparison", "both"],
        default="both",
        help="Type of visualization"
    )
    args = parser.parse_args()
    
    print("Generating visualizations...")
    print("Close the plot windows to continue.\n")
    
    if args.type in ["mppi", "both"]:
        print("1. MPPI Sampling Visualization")
        visualize_mppi_sampling()
    
    if args.type in ["comparison", "both"]:
        print("2. Controller Comparison")
        plot_controller_comparison()
    
    print("\nVisualization complete!")


if __name__ == "__main__":
    main()
