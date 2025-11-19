"""Benchmark script to compare MPPI+MPC hybrid vs original MPC.

This script runs both controllers multiple times and compares:
- Completion time
- Success rate
- Gates passed
- Collision rate
"""

import sys
from pathlib import Path
import numpy as np
import time
import json

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from scripts.sim import simulate


def run_benchmark(config_name: str, n_runs: int = 5, label: str = "Controller"):
    """Run multiple episodes and collect statistics.
    
    Args:
        config_name: Name of config file (e.g., 'level1.toml')
        n_runs: Number of episodes to run
        label: Label for printing
        
    Returns:
        Dictionary with statistics
    """
    print(f"\n{'='*60}")
    print(f"Benchmarking: {label}")
    print(f"Config: {config_name}")
    print(f"Runs: {n_runs}")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    # Run simulations (with render=False for speed)
    episode_times = simulate(
        config=config_name,
        controller=None,  # Use controller from config
        n_runs=n_runs,
        render=False,
    )
    
    total_time = time.time() - start_time
    
    # Analyze results
    completed = [t for t in episode_times if t is not None]
    failed = [t for t in episode_times if t is None]
    
    stats = {
        "label": label,
        "config": config_name,
        "n_runs": n_runs,
        "success_rate": len(completed) / n_runs * 100,
        "completed": len(completed),
        "failed": len(failed),
        "mean_time": np.mean(completed) if completed else None,
        "std_time": np.std(completed) if completed else None,
        "min_time": np.min(completed) if completed else None,
        "max_time": np.max(completed) if completed else None,
        "total_benchmark_time": total_time,
    }
    
    # Print results
    print(f"\nResults for {label}:")
    print(f"  Success Rate:    {stats['success_rate']:.1f}% ({stats['completed']}/{n_runs})")
    
    if completed:
        print(f"  Completion Time: {stats['mean_time']:.2f}s ± {stats['std_time']:.2f}s")
        print(f"  Best Time:       {stats['min_time']:.2f}s")
        print(f"  Worst Time:      {stats['max_time']:.2f}s")
    else:
        print(f"  No successful runs")
    
    print(f"  Benchmark Time:  {stats['total_benchmark_time']:.1f}s")
    
    return stats


def compare_controllers(n_runs: int = 5):
    """Compare original MPC vs MPPI+MPC hybrid."""
    print("=" * 60)
    print("CONTROLLER COMPARISON BENCHMARK")
    print("=" * 60)
    
    results = []
    
    # Benchmark 1: Original MPC with spline trajectory
    try:
        stats_original = run_benchmark(
            config_name="level1.toml",
            n_runs=n_runs,
            label="Original MPC (Spline)"
        )
        results.append(stats_original)
    except Exception as e:
        print(f"Error running original MPC: {e}")
        stats_original = None
    
    # Benchmark 2: MPPI+MPC Hybrid
    try:
        stats_hybrid = run_benchmark(
            config_name="level1_mppi_hybrid.toml",
            n_runs=n_runs,
            label="MPPI+MPC Hybrid"
        )
        results.append(stats_hybrid)
    except Exception as e:
        print(f"Error running hybrid controller: {e}")
        stats_hybrid = None
    
    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON SUMMARY")
    print("=" * 60)
    
    if stats_original and stats_hybrid:
        print(f"\n{'Metric':<25} {'Original MPC':<20} {'MPPI+MPC Hybrid':<20}")
        print("-" * 65)
        
        # Success rate
        print(f"{'Success Rate':<25} {stats_original['success_rate']:>6.1f}% "
              f"{stats_hybrid['success_rate']:>20.1f}%")
        
        # Mean time
        if stats_original['mean_time'] and stats_hybrid['mean_time']:
            orig_time = stats_original['mean_time']
            hybrid_time = stats_hybrid['mean_time']
            speedup = (orig_time - hybrid_time) / orig_time * 100
            
            print(f"{'Mean Time':<25} {orig_time:>6.2f}s "
                  f"{hybrid_time:>20.2f}s")
            
            if speedup > 0:
                print(f"{'Speedup':<25} {'':<20} {speedup:>18.1f}% faster")
            else:
                print(f"{'Speedup':<25} {'':<20} {-speedup:>18.1f}% slower")
        
        # Best time
        if stats_original['min_time'] and stats_hybrid['min_time']:
            print(f"{'Best Time':<25} {stats_original['min_time']:>6.2f}s "
                  f"{stats_hybrid['min_time']:>20.2f}s")
        
        # Consistency (lower std = more consistent)
        if stats_original['std_time'] and stats_hybrid['std_time']:
            print(f"{'Consistency (σ)':<25} {stats_original['std_time']:>6.2f}s "
                  f"{stats_hybrid['std_time']:>20.2f}s")
        
        print("\n" + "-" * 65)
        
        # Winner
        winner = None
        if stats_hybrid['success_rate'] > stats_original['success_rate']:
            winner = "MPPI+MPC Hybrid"
            reason = "higher success rate"
        elif stats_hybrid['success_rate'] == stats_original['success_rate']:
            if (stats_hybrid['mean_time'] and stats_original['mean_time'] and
                stats_hybrid['mean_time'] < stats_original['mean_time']):
                winner = "MPPI+MPC Hybrid"
                reason = "faster completion"
            elif (stats_hybrid['mean_time'] and stats_original['mean_time']):
                winner = "Original MPC"
                reason = "faster completion"
        
        if winner:
            print(f"\n🏆 Winner: {winner} ({reason})")
        else:
            print(f"\n🤝 Tie: Both controllers performed similarly")
    
    # Save results
    output_file = Path(__file__).parent.parent / "benchmark_results.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}")
    
    return results


def main():
    """Main benchmark function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Benchmark controllers")
    parser.add_argument(
        "--n_runs",
        type=int,
        default=5,
        help="Number of runs per controller (default: 5)"
    )
    args = parser.parse_args()
    
    print(f"\nStarting benchmark with {args.n_runs} runs per controller...")
    print("This may take several minutes.\n")
    
    results = compare_controllers(n_runs=args.n_runs)
    
    print("\n" + "=" * 60)
    print("Benchmark complete!")
    print("=" * 60)
    
    return results


if __name__ == "__main__":
    main()
