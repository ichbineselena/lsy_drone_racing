"""Test script for MPPI + MPC hybrid controller.

This script verifies that the hybrid controller can be instantiated and run.
It's useful for debugging before running the full simulation.
"""

import numpy as np
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from lsy_drone_racing.utils import load_config
from lsy_drone_racing.control.trajectory_builders import MPPIBuilderAdvanced


def test_mppi_builder():
    """Test the advanced MPPI trajectory builder."""
    print("=" * 60)
    print("Testing MPPI Trajectory Builder")
    print("=" * 60)
    
    # Define test gates and obstacles
    gates = [
        np.array([0.5, 0.25, 0.7]),
        np.array([1.05, 0.75, 1.2]),
        np.array([-1.0, -0.25, 0.7]),
        np.array([0.0, -0.75, 1.2]),
    ]
    
    obstacles = [
        np.array([0.0, 0.75, 1.55]),
        np.array([1.0, 0.25, 1.55]),
    ]
    
    # Create MPPI builder
    mppi = MPPIBuilderAdvanced(
        gates=gates,
        obstacles=obstacles,
        K=100,  # Reduced for testing
        lambda_=0.8,
        sigma_u=0.4,
    )
    
    # Test initial state
    initial_state = np.array([
        -1.5, 0.75, 0.01,  # pos
        0.0, 0.0, 0.0,     # rpy
        0.0, 0.0, 0.0,     # vel
        0.0, 0.0, 0.0,     # drpy
    ])
    
    mppi.reset(initial_state, t0=0.0)
    print(f"✓ MPPI builder initialized")
    print(f"  Initial position: {initial_state[0:3]}")
    print(f"  Target gate: {gates[0]}")
    
    # Generate trajectory
    print("\nGenerating trajectory...")
    horizon = mppi.get_horizon(t_now=0.0, N=25, dt=0.02)
    
    print(f"✓ Trajectory generated")
    print(f"  Position shape: {horizon['pos'].shape}")
    print(f"  Velocity shape: {horizon['vel'].shape}")
    print(f"  Yaw shape: {horizon['yaw'].shape}")
    print(f"  First position: {horizon['pos'][0]}")
    print(f"  Last position: {horizon['pos'][-1]}")
    
    # Check that trajectory moves toward gate
    start_dist = np.linalg.norm(initial_state[0:3] - gates[0])
    end_dist = np.linalg.norm(horizon['pos'][-1] - gates[0])
    print(f"\n  Distance to gate 1:")
    print(f"    Start: {start_dist:.3f}m")
    print(f"    End:   {end_dist:.3f}m")
    
    if end_dist < start_dist:
        print("  ✓ Trajectory moves toward gate")
    else:
        print("  ⚠ Warning: Trajectory may not be optimal")
    
    return True


def test_hybrid_controller():
    """Test the hybrid controller instantiation."""
    print("\n" + "=" * 60)
    print("Testing Hybrid Controller")
    print("=" * 60)
    
    try:
        from lsy_drone_racing.control.attitude_mpc_mppi_hybrid_advanced import (
            AttitudeMPCMPPIHybridAdvanced
        )
        print("✓ Successfully imported hybrid controller")
        
        # Load config
        config_path = Path(__file__).parent.parent / "config" / "level1_mppi_hybrid.toml"
        if not config_path.exists():
            print(f"⚠ Config file not found: {config_path}")
            return False
        
        config = load_config(config_path)
        print(f"✓ Loaded configuration from {config_path.name}")
        
        # Create mock observation
        obs = {
            "pos": np.array([-1.5, 0.75, 0.01]),
            "quat": np.array([0.0, 0.0, 0.0, 1.0]),
            "vel": np.array([0.0, 0.0, 0.0]),
            "ang_vel": np.array([0.0, 0.0, 0.0]),
        }
        
        info = {}
        
        print("\nInitializing hybrid controller...")
        print("(This may take a moment to compile Acados solver)")
        
        # This will compile the Acados solver
        controller = AttitudeMPCMPPIHybridAdvanced(obs, info, config)
        print("✓ Controller initialized successfully")
        
        # Test control computation
        print("\nComputing control action...")
        action = controller.compute_control(obs, info)
        
        print(f"✓ Control computed: {action}")
        print(f"  Roll:   {action[0]:.3f} rad ({np.degrees(action[0]):.1f}°)")
        print(f"  Pitch:  {action[1]:.3f} rad ({np.degrees(action[1]):.1f}°)")
        print(f"  Yaw:    {action[2]:.3f} rad ({np.degrees(action[2]):.1f}°)")
        print(f"  Thrust: {action[3]:.3f} N")
        
        # Check planned trajectory
        traj = controller.get_planned_trajectory()
        if traj is not None:
            print(f"\n✓ Planned trajectory available: {traj.shape}")
        
        return True
        
    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("MPPI + MPC Hybrid Controller Tests")
    print("=" * 60)
    print()
    
    # Test 1: MPPI builder
    try:
        test1 = test_mppi_builder()
    except Exception as e:
        print(f"✗ MPPI builder test failed: {e}")
        import traceback
        traceback.print_exc()
        test1 = False
    
    # Test 2: Hybrid controller
    try:
        test2 = test_hybrid_controller()
    except Exception as e:
        print(f"✗ Hybrid controller test failed: {e}")
        import traceback
        traceback.print_exc()
        test2 = False
    
    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    print(f"MPPI Builder:       {'✓ PASS' if test1 else '✗ FAIL'}")
    print(f"Hybrid Controller:  {'✓ PASS' if test2 else '✗ FAIL'}")
    print()
    
    if test1 and test2:
        print("All tests passed! You can now run:")
        print("  python scripts/sim.py --config level1_mppi_hybrid.toml")
    else:
        print("Some tests failed. Check the output above for details.")
    
    return test1 and test2


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
