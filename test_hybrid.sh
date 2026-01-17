#!/bin/bash
# Test script for MPPI+MPC Hybrid Controller
# This script runs all tests with proper command syntax

echo "=================================="
echo "MPPI + MPC Hybrid Testing Suite"
echo "=================================="
echo ""

# Test 1: Unit tests
echo "1. Running unit tests..."
pixi run python tests/test_mppi_hybrid.py
TEST_RESULT=$?
echo ""

if [ $TEST_RESULT -ne 0 ]; then
    echo "❌ Unit tests failed. Please fix errors before proceeding."
    exit 1
fi

echo "✅ Unit tests passed!"
echo ""

# Test 2: Single simulation with visualization
echo "2. Running single simulation (with visualization)..."
echo "   Note: This will open a PyBullet window"
pixi run python scripts/sim.py --config level1_mppi_hybrid.toml --render True
echo ""

# Test 3: Multiple runs without visualization (faster)
echo "3. Running 3 episodes without visualization..."
pixi run python scripts/sim.py --config level1_mppi_hybrid.toml --n_runs 3
echo ""

# Test 4: Benchmark comparison (optional - takes time)
read -p "4. Run benchmark comparison? (5 runs, ~2-3 min) [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Running benchmark..."
    pixi run python scripts/benchmark_hybrid.py --n_runs 5
    echo ""
fi

# Test 5: Visualization (optional)
read -p "5. Generate visualizations? [y/N]: " -n 1 -r
echo ""
if [[ $REPLY =~ ^[Yy]$ ]]; then
    echo "Generating visualizations..."
    pixi run python scripts/visualize_hybrid.py
    echo ""
fi

echo "=================================="
echo "Testing complete!"
echo "=================================="
echo ""
echo "Next steps:"
echo "  - Review the output above for any warnings"
echo "  - Check generated images (if visualizations were run)"
echo "  - Try tuning parameters in the controller files"
echo ""
