#!/bin/bash

# Script to run sim.py multiple times and collect statistics
# Usage: ./scripts/run_sim_multiple.sh [config] [n_runs]
# Example: ./scripts/run_sim_multiple.sh level0.toml 20

CONFIG="${1:-level2.toml}"
N_RUNS="${2:-20}"
RENDER="${3:-false}"

echo "Running sim.py $N_RUNS times with config: $CONFIG"
echo "Render: $RENDER"
echo "================================================"
echo ""

# Arrays to store results
declare -a results
declare -a gates_passed
declare -a flight_times

# Run sim.py N_RUNS times
for i in $(seq 1 $N_RUNS); do
    echo "Run $i/$N_RUNS..."
    
    # Capture output
    output=$(python scripts/sim.py --config "$CONFIG" --n_runs 1 --render "$RENDER" 2>&1)
    
    # Extract gates passed (format: "Gates passed: X" - extract the number at the end)
    gates=$(echo "$output" | grep "Gates passed:" | tail -1 | grep -oE '[0-9]+$')
    
    # Extract flight time (format: "Flight time (s): X.XX" - extract the number after the colon)
    time=$(echo "$output" | grep "Flight time" | tail -1 | grep -oE '[0-9]+\.?[0-9]*$')
    
    # Extract finished status
    finished=$(echo "$output" | grep "Finished:" | tail -1 | grep -oE '(True|False)$')
    
    # Determine total gates (assume 4 if not specified)
    if [ "$finished" == "True" ]; then
        total_gates="$gates"
    else
        # Try to infer from common configurations
        total_gates="4"  # Default assumption
    fi
    
    # Store result
    if [ -n "$gates" ]; then
        result="${gates}/${total_gates} gates"
        results+=("$result")
        gates_passed+=("$gates")
        flight_times+=("$time")
    else
        result="Error/Unknown"
        results+=("$result")
        gates_passed+=("0")
        flight_times+=("N/A")
    fi
    
    echo "  Result: $result (Time: ${time}s)"
    echo ""
done

echo "================================================"
echo "SUMMARY OF ALL $N_RUNS RUNS:"
echo "================================================"

# Print individual run results
for i in $(seq 1 $N_RUNS); do
    idx=$((i-1))
    printf "Run %2d: %s (Time: %ss)\n" "$i" "${results[$idx]}" "${flight_times[$idx]}"
done

echo ""
echo "================================================"
echo "STATISTICS:"
echo "================================================"

# Count successes (completed all gates)
success_count=0
for gates in "${gates_passed[@]}"; do
    if [ "$gates" -ge 4 ]; then  # Assuming 4 gates is success
        ((success_count++))
    fi
done

# Count distribution of gates passed
declare -A gate_distribution
for gates in "${gates_passed[@]}"; do
    if [ -n "$gates" ]; then
        ((gate_distribution[$gates]++))
    fi
done

echo "Success Rate: $success_count/$N_RUNS ($(echo "scale=1; $success_count * 100 / $N_RUNS" | bc)%)"
echo ""
echo "Gates Passed Distribution:"
for key in $(echo "${!gate_distribution[@]}" | tr ' ' '\n' | sort -n); do
    count="${gate_distribution[$key]}"
    percentage=$(echo "scale=1; $count * 100 / $N_RUNS" | bc)
    printf "  %s gates: %2d runs (%.1f%%)\n" "$key" "$count" "$percentage"
done

# Calculate average flight time for successful runs
total_time=0
time_count=0
for time in "${flight_times[@]}"; do
    if [ "$time" != "N/A" ] && [ -n "$time" ]; then
        total_time=$(echo "$total_time + $time" | bc)
        ((time_count++))
    fi
done

if [ $time_count -gt 0 ]; then
    avg_time=$(echo "scale=3; $total_time / $time_count" | bc)
    echo ""
    echo "Average Flight Time: ${avg_time}s (across $time_count completed runs)"
fi

echo "================================================"
