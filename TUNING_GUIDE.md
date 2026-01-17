# MPPI Cost Function Tuning Guide

## Current Cost Weights (in `mppi_builder_advanced.py`)

### Gate Attraction
```python
cost += 300.0 * terminal_distance      # Distance to gate at end of horizon
cost += 10.0 * running_distance        # Distance to gate at each timestep
cost -= 500.0 * gate_bonus             # Reward for getting close to gate
```

**Effect**: Higher values = more aggressive toward gates
**Tuning**: 
- If drone doesn't reach gates: ↑ these values (400, 15, 700)
- If drone is too aggressive/unstable: ↓ these values (200, 5, 300)

### Obstacle Avoidance
```python
cost += 500.0 * penetration**2         # Quadratic penalty for being too close
safety_margin = obstacle_radius + 0.2  # 20cm extra safety
```

**Effect**: Higher values = more conservative around obstacles
**Tuning**:
- If drone hits obstacles: ↑ penalty (800), ↑ margin (0.3)
- If drone is too cautious: ↓ penalty (300), ↓ margin (0.1)

### Speed Preference
```python
preferred_speed = 1.5  # m/s
cost += 0.5 * speed_error**2
```

**Effect**: Encourages maintaining preferred speed
**Tuning**:
- For faster racing: ↑ preferred_speed (2.0-3.0)
- For more conservative: ↓ preferred_speed (1.0-1.5)
- For less strict: ↓ weight (0.1-0.3)

### Control Effort
```python
cost += 0.1 * acceleration**2
```

**Effect**: Penalizes large accelerations
**Tuning**:
- For smoother trajectories: ↑ weight (0.3-0.5)
- For more aggressive: ↓ weight (0.05-0.1)

### Control Smoothness
```python
cost += 1.0 * (acceleration_change)**2
```

**Effect**: Penalizes jerky motion
**Tuning**:
- For smoother: ↑ weight (2.0-5.0)
- For more responsive: ↓ weight (0.5-1.0)

### Height Safety
```python
min_height = 0.1  # meters
cost += 500.0 * height_violation**2
```

**Effect**: Prevents flying too low
**Tuning**:
- For safer flight: ↑ min_height (0.2-0.3), ↑ weight (1000)
- If drone gets stuck: ↓ min_height (0.05), ↓ weight (300)

## Common Issues & Fixes

### Issue: Drone doesn't reach gates
**Symptoms**: High costs (>2000), drone wanders aimlessly

**Solution**:
```python
# In mppi_builder_advanced.py _cost() method:
cost += 400.0 * dist_to_gate      # Was 300
cost += 15.0 * dist_t             # Was 10
cost -= 700.0 * gate_bonus        # Was 500
```

### Issue: Drone crashes into ground
**Symptoms**: Height violations, episode ends early

**Solution**:
```python
min_height = 0.2                  # Was 0.1
cost += 800.0 * height_violation  # Was 500
```

### Issue: Trajectory too jerky/unstable
**Symptoms**: Oscillations, MPC solver warnings

**Solution**:
```python
cost += 0.3 * np.sum(U**2)        # Was 0.1 (more smooth)
cost += 3.0 * np.sum(dU**2)       # Was 1.0 (less jerky)
preferred_speed = 1.0             # Was 1.5 (slower)
```

### Issue: Drone hits obstacles
**Symptoms**: Collision detection, episode ends

**Solution**:
```python
safety_margin = self.obstacle_radius + 0.3  # Was 0.2
cost += 800.0 * penetration**2              # Was 500
```

### Issue: Drone too slow/conservative
**Symptoms**: Takes too long, doesn't explore

**Solution**:
```python
preferred_speed = 2.5             # Was 1.5
cost += 0.05 * np.sum(U**2)       # Was 0.1 (allow bigger accel)
cost += 0.5 * np.sum(dU**2)       # Was 1.0 (allow jerk)
```

## MPPI Hyperparameters

In `attitude_mpc_mppi_hybrid_advanced.py`:

```python
K = 500              # Number of samples
lambda_ = 0.8        # Temperature
sigma_u = 0.4        # Control noise
```

### K (Number of Samples)
- **Higher** (800-1000): Better quality, slower
- **Lower** (200-300): Faster, less optimal
- **Current** (500): Good balance

### lambda_ (Temperature)
- **Lower** (0.3-0.5): More exploitation, sharper, riskier
- **Higher** (1.0-2.0): More exploration, smoother, safer
- **Current** (0.8): Balanced

### sigma_u (Control Noise)
- **Lower** (0.2-0.3): Less exploration, converges faster
- **Higher** (0.5-0.8): More exploration, finds alternatives
- **Current** (0.4): Good for real-time

## Testing Your Changes

After editing the cost function:

```bash
# Quick test
pixi run python tests/test_mppi_hybrid.py

# Single run to observe behavior
pixi run python scripts/sim.py --config level1_mppi_hybrid.toml --render True

# Multiple runs for statistics
pixi run python scripts/sim.py --config level1_mppi_hybrid.toml --n_runs 5
```

## Monitoring Cost Values

The console shows:
```
[MPPIBuilderAdvanced] t=0.0s, gate=1/4, cost_min=2539.0, pos=[...]
```

**Good cost values**: 500-1500 (drone on track)
**High cost values**: 2000+ (drone struggling, far from goal)
**Very low costs**: <300 (very close to goal)

## Recommended Starting Points

### Conservative/Safe (for testing)
```python
cost += 200.0 * dist_to_gate
cost += 5.0 * dist_t
cost -= 300.0 * gate_bonus
cost += 800.0 * penetration**2
preferred_speed = 1.0
```

### Aggressive/Fast (for racing)
```python
cost += 500.0 * dist_to_gate
cost += 20.0 * dist_t
cost -= 1000.0 * gate_bonus
cost += 400.0 * penetration**2
preferred_speed = 3.0
```

### Balanced (current default)
```python
cost += 300.0 * dist_to_gate
cost += 10.0 * dist_t
cost -= 500.0 * gate_bonus
cost += 500.0 * penetration**2
preferred_speed = 1.5
```

## Debug Tips

1. **Check costs in console**: Should decrease as drone approaches gates
2. **Watch visualization**: Green line shows planned trajectory
3. **Monitor gate progress**: Should see "Advanced to gate X/4"
4. **Check MPC solver status**: Should be 0 (success)
5. **Look for warnings**: Height violations, solver failures

## Example Tuning Session

```bash
# 1. Run and observe
pixi run python scripts/sim.py --config level1_mppi_hybrid.toml --render True

# 2. Note issues (e.g., "too conservative")

# 3. Edit mppi_builder_advanced.py
#    - Increase gate attraction
#    - Increase preferred speed
#    - Decrease smoothness penalty

# 4. Test again
pixi run python scripts/sim.py --config level1_mppi_hybrid.toml --render True

# 5. Repeat until satisfied
```

Save this file as `TUNING_GUIDE.md` for quick reference during development!
