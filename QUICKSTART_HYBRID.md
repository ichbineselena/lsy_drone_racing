# Quick Start: MPPI + MPC Hybrid Controller

## What You Got

I've implemented a **two-level hybrid control system** for your drone racing project:

### 🎯 High-Level: MPPI Trajectory Planner
- Generates adaptive trajectories online using sampling (500 parallel rollouts)
- Handles obstacles and gates intelligently
- Replans every timestep based on current state

### 🎮 Low-Level: MPC Attitude Controller  
- Tracks MPPI trajectories with precise attitude control
- Uses full nonlinear drone dynamics (Acados solver)
- Enforces angle constraints (±30°) and thrust limits

## Files Created

```
lsy_drone_racing/
├── control/
│   ├── attitude_mpc_mppi_hybrid.py              # Basic hybrid controller
│   ├── attitude_mpc_mppi_hybrid_advanced.py     # Full-featured version ⭐
│   └── trajectory_builders/
│       ├── mppi_builder_advanced.py             # Advanced MPPI planner ⭐
│       └── __init__.py (updated)
├── config/
│   └── level1_mppi_hybrid.toml                  # Test configuration ⭐
├── docs/
│   └── MPPI_MPC_HYBRID.md                       # Detailed documentation
└── tests/
    └── test_mppi_hybrid.py                      # Unit tests
```

**⭐ = Main files you'll use**

## How to Run

### Option A: Use the Test Script (Recommended)
```bash
cd /home/elena/lsy_drone_racing
./test_hybrid.sh
```

This interactive script will guide you through all tests.

### Option B: Manual Testing

#### 1. Test the Implementation
```bash
cd /home/elena/lsy_drone_racing
python tests/test_mppi_hybrid.py
```

This will:
- Test MPPI trajectory generation
- Test hybrid controller initialization
- Compile Acados solvers (first time only)

#### 2. Run a Simulation
```bash
# With visualization (watch the drone race!)
# Note: Use True (capital T), not true
python scripts/sim.py --config level1_mppi_hybrid.toml --render True

# Without visualization (faster)
python scripts/sim.py --config level1_mppi_hybrid.toml

# Multiple runs for evaluation
python scripts/sim.py --config level1_mppi_hybrid.toml --n_runs 5
```

#### 3. Compare to Original MPC
```bash
# Run original MPC controller
python scripts/sim.py --config level1.toml --render True

# Run hybrid controller
python scripts/sim.py --config level1_mppi_hybrid.toml --render True
```

## Key Differences from Original

| Feature | Original MPC | Hybrid MPPI+MPC |
|---------|--------------|-----------------|
| **Trajectory** | Fixed spline waypoints | Adaptive online planning |
| **Obstacles** | Not considered | Actively avoided |
| **Gates** | Fixed sequence | Tracked dynamically |
| **Replanning** | No | Every timestep |
| **Robustness** | Good | Better (samples variations) |

## What to Expect

When you run the simulation, you should see:

1. **Console output** showing:
   ```
   [MPPIBuilderAdvanced] Initialized:
     - Gates: 4
     - Obstacles: 4
     - K=500, lambda=0.8, sigma_u=0.4
   
   [AttitudeMPCMPPIHybridAdvanced] Initialized
     - MPC horizon: 25 steps (0.50s)
     - MPPI: K=500, lambda=0.8, sigma=0.4
   ```

2. **During flight** (if you enabled printing):
   ```
   [MPPIBuilderAdvanced] t=2.5s, gate=1/4, cost_min=245.3, pos=[...]
   [MPPIBuilderAdvanced] Advanced to gate 2/4
   ```

3. **Visualization** (if render=true):
   - Green line showing MPPI's planned trajectory
   - Drone following the trajectory
   - Gates and obstacles in the scene

4. **Final statistics**:
   ```
   Flight time (s): 14.2
   Finished: True
   Gates passed: 4
   ```

## Tuning the Controller

### If the drone is too conservative:
```python
# In mppi_builder_advanced.py, reduce obstacle penalties
cost += 200.0 * penetration**2  # Was 500.0
```

### If trajectory is too jerky:
```python
# Increase smoothness weight
cost += 5.0 * np.sum(dU**2, axis=(1, 2))  # Was 2.0
```

### If planning is too slow:
Edit `config/level1_mppi_hybrid.toml`:
```toml
# Reduce MPPI samples (faster but lower quality)
# In the controller file, change K=500 to K=200
```

Or edit the controller initialization:
```python
# In attitude_mpc_mppi_hybrid_advanced.py
mppi_K = 200  # Was 500
```

## Troubleshooting

### "ModuleNotFoundError: No module named acados_template"
```bash
# Install acados (if not already done)
pip install acados_template
```

### "ImportError: cannot import name MPPIBuilderAdvanced"
```bash
# Make sure __init__.py is updated
cd lsy_drone_racing/control/trajectory_builders
cat __init__.py  # Should include MPPIBuilderAdvanced
```

### Acados compilation fails
```bash
# Check that acados is properly installed
cd acados
source env.sh  # or equivalent setup script
```

### Drone crashes / doesn't reach gates
1. Check gate positions in config
2. Increase MPPI samples: `K=1000`
3. Reduce lambda: `lambda_=0.5` (more aggressive)
4. Check MPC solver status in output

## Next Steps

### 1. Experiment with Parameters
Try different MPPI settings in the controller:
```python
MPPIBuilderAdvanced(
    gates=gates,
    obstacles=obstacles,
    K=1000,              # More samples = better quality
    lambda_=0.5,         # Lower = more aggressive
    sigma_u=0.3,         # Lower = less exploration
    obstacle_radius=0.4, # Larger = more conservative
)
```

### 2. Add Custom Costs
Edit `mppi_builder_advanced.py` `_cost()` method:
```python
# Example: penalize high speed
speed = np.linalg.norm(vel_seq, axis=2)
cost += 5.0 * (speed > 3.0) * speed**2
```

### 3. GPU Acceleration
For faster MPPI, check out `highlevel_mppi.py` for PyTorch/GPU version

### 4. Learned Dynamics
Replace double-integrator with a learned model for better accuracy

### 5. Deploy to Real Drone
Use the deployment configuration and test progressively

## Architecture Diagram

```
     Observations               Reference              Control
     (pos, vel, ...)           Trajectory             Commands
          │                    (pos, vel, yaw)        (r,p,y,T)
          │                         │                     │
          ▼                         ▼                     ▼
    ┌──────────┐              ┌──────────┐         ┌──────────┐
    │   MPPI   │─────────────▶│   MPC    │────────▶│  Drone   │
    │ Planner  │  25 steps    │ Tracker  │         │          │
    └──────────┘   @ 50Hz     └──────────┘         └──────────┘
         │                                                │
         │ Uses:                                         │
         │ - Gates list                                   │
         │ - Obstacles                                    │
         │ - Current state                                │
         └────────────────────────────────────────────────┘
                          Feedback Loop
```

## Support

- **Detailed docs**: `docs/MPPI_MPC_HYBRID.md`
- **Code examples**: `attitude_mpc_mppi_hybrid_advanced.py`
- **Test suite**: `tests/test_mppi_hybrid.py`

Good luck with your drone racing! 🚁🏁
