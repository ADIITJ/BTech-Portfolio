# Particle Filter for Robot Localization

Sequential Importance Resampling (SIR) particle filter for 2D robot tracking with landmark measurements. Achieves 0.33 m RMSE with 1000 particles and rejuvenation.

## Repository Structure

```
particle_filter_project/
├── particle_filter.py        # ParticleFilter and RobotSim classes
├── utils.py                  # Resampling, statistics, angle operations
├── visualize.py              # Plotting functions
├── run_experiments.py        # Main experiment runner
├── tests/
│   └── test_particle_filter.py
├── experiments/              # Generated outputs (git-ignored)
│   ├── summary.csv
│   ├── comparison_*.png
│   └── exp_*/
│       ├── frames/
│       ├── results.json
│       └── stats.csv
└── requirements.txt
```

## Installation

Python 3.8+ with NumPy, Matplotlib, Pandas, SciPy, tqdm:

```bash
pip install -r requirements.txt
```

## Quick Start

Run single test experiment (200 particles, 8 timesteps):
```bash
python run_experiments.py --quick
```

Run full suite (10 experiments, ~3 minutes):
```bash
python run_experiments.py
```

Run unit tests:
```bash
python tests/test_particle_filter.py
```

## Algorithm

**State**: `[x, y, θ]` where θ ∈ [-π, π]

**Motion model**: Rotate 10°, move forward 1 m, add Gaussian noise

**Measurement model**: Euclidean distance to landmarks with Gaussian noise

**Pipeline**:
1. **Predict**: Propagate particles with motion model
2. **Update**: Compute weights using measurement likelihood
3. **Resample**: Systematic or residual resampling
4. **Estimate**: Weighted mean (circular mean for θ)

## Experiments

10 configurations varying:
- **Particles**: N ∈ {100, 300, 1000}
- **Motion noise**: σ_d ∈ {0.02, 0.1} m, σ_θ ∈ {1°, 5°}
- **Measurement noise**: σ_m ∈ {0.05, 0.1, 0.2, 0.5} m
- **Resampling**: Systematic, residual
- **Rejuvenation**: With/without Gaussian jitter

Each experiment: 30 timesteps, 6 landmarks, seed=0.

### Key Results

| Experiment | N | σ_m (m) | RMSE (m) | ESS |
|------------|---|---------|----------|-----|
| exp_02 | 300 | 0.05 | 0.40 | 32.46 |
| exp_09 | 1000 | 0.50 | 0.33 | 393.38 |
| exp_08 | 300 | 0.50 | 1.44 | 156.18 |
| exp_03 | 1000 | 0.05 | 3.16 | 1.13 |

**Finding**: Low σ_m achieves better RMSE but causes degeneracy (ESS → 1). High σ_m maintains diversity (ESS > 150) at cost of accuracy. Rejuvenation with N=1000 achieves best balance.

## Outputs

Each experiment generates:
- `experiments/<exp_name>/frames/` - Per-timestep visualizations
- `experiments/<exp_name>/results.json` - Configuration and statistics
- `experiments/<exp_name>/stats.csv` - Time-series data (RMSE, ESS, variance)
- `experiments/summary.csv` - Cross-experiment comparison
- `experiments/comparison_*.png` - RMSE, ESS, variance plots

Visualizations show:
- Blue dots: particles (size ∝ weight)
- Red star: ground truth
- Green square: estimate
- Orange X: landmarks
- Green ellipse: 95% confidence

## Implementation Details

### Critical Bug Fixes

1. **Angle wrapping**: `arctan2(sin(θ), cos(θ))` instead of modulo
2. **Systematic resample**: Bounds checking `indices = min(indices, N-1)`
3. **Residual resample**: Handles zero-sum residuals
4. **Weighted covariance**: Outer product instead of element-wise
5. **Initialization**: θ ∈ [-π, π] matching wrap_angle
6. **Measurements**: Clamped to non-negative
7. **Plot scaling**: Dynamic limits with fixed margins
8. **ESS tracking**: Monitors when ESS < N/3

### Measurement Noise-ESS Trade-off

Low σ_m creates sharp likelihoods → weight concentration → ESS collapse. High σ_m broadens likelihoods → maintains diversity → larger RMSE.

**Recommendation**: Overestimate σ_m by 2-4× to maintain ESS > 50.

## Performance

- Full suite: ~3 minutes (M1 Mac)
- Single experiment (N=300): ~15 seconds
- Quick test: ~5 seconds
- Bottleneck: PNG frame generation

## References

- Thrun, S., Burgard, W., & Fox, D. (2005). *Probabilistic Robotics*. MIT Press.
- Arulampalam, M. S., et al. (2002). "A tutorial on particle filters." *IEEE Trans. Signal Processing*.

## Authors

Atharva Date B22AI045  
Samay Mehar B22AI048  
Gouri Patidar B22AI020  
IIT Jodhpur

