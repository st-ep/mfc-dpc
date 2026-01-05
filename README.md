# Matryoshka Function Encoder + DPC

Matryoshka Function Encoder combined with Differentiable Predictive Control for the Van der Pol oscillator.

## Key Finding

**MFE+DPC outperforms standard FE+DPC by ~17% overall and 3-5x on in-distribution trajectories.**

The Matryoshka nested training loss creates better-structured basis functions that improve policy generalization, without requiring coarse-to-fine training schedules.

## Method

### Function Encoder (FE)
Learns neural network basis functions g_i(x) such that dynamics can be approximated as:
```
dx/dt ≈ Σ c_i * g_i(x)
```
Coefficients c_i are computed via ridge regression from observations.

### Matryoshka Function Encoder (MFE)
Same architecture as FE, but trained with nested reconstruction loss at multiple truncation levels k={2, 4, 8, 16}:
```
L_MFE = mean(L_k2, L_k4, L_k8, L_k16)
```
This forces basis functions to be importance-ordered.

### DPC Policy
Neural network policy that maps (state, reference, FE coefficients) → control action.
Trained via differentiable simulation through the dynamics.

## Results

| μ | FE+DPC | MFE+DPC | Winner |
|---|--------|---------|--------|
| 0.5 | 0.998 | **0.626** | MFE |
| 1.0 | 1.014 | **0.638** | MFE |
| 1.5 | 1.228 | **0.843** | MFE |
| 2.0 | 1.286 | **1.002** | MFE |
| 2.5 | 1.516 | **1.244** | MFE |
| 3.0 (OOD) | 1.529 | **1.408** | MFE |
| 3.5 (OOD) | 1.587 | **1.521** | MFE |
| 4.0 (OOD) | **1.589** | 1.606 | FE |

**Overall Average**: FE=1.34, MFE=**1.11** (17% improvement)
**OOD Average**: FE=1.57, MFE=**1.51** (4% improvement)

### Trajectory Analysis (100-step horizon)

| μ | FE err | MFE err | Improvement |
|---|--------|---------|-------------|
| 0.5 | 0.56 | **0.15** | 3.7x |
| 1.0 | 0.52 | **0.10** | 5.2x |
| 1.5 | 1.16 | **0.42** | 2.8x |
| 2.0 | 1.78 | **0.60** | 3.0x |
| 2.5 | 1.80 | **1.21** | 1.5x |
| 3.0 (OOD) | 1.01 | 1.05 | ~same |

## Why MFE Works

The nested training loss creates basis functions with much better truncation properties:
- At k=2: MFE reconstruction error is **22x better** than FE
- At k=4: MFE reconstruction error is **21x better** than FE
- At k=8: MFE reconstruction error is **50x better** than FE

Even though we use full k=16 coefficients at evaluation, the MFE training acts as implicit regularization, creating more structured representations that help policy generalization.

## Installation

Using uv:
```bash
uv sync
uv run python mfe_dpc.py
```

Using pip:
```bash
pip install -r requirements.txt
python mfe_dpc.py
```

## Output

Running the script produces:
- `mfe_dpc_results.png`: Training curves and performance comparison
- `mfe_trajectories.png`: Detailed trajectory comparisons

## Van der Pol Oscillator

The Van der Pol oscillator is a nonlinear dynamical system:
```
dx1/dt = x2
dx2/dt = μ(1 - x1²)x2 - x1
```

The parameter μ controls the nonlinearity strength. We train on μ ∈ [0.5, 2.5] and test generalization on μ up to 4.0 (out-of-distribution).

## Citation

Based on:
- Function Encoders: Ingebrand et al. (2024). "Zero-Shot Transfer of Neural ODEs." NeurIPS.
- Matryoshka RL: Kusupati et al. (2022). "Matryoshka Representation Learning." NeurIPS.
- DPC: Iqbal et al. (2025). "Zero-Shot Function Encoder-Based Differentiable Predictive Control."
