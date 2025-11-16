# Technical Methodology

**Authors: Prof. Edson Pindza et al.**

This document explains the technical details of the hybrid ML-PDE framework.

---

## Table of Contents

1. [Overview](#overview)
2. [PDE Solvers](#pde-solvers)
3. [Neural Network Architecture](#neural-network-architecture)
4. [Training Procedure](#training-procedure)
5. [Hybrid Blending](#hybrid-blending)
6. [Option Types](#option-types)

---

## Overview

The framework combines three key components:

1. **PDE Solvers**: Crank–Nicolson finite difference for reference pricing
2. **Neural Networks**: Compact feedforward surrogates for fast evaluation
3. **Hybrid Blending**: Damped Gaussian weights to combine predictions

### Key Innovation: PDE Anchoring (Not PINN Residuals)

Unlike Physics-Informed Neural Networks (PINNs) that minimize PDE residuals at collocation points, our approach uses **supervised learning with PDE anchoring**:

```
Loss = MSE(prediction, labels) + λ_aux * MSE(prediction, CN_values) + λ_wd * ||θ||²
       ^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^   ^^^^^^^^^^^^^^^
       Data fidelity              PDE anchoring (auxiliary MSE)     Weight decay
```

**Why this works:**
- Simpler than PINNs (no automatic differentiation through PDE)
- More stable training (supervised labels provide strong gradients)
- Flexible (can use different labels per option type: analytic, binomial, MC, PDE)

---

## PDE Solvers

### Crank–Nicolson for European Calls

**Governing PDE:**
```
∂V/∂t + (1/2)σ²S² ∂²V/∂S² + rS ∂V/∂S - rV = 0
V(S,T) = max(S - K, 0)
```

**Discretization:**
- Spatial grid: `S ∈ [0, S_max]` with `n_space + 1` points
- Temporal grid: `t ∈ [0, T]` with `n_time + 1` steps
- Scheme: Crank–Nicolson (implicit, second-order accurate)

**Key Implementation Details:**

1. **Spatial Scaling**: Coefficients scaled by `S_i` at each interior node
   ```python
   a = 0.25 * dt * (σ²S² / dS² - rS / dS)
   b = -0.5 * dt * (σ²S² / dS² + r)
   c = 0.25 * dt * (σ²S² / dS² + rS / dS)
   ```

2. **Rannacher Smoothing**: Two half-steps of backward Euler at maturity to handle payoff kink
   ```python
   dt_half = 0.5 * dt
   # Step 1: T - dt_half
   # Step 2: T - dt
   # Then switch to Crank–Nicolson
   ```

3. **Boundary Conditions**:
   - Low boundary: `V(0, t) = 0`
   - High boundary: `V(S_max, t) = S_max - K*exp(-r*(T-t))`

4. **Thomas Algorithm**: Tridiagonal system solver (O(n) complexity)

**Function:**
```python
def crank_nicolson_european_call(strike, rate, vol, maturity, s_max, n_space, n_time):
    # Returns (grid, values)
```

### Binomial Tree for American Puts

**Algorithm:**
```
1. Build asset price tree: S_i,j = S_0 * u^(j-i) * d^i
2. Initialize terminal payoff: V_N,i = max(K - S_N,i, 0)
3. Backward recursion:
   V_n,i = max(payoff, e^(-r*dt) * (p*V_{n+1,i} + (1-p)*V_{n+1,i+1}))
```

**Function:**
```python
def american_put_binomial(S0, strike, rate, vol, maturity, steps):
    # Returns option value at S0
```

---

## Neural Network Architecture

### FeedForwardRegressor Class

**Structure:**
```
Input → Normalization → Hidden Layer 1 → Activation → ... → Hidden Layer L → Output
```

**Components:**

1. **Input Normalization**:
   ```python
   x_norm = (x - x_mean) / x_std
   ```
   Crucial for stable training!

2. **Hidden Layers**:
   - Typical configuration: `(64, 64)` or `(128, 128)`
   - Activations: tanh, ReLU, leaky_relu, softplus, RBF

3. **Output Layer**:
   - Single neuron (regression)
   - No activation (linear output)

**Parameter Count:**
```
For input_dim=1, hidden_layers=(64,64):
  Layer 1: 1*64 + 64 = 128
  Layer 2: 64*64 + 64 = 4160
  Output: 64*1 + 1 = 65
  Total: 4353 parameters
```

**Forward Pass:**
```python
def forward(self, x):
    x = (x - self.x_mean) / self.x_std
    for W, b, act in zip(self.weights, self.biases, self.activations):
        x = act(x @ W + b)
    return x  # Linear output
```

---

## Training Procedure

### Loss Function

**Full Objective:**
```
L(θ) = L_data + λ_aux * L_anchor + λ_wd * ||θ||²
```

**Components:**

1. **Data Loss** (primary):
   ```
   L_data = (1/N) Σ ||u_θ(x_i) - y_i||²
   ```
   Where `y_i` are:
   - European: Black–Scholes analytic
   - American: Binomial tree values
   - Barrier: Crank–Nicolson grid
   - Basket/Asian: Monte Carlo estimates

2. **Anchor Loss** (auxiliary):
   ```
   L_anchor = (1/M) Σ ||u_θ(x̃_j) - V_CN(x̃_j)||²
   ```
   Where `V_CN` are Crank–Nicolson values at the same inputs

3. **Weight Decay** (regularization):
   ```
   ||θ||² = Σ ||W||_F² + Σ ||b||²
   ```

### Gradient Descent

**Full-Batch Updates:**
```python
for epoch in range(epochs):
    # Forward pass
    pred = forward(X_train)
    
    # Compute losses
    data_loss = mse(pred, y_train)
    anchor_loss = mse(forward(X_aux), y_aux)
    wd_loss = sum(||W||² for W in weights)
    
    total_loss = data_loss + λ_aux * anchor_loss + λ_wd * wd_loss
    
    # Backward pass
    grads = compute_gradients(total_loss)
    
    # Update
    θ ← θ - η * grads
```

**Why Full-Batch?**
- Stable convergence (no stochastic noise)
- Datasets are small (~100–1000 samples)
- CPU-friendly (no GPU mini-batch overhead)

**Typical Hyperparameters:**
```python
lr = 1e-3          # Learning rate
epochs = 5000      # Training iterations
λ_aux = 0.001      # PDE anchor weight
λ_wd = 1e-4        # Weight decay
```

---

## Hybrid Blending

### Motivation

Neural networks can extrapolate poorly near boundaries. PDE solvers are reliable but slow. **Solution:** Blend them!

### Blending Formula

```
V_hyb(S) = w_eff(S) * V_NN(S) + (1 - w_eff(S)) * V_CN(S)
```

Where:
```
w_eff(S) = 0.1 * exp(-((S - S_c) / (0.5 * span))²)
S_c = (S_min + S_max) / 2
span = max(S_max - S_c, 1)
```

**Key Feature:** Damping factor `0.1`

### Interpretation

```
Near boundaries (S → 0 or S → S_max):
  w_eff → 0  ⇒  V_hyb → V_CN (PDE dominates)

At-the-money (S ≈ S_c):
  w_eff ≈ 0.1  ⇒  V_hyb ≈ 0.1*V_NN + 0.9*V_CN (PDE still dominates!)
```

**Why 0.1?**
- Prevents NN from dominating even at the money
- Keeps hybrid stable under extrapolation
- Empirically determined through experiments

**Function:**
```python
def blend_predictions(s_values, nn_values, pde_values):
    s_c = (s_values.min() + s_values.max()) / 2
    span = max(s_values.max() - s_c, 1.0)
    weights = np.exp(-((s_values - s_c) / (0.5 * span))**2)
    effective = 0.1 * weights
    return effective * nn_values + (1 - effective) * pde_values
```

---

## Option Types

### European Call Options

**Label Generation:**
```python
def black_scholes_call_price(S, K, T, r, sigma):
    d1 = (log(S/K) + (r + 0.5*sigma**2)*T) / (sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)
    return S*norm_cdf(d1) - K*exp(-r*T)*norm_cdf(d2)
```

**Training:**
- Labels: Black–Scholes analytic
- Anchors: Crank–Nicolson values

### American Put Options

**Label Generation:**
- Binomial tree with `steps=1000`

**Training:**
- Labels: Binomial values
- Anchors: Crank–Nicolson values (no early exercise)

### Barrier Options (Up-and-Out Call)

**PDE Grid:**
- Domain: `S ∈ [0, B]` where `B` is the barrier
- Absorbing boundary at `S = B`: `V(B, t) = 0`

**Training:**
- Labels: PDE grid values
- Anchors: Same PDE grid
- **Special:** Adaptive oversampling near barrier (±2.5% of B)

### Asian Options

**Monte Carlo:**
```python
paths = simulate_GBM(S0, r, sigma, T, steps, n_paths)
running_avg = np.cumsum(paths, axis=1) / np.arange(1, steps+1)
payoff = max(running_avg[-1] - K, 0)
price = exp(-r*T) * mean(payoff)
```

**Training:**
- Labels: Monte Carlo estimates
- Anchors: None (MC is the reference)

### Basket Options

**Dimensionality:** `d ∈ {5, 10, 20, 30}` assets

**Monte Carlo:**
```python
# Correlated GBM
S = S0 * exp((r - 0.5*sigma²)*T + sigma*sqrt(T)*Z)
Z ~ MVN(0, Σ) with Σ_ij = ρ
basket_value = mean(S) - K
```

**Training:**
- Labels: Monte Carlo estimates
- Network input: concatenated asset prices `[S1, S2, ..., Sd]`

---

## Computational Complexity

### PDE Solver

**Crank–Nicolson:**
- Spatial grid: O(N_S)
- Temporal steps: O(N_T)
- Tridiagonal solve per step: O(N_S)
- **Total**: O(N_S * N_T)

**Typical:** N_S = 320, N_T = 320 ⇒ ~100K operations

### Neural Network

**Training:**
- Forward pass: O(L * W²) per epoch
- Backward pass: O(L * W²) per epoch
- **Total**: O(epochs * L * W²)

**Typical:** epochs=5000, L=2, W=64 ⇒ ~40M operations

**Inference:**
- Forward pass only: O(L * W²)
- **Single evaluation**: ~4K operations

### Speedup

For repeated evaluations (e.g., risk analysis):
- PDE: 100K operations per evaluation
- NN: 4K operations per evaluation
- **Speedup**: ~25×

---

## Validation

### Convergence Checks

1. **PDE Convergence:**
   - Refine grid: N_S, N_T → 2N_S, 2N_T
   - Measure: ||V_fine - V_coarse||
   - Expected: O(ΔS² + Δt²)

2. **NN Convergence:**
   - Increase width: W → 2W
   - Measure: ||V_NN - V_ref||
   - Expected: Sublinear decay

### Accuracy Metrics

```python
# Relative L2 error
rel_L2 = ||V_NN - V_ref||_2 / ||V_ref||_2

# Mean Absolute Error
MAE = mean(|V_NN - V_ref|)

# Maximum Error
Max_err = max(|V_NN - V_ref|)
```

**Typical Results:**
- European: rel_L2 < 1e-2
- American: rel_L2 < 2e-2
- Barrier: rel_L2 < 5e-2 (near-barrier challenging)

---

## References

For full details, see the manuscript:

> "Hybrid Machine Learning and Partial Differential Equation Framework for Modern Option Pricing"  
> Pindza, E., Owolabi, K. M., and Mare, E. (2025)

---

**For usage instructions, see USAGE.md**  
**For code organization, see README.md**
