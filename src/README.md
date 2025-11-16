# Source Code Directory

**Authors: Prof. Edson Pindza et al.**

This directory contains all Python scripts for reproducing the manuscript figures.

---

## Main Script

### `reproduce_figures.py` (58 KB)
**Purpose:** Orchestrates generation of all 11 manuscript figures  
**Usage:** `python reproduce_figures.py`  
**Output:** All figures saved to `../figures/`  
**Runtime:** ~10-15 minutes on modern CPU

**Key Components:**
- PDE solvers (Crank–Nicolson, binomial)
- Neural network class (`FeedForwardRegressor`)
- Data generation (Black–Scholes, Monte Carlo)
- Hybrid blending function
- Plotting utilities

---

## Individual Figure Scripts

Each script generates one figure independently for faster iteration.

### Figure Generation Scripts

| Script | Output | Figure # | Description |
|--------|--------|----------|-------------|
| `fig01_architecture.py` | `fig01_hybrid_architecture.png` | 1 | Workflow diagram |
| `fig02_price_comparison.py` | `a.png` | 2 | European call & American put |
| `fig03_training_history.py` | `a3.png` | 3 | Training convergence |
| `fig04_asian_surface.py` | `a1.png` | 4 | Asian option 2D surface |
| `fig05_barrier_surfaces.py` | `a2.png` | 5 | Barrier option surfaces |
| `fig06_basket_scaling.py` | `a4.png`, `a4a.png` | 6 | Basket scalability |
| `fig07_convergence.py` | `a5a.png`, `a5b.png` | 7 | Numerical convergence |
| `fig08_benchmark_comparison.py` | `a6.png` | 8 | Method comparison |
| `fig09_error_time_tradeoff.py` | `a7.png` | 9 | Error-time tradeoff |
| `fig10_pricing_error_vs_strike.py` | `a7a.png` | 10 | Error vs strike |
| `fig11_robustness.py` | `a7b.png` | 11 | Robustness analysis |

### Utility Scripts

| Script | Purpose |
|--------|---------|
| `compute_metrics_path_dependents.py` | Metrics for barrier/Asian options |

---

## Usage Examples

### Generate All Figures

```bash
cd src
python reproduce_figures.py
```

### Generate Single Figure

```bash
cd src
python fig02_price_comparison.py
```

### Custom Parameter Testing

Edit `reproduce_figures.py` to modify:

```python
# Grid resolution
n_space = 320  # Lower for faster testing
n_time = 320

# Neural network
hidden_layers = (64, 64)  # Smaller/larger as needed

# Training
epochs = 5000  # Fewer for quick tests

# Monte Carlo
paths = 400  # Fewer for faster generation
```

---

## Code Structure

### Main Components in `reproduce_figures.py`

1. **PDE Solvers**
   ```python
   crank_nicolson_european_call(strike, rate, vol, maturity, s_max, n_space, n_time)
   american_put_binomial(S0, strike, rate, vol, maturity, steps)
   ```

2. **Neural Network**
   ```python
   class FeedForwardRegressor:
       def __init__(self, input_dim, hidden_layers, lr, seed)
       def train(self, X, y, epochs, aux_inputs=None, aux_targets=None)
       def predict(self, X)
   ```

3. **Data Generation**
   ```python
   black_scholes_call_price(S, K, T, r, sigma)
   basket_option_dataset(dimension, n_samples, ...)
   ```

4. **Hybrid Blending**
   ```python
   blend_predictions(s_values, nn_values, pde_values)
   # Returns: w_eff * nn + (1 - w_eff) * pde
   ```

---

## Dependencies

All scripts require:
- Python 3.8+
- NumPy >= 1.20.0
- Matplotlib >= 3.3.0

Install via:
```bash
pip install -r ../requirements.txt
```

---

## Output

All figures are saved to `../figures/` with:
- High resolution (300 DPI)
- PNG format
- Publication-ready quality

---

## Customization

### Add New Activation Function

Edit `FeedForwardRegressor` class:

```python
def _get_activation(name: str):
    activations = {
        "tanh": np.tanh,
        "relu": lambda x: np.maximum(0, x),
        # Add your activation here
        "my_activation": lambda x: ...
    }
    return activations[name]
```

### Modify Blending Weight

Edit `blend_predictions()`:

```python
# Current: 0.1 damping
effective = 0.1 * weights

# Custom damping
effective = YOUR_FACTOR * weights
```

---

## Testing

### Quick Test (Fast)
```python
# Reduce computational load
n_space = 80
epochs = 1000
paths = 100
```

### Full Test (Accurate)
```python
# Use production settings
n_space = 320
epochs = 5000
paths = 400
```

---

## Common Issues

### Import Error
**Problem:** `ModuleNotFoundError: No module named 'numpy'`  
**Solution:**
```bash
pip install -r ../requirements.txt
```

### Path Error
**Problem:** `FileNotFoundError: [Errno 2] No such file or directory: '../figures'`  
**Solution:** Run from `src/` directory:
```bash
cd src
python reproduce_figures.py
```

### Memory Error
**Problem:** `MemoryError` during basket/Asian options  
**Solution:** Reduce parameters:
```python
n_samples = 36  # Down from 60
paths = 200     # Down from 400
```

---

## Performance

### Typical Runtimes (Apple M4 Max, 16-core CPU)

| Script | Runtime | Bottleneck |
|--------|---------|------------|
| `reproduce_figures.py` | 10-15 min | Basket Monte Carlo |
| `fig02_price_comparison.py` | ~30 sec | PDE grids |
| `fig06_basket_scaling.py` | ~5 min | Monte Carlo labels |
| `fig07_convergence.py` | ~2 min | Multiple grid refinements |

**Speedup Tips:**
- Lower PDE grid resolution
- Fewer Monte Carlo paths
- Fewer training epochs for quick tests

---

## Contributing

To add a new figure:

1. Create `figXX_description.py`
2. Import from `reproduce_figures.py`:
   ```python
   from reproduce_figures import (
       crank_nicolson_european_call,
       FeedForwardRegressor,
       blend_predictions,
       ...
   )
   ```
3. Save output to `../figures/`
4. Update this README

---

## Citation

If you use or modify this code:

```bibtex
@article{pindza2025hybrid,
  title={Hybrid Machine Learning and Partial Differential Equation Framework for Modern Option Pricing},
  author={Pindza, Edson and Owolabi, Kolade M. and Mare, Eben},
  year={2025}
}
```

---

**For detailed methodology, see `../docs/METHODS.md`**  
**For usage instructions, see `../docs/USAGE.md`**
