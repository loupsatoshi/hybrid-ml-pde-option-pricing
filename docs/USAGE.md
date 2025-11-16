# Detailed Usage Guide

**Authors: Prof. Edson Pindza et al.**

This guide provides detailed instructions for using the codebase.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Running Individual Scripts](#running-individual-scripts)
3. [Understanding the Output](#understanding-the-output)
4. [Customizing Parameters](#customizing-parameters)
5. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Step 1: Installation

```bash
# Clone repository
git clone https://github.com/loupsatoshi/hybrid-ml-pde-option-pricing.git
cd hybrid-ml-pde-option-pricing

# Create virtual environment
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### Step 2: Generate All Figures

```bash
cd src
python reproduce_figures.py
```

**Expected Output:**
```
=== Reproducing manuscript figures ===
Computing PDE reference (European call, 320x320 grid)... done in 0.42s
Computing PDE reference (American put, 320x320 grid)... done in 0.38s
Computing binomial reference (American put, 1000 steps)... done in 0.28s
Training European call network (5000 epochs)... done in 1.24s
Training American put network (5000 epochs)... done in 1.31s
...
All figures saved to ../figures/
```

**Runtime:** ~10–15 minutes on a modern CPU

---

## Running Individual Scripts

Each figure can be generated separately for faster iteration.

### Figure 1: Workflow Diagram

```bash
cd src
python fig01_architecture.py
```

**Output:** `../figures/fig01_hybrid_architecture.png`  
**Purpose:** Conceptual diagram of the hybrid ML-PDE pipeline

### Figure 2: European Call & American Put Pricing

```bash
python fig02_price_comparison.py
```

**Output:** `../figures/a.png`  
**Purpose:** Compare analytical, PDE, hybrid, and binomial methods

### Figure 3: Training History

```bash
python fig03_training_history.py
```

**Output:** `../figures/a3.png`  
**Purpose:** Show convergence of data loss and PDE anchor loss

### Figure 4: Asian Option Surface

```bash
python fig04_asian_surface.py
```

**Output:** `../figures/a1.png`  
**Purpose:** 2D surface for Asian option pricing

### Figure 5: Barrier Option Surfaces

```bash
python fig05_barrier_surfaces.py
```

**Output:** `../figures/a2.png`  
**Purpose:** PDE vs ML surrogates for barrier options

### Figure 6: Basket Scalability

```bash
python fig06_basket_scaling.py
```

**Output:** `../figures/a4.png` and `a4a.png`  
**Purpose:** NRMSE, timing, and parity plots for d=5,10,20,30

### Figure 7: Numerical Convergence

```bash
python fig07_convergence.py
```

**Output:** `../figures/a5a.png` and `a5b.png`  
**Purpose:** Log-log convergence analysis

### Figure 8: Benchmark Comparison

```bash
python fig08_benchmark_comparison.py
```

**Output:** `../figures/a6.png`  
**Purpose:** Compare FDM, LSM, and hybrid methods

### Figure 9: Error-Time Tradeoff

```bash
python fig09_error_time_tradeoff.py
```

**Output:** `../figures/a7.png`  
**Purpose:** Pareto frontier for accuracy vs speed

### Figure 10: Pricing Error vs Strike

```bash
python fig10_pricing_error_vs_strike.py
```

**Output:** `../figures/a7a.png`  
**Purpose:** Error analysis across strikes

### Figure 11: Robustness Analysis

```bash
python fig11_robustness.py
```

**Output:** `../figures/a7b.png`  
**Purpose:** Sensitivity to parameter changes

---

## Understanding the Output

### Figures Directory Structure

After running the scripts, `figures/` contains:

```
figures/
├── fig01_hybrid_architecture.png  # Figure 1
├── a.png                           # Figure 2
├── a3.png                          # Figure 3
├── a1.png                          # Figure 4
├── a2.png                          # Figure 5
├── a4.png                          # Figure 6 (top panels)
├── a4a.png                         # Figure 6 (bottom panels)
├── a5a.png                         # Figure 7 (left)
├── a5b.png                         # Figure 7 (right)
├── a6.png                          # Figure 8
├── a7.png                          # Figure 9
├── a7a.png                         # Figure 10
└── a7b.png                         # Figure 11
```

### Console Output

The scripts provide real-time progress updates:

```
Computing PDE reference (European call, 320x320 grid)... done in 0.42s
Training European call network (5000 epochs)... done in 1.24s
```

**Key metrics reported:**
- **PDE grid resolution**: e.g., `320x320` (spatial × temporal)
- **Training epochs**: typically `5000` for convergence
- **Timing**: wall-clock seconds for each step

---

## Customizing Parameters

### Modify Grid Resolution

Edit `src/reproduce_figures.py`:

```python
# Original (high resolution)
n_space = 320
n_time = 320

# Faster (lower resolution)
n_space = 160
n_time = 160
```

### Adjust Neural Network Size

```python
# Original
net = FeedForwardRegressor(input_dim=1, hidden_layers=(64, 64), lr=1e-3)

# Smaller/faster
net = FeedForwardRegressor(input_dim=1, hidden_layers=(32, 32), lr=1e-3)

# Larger/more accurate
net = FeedForwardRegressor(input_dim=1, hidden_layers=(128, 128, 64), lr=1e-3)
```

### Change Training Epochs

```python
# Original
net.train(X_train, y_train, epochs=5000)

# Faster testing
net.train(X_train, y_train, epochs=1000)
```

### Basket Option Dimensionality

```python
# Original
dims = [5, 10, 20, 30]

# Faster (fewer dimensions)
dims = [5, 10]
```

---

## Troubleshooting

### Issue: Script crashes with "Out of Memory"

**Solution:** Reduce batch sizes or grid resolution:

```python
# For basket options
n_samples = 36  # Down from 60
paths = 400     # Down from 800

# For PDE grids
n_space = 160   # Down from 320
```

### Issue: Figures look different from paper

**Solution:** Check random seed consistency:

```python
rng = np.random.default_rng(7)  # Must match paper
```

### Issue: Slow execution

**Solution:** 
1. Use fewer Monte Carlo paths for basket/Asian options
2. Reduce training epochs to 1000 for quick tests
3. Lower PDE grid resolution

### Issue: Import errors

**Solution:**

```bash
# Ensure you're in the virtual environment
source .venv/bin/activate

# Reinstall dependencies
pip install --upgrade -r requirements.txt
```

### Issue: Path not found errors

**Solution:** Scripts assume you're running from `src/` directory:

```bash
cd src
python reproduce_figures.py  # ✓ Correct
```

Not:
```bash
python src/reproduce_figures.py  # ✗ Will fail
```

---

## Advanced Usage

### Running on a Cluster

For HPC environments:

```bash
#!/bin/bash
#SBATCH --job-name=option_pricing
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=01:00:00

module load python/3.8
source .venv/bin/activate
cd src
python reproduce_figures.py
```

### Parallel Execution

Individual figures can be run in parallel:

```bash
cd src
python fig02_price_comparison.py &
python fig04_asian_surface.py &
python fig05_barrier_surfaces.py &
wait
```

---

## Getting Help

If you encounter issues:

1. Check this guide's Troubleshooting section
2. Review the code comments in `src/reproduce_figures.py`
3. Contact the authors (see README.md)

---

**For methodology details, see METHODS.md**  
**For code organization, see README.md**
