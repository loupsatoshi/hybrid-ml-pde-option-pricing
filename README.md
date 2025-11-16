# Hybrid Machine Learning and Partial Differential Equation Framework for Modern Option Pricing

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Authors

**Prof. Edson Pindza et al.**
- Edson Pindza, University of South Africa (UNISA)
- Kolade M. Owolabi, Federal University of Technology Akure & Sefako Makgatho Health Sciences University
- Eben Mare, University of Pretoria

---

## Overview

This repository contains the complete implementation and reproducibility code for the paper:

> **"Hybrid Machine Learning and Partial Differential Equation Framework for Modern Option Pricing"**

The codebase demonstrates a novel approach combining:
- **Supervised neural network training** with PDE anchoring (not PINN residuals)
- **Crank–Nicolson finite difference solvers** for reference pricing
- **Damped Gaussian blending** for robust hybrid predictions
- **Full-batch CPU training** using NumPy (no GPU required)

### Key Features

✅ **Comprehensive option pricing**: European, American, barrier, Asian, basket options  
✅ **Pure NumPy implementation**: No TensorFlow/PyTorch dependencies  
✅ **Reproducible results**: All figures from the manuscript can be regenerated  
✅ **Well-documented code**: Clear docstrings and inline comments  
✅ **Fast execution**: All experiments run in minutes on CPU  

---

## Repository Structure

```
├── README.md                          # This file
├── requirements.txt                   # Python dependencies
├── LICENSE                            # MIT License
├── CITATION.cff                       # Citation metadata
├── .gitignore                         # Git ignore rules
│
├── src/                               # Source code
│   ├── reproduce_figures.py          # Main script (generates all figures)
│   ├── fig01_architecture.py         # Individual figure scripts
│   ├── fig02_price_comparison.py
│   ├── fig03_training_history.py
│   ├── fig04_asian_surface.py
│   ├── fig05_barrier_surfaces.py
│   ├── fig06_basket_scaling.py
│   ├── fig07_convergence.py
│   ├── fig08_benchmark_comparison.py
│   ├── fig09_error_time_tradeoff.py
│   ├── fig10_pricing_error_vs_strike.py
│   ├── fig11_robustness.py
│   └── compute_metrics_path_dependents.py
│
├── manuscript/                        # LaTeX manuscript
│   ├── manuscript.tex                 # Main LaTeX file
│   └── manuscript.pdf                 # Compiled PDF
│
├── figures/                           # Generated figures (output)
│   └── (auto-generated .png files)
│
└── docs/                              # Additional documentation
    ├── USAGE.md                       # Detailed usage guide
    └── METHODS.md                     # Technical methodology
```

---

## Installation

### Prerequisites

- Python 3.8 or higher
- pip package manager
- (Optional) Virtual environment tool

### Quick Start

1. **Clone the repository**
```bash
git clone https://github.com/loupsatoshi/hybrid-ml-pde-option-pricing.git
cd hybrid-ml-pde-option-pricing
```

2. **Create a virtual environment** (recommended)
```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

---

## Usage

### Generate All Figures (One Command)

To reproduce all figures from the manuscript:

```bash
cd src
python reproduce_figures.py
```

**Output**: All figures saved to `../figures/` directory  
**Runtime**: ~10–15 minutes on a modern CPU  

### Generate Individual Figures

Each figure can be generated separately:

```bash
cd src
python fig01_architecture.py    # Figure 1: Workflow diagram
python fig02_price_comparison.py # Figure 2: European/American pricing
python fig03_training_history.py # Figure 3: Training convergence
# ... and so on
```

### Hardware Used

All experiments were conducted on:
- **CPU**: Apple M4 Max (16 cores)
- **Memory**: 48 GB unified
- **Implementation**: CPU-only, no GPU/Neural Engine used

---

## Key Components

### 1. **PDE Solvers**
- `crank_nicolson_european_call()`: Crank–Nicolson for European calls with Rannacher smoothing
- `american_put_binomial()`: Binomial tree for American puts
- `barrier_pde_surface()`: PDE solver for barrier options

### 2. **Neural Network**
- `FeedForwardRegressor`: Compact neural surrogate (2–3 hidden layers)
- Activation functions: tanh, ReLU, leaky_relu, softplus, RBF
- Training: Full-batch gradient descent with optional PDE anchoring

### 3. **Hybrid Blending**
- `blend_predictions()`: Damped Gaussian blend
- Weight formula: `w_eff(S) = 0.1 * exp(-((S-S_c)/(0.5*span))^2)`
- Combines neural network and PDE values for robust extrapolation

### 4. **Data Generation**
- `basket_option_dataset()`: Monte Carlo for basket options
- Black–Scholes analytic formulas for European options
- Binomial trees for American options

---

## Figures Overview

| Figure | Description | Script |
|--------|-------------|--------|
| **1** | Hybrid ML-PDE workflow diagram | `fig01_architecture.py` |
| **2** | European call & American put pricing | `fig02_price_comparison.py` |
| **3** | Training history (data + anchor loss) | `fig03_training_history.py` |
| **4** | Asian option 2D surface | `fig04_asian_surface.py` |
| **5** | Barrier option surfaces (PDE vs ML) | `fig05_barrier_surfaces.py` |
| **6** | Basket scalability (NRMSE, timing) | `fig06_basket_scaling.py` |
| **7** | Numerical convergence analysis | `fig07_convergence.py` |
| **8** | Benchmark comparison | `fig08_benchmark_comparison.py` |
| **9** | Error-time tradeoff | `fig09_error_time_tradeoff.py` |
| **10** | Pricing error vs strike | `fig10_pricing_error_vs_strike.py` |
| **11** | Robustness to parameter changes | `fig11_robustness.py` |

---

## Methodology Summary

### Training Pipeline
1. **Label Generation**: Use analytic/binomial/MC/PDE references depending on option type
2. **PDE Anchoring**: Compute Crank–Nicolson values at same inputs for auxiliary loss
3. **Full-Batch Training**: Minimize `MSE(data) + λ_aux * MSE(anchor) + λ_wd * ||θ||²`
4. **Blending**: Combine neural output and PDE solution via damped Gaussian weights

### Key Differences from Standard Approaches
- ❌ **No PINN residuals**: We do not optimize PDE collocation loss
- ❌ **No reinforcement learning**: Purely supervised training
- ✅ **PDE anchoring**: Auxiliary MSE to Crank–Nicolson values
- ✅ **Damped blending**: 0.1 factor ensures PDE dominance near boundaries

---

## Citation

If you use this code in your research, please cite:

```bibtex
@article{pindza2025hybrid,
  title={Hybrid Machine Learning and Partial Differential Equation Framework for Modern Option Pricing},
  author={Pindza, Edson and Owolabi, Kolade M. and Mare, Eben},
  year={2025},
  note={Manuscript in preparation}
}
```

See `CITATION.cff` for machine-readable citation metadata.

---

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## Contributing

Contributions, issues, and feature requests are welcome! Feel free to:
1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## Contact

- **Edson Pindza**: edsonpindza@gmail.com
- **Kolade M. Owolabi**: kmowolabi@futa.edu.ng
- **Eben Mare**: eben.mare@up.ac.za

---

## Acknowledgments

- Implementation: Python 3 + NumPy + Matplotlib
- Hardware: Apple M4 Max (CPU-only execution)
- No external ML frameworks (TensorFlow/PyTorch) required

---

## Troubleshooting

### Common Issues

**Q: Figures look different from the paper**  
A: Ensure you're using the same random seed (default: `rng = np.random.default_rng(42)` in `reproduce_figures.py`)

**Q: Out of memory error**  
A: Reduce `n_samples`, `paths`, or `steps` in basket/Asian options. Current settings are optimized for 48GB RAM.

**Q: Slow execution**  
A: Most time is spent in Monte Carlo label generation. Use fewer paths or samples for faster testing.

---

## Version History

- **v1.0.0** (2025): Initial release with manuscript submission

---

**For detailed methodology, see the manuscript in `manuscript/manuscript.pdf`**
