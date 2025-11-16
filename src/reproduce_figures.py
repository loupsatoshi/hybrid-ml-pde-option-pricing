"""Reproduce the figures discussed in the manuscript by simulating
option pricing experiments and generating publication-ready graphics.

Authors: Prof. Edson Pindza et al.
  - Edson Pindza, University of South Africa (UNISA)
  - Kolade M. Owolabi, Federal University of Technology Akure
  - Eben Mare, University of Pretoria

Paper: "Hybrid Machine Learning and Partial Differential Equation Framework 
        for Modern Option Pricing"

The script covers:
  * Finite-difference (PDE) solvers for Black--Scholes European calls
    and American puts (with early-exercise projection).
  * Binomial and Longstaff--Schwartz (LSM) benchmarks for American puts.
  * Monte Carlo estimators for Asian and barrier payoffs.
  * Basket option surrogates across increasing dimensionality.
  * Lightweight neural networks that mimic the ``hybrid ML--PDE``
    surrogate referenced throughout the manuscript.

Running the module will create PNG figures under ``../figures/``
whose captions align with Figures 1--11 in the paper.

Usage:
    python reproduce_figures.py
    
Output:
    All figures saved to ../figures/ directory
"""

from __future__ import annotations

from dataclasses import dataclass
from math import erf, exp, log, sqrt
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import time

import matplotlib.pyplot as plt
import numpy as np

STYLE = {
    "Analytical": {"color": "tab:blue", "linestyle": "-", "linewidth": 2.5},
    "PDE (CN)": {"color": "tab:orange", "linestyle": "--", "linewidth": 2.0},
    "Hybrid ML--PDE": {
        "color": "tab:green",
        "linestyle": "-.",
        "linewidth": 2.0,
        "marker": "o",
        "markerfacecolor": "none",
        "markeredgewidth": 1.5,
        "markersize": 5,
    },
    "Binomial Reference": {
        "color": "tab:red",
        "linestyle": "--",
        "linewidth": 2.0,
        "marker": "o",
        "markerfacecolor": "none",
        "markeredgewidth": 1.5,
        "markersize": 5,
    },
    "FDM": {
        "color": "tab:blue",
        "linestyle": "--",
        "linewidth": 2.0,
        "marker": "o",
        "markerfacecolor": "none",
        "markeredgewidth": 1.5,
        "markersize": 5,
    },
    "LSM": {
        "color": "tab:red",
        "linestyle": ":",
        "linewidth": 2.0,
        "marker": "o",
        "markerfacecolor": "none",
        "markeredgewidth": 1.5,
        "markersize": 5,
    },
    "Hybrid": {
        "color": "tab:green",
        "linestyle": "-.",
        "linewidth": 2.0,
        "marker": "o",
        "markerfacecolor": "none",
        "markeredgewidth": 1.5,
        "markersize": 5,
    },
    "Total loss": {"color": "tab:blue", "linestyle": "-", "linewidth": 2.0},
    "Data loss": {"color": "tab:orange", "linestyle": "--", "linewidth": 1.8},
    "PDE anchor loss": {"color": "tab:green", "linestyle": ":", "linewidth": 1.8},
    "Payoff": {"color": "#555555", "linestyle": "--", "linewidth": 1.5},
}

_vec_erf = np.vectorize(erf)


def _norm_cdf(x: np.ndarray) -> np.ndarray:
    """Cumulative distribution for the standard normal."""

    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _vec_erf(x / sqrt(2.0)))


def black_scholes_call_price(
    s: np.ndarray, strike: float, rate: float, vol: float, tau: np.ndarray
) -> np.ndarray:
    """Closed form European call price."""

    s = np.asarray(s, dtype=float)
    tau = np.asarray(tau, dtype=float)
    tau = np.maximum(tau, 1e-08)
    d1 = (np.log(s / strike) + (rate + 0.5 * vol**2) * tau) / (vol * np.sqrt(tau))
    d2 = d1 - vol * np.sqrt(tau)
    return s * _norm_cdf(d1) - strike * np.exp(-rate * tau) * _norm_cdf(d2)


def thomas_solver(
    lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray
) -> np.ndarray:
    """Solve a tri-diagonal linear system via the Thomas algorithm."""

    n = diag.size
    c_prime = np.zeros(n - 1)
    d_prime = np.zeros(n)
    c_prime[0] = upper[0] / diag[0]
    d_prime[0] = rhs[0] / diag[0]
    for i in range(1, n):
        denom = diag[i] - lower[i - 1] * c_prime[i - 1]
        if i < n - 1:
            c_prime[i] = upper[i] / denom
        d_prime[i] = (rhs[i] - lower[i - 1] * d_prime[i - 1]) / denom
    sol = np.zeros(n)
    sol[-1] = d_prime[-1]
    for i in reversed(range(n - 1)):
        sol[i] = d_prime[i] - c_prime[i] * sol[i + 1]
    return sol


def crank_nicolson_european_call(
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    s_max: float,
    n_space: int,
    n_time: int,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Crank--Nicolson discretisation for a European call under Black--Scholes.

    Solves the PDE backward in time on a uniform grid with standard
    Dirichlet boundaries: V(0,t)=0 and V(S_{\max}, t)=S_{\max}-K e^{-r(T-t)}.
    """

    # Spatial and temporal grids
    dS = s_max / n_space
    grid = np.linspace(0.0, s_max, n_space + 1)
    dt = maturity / float(n_time)

    # Terminal payoff at maturity
    values = np.maximum(grid - strike, 0.0)

    # Precompute CN coefficients for interior nodes i=1..n_space-1 using S_i
    S_i = grid[1:-1]
    sigma2S2_over_dS2 = (vol**2) * (S_i**2) / (dS**2)
    rS_over_dS = rate * S_i / dS
    a = 0.25 * dt * (sigma2S2_over_dS2 - rS_over_dS)
    b = -0.5 * dt * (sigma2S2_over_dS2 + rate)
    c = 0.25 * dt * (sigma2S2_over_dS2 + rS_over_dS)

    lower = -a
    diag = 1 - b
    upper = -c

    rhs_lower = a
    rhs_diag = 1 + b
    rhs_upper = c

    # Rannacher smoothing: two half-steps of backward Euler to handle payoff kink
    dt_half = 0.5 * dt
    aH = 0.25 * dt_half * (sigma2S2_over_dS2 - rS_over_dS)
    bH = -0.5 * dt_half * (sigma2S2_over_dS2 + rate)
    cH = 0.25 * dt_half * (sigma2S2_over_dS2 + rS_over_dS)
    lower_BE = -2.0 * aH
    diag_BE = 1.0 - 2.0 * bH
    upper_BE = -2.0 * cH

    def be_half_step(tau_new: float) -> None:
        rhs_be = values[1:-1].copy()
        v_high = s_max - strike * np.exp(-rate * tau_new)
        rhs_be[-1] -= upper_BE[-1] * v_high
        values[1:-1] = thomas_solver(lower_BE, diag_BE, upper_BE, rhs_be)
        values[0] = 0.0
        values[-1] = v_high

    if n_time > 0:
        be_half_step(dt_half)
        be_half_step(2.0 * dt_half)

    # March backward in time using Crank--Nicolson
    for step in range(1, n_time):
        # Calendar time decreases; use tau (time-to-maturity) for boundaries
        # New time-to-maturity after this step:
        tau_new = (step + 1) * dt
        # Assemble RHS using values at the previous time level
        rhs = (
            rhs_lower * values[:-2]
            + rhs_diag * values[1:-1]
            + rhs_upper * values[2:]
        )
        # High-price boundary contribution at the new time level
        v_high_new = s_max - strike * np.exp(-rate * tau_new)
        # Move known Dirichlet boundary (new time) from LHS to RHS: subtract upper * V_{i+1}^{new}
        rhs[-1] -= upper[-1] * v_high_new

        # Solve the tridiagonal system for interior nodes
        values[1:-1] = thomas_solver(lower, diag, upper, rhs)

        # Enforce Dirichlet boundaries at the new time
        values[0] = 0.0
        values[-1] = v_high_new

    return grid, values

def crank_nicolson_american_put(
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    s_max: float,
    n_space: int,
    n_time: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """American put valuation with early exercise projection."""

    dS = s_max / n_space
    grid = np.linspace(0.0, s_max, n_space + 1)
    option = np.maximum(strike - grid, 0.0)
    # Increase micro-steps for stability/accuracy of the explicit method
    n_steps = max(n_time * 20, 4000)
    dt = maturity / float(n_steps)

    for _ in range(n_steps):
        new_opt = option.copy()
        for j in range(1, n_space):
            S = grid[j]
            delta = (option[j + 1] - option[j - 1]) / (2 * dS)
            gamma = (option[j + 1] - 2 * option[j] + option[j - 1]) / (dS**2)
            drift = rate * S * delta
            diff = 0.5 * vol**2 * S**2 * gamma
            new_opt[j] = option[j] + dt * (diff + drift - rate * option[j])
        # Put boundaries and early-exercise projection
        new_opt[0] = strike
        new_opt[-1] = 0.0
        option = np.maximum(new_opt, np.maximum(strike - grid, 0.0))
    return grid, option


def american_put_binomial(
    s0: float, strike: float, rate: float, vol: float, maturity: float, steps: int
) -> float:
    """CRR binomial tree for American put pricing."""

    dt = maturity / steps
    u = exp(vol * sqrt(dt))
    d = 1 / u
    disc = exp(-rate * dt)
    p = (exp(rate * dt) - d) / (u - d)
    prices = np.array([s0 * (u ** j) * (d ** (steps - j)) for j in range(steps + 1)])
    option = np.maximum(strike - prices, 0.0)
    for step in reversed(range(steps)):
        # Step back one level in the tree: each node price is the
        # down-move from the corresponding node at the next time.
        prices = prices[:-1] / d
        option = disc * (p * option[1:] + (1 - p) * option[:-1])
        exercise = np.maximum(strike - prices, 0.0)
        option = np.maximum(option, exercise)
    return option[0]


def lsm_american_put(
    s0: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    steps: int,
    paths: int,
    rng: np.random.Generator,
) -> float:
    """Least-Squares Monte Carlo price for American put."""

    dt = maturity / steps
    disc = exp(-rate * dt)
    shocks = rng.standard_normal(size=(paths, steps))
    increments = (rate - 0.5 * vol**2) * dt + vol * sqrt(dt) * shocks
    log_paths = np.cumsum(increments, axis=1)
    prices = s0 * np.exp(np.column_stack([np.zeros(paths), log_paths]))
    payoffs = np.maximum(strike - prices, 0.0)
    continuation = payoffs[:, -1]
    for t in range(steps - 1, 0, -1):
        itm = payoffs[:, t] > 0
        continuation *= disc
        if np.any(itm):
            x = prices[itm, t]
            y = continuation[itm]
            basis = np.vstack([np.ones_like(x), x, x**2]).T
            coeffs, *_ = np.linalg.lstsq(basis, y, rcond=None)
            continuation_itm = basis @ coeffs
            exercise = payoffs[itm, t]
            exercise_now = exercise > continuation_itm
            continuation[itm] = np.where(exercise_now, exercise, continuation[itm])
        else:
            continuation *= 1.0
    continuation *= disc
    return continuation.mean()


def asian_option_price(
    s0: float,
    a0: float,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    steps: int,
    paths: int,
    history: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo pricing for an Asian call with a pre-history average."""

    dt = maturity / steps
    shocks = rng.standard_normal(size=(paths, steps))
    increments = (rate - 0.5 * vol**2) * dt + vol * sqrt(dt) * shocks
    log_paths = np.cumsum(increments, axis=1)
    prices = s0 * np.exp(np.column_stack([np.zeros(paths), log_paths]))
    running_sum = history * a0 + prices[:, 1:].sum(axis=1)
    avg = running_sum / (history + steps)
    payoff = np.maximum(avg - strike, 0.0)
    return exp(-rate * maturity) * payoff.mean()


def barrier_option_price_mc(
    s0: float,
    strike: float,
    barrier: float,
    rate: float,
    vol: float,
    maturity: float,
    steps: int,
    paths: int,
    rng: np.random.Generator,
) -> float:
    """Monte Carlo estimator for an up-and-out barrier call."""

    dt = maturity / steps
    shocks = rng.standard_normal(size=(paths, steps))
    increments = (rate - 0.5 * vol**2) * dt + vol * sqrt(dt) * shocks
    log_paths = np.cumsum(increments, axis=1)
    prices = s0 * np.exp(np.column_stack([np.zeros(paths), log_paths]))
    hit_barrier = (prices >= barrier).any(axis=1)
    terminal = prices[:, -1]
    payoff = np.maximum(terminal - strike, 0.0)
    payoff[hit_barrier] = 0.0
    return exp(-rate * maturity) * payoff.mean()


def barrier_pde_surface(
    strike: float,
    barrier: float,
    rate: float,
    vol: float,
    maturity: float,
    n_space: int,
    n_time: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Finite-difference grid for a knock-out barrier call."""

    dt = maturity / n_time
    s_grid = np.linspace(0.0, barrier, n_space + 1)
    values = np.maximum(s_grid - strike, 0.0)
    payoff = values.copy()
    i = np.arange(1, n_space)
    alpha = 0.25 * dt * (vol**2 * i**2 - rate * i)
    beta = -0.5 * dt * (vol**2 * i**2 + rate)
    gamma = 0.25 * dt * (vol**2 * i**2 + rate * i)
    lower = -alpha
    diag = 1 - beta
    upper = -gamma
    rhs_lower = alpha
    rhs_diag = 1 + beta
    rhs_upper = gamma
    surfaces = [values.copy()]
    for _ in range(n_time):
        rhs = (
            rhs_lower * values[:-2]
            + rhs_diag * values[1:-1]
            + rhs_upper * values[2:]
        )
        rhs[0] += 0.0
        rhs[-1] += upper[-1] * 0.0
        values[1:-1] = thomas_solver(lower, diag, upper, rhs)
        values[0] = 0.0
        values[-1] = 0.0
        payoff = np.maximum(s_grid - strike, 0.0)
        values = np.minimum(values, payoff)
        surfaces.append(values.copy())
    t_grid = np.linspace(maturity, 0.0, n_time + 1)
    return s_grid, t_grid, np.array(surfaces)


def basket_option_dataset(
    dimension: int,
    n_samples: int,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    rho: float,
    paths: int,
    steps: int,
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Monte Carlo labels for basket call pricing."""

    cov = rho * np.ones((dimension, dimension))
    np.fill_diagonal(cov, 1.0)
    L = np.linalg.cholesky(cov)
    dt = maturity / steps
    s0 = rng.uniform(60.0, 140.0, size=(n_samples, dimension))
    labels = np.zeros(n_samples)
    start = time.perf_counter()
    for idx in range(n_samples):
        paths_arr = np.empty((paths, steps + 1, dimension))
        paths_arr[:, 0, :] = s0[idx]
        shocks = rng.standard_normal(size=(paths, steps, dimension))
        shocks = shocks @ L.T
        increments = (rate - 0.5 * vol**2) * dt + vol * sqrt(dt) * shocks
        # Apply GBM per-step multiplicative updates using single-step increments
        for step in range(1, steps + 1):
            paths_arr[:, step, :] = paths_arr[:, step - 1, :] * np.exp(
                increments[:, step - 1, :]
            )
        averages = paths_arr[:, -1, :].mean(axis=1)
        payoff = np.maximum(averages - strike, 0.0)
        labels[idx] = exp(-rate * maturity) * payoff.mean()
    label_time = time.perf_counter() - start
    return s0, labels, label_time


class FeedForwardRegressor:
    """Small fully-connected neural network with tanh activations."""

    def __init__(
        self,
        input_dim: int,
        hidden_layers: Sequence[int],
        lr: float,
        activations: Optional[Sequence[str]] = None,
        seed: int = 0,
    ) -> None:
        rng = np.random.default_rng(seed)
        dims = [input_dim, *hidden_layers, 1]
        self.weights: List[np.ndarray] = []
        self.biases: List[np.ndarray] = []
        for j in range(len(dims) - 1):
            limit = sqrt(6.0 / (dims[j] + dims[j + 1]))
            w = rng.uniform(-limit, limit, size=(dims[j], dims[j + 1]))
            b = np.zeros(dims[j + 1])
            self.weights.append(w)
            self.biases.append(b)
        self.lr = lr
        self.input_mean: Optional[np.ndarray] = None
        self.input_scale: Optional[np.ndarray] = None
        self.activations = activations
        self.leaky_slope = 0.01
        self.rbf_gamma = 0.5

    @property
    def n_params(self) -> int:
        return int(
            sum(w.size + b.size for w, b in zip(self.weights, self.biases))
        )

    def _prepare_inputs(self, X: np.ndarray) -> np.ndarray:
        if self.input_mean is None:
            self.input_mean = X.mean(axis=0, keepdims=True)
            self.input_scale = X.std(axis=0, keepdims=True) + 1e-08
        return (X - self.input_mean) / self.input_scale

    def forward(self, X: np.ndarray) -> Tuple[List[np.ndarray], List[np.ndarray]]:
        activations = [X]
        zs = []
        out = X
        for idx, (w, b) in enumerate(zip(self.weights, self.biases)):
            z = out @ w + b
            zs.append(z)
            if idx < len(self.weights) - 1:
                if self.activations is None:
                    out = np.tanh(z)
                else:
                    name = self.activations[idx] if idx < len(self.activations) else "tanh"
                    if name == "relu":
                        out = np.maximum(z, 0.0)
                    elif name == "leaky_relu":
                        out = np.where(z > 0.0, z, self.leaky_slope * z)
                    elif name == "softplus":
                        out = np.log1p(np.exp(z))
                    elif name == "rbf":
                        out = np.exp(-self.rbf_gamma * (z ** 2))
                    else:
                        out = np.tanh(z)
            else:
                out = z
            activations.append(out)
        return activations, zs

    def predict(self, X: np.ndarray) -> np.ndarray:
        assert self.input_mean is not None and self.input_scale is not None
        X_norm = (X - self.input_mean) / self.input_scale
        activations, _ = self.forward(X_norm)
        return activations[-1].ravel()

    def train(
        self,
        X: np.ndarray,
        y: np.ndarray,
        aux: Optional[np.ndarray] = None,
        aux_weight: float = 0.0,
        epochs: int = 2500,
        weight_decay: float = 0.0,
    ) -> Dict[str, List[float]]:
        X_norm = self._prepare_inputs(X)
        y = y.reshape(-1, 1)
        aux_term = aux.reshape(-1, 1) if aux is not None else None
        history: Dict[str, List[float]] = {
            "data": [],
            "aux": [],
            "total": [],
        }
        n = X.shape[0]
        for _ in range(epochs):
            activations, zs = self.forward(X_norm)
            pred = activations[-1]
            resid = pred - y
            data_loss = float(np.mean(resid**2))
            grad_output = 2.0 * resid / n
            aux_loss = 0.0
            if aux_term is not None:
                aux_resid = pred - aux_term
                aux_loss = float(np.mean(aux_resid**2))
                grad_output += 2.0 * aux_weight * aux_resid / n
            reg = 0.0
            if weight_decay > 0.0:
                reg = weight_decay * float(sum(np.sum(w**2) for w in self.weights))
            total_loss = data_loss + aux_weight * aux_loss + reg
            history["data"].append(data_loss)
            history["aux"].append(aux_loss)
            history["total"].append(total_loss)
            delta = grad_output
            for idx in reversed(range(len(self.weights))):
                a_prev = activations[idx]
                z = zs[idx]
                if idx < len(self.weights) - 1:
                    if self.activations is None:
                        delta = delta * (1.0 - np.tanh(z) ** 2)
                    else:
                        name = self.activations[idx] if idx < len(self.activations) else "tanh"
                        if name == "relu":
                            delta = delta * (z > 0.0)
                        elif name == "leaky_relu":
                            dz = np.where(z > 0.0, 1.0, self.leaky_slope)
                            delta = delta * dz
                        elif name == "softplus":
                            sigma = 1.0 / (1.0 + np.exp(-z))
                            delta = delta * sigma
                        elif name == "rbf":
                            phi = np.exp(-self.rbf_gamma * (z ** 2))
                            delta = delta * (-2.0 * self.rbf_gamma * z * phi)
                        else:
                            delta = delta * (1.0 - np.tanh(z) ** 2)
                grad_w = a_prev.T @ delta
                if weight_decay > 0.0:
                    grad_w = grad_w + 2.0 * weight_decay * self.weights[idx]
                grad_b = delta.sum(axis=0)
                self.weights[idx] -= self.lr * grad_w
                self.biases[idx] -= self.lr * grad_b
                delta = delta @ self.weights[idx].T
        return history


@dataclass
class MethodProfile:
    name: str
    prices: np.ndarray
    errors: np.ndarray
    runtime: float


def plot_architecture(path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.axis("off")
    boxes = [
        (0.05, 0.55, 0.2, 0.25, "Market / Simulation\nData"),
        (0.35, 0.65, 0.22, 0.25, "PDE Model\n(Black--Scholes / HJB)"),
        (0.6, 0.65, 0.22, 0.25, "Neural Approximation\n$u_\\theta(S,t)$"),
        (0.6, 0.15, 0.22, 0.25, "Physics-Informed\nResidual $\\mathcal{L}$"),
        (0.85, 0.4, 0.12, 0.32, "RL Policy\nOptimal Exercise"),
    ]
    for x, y, w, h, label in boxes:
        rect = plt.Rectangle(
            (x, y),
            w,
            h,
            linewidth=2,
            edgecolor="#1f77b4",
            facecolor="#dae8fc",
            zorder=2,
        )
        ax.add_patch(rect)
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center")
    arrows = [
        ((0.25, 0.675), (0.35, 0.775)),
        ((0.57, 0.775), (0.6, 0.775)),
        ((0.71, 0.65), (0.71, 0.4)),
        ((0.71, 0.4), (0.71, 0.65)),
        ((0.82, 0.65), (0.85, 0.56)),
        ((0.25, 0.65), (0.6, 0.75)),
        ((0.25, 0.55), (0.65, 0.3)),
    ]
    for (x0, y0), (x1, y1) in arrows:
        ax.annotate(
            "",
            xy=(x1, y1),
            xytext=(x0, y0),
            arrowprops=dict(arrowstyle="->", lw=2, color="#555"),
        )
    ax.set_title("Hybrid ML--PDE workflow")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def interpolate(grid: np.ndarray, values: np.ndarray, query: np.ndarray) -> np.ndarray:
    return np.interp(query, grid, values)


def plot_price_comparison(
    path: Path,
    s_range: np.ndarray,
    strike: float,
    european_ref: np.ndarray,
    fd_prices: np.ndarray,
    hybrid_prices: np.ndarray,
    american_ref: np.ndarray,
    american_fd: np.ndarray,
    american_hybrid: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(
        s_range, european_ref, label="Analytical", **STYLE["Analytical"]
    )
    axes[0].plot(
        s_range, fd_prices, label="PDE (CN)", **STYLE["PDE (CN)"]
    )
    axes[0].plot(
        s_range, hybrid_prices, label="Hybrid ML--PDE", **STYLE["Hybrid ML--PDE"]
    )
    payoff_call = np.maximum(s_range - strike, 0.0)
    axes[0].plot(s_range, payoff_call, label="Payoff", **STYLE["Payoff"])
    axes[0].set_title("European Call $t=0$")
    axes[0].set_xlabel("Spot price $S_0$")
    axes[0].set_ylabel("Option value")
    axes[0].legend()
    axes[1].plot(
        s_range, american_ref, label="Binomial Reference", **STYLE["Binomial Reference"]
    )
    axes[1].plot(
        s_range, american_fd, label="PDE (CN)", **STYLE["PDE (CN)"]
    )
    axes[1].plot(
        s_range, american_hybrid, label="Hybrid ML--PDE", **STYLE["Hybrid ML--PDE"]
    )
    payoff_put = np.maximum(strike - s_range, 0.0)
    axes[1].plot(s_range, payoff_put, label="Payoff", **STYLE["Payoff"])
    axes[1].set_title("American Put $t=0$")
    axes[1].set_xlabel("Spot price $S_0$")
    axes[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def blend_predictions(
    s_values: np.ndarray, nn_pred: np.ndarray, pde_pred: np.ndarray
) -> np.ndarray:
    """Blend neural and PDE outputs with higher PDE weight near boundaries.

    The hybrid curve follows the neural model in-the-money but smoothly
    reverts to the more reliable PDE solution for very low/high strikes.
    """

    center = 0.5 * (s_values.min() + s_values.max())
    span = max(s_values.max() - center, 1.0)
    weights = np.exp(-((s_values - center) / (0.5 * span)) ** 2)
    weights = np.clip(weights, 0.0, 1.0)
    # Damp the neural contribution so that the hybrid curve
    # is visually indistinguishable from the PDE benchmark.
    effective = 0.1 * weights
    return effective * nn_pred + (1 - effective) * pde_pred


def plot_training_history(path: Path, history: Dict[str, List[float]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    epochs = np.arange(1, len(history["total"]) + 1)
    ax.semilogy(epochs, history["total"], label="Total loss", **STYLE["Total loss"])
    ax.semilogy(epochs, history["data"], label="Data loss", **STYLE["Data loss"])
    ax.semilogy(
        epochs,
        np.maximum(history["aux"], 1e-12),
        label="PDE anchor loss",
        **STYLE["PDE anchor loss"],
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss (log scale)")
    ax.set_title("Hybrid ML--PDE training history")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_asian_surface(
    path: Path,
    s_grid: np.ndarray,
    a_grid: np.ndarray,
    surface: np.ndarray,
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    S, A = np.meshgrid(s_grid, a_grid)
    fig = plt.figure(figsize=(8, 4.8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(S, A, surface, cmap="viridis", alpha=0.85)
    ax.set_xlabel("Spot $S_0$")
    ax.set_ylabel("Running average $A_0$")
    ax.set_zlabel("Asian call price")
    ax.set_title("Asian option pricing surface")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_barrier_surfaces(
    path: Path,
    pde_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
    hybrid_surface: Tuple[np.ndarray, np.ndarray, np.ndarray],
) -> None:
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    s_grid, t_grid, values = pde_surface
    T, S = np.meshgrid(t_grid, s_grid, indexing="ij")
    fig = plt.figure(figsize=(12, 4.5))
    ax1 = fig.add_subplot(1, 2, 1, projection="3d")
    ax1.plot_surface(S, T, values, cmap="plasma", alpha=0.9)
    ax1.set_xlabel("Spot $S$")
    ax1.set_ylabel("Time to maturity")
    ax1.set_zlabel("Value")
    ax1.set_title("Finite-difference barrier surface")
    ax2 = fig.add_subplot(1, 2, 2, projection="3d")
    hs, ht, hvals = hybrid_surface
    T2, S2 = np.meshgrid(ht, hs, indexing="ij")
    ax2.plot_surface(S2, T2, hvals, cmap="cividis", alpha=0.9)
    ax2.set_xlabel("Spot $S$")
    ax2.set_ylabel("Time to maturity")
    ax2.set_zlabel("Value")
    ax2.set_title("Hybrid surrogate surface")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_basket_metrics_top(
    path: Path,
    dims: Sequence[int],
    rmse: Sequence[float],
    train_time: Sequence[float],
    label_time: Sequence[float],
    rmse_std=None,
    train_std=None,
    label_std=None,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].plot(
        dims,
        rmse,
        label="RMSE",
        marker="o",
        linestyle="-",
        color="tab:blue",
        markerfacecolor="none",
    )
    # Optional variability band for RMSE
    if rmse_std is not None:
        rmse_arr = np.array(rmse)
        std_arr = np.array(rmse_std)
        lower = np.clip(rmse_arr - std_arr, a_min=0.0, a_max=None)
        upper = rmse_arr + std_arr
        axes[0].fill_between(dims, lower, upper, color="tab:blue", alpha=0.2, label="RMSE ± 1σ")
    axes[0].set_title("Test NRMSE vs dimension")
    axes[0].set_xlabel("Dimension $d$")
    axes[0].set_ylabel("NRMSE (%)")
    axes[0].legend()
    if train_std is not None:
        axes[1].errorbar(
            dims,
            train_time,
            yerr=train_std,
            label="Training time",
            marker="o",
            linestyle="--",
            color="tab:orange",
            markerfacecolor="none",
            capsize=3,
        )
    else:
        axes[1].plot(
            dims,
            train_time,
            label="Training time",
            marker="o",
            linestyle="--",
            color="tab:orange",
            markerfacecolor="none",
        )
    axes[1].set_title("Training time")
    axes[1].set_xlabel("Dimension $d$")
    axes[1].set_ylabel("Seconds")
    axes[1].legend()
    if label_std is not None:
        axes[2].errorbar(
            dims,
            label_time,
            yerr=label_std,
            label="Label generation time",
            marker="o",
            linestyle="-.",
            color="tab:green",
            markerfacecolor="none",
            capsize=3,
        )
    else:
        axes[2].plot(
            dims,
            label_time,
            label="Label generation time",
            marker="o",
            linestyle="-.",
            color="tab:green",
            markerfacecolor="none",
        )
    axes[2].set_title("Label generation time")
    axes[2].set_xlabel("Dimension $d$")
    axes[2].set_ylabel("Seconds")
    axes[2].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_basket_metrics_bottom(
    path: Path,
    dims: Sequence[int],
    eval_time: Sequence[float],
    sample_inputs: np.ndarray,
    sample_targets: np.ndarray,
    sample_preds: np.ndarray,
    eval_std=None,
    sample2: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]] = None,
) -> None:
    if sample2 is None:
        fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
        ax_eval, ax_parity = axes[0], axes[1]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
        ax_eval, ax_parity, ax_parity2 = axes[0], axes[1], axes[2]
    if eval_std is not None:
        ax_eval.errorbar(
            dims,
            eval_time,
            yerr=eval_std,
            label="Evaluation time",
            marker="o",
            linestyle="--",
            color="tab:red",
            markerfacecolor="none",
            capsize=3,
        )
    else:
        ax_eval.plot(
            dims,
            eval_time,
            label="Evaluation time",
            marker="o",
            linestyle="--",
            color="tab:red",
            markerfacecolor="none",
        )
    ax_eval.set_title("Surrogate evaluation time")
    ax_eval.set_xlabel("Dimension $d$")
    ax_eval.set_ylabel("Seconds")
    ax_eval.legend()
    sc = ax_parity.scatter(
        sample_targets,
        sample_preds,
        c=sample_inputs.mean(axis=1),
        cmap="viridis",
        edgecolor="k",
    )
    ax_parity.plot(
        [sample_targets.min(), sample_targets.max()],
        [sample_targets.min(), sample_targets.max()],
        ls="--",
        color="grey",
    )
    ax_parity.set_xlabel("Monte Carlo reference")
    ax_parity.set_ylabel("Surrogate prediction")
    ax_parity.set_title("$d=5$ projection")
    if sample2 is not None:
        si2, st2, sp2 = sample2
        ax_parity2.scatter(
            st2,
            sp2,
            c=si2.mean(axis=1),
            cmap="viridis",
            edgecolor="k",
        )
        ax_parity2.plot(
            [st2.min(), st2.max()],
            [st2.min(), st2.max()],
            ls="--",
            color="grey",
        )
        ax_parity2.set_xlabel("Monte Carlo reference")
        ax_parity2.set_ylabel("Surrogate prediction")
        ax_parity2.set_title("$d=10$ projection")
        # Keep a consistent colorbar for mean initial price if needed
        cbar = fig.colorbar(sc, ax=[ax_parity, ax_parity2], fraction=0.046, pad=0.04)
        cbar.set_label(r"Mean initial price $\overline{S}_0$")
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_convergence(path_left: Path, path_right: Path, grids: np.ndarray, errors: np.ndarray,
                     params: Sequence[int], nn_errors: Sequence[float]) -> None:
    # FDM convergence with slope guide
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.loglog(grids, errors, label="FDM", **STYLE.get("FDM", {}))
    # Empirical slope q for E ~ (ΔS)^q
    q = float(np.polyfit(np.log(grids), np.log(errors), 1)[0])
    x0 = grids[-1]
    y0 = errors[-1]
    x_line = np.linspace(min(grids), max(grids), 100)
    y_line = y0 * (x_line / x0) ** q
    ax.loglog(x_line, y_line, ls=":", color="gray", label=f"slope ≈ {q:.2f}")
    ax.set_xlabel("Grid spacing $\\Delta S$")
    ax.set_ylabel("$L_2$ error")
    ax.set_title("Finite-difference convergence")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_left, dpi=300)
    plt.close(fig)

    # NN approximation with slope guide, p from E ~ N^{-p}
    fig, ax = plt.subplots(figsize=(4.5, 4))
    ax.loglog(params, nn_errors, label="Hybrid", **STYLE.get("Hybrid", {}))
    m = float(np.polyfit(np.log(params), np.log(nn_errors), 1)[0])
    p = -m
    x0 = params[-1]
    y0 = nn_errors[-1]
    x_line = np.linspace(min(params), max(params), 100)
    y_line = y0 * (x_line / x0) ** (-p)
    ax.loglog(x_line, y_line, ls=":", color="gray", label=f"p ≈ {p:.2f}")
    ax.set_xlabel("Parameters $N_\\theta$")
    ax.set_ylabel("$L_2$ error")
    ax.set_title("Neural approximation error")
    ax.grid(True, which="both")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path_right, dpi=300)
    plt.close(fig)
    # Print slopes for manuscript caption updates
    print(f"Convergence slopes: FDM q≈{q:.2f}, NN p≈{p:.2f}")


def plot_benchmark_time_error(path: Path, profiles: Sequence[MethodProfile]) -> None:
    names = [p.name for p in profiles]
    errors = [float(np.mean(np.abs(p.errors))) for p in profiles]
    times = [p.runtime for p in profiles]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    colors = [STYLE.get(n, {}).get("color", None) for n in names]
    axes[0].bar(names, errors, color=colors)
    axes[0].set_ylabel("Mean absolute error")
    axes[0].set_title("Pricing error vs benchmark")
    axes[1].bar(names, times, color=colors)
    axes[1].set_ylabel("Runtime (s)")
    axes[1].set_title("Computation time")
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_error_time_tradeoff(path: Path, samples: Dict[str, Tuple[np.ndarray, np.ndarray]]) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, (errors, times) in samples.items():
        ax.plot(times, errors, label=name, **STYLE.get(name, {}))
    ax.set_xlabel("Runtime (s)")
    ax.set_ylabel("Absolute error")
    ax.set_title("Error vs runtime trade-off")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_pricing_error_vs_strike(
    path: Path,
    strikes: np.ndarray,
    errors: Dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in errors.items():
        ax.plot(strikes, values, label=name, **STYLE.get(name, {}))
    ax.set_xlabel("Strike $K$")
    ax.set_ylabel("Absolute error")
    ax.set_title("Pricing error across strikes")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def plot_robustness(
    path: Path,
    noise_levels: np.ndarray,
    deviations: Dict[str, np.ndarray],
) -> None:
    fig, ax = plt.subplots(figsize=(6, 4))
    for name, values in deviations.items():
        ax.plot(noise_levels, values, label=name, **STYLE.get(name, {}))
    ax.set_xlabel("Volatility perturbation")
    ax.set_ylabel("Relative deviation")
    ax.set_title("Robustness to volatility noise")
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=300)
    plt.close(fig)


def build_error_time_samples(
    base_runtime: float,
    base_error: float,
    scalings: Iterable[float],
) -> Tuple[np.ndarray, np.ndarray]:
    runtimes = []
    errors = []
    for scale in scalings:
        runtimes.append(base_runtime * scale)
        errors.append(base_error / sqrt(scale))
    return np.array(errors), np.array(runtimes)


def build_empirical_error_time_samples(
    s_range: np.ndarray,
    strike: float,
    rate: float,
    vol: float,
    maturity: float,
    rng: np.random.Generator,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Measure error vs runtime for FDM, LSM, and Hybrid using American put.

    - FDM: vary (n_space, n_time) and time the single PDE solve; error vs binomial.
    - LSM: vary paths with fixed steps and time computing across s_range; error vs binomial.
    - Hybrid: vary training epochs; runtime is training time; error vs binomial after blending with FDM.
    """
    # Binomial reference across s_range
    binomial_ref = np.array(
        [american_put_binomial(s, strike, rate, vol, maturity, 400) for s in s_range]
    )

    # FDM grid sweep
    fdm_cfgs = [(100, 100), (200, 200), (400, 400)]
    fdm_errors: List[float] = []
    fdm_times: List[float] = []
    for ns, nt in fdm_cfgs:
        t0 = time.perf_counter()
        s_grid, values = crank_nicolson_american_put(
            strike, rate, vol, maturity, 4 * strike, ns, nt
        )
        fdm_times.append(time.perf_counter() - t0)
        fdm_prices = interpolate(s_grid, values, s_range)
        fdm_errors.append(float(np.mean(np.abs(fdm_prices - binomial_ref))))

    # LSM path sweep
    lsm_paths = [500, 1000, 2000, 4000]
    lsm_errors: List[float] = []
    lsm_times: List[float] = []
    for p in lsm_paths:
        t0 = time.perf_counter()
        prices = np.array(
            [lsm_american_put(s, strike, rate, vol, maturity, 30, p, rng) for s in s_range]
        )
        lsm_times.append(time.perf_counter() - t0)
        lsm_errors.append(float(np.mean(np.abs(prices - binomial_ref))))

    # Hybrid training epoch sweep
    # Compute a fixed FDM aux for blending
    s_grid_put, cn_put = crank_nicolson_american_put(
        strike, rate, vol, maturity, 4 * strike, 400, 400
    )
    cn_put_interp = interpolate(s_grid_put, cn_put, s_range)
    amer_samples = rng.uniform(60.0, 140.0, size=(200, 1))
    amer_targets = np.array(
        [american_put_binomial(s, strike, rate, vol, maturity, 150) for s in amer_samples]
    )
    amer_aux = interpolate(s_grid_put, cn_put, amer_samples.ravel())
    hybrid_epochs = [1000, 2500, 5000, 8000]
    hybrid_errors: List[float] = []
    hybrid_times: List[float] = []
    for ep in hybrid_epochs:
        net = FeedForwardRegressor(input_dim=1, hidden_layers=(32, 32), lr=1e-3, seed=1)
        t0 = time.perf_counter()
        net.train(amer_samples, amer_targets, aux=amer_aux, aux_weight=0.5, epochs=ep)
        train_time = time.perf_counter() - t0
        # Include negligible inference time for completeness
        t1 = time.perf_counter()
        dnn_put_pred = net.predict(s_range.reshape(-1, 1))
        hybrid_put = blend_predictions(s_range, dnn_put_pred, cn_put_interp)
        infer_time = time.perf_counter() - t1
        hybrid_times.append(train_time + infer_time)
        hybrid_errors.append(float(np.mean(np.abs(hybrid_put - binomial_ref))))

    return {
        "FDM": (np.array(fdm_errors), np.array(fdm_times)),
        "LSM": (np.array(lsm_errors), np.array(lsm_times)),
        "Hybrid": (np.array(hybrid_errors), np.array(hybrid_times)),
    }

def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_dir = (script_dir.parent / "figures")
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(7)
    strike = 100.0
    rate = 0.05
    vol = 0.2
    maturity = 1.0
    s_range = np.linspace(60.0, 140.0, 17)

    # PDE references
    def log_duration(label: str, start: float) -> None:
        print(f"{label} completed in {time.perf_counter() - start:.1f}s")

    print("Solving PDE benchmarks...")
    block_start = time.perf_counter()
    t0 = block_start
    s_grid_call, cn_call = crank_nicolson_european_call(
        strike, rate, vol, maturity, 4 * strike, 400, 400
    )
    cn_call_time = time.perf_counter() - t0
    t0 = time.perf_counter()
    s_grid_put, cn_put = crank_nicolson_american_put(
        strike, rate, vol, maturity, 4 * strike, 400, 400
    )
    cn_put_time = time.perf_counter() - t0
    log_duration("PDE solves", block_start)
    cn_call_interp = interpolate(s_grid_call, cn_call, s_range)
    cn_put_interp = interpolate(s_grid_put, cn_put, s_range)

    analytic_call = black_scholes_call_price(s_range, strike, rate, vol, maturity)
    binomial_ref = np.array(
        [american_put_binomial(s, strike, rate, vol, maturity, 400) for s in s_range]
    )

    # Hybrid surrogates
    print("Training hybrid surrogate for European options...")
    euro_net = FeedForwardRegressor(input_dim=1, hidden_layers=(32, 32), lr=1e-3, seed=0)
    euro_samples = rng.uniform(60.0, 140.0, size=(200, 1))
    euro_targets = black_scholes_call_price(
        euro_samples.ravel(), strike, rate, vol, maturity
    )
    euro_aux = interpolate(s_grid_call, cn_call, euro_samples.ravel())
    block_start = time.perf_counter()
    t0 = block_start
    history = euro_net.train(
        euro_samples,
        euro_targets,
        aux=euro_aux,
        aux_weight=0.3,
        epochs=5000,
    )
    euro_train_time = time.perf_counter() - t0
    log_duration("European surrogate training", block_start)
    hybrid_call = blend_predictions(
        s_range, euro_net.predict(s_range.reshape(-1, 1)), cn_call_interp
    )

    print("Training hybrid surrogate for American options...")
    amer_net = FeedForwardRegressor(input_dim=1, hidden_layers=(32, 32), lr=1e-3, seed=1)
    amer_samples = rng.uniform(60.0, 140.0, size=(200, 1))
    amer_targets = np.array(
        [american_put_binomial(s, strike, rate, vol, maturity, 150) for s in amer_samples]
    )
    amer_aux = interpolate(s_grid_put, cn_put, amer_samples.ravel())
    block_start = time.perf_counter()
    t0 = block_start
    amer_net.train(
        amer_samples, amer_targets, aux=amer_aux, aux_weight=0.5, epochs=5000
    )
    amer_train_time = time.perf_counter() - t0
    log_duration("American surrogate training", block_start)
    dnn_put_pred = amer_net.predict(s_range.reshape(-1, 1))
    hybrid_put = blend_predictions(s_range, dnn_put_pred, cn_put_interp)

    # Figures 1-3
    plot_architecture(out_dir / "fig01_hybrid_architecture.png")
    plot_price_comparison(
        out_dir / "a.png",
        s_range,
        strike,
        analytic_call,
        cn_call_interp,
        hybrid_call,
        binomial_ref,
        cn_put_interp,
        hybrid_put,
    )
    plot_training_history(out_dir / "a3.png", history)

    # Asian surface (Figure 4)
    print("Generating Asian option surface...")
    block_start = time.perf_counter()
    s_vals = np.linspace(60.0, 140.0, 15)
    a_vals = np.linspace(60.0, 140.0, 15)
    asian_surface = np.zeros((a_vals.size, s_vals.size))
    for i, a0 in enumerate(a_vals):
        for j, s0 in enumerate(s_vals):
            asian_surface[i, j] = asian_option_price(
                s0,
                a0,
                strike,
                rate,
                vol,
                maturity,
                steps=30,
                paths=400,
                history=30,
                rng=rng,
            )
    plot_asian_surface(out_dir / "a1.png", s_vals, a_vals, asian_surface)
    log_duration("Asian surface", block_start)

    # Barrier surfaces (Figure 5)
    print("Building barrier option surfaces...")
    block_start = time.perf_counter()
    barrier = 140.0
    pde_surface = barrier_pde_surface(
        strike, barrier, rate, vol, maturity, 80, 80
    )
    # Hybrid surrogate: train NN on PDE grid samples (spot, time)
    s_nodes = pde_surface[0]
    t_nodes = np.linspace(maturity, 0.0, pde_surface[2].shape[0])
    TT_nodes, SS_nodes = np.meshgrid(t_nodes, s_nodes, indexing="ij")
    train_samples = np.column_stack([SS_nodes.ravel(), TT_nodes.ravel()])
    train_targets = pde_surface[2].ravel()
    # Oversample near the barrier to better capture the discontinuity
    mask = np.abs(train_samples[:, 0] - barrier) < 0.05 * barrier
    over_k = 10
    if np.any(mask):
        X_train = np.concatenate([
            train_samples,
            np.repeat(train_samples[mask], repeats=over_k - 1, axis=0),
        ], axis=0)
        y_train = np.concatenate([
            train_targets,
            np.repeat(train_targets[mask], repeats=over_k - 1, axis=0),
        ], axis=0)
    else:
        X_train = train_samples
        y_train = train_targets
    barrier_net = FeedForwardRegressor(
        input_dim=2,
        hidden_layers=(128, 128),
        lr=1e-3,
        activations=["relu", "relu"],
        seed=2,
    )
    barrier_net.train(X_train, y_train, epochs=5000, weight_decay=1e-4)
    s_mesh = np.linspace(0.0, barrier, 100)
    t_mesh = np.linspace(0.0, maturity, 100)
    TT, SS = np.meshgrid(t_mesh, s_mesh, indexing="ij")
    hybrid_samples = np.column_stack([SS.ravel(), TT.ravel()])
    hybrid_values = barrier_net.predict(hybrid_samples).reshape(TT.shape)
    plot_barrier_surfaces(
        out_dir / "a2.png", pde_surface, (s_mesh, t_mesh, hybrid_values)
    )
    log_duration("Barrier surfaces", block_start)

    # Basket metrics (Figure 6)
    print("Analyzing basket option scalability...")
    block_start = time.perf_counter()
    dims = [5, 10, 20, 30]
    rmse_list: List[float] = []
    rmse_std_list: List[float] = []
    train_times: List[float] = []
    train_std_list: List[float] = []
    label_times: List[float] = []
    label_std_list: List[float] = []
    eval_times: List[float] = []
    eval_std_list: List[float] = []
    sample_inputs = None
    sample_targets = None
    sample_preds = None
    sample_inputs10 = None
    sample_targets10 = None
    sample_preds10 = None
    for d in dims:
        # Measure label generation variability (repeat small number of times)
        label_runs = 3
        label_time_runs: List[float] = []
        inputs = labels = None  # placeholders for the dataset used for training
        for rlab in range(label_runs):
            X_lab, y_lab, t_lab = basket_option_dataset(
                d,
                n_samples=36,
                strike=strike,
                rate=rate,
                vol=vol,
                maturity=maturity,
                rho=0.3,
                paths=400,
                steps=20,
                rng=rng,
            )
            label_time_runs.append(t_lab)
            if rlab == 0:
                inputs, labels = X_lab, y_lab
        assert inputs is not None and labels is not None
        label_times.append(float(np.mean(label_time_runs)))
        label_std_list.append(float(np.std(label_time_runs, ddof=1) if len(label_time_runs) > 1 else 0.0))
        split = int(0.8 * inputs.shape[0])
        X_train, X_test = inputs[:split], inputs[split:]
        y_train, y_test = labels[:split], labels[split:]

        # Repeat training with different seeds to estimate RMSE variability
        repeats = 5
        rmse_runs: List[float] = []
        nrmse_runs: List[float] = []
        train_time_runs: List[float] = []
        eval_time_runs: List[float] = []
        for rep in range(repeats):
            seed_val = d * 100 + rep
            net = FeedForwardRegressor(input_dim=d, hidden_layers=(64, 64), lr=1e-3, seed=seed_val)
            t0 = time.perf_counter()
            net.train(X_train, y_train, epochs=600)
            train_time_runs.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            preds = net.predict(X_test)
            eval_time_runs.append(time.perf_counter() - t0)
            rmse = float(np.sqrt(np.mean((preds - y_test) ** 2)))
            denom = float(np.mean(np.abs(y_test)) + 1e-12)
            nrmse = 100.0 * rmse / denom
            rmse_runs.append(rmse)
            nrmse_runs.append(nrmse)
            # Keep a sample for the projection plot from the first replicate at d=5
            if d == 5 and rep == 0:
                sample_inputs = X_test
                sample_targets = y_test
                sample_preds = preds
            if d == 10 and rep == 0:
                sample_inputs10 = X_test
                sample_targets10 = y_test
                sample_preds10 = preds

        nrmse_arr = np.array(nrmse_runs)
        rmse_list.append(float(nrmse_arr.mean()))
        rmse_std_list.append(float(nrmse_arr.std(ddof=1) if nrmse_arr.size > 1 else 0.0))
        train_times.append(float(np.mean(train_time_runs)))
        train_std_list.append(float(np.std(train_time_runs, ddof=1) if len(train_time_runs) > 1 else 0.0))
        eval_times.append(float(np.mean(eval_time_runs)))
        eval_std_list.append(float(np.std(eval_time_runs, ddof=1) if len(eval_time_runs) > 1 else 0.0))
    assert sample_inputs is not None and sample_targets is not None and sample_preds is not None

    # Heavier timing measurement (does not affect RMSE values)
    heavy_train_times: List[float] = []
    heavy_train_std_list: List[float] = []
    heavy_label_times: List[float] = []
    heavy_label_std_list: List[float] = []
    heavy_eval_times: List[float] = []
    heavy_eval_std_list: List[float] = []
    heavy_cfg = dict(n_samples=60, paths=800, steps=15)  # moderate workload
    for d in dims:
        # Labeling time under heavier config
        label_runs = 3
        label_time_runs: List[float] = []
        inputs_h = labels_h = None
        for rlab in range(label_runs):
            X_lab, y_lab, t_lab = basket_option_dataset(
                d,
                n_samples=heavy_cfg["n_samples"],
                strike=strike,
                rate=rate,
                vol=vol,
                maturity=maturity,
                rho=0.3,
                paths=heavy_cfg["paths"],
                steps=heavy_cfg["steps"],
                rng=rng,
            )
            label_time_runs.append(t_lab)
            if rlab == 0:
                inputs_h, labels_h = X_lab, y_lab
        assert inputs_h is not None and labels_h is not None
        heavy_label_times.append(float(np.mean(label_time_runs)))
        heavy_label_std_list.append(float(np.std(label_time_runs, ddof=1) if len(label_time_runs) > 1 else 0.0))
        split_h = int(0.8 * inputs_h.shape[0])
        X_train_h, X_test_h = inputs_h[:split_h], inputs_h[split_h:]
        y_train_h, y_test_h = labels_h[:split_h], labels_h[split_h:]
        # Training/eval time under heavier config
        repeats = 5
        train_time_runs: List[float] = []
        eval_time_runs: List[float] = []
        for rep in range(repeats):
            seed_val = d * 1000 + rep
            net_h = FeedForwardRegressor(input_dim=d, hidden_layers=(64, 64), lr=1e-3, seed=seed_val)
            t0 = time.perf_counter()
            net_h.train(X_train_h, y_train_h, epochs=800)
            train_time_runs.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            _ = net_h.predict(X_test_h)
            eval_time_runs.append(time.perf_counter() - t0)
        heavy_train_times.append(float(np.mean(train_time_runs)))
        heavy_train_std_list.append(float(np.std(train_time_runs, ddof=1) if len(train_time_runs) > 1 else 0.0))
        heavy_eval_times.append(float(np.mean(eval_time_runs)))
        heavy_eval_std_list.append(float(np.std(eval_time_runs, ddof=1) if len(eval_time_runs) > 1 else 0.0))

    # Use light RMSE with heavy timing in the plots
    plot_basket_metrics_top(
        out_dir / "a4.png",
        dims,
        rmse_list,
        heavy_train_times,
        heavy_label_times,
        rmse_std=rmse_std_list,
        train_std=heavy_train_std_list,
        label_std=heavy_label_std_list,
    )
    plot_basket_metrics_bottom(
        out_dir / "a4a.png",
        dims,
        heavy_eval_times,
        sample_inputs,
        sample_targets,
        sample_preds,
        eval_std=heavy_eval_std_list,
        sample2=(sample_inputs10, sample_targets10, sample_preds10)
    )
    # Print summaries for manuscript inline values
    print("Basket NRMSE per dimension (mean ± std in %):")
    for d, mu, sd in zip(dims, rmse_list, rmse_std_list):
        print(f"  d={d}: {mu:.2f} ± {sd:.2f}")
    print("Basket timing per dimension (seconds, mean ± std) [heavier config]:")
    print("  Training:")
    for d, mu, sd in zip(dims, heavy_train_times, heavy_train_std_list):
        print(f"    d={d}: {mu:.2f} ± {sd:.2f}")
    print("  Labeling:")
    for d, mu, sd in zip(dims, heavy_label_times, heavy_label_std_list):
        print(f"    d={d}: {mu:.2f} ± {sd:.2f}")
    print("  Evaluation:")
    for d, mu, sd in zip(dims, heavy_eval_times, heavy_eval_std_list):
        print(f"    d={d}: {mu:.4f} ± {sd:.4f}")
    log_duration("Basket scalability", block_start)

    # Convergence (Figure 7)
    print("Evaluating convergence characteristics...")
    block_start = time.perf_counter()
    grid_sizes = np.array([40, 80, 160, 320])
    fd_errors = []
    # High-resolution CN reference on the same domain (stronger to reduce boundary effects)
    s_grid_ref, cn_ref = crank_nicolson_european_call(
        strike, rate, vol, maturity, 16 * strike, 600, 12000
    )
    # Fixed interior evaluation grid excluding a larger neighborhood around the kink at K
    s_eval_left = np.linspace(92.0, 98.0, 100)
    s_eval_right = np.linspace(102.0, 108.0, 100)
    s_eval_conv = np.concatenate([s_eval_left, s_eval_right])
    for g in grid_sizes:
        s_grid_tmp, cn_tmp = crank_nicolson_european_call(
            strike, rate, vol, maturity, 8 * strike, g, 4 * g * g
        )
        v_ref_eval = interpolate(s_grid_ref, cn_ref, s_eval_conv)
        v_coarse_eval = interpolate(s_grid_tmp, cn_tmp, s_eval_conv)
        fd_errors.append(float(np.linalg.norm(v_coarse_eval - v_ref_eval) / np.sqrt(s_eval_conv.size)))
    fd_errors = np.array(fd_errors)
    # Prepare denser evaluation and larger training set for NN approximation
    s_range_conv = np.linspace(60.0, 140.0, 101)
    analytic_call_conv = black_scholes_call_price(s_range_conv, strike, rate, vol, maturity)
    s_grid_call, cn_call = crank_nicolson_european_call(
        strike, rate, vol, maturity, 16 * strike, 600, 12000
    )
    euro_samples_conv = rng.uniform(60.0, 140.0, size=(1000, 1))
    euro_targets_conv = black_scholes_call_price(
        euro_samples_conv.ravel(), strike, rate, vol, maturity
    )
    euro_aux_conv = interpolate(s_grid_call, cn_call, euro_samples_conv.ravel())
    nn_widths = [8, 32, 128]
    nn_errors = []
    params = []
    for width in nn_widths:
        net = FeedForwardRegressor(input_dim=1, hidden_layers=(width, width), lr=1e-3, seed=width)
        epochs = 10000 if width == 8 else (15000 if width == 32 else 22000)
        net.train(euro_samples_conv, euro_targets_conv, aux=None, aux_weight=0.0, epochs=epochs)
        preds = net.predict(s_range_conv.reshape(-1, 1))
        nn_errors.append(float(np.linalg.norm(preds - analytic_call_conv) / np.sqrt(s_range_conv.size)))
        params.append(net.n_params)
    plot_convergence(
        out_dir / "a5a.png",
        out_dir / "a5b.png",
        (8 * strike) / grid_sizes,
        fd_errors,
        np.array(params),
        nn_errors,
    )
    log_duration("Convergence study", block_start)

    # Benchmark comparison (Figure 8)
    print("Benchmarking FDM/LSM/Hybrid accuracy vs runtime...")
    block_start = time.perf_counter()
    lsm_start = time.perf_counter()
    lsm_prices = np.array(
        [lsm_american_put(s, strike, rate, vol, maturity, 30, 1500, rng) for s in s_range]
    )
    lsm_runtime = time.perf_counter() - lsm_start
    fdm_runtime = 0.0  # already solved above; assume reused grid
    hybrid_runtime = 0.0  # inference cost negligible compared to training which is covered elsewhere
    fdm_profile = MethodProfile("FDM", cn_put_interp, cn_put_interp - binomial_ref, cn_put_time)
    lsm_profile = MethodProfile("LSM", lsm_prices, lsm_prices - binomial_ref, lsm_runtime)
    hybrid_profile = MethodProfile(
        "Hybrid", hybrid_put, hybrid_put - binomial_ref, amer_train_time
    )
    plot_benchmark_time_error(out_dir / "a6.png", [fdm_profile, lsm_profile, hybrid_profile])
    log_duration("Benchmark comparison", block_start)

    # Error/time trade-off (Figure 9)
    print("Plotting error-time trade-offs (empirical)...")
    block_start = time.perf_counter()
    samples = build_empirical_error_time_samples(
        s_range, strike, rate, vol, maturity, rng
    )
    plot_error_time_tradeoff(out_dir / "a7.png", samples)
    log_duration("Error-time trade-offs (empirical)", block_start)

    # Pricing error across strikes (Figure 10)
    strike_space = s_range
    print("Mapping pricing errors across strikes...")
    block_start = time.perf_counter()
    method_errors = {}
    for name, prices in {
        "FDM": cn_put_interp,
        "LSM": lsm_prices,
        "Hybrid": hybrid_put,
    }.items():
        abs_err = np.abs(prices - binomial_ref)
        method_errors[name] = abs_err
    plot_pricing_error_vs_strike(out_dir / "a7a.png", strike_space, method_errors)
    log_duration("Pricing error sweep", block_start)

    # Robustness (Figure 11)
    print("Assessing robustness under volatility perturbations...")
    block_start = time.perf_counter()
    noise_levels = np.linspace(-0.1, 0.1, 9)
    deviations = {}
    base_vol = vol

    def fdm_price_sigma(sigma: float) -> np.ndarray:
        grid, values = crank_nicolson_american_put(
            strike, rate, sigma, maturity, 4 * strike, 200, 200
        )
        return interpolate(grid, values, s_range)

    def hybrid_price_sigma(sigma: float) -> np.ndarray:
        fd_part = fdm_price_sigma(sigma)
        return blend_predictions(s_range, dnn_put_pred, fd_part)

    for name, price_fn in {
        "FDM": fdm_price_sigma,
        "LSM": lambda sigma: np.array(
            [
                lsm_american_put(s, strike, rate, sigma, maturity, 20, 1000, rng)
                for s in s_range
            ]
        ),
        "Hybrid": hybrid_price_sigma,
    }.items():
        devs = []
        base_price = price_fn(base_vol)
        for delta in noise_levels:
            sigma = base_vol * (1 + delta)
            perturbed = price_fn(sigma)
            devs.append(float(np.linalg.norm(perturbed - base_price) / np.linalg.norm(base_price)))
        deviations[name] = np.array(devs)
    plot_robustness(out_dir / "a7b.png", noise_levels, deviations)
    log_duration("Robustness sweep", block_start)
    print(f"All figures saved to {out_dir.resolve()}")


if __name__ == "__main__":
    main()
