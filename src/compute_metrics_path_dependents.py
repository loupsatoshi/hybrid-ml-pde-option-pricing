from pathlib import Path
import numpy as np
from reproduce_figures import (
    asian_option_price,
    barrier_pde_surface,
    FeedForwardRegressor,
    plot_barrier_surfaces,
)

def relative_l2(pred: np.ndarray, ref: np.ndarray) -> float:
    num = np.linalg.norm(pred - ref)
    den = np.linalg.norm(ref)
    return float(num / (den + 1e-12))

def mae(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.mean(np.abs(pred - ref)))

def max_err(pred: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(pred - ref)))

def compute_asian_metrics() -> dict:
    rng = np.random.default_rng(7)
    strike = 100.0
    rate = 0.05
    vol = 0.2
    maturity = 1.0
    s_vals = np.linspace(60.0, 140.0, 15)
    a_vals = np.linspace(60.0, 140.0, 15)
    steps = 30
    paths = 400
    history = 30

    # Monte Carlo reference surface
    asian_surface = np.zeros((a_vals.size, s_vals.size))
    for i, a0 in enumerate(a_vals):
        for j, s0 in enumerate(s_vals):
            asian_surface[i, j] = asian_option_price(
                s0, a0, strike, rate, vol, maturity, steps, paths, history, rng
            )

    # Train a small surrogate on (S, A) -> V to benchmark against MC
    A0, S0 = np.meshgrid(a_vals, s_vals, indexing="ij")
    samples = np.column_stack([S0.ravel(), A0.ravel()])
    targets = asian_surface.ravel()

    # Train/val split (hold out some points)
    n = samples.shape[0]
    idx = rng.permutation(n)
    split = int(0.7 * n)
    tr, te = idx[:split], idx[split:]

    net = FeedForwardRegressor(input_dim=2, hidden_layers=(64, 64), lr=1e-3, seed=3)
    net.train(samples[tr], targets[tr], epochs=5000)
    preds_all = net.predict(samples)

    # Metrics on all grid points
    rel_l2 = relative_l2(preds_all, targets)
    abs_mae = mae(preds_all, targets)
    abs_max = max_err(preds_all, targets)

    return {
        "asian_rel_l2": rel_l2,
        "asian_mae": abs_mae,
        "asian_max": abs_max,
    }


def compute_barrier_metrics(regen_figure: bool = True) -> dict:
    rng = np.random.default_rng(7)
    strike = 100.0
    rate = 0.05
    vol = 0.2
    maturity = 1.0
    barrier = 140.0

    # PDE surface on nodes using the final configuration (120x120)
    s_nodes, t_nodes, values = barrier_pde_surface(
        strike, barrier, rate, vol, maturity, 120, 120
    )
    # Build (S,T) grid and dataset
    TT_nodes, SS_nodes = np.meshgrid(
        np.linspace(maturity, 0.0, values.shape[0]), s_nodes, indexing="ij"
    )
    samples = np.column_stack([SS_nodes.ravel(), TT_nodes.ravel()])
    targets = values.ravel()

    # Oversample near the barrier S≈B to capture the discontinuity and train on full grid
    mask = np.abs(samples[:, 0] - barrier) < 0.025 * barrier
    over_k = 5
    if np.any(mask):
        X_train = np.concatenate([
            samples,
            np.repeat(samples[mask], repeats=over_k - 1, axis=0),
        ], axis=0)
        y_train = np.concatenate([
            targets,
            np.repeat(targets[mask], repeats=over_k - 1, axis=0),
        ], axis=0)
    else:
        X_train = samples
        y_train = targets
    net = FeedForwardRegressor(
        input_dim=2,
        hidden_layers=(128, 128),
        lr=1e-3,
        activations=["leaky_relu", "relu"],
        seed=2,
    )
    net.train(X_train, y_train, epochs=5000, weight_decay=1e-4)

    preds_all = net.predict(samples)

    rel_l2 = relative_l2(preds_all, targets)
    abs_mae = mae(preds_all, targets)
    abs_max = max_err(preds_all, targets)

    metrics = {
        "barrier_rel_l2": rel_l2,
        "barrier_mae": abs_mae,
        "barrier_max": abs_max,
    }

    # Optionally regenerate Figure 6 (a2.png) using the improved barrier surrogate
    if regen_figure:
        out_dir = Path("generated_figures")
        s_mesh = np.linspace(0.0, barrier, 120)
        t_mesh = np.linspace(0.0, maturity, 120)
        TT, SS = np.meshgrid(t_mesh, s_mesh, indexing="ij")
        hybrid_samples = np.column_stack([SS.ravel(), TT.ravel()])
        hybrid_values = net.predict(hybrid_samples).reshape(TT.shape)
        plot_barrier_surfaces(
            out_dir / "a2.png", (s_nodes, t_nodes, values), (s_mesh, t_mesh, hybrid_values)
        )

    return metrics


def main():
    out_dir = Path("generated_figures")
    out_dir.mkdir(exist_ok=True)

    asian = compute_asian_metrics()
    barrier = compute_barrier_metrics()

    report = (
        "Path-dependent metrics (relative to MC/PDE references)\n"
        "-----------------------------------------------------\n"
        f"Asian (surrogate vs MC)\n"
        f"  relative L2: {asian['asian_rel_l2']:.3e}\n"
        f"  MAE:         {asian['asian_mae']:.3e}\n"
        f"  Max error:   {asian['asian_max']:.3e}\n\n"
        f"Barrier (surrogate vs PDE grid)\n"
        f"  relative L2: {barrier['barrier_rel_l2']:.3e}\n"
        f"  MAE:         {barrier['barrier_mae']:.3e}\n"
        f"  Max error:   {barrier['barrier_max']:.3e}\n"
    )

    (out_dir / "metrics_path_dependents.txt").write_text(report)
    print(report)

if __name__ == "__main__":
    main()
