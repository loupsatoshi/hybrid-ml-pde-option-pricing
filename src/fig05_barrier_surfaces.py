"""Generate Figure 5: barrier option surfaces (PDE vs hybrid surrogate)."""

from pathlib import Path

import numpy as np

from reproduce_figures import (
    FeedForwardRegressor,
    barrier_pde_surface,
    plot_barrier_surfaces,
)


def main() -> None:
    out_dir = Path("generated_figures")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(7)
    strike = 100.0
    rate = 0.05
    vol = 0.2
    maturity = 1.0
    barrier = 140.0

    pde_surface = barrier_pde_surface(
        strike, barrier, rate, vol, maturity, 60, 60
    )

    # Train neural surrogate on PDE grid samples (spot, time)
    s_nodes = pde_surface[0]
    t_nodes = np.linspace(maturity, 0.0, pde_surface[2].shape[0])
    TT_nodes, SS_nodes = np.meshgrid(t_nodes, s_nodes, indexing="ij")
    train_samples = np.column_stack([SS_nodes.ravel(), TT_nodes.ravel()])
    train_targets = pde_surface[2].ravel()

    take_idx = rng.choice(
        train_samples.shape[0], size=min(2500, train_samples.shape[0]), replace=False
    )
    net = FeedForwardRegressor(input_dim=2, hidden_layers=(64, 64), lr=1e-3, seed=2)
    net.train(train_samples[take_idx], train_targets[take_idx], epochs=400)

    s_mesh = np.linspace(0.0, barrier, 80)
    t_mesh = np.linspace(0.0, maturity, 80)
    TT, SS = np.meshgrid(t_mesh, s_mesh, indexing="ij")
    hybrid_samples = np.column_stack([SS.ravel(), TT.ravel()])
    hybrid_values = net.predict(hybrid_samples).reshape(TT.shape)

    plot_barrier_surfaces(out_dir / "a2.png", pde_surface, (s_mesh, t_mesh, hybrid_values))


if __name__ == "__main__":  # pragma: no cover
    main()

