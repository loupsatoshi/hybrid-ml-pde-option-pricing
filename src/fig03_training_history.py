"""Generate Figure 3: training history of the hybrid ML--PDE model.

This script trains the European surrogate network and records the
composite loss used in the paper, then plots the convergence curve.
"""

from pathlib import Path

import numpy as np

from reproduce_figures import (
    FeedForwardRegressor,
    black_scholes_call_price,
    crank_nicolson_european_call,
    interpolate,
    plot_training_history,
)


def main() -> None:
    out_dir = Path("generated_figures")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(7)
    strike = 100.0
    rate = 0.05
    vol = 0.2
    maturity = 1.0

    s_grid_call, cn_call = crank_nicolson_european_call(
        strike, rate, vol, maturity, 4 * strike, 400, 400
    )

    euro_net = FeedForwardRegressor(input_dim=1, hidden_layers=(32, 32), lr=1e-3, seed=0)
    euro_samples = rng.uniform(60.0, 140.0, size=(200, 1))
    euro_targets = black_scholes_call_price(
        euro_samples.ravel(), strike, rate, vol, maturity
    )
    euro_aux = interpolate(s_grid_call, cn_call, euro_samples.ravel())
    history = euro_net.train(
        euro_samples,
        euro_targets,
        aux=euro_aux,
        aux_weight=0.3,
        epochs=5000,
    )

    plot_training_history(out_dir / "a3.png", history)


if __name__ == "__main__":  # pragma: no cover
    main()

