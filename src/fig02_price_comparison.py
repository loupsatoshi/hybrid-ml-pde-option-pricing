"""Generate Figure 2: analytical vs PDE vs hybrid ML--PDE prices.

This script reproduces the European and American option comparison
shown in Figure 2 by:

- Solving the Black--Scholes PDE (European call) with a finite
  difference scheme.
- Computing American put prices via PDE and a binomial benchmark.
- Training small neural surrogates and forming the hybrid curve.
"""

from pathlib import Path

import numpy as np

from reproduce_figures import (
    FeedForwardRegressor,
    american_put_binomial,
    black_scholes_call_price,
    blend_predictions,
    crank_nicolson_american_put,
    crank_nicolson_european_call,
    interpolate,
    plot_price_comparison,
)


def main() -> None:
    script_dir = Path(__file__).resolve().parent
    out_dir = script_dir.parent / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(7)
    strike = 100.0
    rate = 0.05
    vol = 0.2
    maturity = 1.0
    s_range = np.linspace(60.0, 140.0, 17)

    # PDE benchmarks
    s_grid_call, cn_call = crank_nicolson_european_call(
        strike, rate, vol, maturity, 4 * strike, 400, 400
    )
    s_grid_put, cn_put = crank_nicolson_american_put(
        strike, rate, vol, maturity, 4 * strike, 400, 400
    )
    cn_call_interp = interpolate(s_grid_call, cn_call, s_range)
    cn_put_interp = interpolate(s_grid_put, cn_put, s_range)

    analytic_call = black_scholes_call_price(s_range, strike, rate, vol, maturity)
    # Use a higher-resolution binomial tree to reduce discretization error
    binomial_ref = np.array(
        [american_put_binomial(s, strike, rate, vol, maturity, 1200) for s in s_range]
    )

    # European surrogate + hybrid blend
    euro_net = FeedForwardRegressor(input_dim=1, hidden_layers=(32, 32), lr=1e-3, seed=0)
    euro_samples = rng.uniform(60.0, 140.0, size=(200, 1))
    euro_targets = black_scholes_call_price(
        euro_samples.ravel(), strike, rate, vol, maturity
    )
    euro_aux = interpolate(s_grid_call, cn_call, euro_samples.ravel())
    euro_net.train(
        euro_samples,
        euro_targets,
        aux=euro_aux,
        aux_weight=0.3,
        epochs=5000,
    )
    hybrid_call = blend_predictions(
        s_range, euro_net.predict(s_range.reshape(-1, 1)), cn_call_interp
    )

    # American surrogate + hybrid blend
    amer_net = FeedForwardRegressor(input_dim=1, hidden_layers=(32, 32), lr=1e-3, seed=1)
    amer_samples = rng.uniform(60.0, 140.0, size=(200, 1))
    amer_targets = np.array(
        [american_put_binomial(s, strike, rate, vol, maturity, 150) for s in amer_samples]
    )
    amer_aux = interpolate(s_grid_put, cn_put, amer_samples.ravel())
    amer_net.train(
        amer_samples, amer_targets, aux=amer_aux, aux_weight=0.5, epochs=5000
    )
    dnn_put_pred = amer_net.predict(s_range.reshape(-1, 1))
    hybrid_put = blend_predictions(s_range, dnn_put_pred, cn_put_interp)

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


if __name__ == "__main__":  # pragma: no cover
    main()

