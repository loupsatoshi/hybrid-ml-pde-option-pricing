"""Generate Figure 10: pricing error across strikes."""

from pathlib import Path

import numpy as np

from reproduce_figures import (
    FeedForwardRegressor,
    american_put_binomial,
    blend_predictions,
    crank_nicolson_american_put,
    interpolate,
    lsm_american_put,
    plot_pricing_error_vs_strike,
)


def main() -> None:
    out_dir = Path("generated_figures")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(7)
    strike = 100.0
    rate = 0.05
    vol = 0.2
    maturity = 1.0
    s_range = np.linspace(60.0, 140.0, 17)

    s_grid_put, cn_put = crank_nicolson_american_put(
        strike, rate, vol, maturity, 4 * strike, 400, 400
    )
    cn_put_interp = interpolate(s_grid_put, cn_put, s_range)
    binomial_ref = np.array(
        [american_put_binomial(s, strike, rate, vol, maturity, 400) for s in s_range]
    )
    lsm_prices = np.array(
        [lsm_american_put(s, strike, rate, vol, maturity, 30, 1500, rng) for s in s_range]
    )

    # Train a genuine Hybrid surrogate and blend with FDM
    rng = np.random.default_rng(7)
    amer_net = FeedForwardRegressor(input_dim=1, hidden_layers=(32, 32), lr=1e-3, seed=1)
    amer_samples = rng.uniform(60.0, 140.0, size=(200, 1))
    amer_targets = np.array(
        [american_put_binomial(s, strike, rate, vol, maturity, 150) for s in amer_samples]
    )
    amer_aux = interpolate(s_grid_put, cn_put, amer_samples.ravel())
    amer_net.train(amer_samples, amer_targets, aux=amer_aux, aux_weight=0.5, epochs=5000)
    dnn_put_pred = amer_net.predict(s_range.reshape(-1, 1))
    hybrid_put = blend_predictions(s_range, dnn_put_pred, cn_put_interp)

    method_errors = {
        "FDM": np.abs(cn_put_interp - binomial_ref),
        "LSM": np.abs(lsm_prices - binomial_ref),
        "Hybrid": np.abs(hybrid_put - binomial_ref),
    }

    plot_pricing_error_vs_strike(out_dir / "a7a.png", s_range, method_errors)


if __name__ == "__main__":  # pragma: no cover
    main()

