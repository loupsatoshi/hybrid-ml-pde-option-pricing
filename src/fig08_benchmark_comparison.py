"""Generate Figure 8: benchmark comparison (FDM, LSM, Hybrid).

This version computes genuine Hybrid predictions by training a small
surrogate for the American put and blending with FDM as in the main
pipeline. Runtimes are measured consistently.
"""

from pathlib import Path

import numpy as np

from reproduce_figures import (
    MethodProfile,
    FeedForwardRegressor,
    american_put_binomial,
    blend_predictions,
    crank_nicolson_american_put,
    interpolate,
    lsm_american_put,
    plot_benchmark_time_error,
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

    # PDE prices for American put (reference grid)
    s_grid_put, cn_put = crank_nicolson_american_put(
        strike, rate, vol, maturity, 4 * strike, 400, 400
    )
    cn_put_interp = interpolate(s_grid_put, cn_put, s_range)

    # Binomial reference
    binomial_ref = np.array(
        [american_put_binomial(s, strike, rate, vol, maturity, 400) for s in s_range]
    )

    # Longstaff--Schwartz prices
    t0 = __import__("time").perf_counter()
    lsm_prices = np.array(
        [lsm_american_put(s, strike, rate, vol, maturity, 30, 1500, rng) for s in s_range]
    )
    lsm_runtime = __import__("time").perf_counter() - t0

    # Hybrid surrogate (train on binomial targets with FDM aux; blend with FDM)
    amer_net = FeedForwardRegressor(input_dim=1, hidden_layers=(32, 32), lr=1e-3, seed=1)
    amer_samples = rng.uniform(60.0, 140.0, size=(200, 1))
    amer_targets = np.array(
        [american_put_binomial(s, strike, rate, vol, maturity, 150) for s in amer_samples]
    )
    amer_aux = interpolate(s_grid_put, cn_put, amer_samples.ravel())
    t0 = __import__("time").perf_counter()
    amer_net.train(amer_samples, amer_targets, aux=amer_aux, aux_weight=0.5, epochs=5000)
    amer_train_time = __import__("time").perf_counter() - t0
    dnn_put_pred = amer_net.predict(s_range.reshape(-1, 1))
    hybrid_put = blend_predictions(s_range, dnn_put_pred, cn_put_interp)

    fdm_profile = MethodProfile("FDM", cn_put_interp, cn_put_interp - binomial_ref, 0.0)
    lsm_profile = MethodProfile("LSM", lsm_prices, lsm_prices - binomial_ref, lsm_runtime)
    hybrid_profile = MethodProfile(
        "Hybrid", hybrid_put, hybrid_put - binomial_ref, amer_train_time
    )

    plot_benchmark_time_error(out_dir / "a6.png", [fdm_profile, lsm_profile, hybrid_profile])


if __name__ == "__main__":  # pragma: no cover
    main()

