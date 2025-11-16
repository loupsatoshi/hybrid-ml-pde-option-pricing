"""Generate Figure 11: robustness under volatility perturbations."""

from pathlib import Path

import numpy as np

from reproduce_figures import (
    FeedForwardRegressor,
    american_put_binomial,
    blend_predictions,
    crank_nicolson_american_put,
    interpolate,
    lsm_american_put,
    plot_robustness,
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

    def fdm_price_sigma(sigma: float) -> np.ndarray:
        grid, values = crank_nicolson_american_put(
            strike, rate, sigma, maturity, 4 * strike, 200, 200
        )
        return interpolate(grid, values, s_range)

    # Train a genuine Hybrid surrogate at base volatility
    base_vol = vol
    s_grid_base, cn_base = crank_nicolson_american_put(
        strike, rate, base_vol, maturity, 4 * strike, 400, 400
    )
    cn_base_interp = interpolate(s_grid_base, cn_base, s_range)
    rng_local = np.random.default_rng(7)
    amer_net = FeedForwardRegressor(input_dim=1, hidden_layers=(32, 32), lr=1e-3, seed=1)
    amer_samples = rng_local.uniform(60.0, 140.0, size=(200, 1))
    amer_targets = np.array(
        [american_put_binomial(s, strike, rate, base_vol, maturity, 150) for s in amer_samples]
    )
    amer_aux = interpolate(s_grid_base, cn_base, amer_samples.ravel())
    amer_net.train(amer_samples, amer_targets, aux=amer_aux, aux_weight=0.5, epochs=5000)
    dnn_put_pred = amer_net.predict(s_range.reshape(-1, 1))

    noise_levels = np.linspace(-0.1, 0.1, 9)
    deviations = {}

    def hybrid_price_sigma(sigma: float) -> np.ndarray:
        fd_part = fdm_price_sigma(sigma)
        return blend_predictions(s_range, dnn_put_pred, fd_part)

    for name, price_fn in {
        "FDM": fdm_price_sigma,
        "LSM": lambda sigma: np.array(
            [lsm_american_put(s, strike, rate, sigma, maturity, 20, 1000, rng) for s in s_range]
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


if __name__ == "__main__":  # pragma: no cover
    main()

