"""Generate Figure 7: convergence of PDE and neural components."""

from pathlib import Path

import numpy as np

from reproduce_figures import (
    FeedForwardRegressor,
    black_scholes_call_price,
    crank_nicolson_european_call,
    interpolate,
    plot_convergence,
)


def main() -> None:
    out_dir = Path("generated_figures")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(7)
    strike = 100.0
    rate = 0.05
    vol = 0.2
    maturity = 1.0
    s_range = np.linspace(60.0, 140.0, 101)
    analytic_call = black_scholes_call_price(s_range, strike, rate, vol, maturity)

    # PDE convergence with grid refinement (interior-domain error vs high-res CN reference)
    grid_sizes = np.array([40, 80, 160, 320])
    fd_errors = []
    # High-resolution CN reference on larger domain to reduce boundary effects
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
        fd_errors.append(
            float(np.linalg.norm(v_coarse_eval - v_ref_eval) / np.sqrt(s_eval_conv.size))
        )
    fd_errors = np.array(fd_errors)

    # Neural approximation error vs parameter count
    s_grid_call, cn_call = crank_nicolson_european_call(
        strike, rate, vol, maturity, 16 * strike, 600, 12000
    )
    euro_samples = rng.uniform(60.0, 140.0, size=(1000, 1))
    euro_targets = black_scholes_call_price(
        euro_samples.ravel(), strike, rate, vol, maturity
    )
    euro_aux = interpolate(s_grid_call, cn_call, euro_samples.ravel())

    nn_widths = [8, 32, 128]
    nn_errors = []
    params = []
    for width in nn_widths:
        net = FeedForwardRegressor(input_dim=1, hidden_layers=(width, width), lr=1e-3, seed=width)
        epochs = 10000 if width == 8 else (15000 if width == 32 else 22000)
        net.train(euro_samples, euro_targets, aux=None, aux_weight=0.0, epochs=epochs)
        preds = net.predict(s_range.reshape(-1, 1))
        nn_errors.append(
            float(np.linalg.norm(preds - analytic_call) / np.sqrt(s_range.size))
        )
        params.append(net.n_params)

    plot_convergence(
        out_dir / "a5a.png",
        out_dir / "a5b.png",
        (8 * strike) / grid_sizes,
        fd_errors,
        np.array(params),
        nn_errors,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

