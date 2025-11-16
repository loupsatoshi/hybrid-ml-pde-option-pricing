"""Generate Figure 6: basket option scalability metrics."""

from pathlib import Path

import numpy as np

from reproduce_figures import (
    FeedForwardRegressor,
    basket_option_dataset,
    plot_basket_metrics_bottom,
    plot_basket_metrics_top,
)


def main() -> None:
    out_dir = Path("generated_figures")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(7)
    strike = 100.0
    rate = 0.05
    vol = 0.25
    maturity = 1.0

    dims = [5, 10, 20]
    rmse_list = []
    train_times = []
    label_times = []
    eval_times = []
    sample_inputs = sample_targets = sample_preds = None

    for d in dims:
        inputs, labels, label_time = basket_option_dataset(
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
        label_times.append(label_time)
        split = int(0.8 * inputs.shape[0])
        X_train, X_test = inputs[:split], inputs[split:]
        y_train, y_test = labels[:split], labels[split:]
        net = FeedForwardRegressor(input_dim=d, hidden_layers=(64, 64), lr=1e-3, seed=d)
        t0 = __import__("time").perf_counter()
        net.train(X_train, y_train, epochs=600)
        train_times.append(__import__("time").perf_counter() - t0)
        t0 = __import__("time").perf_counter()
        preds = net.predict(X_test)
        eval_times.append(__import__("time").perf_counter() - t0)
        rmse_list.append(float(np.sqrt(np.mean((preds - y_test) ** 2))))
        if d == 5:
            sample_inputs, sample_targets, sample_preds = X_test, y_test, preds

    assert sample_inputs is not None

    plot_basket_metrics_top(out_dir / "a4.png", dims, rmse_list, train_times, label_times)
    plot_basket_metrics_bottom(
        out_dir / "a4a.png",
        dims,
        eval_times,
        sample_inputs,
        sample_targets,
        sample_preds,
    )


if __name__ == "__main__":  # pragma: no cover
    main()

