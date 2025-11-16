"""Generate Figure 9: absolute error vs runtime trade-offs."""

from pathlib import Path

import numpy as np

from reproduce_figures import (
    build_empirical_error_time_samples,
    plot_error_time_tradeoff,
)


def main() -> None:
    out_dir = Path("generated_figures")
    out_dir.mkdir(exist_ok=True)

    s_range = np.linspace(60.0, 140.0, 17)
    strike = 100.0
    rate = 0.05
    vol = 0.2
    maturity = 1.0
    rng = np.random.default_rng(7)
    samples = build_empirical_error_time_samples(
        s_range, strike, rate, vol, maturity, rng
    )
    plot_error_time_tradeoff(out_dir / "a7.png", samples)


if __name__ == "__main__":  # pragma: no cover
    main()

