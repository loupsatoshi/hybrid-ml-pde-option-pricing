"""Generate Figure 4: Asian option pricing surface.

Computes Monte Carlo prices for an arithmetic-average Asian call
across a grid of initial spot and running-average values, then
plots the resulting surface.
"""

from pathlib import Path

import numpy as np

from reproduce_figures import asian_option_price, plot_asian_surface


def main() -> None:
    out_dir = Path("generated_figures")
    out_dir.mkdir(exist_ok=True)

    rng = np.random.default_rng(7)
    strike = 100.0
    rate = 0.05
    vol = 0.2
    maturity = 1.0

    s_vals = np.linspace(60.0, 140.0, 15)
    a_vals = np.linspace(60.0, 140.0, 15)
    asian_surface = np.zeros((a_vals.size, s_vals.size))
    for i, a0 in enumerate(a_vals):
        for j, s0 in enumerate(s_vals):
            asian_surface[i, j] = asian_option_price(
                s0,
                a0,
                strike,
                rate,
                vol,
                maturity,
                steps=30,
                paths=400,
                history=30,
                rng=rng,
            )

    plot_asian_surface(out_dir / "a1.png", s_vals, a_vals, asian_surface)


if __name__ == "__main__":  # pragma: no cover
    main()

