"""Generate Figure 1: hybrid ML--PDE architecture diagram.

This script is intentionally minimal and only calls the plotting
utility defined in ``reproduce_figures.py``. It writes the PNG
used by the LaTeX manuscript under ``generated_figures``.
"""

from pathlib import Path

from reproduce_figures import plot_architecture


def main() -> None:
    out_dir = Path("generated_figures")
    out_dir.mkdir(exist_ok=True)
    plot_architecture(out_dir / "fig01_hybrid_architecture.png")


if __name__ == "__main__":  # pragma: no cover
    main()

