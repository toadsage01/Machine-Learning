"""
shared
======

Cross-project helpers for the `Machine-Learning` monorepo.

Currently exposes:
    - `apply_style()` : applies `shared/plot_style.mplstyle` to matplotlib.
    - `REPO_ROOT`     : absolute path to the monorepo root.
    - `load_style()`  : returns the path to the mplstyle file.

Every project (P1–P16) imports `shared` instead of redefining its own theme so
that all figures share the same look-and-feel across the suite.
"""

from __future__ import annotations

import os
from pathlib import Path

__all__ = ["REPO_ROOT", "STYLE_PATH", "apply_style", "load_style"]

# shared/__init__.py lives at <repo_root>/shared/__init__.py
REPO_ROOT: Path = Path(__file__).resolve().parent.parent
STYLE_PATH: Path = Path(__file__).resolve().parent / "plot_style.mplstyle"


def load_style() -> str:
    """Return the absolute path to `plot_style.mplstyle` as a string."""
    return str(STYLE_PATH)


def apply_style() -> None:
    """Apply the project-wide matplotlib style.

    Importing matplotlib is deferred so that headless services that never plot
    are not forced to pay the matplotlib import cost.
    """
    import matplotlib.pyplot as plt  # local import keeps the module lightweight

    plt.style.use(load_style())


# Convenience: print the resolved paths when run directly (`python -m shared`).
if __name__ == "__main__":  # pragma: no cover
    print(f"REPO_ROOT  : {REPO_ROOT}")
    print(f"STYLE_PATH : {STYLE_PATH}")
