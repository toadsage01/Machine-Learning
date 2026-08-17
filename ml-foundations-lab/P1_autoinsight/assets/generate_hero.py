"""
generate_hero
=============

Generate the hero image for the AutoInsight README.

Composes a 2×2 panel showing:
    - top-left  : missingness matrix (sample dataset).
    - top-right : numeric correlation heatmap.
    - bottom-left: histogram of a numeric column with summary annotations.
    - bottom-right: drift PSI bar chart (current vs reference).

The image is saved to ``assets/hero.png`` and used by ``metadata.json``
and the README. Re-run after changing the shared style to refresh.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]   # Machine-Learning/
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # P1_autoinsight/
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from shared import apply_style  # noqa: E402
apply_style()

from dataset import load_csv  # noqa: E402
from model import ProfileBuilder, compute_drift, PSI_NO_DRIFT, PSI_MODERATE_DRIFT  # noqa: E402

# Ensure sample data exists.
_SAMPLE_DIR = PROJECT_ROOT / "sample_data"
_CURRENT_CSV = _SAMPLE_DIR / "sample_current.csv"
_REFERENCE_CSV = _SAMPLE_DIR / "sample_reference.csv"
if not _CURRENT_CSV.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_sample_data", _SAMPLE_DIR / "make_sample_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    mod.main()


def main() -> None:
    cur = load_csv(str(_CURRENT_CSV), name="hero_current")
    ref = load_csv(str(_REFERENCE_CSV), name="hero_reference")
    cur_profile = ProfileBuilder(cur).build()
    ref_profile = ProfileBuilder(ref).build()
    drift = compute_drift(
        current=cur_profile, current_df=cur.df,
        reference=ref_profile, reference_df=ref.df,
    )

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)

    # --- Top-left: missingness matrix -----------------------------------------
    ax = axes[0, 0]
    matrix = cur_profile.missingness_matrix
    if matrix.shape[0] > 1000:
        idx = np.linspace(0, matrix.shape[0] - 1, 1000).astype(int)
        matrix = matrix[idx]
    ax.imshow(
        matrix.T, aspect="auto",
        cmap=matplotlib.colors.ListedColormap(["#f4f4f4", "#D55E00"]),
        interpolation="nearest",
    )
    ax.set_title("Missingness Matrix", loc="left")
    ax.set_xlabel("Row (sampled)")
    ax.set_ylabel("Column")
    ax.set_yticks(range(matrix.shape[1]))
    ax.set_yticklabels(list(cur_profile.column_types.keys()), fontsize=7)
    ax.tick_params(axis="x", labelsize=7)

    # --- Top-right: numeric correlation ---------------------------------------
    ax = axes[0, 1]
    if cur_profile.correlation_numeric is not None:
        cols = cur_profile.correlation_numeric["columns"]
        mat = np.array(cur_profile.correlation_numeric["matrix"])
        im = ax.imshow(mat, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.set_xticks(range(len(cols)))
        ax.set_yticks(range(len(cols)))
        ax.set_xticklabels(cols, rotation=45, ha="right", fontsize=7)
        ax.set_yticklabels(cols, fontsize=7)
        ax.set_title("Numeric Correlation (Pearson)", loc="left")
        for i in range(len(cols)):
            for j in range(len(cols)):
                val = mat[i, j]
                if not np.isnan(val):
                    color = "white" if abs(val) > 0.55 else "#2b2b2b"
                    ax.text(j, i, f"{val:.2f}", ha="center", va="center", color=color, fontsize=6.5)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    else:
        ax.set_axis_off()
        ax.text(0.5, 0.5, "No numeric columns", ha="center", va="center")

    # --- Bottom-left: histogram of rainfall_mm --------------------------------
    ax = axes[1, 0]
    rainfall = cur.df["rainfall_mm"].dropna().astype(float).values
    ax.hist(rainfall, bins=30, color="#0072B2", alpha=0.85)
    ax.set_title("Distribution: rainfall_mm", loc="left")
    ax.set_xlabel("rainfall_mm")
    ax.set_ylabel("Count")

    # --- Bottom-right: drift PSI bars -----------------------------------------
    ax = axes[1, 1]
    cols = [c.name for c in drift.columns]
    psis = [c.psi for c in drift.columns]
    colours = []
    for c in drift.columns:
        if c.psi_label == "no_drift":
            colours.append("#009E73")
        elif c.psi_label == "moderate_drift":
            colours.append("#E69F00")
        else:
            colours.append("#D55E00")
    y_pos = np.arange(len(cols))
    ax.barh(y_pos, psis, color=colours)
    ax.axvline(PSI_NO_DRIFT, color="#2b2b2b", linestyle=":", linewidth=0.7, label="no-drift")
    ax.axvline(PSI_MODERATE_DRIFT, color="#2b2b2b", linestyle="--", linewidth=0.7, label="moderate")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(cols, fontsize=7)
    ax.set_xlabel("PSI")
    ax.set_title(f"Drift — {drift.current_name} vs {drift.reference_name}", loc="left")
    ax.legend(loc="lower right", fontsize=7)

    # Suptitle with subtle branding.
    fig.suptitle("AutoInsight — Automated EDA & Drift Report",
                 fontsize=16, fontweight="bold", x=0.01, ha="left", y=1.02)

    out_path = PROJECT_ROOT / "assets" / "hero.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    print(f"Wrote hero image: {out_path}  ({out_path.stat().st_size // 1024} KB)")


if __name__ == "__main__":
    main()
