"""generate_hero — Hero image for P15 Experiment Kit README."""
from __future__ import annotations
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats as sp_stats

REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import warnings; warnings.filterwarnings("ignore")
from shared import apply_style; apply_style()

from dataset import generate_ab_experiment, ExperimentConfig
from model import ExperimentKit


def main():
    config = ExperimentConfig(n_users=2000, true_lift=0.15, seed=42)
    data = generate_ab_experiment(config)
    kit = ExperimentKit(data)
    freq = kit.run_frequentist()
    cuped = kit.run_cuped()
    seq = kit.run_sequential(peek=1, total_peeks=5)
    bayes = kit.run_bayesian(n_samples=100000)

    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # Top-left: Frequentist p-values.
    ax = axes[0, 0]
    names = list(freq.keys())
    pvals = [freq[n].p_value for n in names]
    colors = ["#D55E00" if p < 0.05 else "#0072B2" for p in pvals]
    ax.bar(names, pvals, color=colors)
    ax.axhline(0.05, color="#2b2b2b", linestyle="--", linewidth=0.8, label="α=0.05")
    ax.set_ylabel("p-value")
    ax.set_title("Frequentist tests", loc="left", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", alpha=0.3)

    # Top-right: CUPED variance.
    ax = axes[0, 1]
    ax.bar(["Raw", "CUPED"], [cuped.raw_variance, cuped.adjusted_variance],
           color=["#0072B2", "#009E73"])
    ax.set_ylabel("Variance")
    ax.set_title(f"CUPED: {cuped.variance_reduction_pct:.2f}% reduction", loc="left", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Bottom-left: Bayesian posteriors.
    ax = axes[1, 0]
    x = np.linspace(0, 0.3, 200)
    post_c = sp_stats.beta.pdf(x, bayes.alpha_control, bayes.beta_control)
    post_t = sp_stats.beta.pdf(x, bayes.alpha_treatment, bayes.beta_treatment)
    ax.plot(x, post_c, "-", color="#0072B2", linewidth=2, label=f"Control: Beta({bayes.alpha_control:.0f},{bayes.beta_control:.0f})")
    ax.plot(x, post_t, "-", color="#D55E00", linewidth=2, label=f"Treatment: Beta({bayes.alpha_treatment:.0f},{bayes.beta_treatment:.0f})")
    ax.set_xlabel("Conversion rate")
    ax.set_ylabel("Density")
    ax.set_title(f"Bayesian posteriors (P(treat>ctrl)={bayes.p_superiority:.2%})", loc="left", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Bottom-right: Sequential.
    ax = axes[1, 1]
    ax.bar(["mSPRT stat", "Boundary"], [seq.statistic, seq.boundary],
           color=["#009E73" if seq.should_stop else "#0072B2", "#D55E00"])
    ax.set_yscale("log")
    ax.set_ylabel("Value (log)")
    ax.set_title(f"Sequential: {seq.recommendation}", loc="left", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Experiment Kit — Frequentist + CUPED + Sequential + Bayesian",
                 fontsize=14, fontweight="bold", x=0.01, ha="left", y=1.02)
    out = PROJECT_ROOT / "assets" / "hero.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=140); plt.close(fig)
    print(f"Wrote hero image: {out}  ({out.stat().st_size // 1024} KB)")

if __name__ == "__main__":
    main()
