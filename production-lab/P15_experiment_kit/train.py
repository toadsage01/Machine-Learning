#!/usr/bin/env python3
"""
train
=====

CLI entry-point for P15_experiment_kit — evaluates A/B experiments with
Frequentist, CUPED, Sequential, and Bayesian engines.

Usage
-----
::

    # 1. Default: synthetic A/B, all engines
    python train.py

    # 2. A/B/C with larger lift
    python train.py --n-variants 3 --true-lift 0.10

    # 3. Real CSV
    python train.py --csv experiment.csv

    # 4. Save metrics + plots
    python train.py --metrics-json metrics.json --plot assets/experiment.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
try:
    from shared import apply_style
    apply_style()
except Exception:  # pragma: no cover
    pass

from dataset import ExperimentConfig, load_experiment_data  # noqa: E402
from model import ExperimentKit  # noqa: E402


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("exp_train")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="exp_train",
        description="P15 Experiment Kit — A/B testing statistical engine.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--csv", default=None, help="Path to experiment CSV.")
    parser.add_argument("--n-users", type=int, default=2000)
    parser.add_argument("--n-variants", type=int, default=2, choices=[2, 3])
    parser.add_argument("--true-lift", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--total-peeks", type=int, default=5, help="Planned sequential peeks.")
    parser.add_argument("--bayes-samples", type=int, default=100000)
    parser.add_argument("--metrics-json", default=None)
    parser.add_argument("--plot", default=None)
    parser.add_argument("--verbose", "-v", action="count", default=0)
    return parser


def _plot_results(freq, cuped, seq, bayes, output_path: Path) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(15, 11), constrained_layout=True)

    # Top-left: frequentist p-values.
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
    bars = ax.bar(["Raw", "CUPED-adjusted"], [cuped.raw_variance, cuped.adjusted_variance],
                  color=["#0072B2", "#009E73"])
    ax.text(0, cuped.raw_variance + 0.001, f"{cuped.raw_variance:.6f}", ha="center", fontsize=10)
    ax.text(1, cuped.adjusted_variance + 0.001, f"{cuped.adjusted_variance:.6f}", ha="center", fontsize=10)
    ax.set_ylabel("Variance")
    ax.set_title(f"CUPED variance reduction: {cuped.variance_reduction_pct:.2f}%", loc="left", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    # Bottom-left: Bayesian posteriors.
    ax = axes[1, 0]
    from scipy import stats as sp_stats
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

    # Bottom-right: sequential.
    ax = axes[1, 1]
    ax.bar(["mSPRT statistic", "Boundary"], [seq.statistic, seq.boundary],
           color=["#009E73" if seq.should_stop else "#0072B2", "#D55E00"])
    ax.set_yscale("log")
    ax.set_ylabel("Value (log scale)")
    ax.set_title(f"Sequential (peek {seq.peek_number}): {seq.recommendation}", loc="left", fontsize=11)
    ax.grid(True, axis="y", alpha=0.3)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if args.verbose >= 2:
        log.setLevel(logging.DEBUG)

    # Step 1 — load data.
    config = ExperimentConfig(
        n_users=args.n_users, n_variants=args.n_variants,
        true_lift=args.true_lift, seed=args.seed,
    )
    try:
        data = load_experiment_data(csv_path=args.csv, config=config)
        log.info("Loaded experiment: %d users, %d variants (source=%s)",
                 data.n_users, data.n_variants, data.source)
        log.info("  Variants: %s", data.variants)
    except Exception as exc:
        log.error("Data loading failed: %s", exc)
        return 2

    # Step 2 — run all engines.
    kit = ExperimentKit(data)

    log.info("Running Frequentist tests ...")
    freq = kit.run_frequentist()
    for name, r in freq.items():
        log.info("  %s: p=%.4f, effect=%.4f, CI=[%.4f, %.4f], sig=%s",
                 name, r.p_value, r.effect_size, r.ci_lower, r.ci_upper, r.significant)

    log.info("Running CUPED variance reduction ...")
    cuped = kit.run_cuped()
    log.info("  theta=%.4f, var_reduction=%.2f%%, raw_var=%.6f, adj_var=%.6f",
             cuped.theta, cuped.variance_reduction_pct,
             cuped.raw_variance, cuped.adjusted_variance)
    log.info("  raw_effect=%.4f, adj_effect=%.4f", cuped.raw_effect, cuped.adjusted_effect)

    log.info("Running Sequential (mSPRT, peek 1/%d) ...", args.total_peeks)
    seq = kit.run_sequential(peek=1, total_peeks=args.total_peeks)
    log.info("  statistic=%.4f, boundary=%.4f, should_stop=%s, rec=%s",
             seq.statistic, seq.boundary, seq.should_stop, seq.recommendation)

    log.info("Running Bayesian (Beta-Binomial, %d MC samples) ...", args.bayes_samples)
    bayes = kit.run_bayesian(n_samples=args.bayes_samples)
    log.info("  P(treatment > control)=%.4f", bayes.p_superiority)
    log.info("  Posterior means: control=%.4f, treatment=%.4f",
             bayes.posterior_mean_control, bayes.posterior_mean_treatment)
    log.info("  Credible interval: [%.4f, %.4f]", bayes.credible_lower, bayes.credible_upper)
    log.info("  ROPE probability: %.4f", bayes.rope_probability)

    # Step 3 — metrics JSON.
    if args.metrics_json:
        payload = {
            "config": {"n_users": data.n_users, "n_variants": data.n_variants,
                        "true_lift": args.true_lift, "seed": args.seed},
            "frequentist": {k: v.to_dict() for k, v in freq.items()},
            "cuped": cuped.to_dict(),
            "sequential": seq.to_dict(),
            "bayesian": bayes.to_dict(),
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        path = Path(args.metrics_json)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
        log.info("Saved metrics JSON → %s", path)

    # Step 4 — plot.
    if args.plot:
        try:
            _plot_results(freq, cuped, seq, bayes, Path(args.plot))
            log.info("Saved experiment plot → %s", args.plot)
        except Exception as exc:
            log.warning("Failed to render plot: %s", exc)

    # Summary.
    best_freq = min(freq.values(), key=lambda r: r.p_value)
    print(f"FREQUENTIST_P_VALUE={best_freq.p_value:.4f}")
    print(f"FREQUENTIST_SIGNIFICANT={best_freq.significant}")
    print(f"CUPED_VAR_REDUCTION_PCT={cuped.variance_reduction_pct:.2f}")
    print(f"SEQUENTIAL_RECOMMENDATION={seq.recommendation}")
    print(f"BAYES_P_SUPERIORITY={bayes.p_superiority:.4f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
