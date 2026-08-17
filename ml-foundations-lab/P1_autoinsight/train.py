#!/usr/bin/env python3
"""
train
=====

CLI entry-point for AutoInsight (P1).

Despite the name "train" (kept for consistency with the rest of the
monorepo's file convention), this module does not train any model — it
runs the EDA + drift pipeline and writes a self-contained HTML report.

Usage
-----
::

    # 1. Basic — local CSV
    python train.py --data ./my_data.csv --out report.html

    # 2. Remote — data.gov.in CSV URL
    python train.py \\
        --data https://data.gov.in/sites/default/files/...csv \\
        --name "Daily Rainfall" --out rainfall.html

    # 3. Drift mode — compare current vs reference
    python train.py \\
        --data ./current.csv --reference ./reference.csv \\
        --out drift_report.html

    # 4. Force re-download (bypass cache for http sources)
    python train.py --data <url> --no-cache --out fresh.html

Exit codes
----------
* 0  : report written successfully.
* 1  : usage error / bad CLI args.
* 2  : data loading failed.
* 3  : profiling failed.
* 4  : rendering failed.
"""

from __future__ import annotations

import argparse
import logging
import sys
import traceback
from pathlib import Path
from typing import Optional

# Make repo root importable so `from dataset import ...` works regardless of CWD.
_REPO_ROOT = Path(__file__).resolve().parents[2]
_PROJECT_ROOT = Path(__file__).resolve().parent
for p in (_REPO_ROOT, _PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from dataset import Dataset, load_csv  # noqa: E402
from model import ProfileBuilder, compute_drift  # noqa: E402
from report import HTMLReportBuilder  # noqa: E402

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("autoinsight")


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="autoinsight",
        description="AutoInsight — automated EDA + drift report generator.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
examples
--------
  # Local CSV
  python train.py --data sample.csv --out report.html

  # data.gov.in resource (cached after first download)
  python train.py --data https://data.gov.in/.../file.csv --out gov.html

  # Drift detection against a reference snapshot
  python train.py --data current.csv --reference reference.csv --out drift.html
""",
    )

    parser.add_argument(
        "--data", "-d", required=True,
        help="Path or URL to the CSV file to profile.",
    )
    parser.add_argument(
        "--reference", "-r", default=None,
        help="Optional path/URL to a reference CSV for drift detection.",
    )
    parser.add_argument(
        "--out", "-o", default="autoinsight_report.html",
        help="Output HTML report path (default: ./autoinsight_report.html).",
    )
    parser.add_argument(
        "--name", "-n", default=None,
        help="Display name for the dataset (defaults to file stem).",
    )
    parser.add_argument(
        "--no-cache", action="store_true",
        help="Bypass the on-disk cache for http/data.gov.in sources.",
    )
    parser.add_argument(
        "--verbose", "-v", action="count", default=0,
        help="Increase log verbosity (use -v for INFO, -vv for DEBUG).",
    )
    return parser


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _load(source: str, name: Optional[str], no_cache: bool) -> Dataset:
    """Wrap ``load_csv`` with logging + cache bypass."""
    log.info("Loading dataset from %s", source)
    # If no_cache, remove the cached file for this URL before loading.
    if no_cache:
        from dataset import _cache_path_for, _detect_source_kind  # type: ignore
        kind = _detect_source_kind(source)
        if kind != "local":
            cache_path = _cache_path_for(source, kind)
            if cache_path.exists():
                cache_path.unlink()
                log.info("Cleared cache file: %s", cache_path)
    ds = load_csv(source, name=name)
    log.info("Loaded '%s' — %d rows × %d cols (sha256=%s…)",
             ds.name, ds.shape[0], ds.shape[1], ds.sha256[:12])
    return ds


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main(argv: Optional[list[str]] = None) -> int:
    args = build_parser().parse_args(argv)

    # Verbosity
    if args.verbose >= 2:
        log.setLevel(logging.DEBUG)
    elif args.verbose == 1:
        log.setLevel(logging.DEBUG)
    else:
        log.setLevel(logging.INFO)

    # Step 1 — load current dataset.
    try:
        current_ds = _load(args.data, args.name, args.no_cache)
    except Exception as exc:
        log.error("Failed to load current dataset: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 2

    # Step 2 — profile current dataset.
    try:
        log.info("Building profile for '%s'…", current_ds.name)
        current_profile = ProfileBuilder(current_ds).build()
        log.info("Profiled %d columns — %d numeric, %s missing cells (%.2f%%)",
                 current_profile.n_cols,
                 sum(1 for c in current_profile.columns if c.type == "numeric"),
                 current_profile.missing_cells,
                 current_profile.missing_pct)
    except Exception as exc:
        log.error("Profiling failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 3

    # Step 3 — optional drift computation against reference.
    drift = None
    if args.reference:
        try:
            log.info("Loading reference dataset from %s", args.reference)
            ref_ds = _load(args.reference, args.reference.split("/")[-1].rsplit(".", 1)[0], args.no_cache)
            ref_profile = ProfileBuilder(ref_ds).build()
            log.info("Computing drift (current vs reference)…")
            drift = compute_drift(
                current=current_profile, current_df=current_ds.df,
                reference=ref_profile, reference_df=ref_ds.df,
            )
            log.info("Drift summary: %d no-drift, %d moderate, %d severe (max PSI=%.4f)",
                     drift.n_no_drift, drift.n_moderate, drift.n_severe, drift.max_psi)
        except Exception as exc:
            log.warning("Drift computation failed (continuing without drift section): %s", exc)
            if args.verbose:
                traceback.print_exc()

    # Step 4 — render HTML.
    try:
        log.info("Rendering HTML report → %s", args.out)
        builder = HTMLReportBuilder(
            profile=current_profile,
            raw_df=current_ds.df,
            drift=drift,
            sha256=current_ds.sha256,
        )
        out_path = builder.save(Path(args.out))
        log.info("✓ Report written: %s (%.1f KB)",
                 out_path.resolve(), out_path.stat().st_size / 1024)
        # Print a single line summary for shell scripting convenience.
        print(f"REPORT_PATH={out_path.resolve()}")
        return 0
    except Exception as exc:
        log.error("Rendering failed: %s", exc)
        if args.verbose:
            traceback.print_exc()
        return 4


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
