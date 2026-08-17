"""
tests/test_pipeline
===================

End-to-end smoke test for the AutoInsight pipeline.

Runs the full flow on the synthetic sample dataset:

    load_csv -> ProfileBuilder -> compute_drift -> HTMLReportBuilder -> save

Verifies:
    * Dataset loads with the expected shape.
    * ProfileBuilder produces a non-empty DatasetProfile.
    * DriftReport has the right number of columns & non-negative max PSI.
    * HTMLReportBuilder.save writes a non-empty HTML file containing
      the expected section markers.

Run with::

    cd ml-foundations-lab/P1_autoinsight
    python -m pytest tests/ -v

or::

    python tests/test_pipeline.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Make the project & repo root importable when run as a standalone script.
REPO_ROOT = Path(__file__).resolve().parents[3]
PROJECT_ROOT = Path(__file__).resolve().parents[1]
for p in (REPO_ROOT, PROJECT_ROOT):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

# Ensure the sample data exists; generate it if not.
_SAMPLE_DIR = PROJECT_ROOT / "sample_data"
_CURRENT_CSV = _SAMPLE_DIR / "sample_current.csv"
_REFERENCE_CSV = _SAMPLE_DIR / "sample_reference.csv"

if not _CURRENT_CSV.exists():
    import importlib.util
    spec = importlib.util.spec_from_file_location("make_sample_data", _SAMPLE_DIR / "make_sample_data.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)  # type: ignore[arg-type]
    mod.main()

from dataset import load_csv, ColumnType, infer_column_types  # noqa: E402
from model import ProfileBuilder, compute_drift  # noqa: E402
from report import HTMLReportBuilder  # noqa: E402


def test_dataset_loads():
    ds = load_csv(str(_CURRENT_CSV), name="sample_current")
    assert ds.shape[0] > 700, f"Expected ~800 rows, got {ds.shape[0]}"
    assert ds.shape[1] >= 12, f"Expected >=12 cols, got {ds.shape[1]}"
    assert ds.sha256 and len(ds.sha256) == 64
    assert ds.source_kind == "local"


def test_type_inference():
    ds = load_csv(str(_CURRENT_CSV), name="sample_current")
    types = infer_column_types(ds.df)
    # Numeric detection should still work despite the "thirty" string injection.
    assert types["rainfall_mm"] == ColumnType.NUMERIC
    assert types["temperature_c"] == ColumnType.NUMERIC
    assert types["district"] == ColumnType.CATEGORICAL
    assert types["date"] == ColumnType.DATETIME
    assert types["is_coastal"] == ColumnType.BOOLEAN
    # Remarks has only 7 unique values -> CATEGORICAL (low-cardinality enum-like).
    assert types["remarks"] == ColumnType.CATEGORICAL
    # FreeText_Notes is genuinely free-form -> TEXT.
    # (After snake_case normalization: "FreeText_Notes" -> "free_text_notes".)
    assert types["free_text_notes"] == ColumnType.TEXT


def test_profile_built():
    ds = load_csv(str(_CURRENT_CSV), name="sample_current")
    profile = ProfileBuilder(ds).build()
    assert profile.n_cols == ds.shape[1]
    assert profile.n_rows > 0
    assert profile.missing_pct >= 0.0
    assert profile.missingness_matrix.shape == (ds.shape[0], ds.shape[1])
    # Each column should have a profile entry.
    assert len(profile.columns) == ds.shape[1]


def test_drift_report():
    current_ds = load_csv(str(_CURRENT_CSV), name="sample_current")
    reference_ds = load_csv(str(_REFERENCE_CSV), name="sample_reference")
    cur_profile = ProfileBuilder(current_ds).build()
    ref_profile = ProfileBuilder(reference_ds).build()
    drift = compute_drift(
        current=cur_profile, current_df=current_ds.df,
        reference=ref_profile, reference_df=reference_ds.df,
    )
    # Every column in the current dataset (that exists in the reference) should
    # have a drift entry.
    assert len(drift.columns) > 0
    assert drift.max_psi >= 0.0
    # The rainfall_mm and temperature_c columns were deliberately shifted between
    # the two datasets — at least one of them should be flagged moderate+.
    severe_or_moderate = [c for c in drift.columns if c.psi_label in ("moderate_drift", "severe_drift")]
    assert len(severe_or_moderate) > 0, "Expected at least one column with moderate+ drift"


def test_html_report_renders():
    current_ds = load_csv(str(_CURRENT_CSV), name="sample_current")
    reference_ds = load_csv(str(_REFERENCE_CSV), name="sample_reference")
    cur_profile = ProfileBuilder(current_ds).build()
    ref_profile = ProfileBuilder(reference_ds).build()
    drift = compute_drift(
        current=cur_profile, current_df=current_ds.df,
        reference=ref_profile, reference_df=reference_ds.df,
    )
    builder = HTMLReportBuilder(
        profile=cur_profile,
        raw_df=current_ds.df,
        drift=drift,
        sha256=current_ds.sha256,
    )
    with tempfile.TemporaryDirectory() as tmp:
        out_path = Path(tmp) / "report.html"
        result_path = builder.save(out_path)
        assert result_path.exists()
        assert result_path.stat().st_size > 10_000, "HTML report too small — likely empty"
        html = result_path.read_text(encoding="utf-8")
        assert "AutoInsight Report" in html
        assert "Missingness Map" in html
        assert "Drift Report" in html
        assert "Column Profiles" in html
        assert "Per-column Distributions" in html


if __name__ == "__main__":
    # Allow `python tests/test_pipeline.py` for ad-hoc runs.
    tests = [
        test_dataset_loads,
        test_type_inference,
        test_profile_built,
        test_drift_report,
        test_html_report_renders,
    ]
    for t in tests:
        t()
        print(f"PASS  {t.__name__}")
    print(f"\nAll {len(tests)} tests passed.")
