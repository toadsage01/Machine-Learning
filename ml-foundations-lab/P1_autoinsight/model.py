"""
model
=====

Profiling engine for AutoInsight.

This module is **pure** — it never touches the filesystem and never makes
network calls. It takes a ``Dataset`` (from ``dataset.py``) and produces
``DatasetProfile`` / ``DriftReport`` value objects that ``report.py``
serializes into HTML.

Two concepts
------------
1. **ColumnProfile** : per-column summary stats (one per column).
2. **DatasetProfile** : the roll-up of all column profiles + the missingness
   matrix + the correlation matrix + duplicate stats.

Drift detection
---------------
``compute_drift`` compares two ``DatasetProfile`` objects (current vs.
reference) and emits a per-column ``DriftReport`` with:

* **PSI** (Population Stability Index) — the production-standard drift
  metric for binned distributions. PSI < 0.10 = no drift, 0.10–0.25 =
  moderate, > 0.25 = severe.
* **KS statistic + p-value** — non-parametric test for numeric columns
  (``scipy.stats.ks_2samp``). Useful for sanity-checking PSI on continuous
  variables.

Choice of PSI over KL-divergence / Wasserstein: PSI is symmetric in practice
because we always bin the *current* distribution using the *reference*'s
bin edges, and it has well-understood thresholds that non-technical
stakeholders recognize.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

# Boot shared style on import — every figure produced here will use the
# project-wide theme.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from dataset import ColumnType, Dataset, infer_column_types  # noqa: E402

# ---------------------------------------------------------------------------
# PSI thresholds (industry-standard)
# ---------------------------------------------------------------------------
PSI_NO_DRIFT = 0.10
PSI_MODERATE_DRIFT = 0.25


def _psi_bucket_label(psi: float) -> str:
    if psi < PSI_NO_DRIFT:
        return "no_drift"
    if psi < PSI_MODERATE_DRIFT:
        return "moderate_drift"
    return "severe_drift"


# ---------------------------------------------------------------------------
# Per-column profile
# ---------------------------------------------------------------------------
@dataclass
class ColumnProfile:
    """Summary statistics for a single column."""

    name: str
    type: str  # ColumnType.value
    n_total: int
    n_missing: int
    n_unique: int
    missing_pct: float
    unique_pct: float
    memory_bytes: int

    # Numeric-only fields (NaN-safe)
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    p05: Optional[float] = None
    p25: Optional[float] = None
    median: Optional[float] = None
    p75: Optional[float] = None
    p95: Optional[float] = None
    max: Optional[float] = None
    skew: Optional[float] = None
    kurtosis: Optional[float] = None
    # Shapiro normality p-value (computed only for n <= 5000 due to scipy limit)
    shapiro_p: Optional[float] = None

    # Categorical-only fields
    top_categories: Optional[List[Tuple[str, int]]] = None  # top 10 (label, count)
    cardinality_flag: Optional[str] = None  # "high" if > MAX_CATEGORY_CARDINALITY

    # Datetime-only fields
    datetime_min: Optional[str] = None
    datetime_max: Optional[str] = None

    # Text-only fields
    mean_token_len: Optional[float] = None
    avg_char_len: Optional[float] = None


# ---------------------------------------------------------------------------
# Dataset profile (roll-up)
# ---------------------------------------------------------------------------
@dataclass
class DatasetProfile:
    """Roll-up statistics for an entire DataFrame."""

    name: str
    n_rows: int
    n_cols: int
    n_duplicate_rows: int
    duplicate_pct: float
    total_cells: int
    missing_cells: int
    missing_pct: float
    memory_mb: float
    columns: List[ColumnProfile]
    column_types: Dict[str, str]
    missingness_matrix: Any  # np.ndarray [n_rows, n_cols], 1=missing 0=present
    correlation_numeric: Optional[Dict[str, Any]] = None  # Pearson corr (numeric only)
    sample_rows: Optional[List[Dict[str, Any]]] = None  # first 5 rows for the report

    def column_by_name(self, name: str) -> Optional[ColumnProfile]:
        for c in self.columns:
            if c.name == name:
                return c
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Plain dict representation (used by the Jinja2 template)."""
        return {
            **asdict(self),
            "columns": [asdict(c) for c in self.columns],
        }


# ---------------------------------------------------------------------------
# Per-column statistic computation
# ---------------------------------------------------------------------------
def _profile_numeric(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return {}
    quantiles = s.quantile([0.05, 0.25, 0.5, 0.75, 0.95]).to_list()
    shapiro_p: Optional[float] = None
    # Shapiro is unreliable for n > 5000 — scipy caps it at 5000 anyway.
    if 3 <= len(s) <= 5000:
        try:
            shapiro_p = float(stats.shapiro(s.values)[1])
        except Exception:
            shapiro_p = None
    return {
        "mean": float(s.mean()),
        "std": float(s.std(ddof=1)) if len(s) > 1 else 0.0,
        "min": float(s.min()),
        "p05": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p95": float(quantiles[4]),
        "max": float(s.max()),
        "skew": float(s.skew()) if len(s) > 2 else 0.0,
        "kurtosis": float(s.kurtosis()) if len(s) > 3 else 0.0,
        "shapiro_p": shapiro_p,
    }


def _profile_categorical(series: pd.Series) -> Dict[str, Any]:
    counts = series.value_counts(dropna=True)
    top = [(str(idx), int(cnt)) for idx, cnt in counts.head(10).items()]
    cardinality_flag = "high" if len(counts) > 50 else None
    return {
        "top_categories": top,
        "cardinality_flag": cardinality_flag,
    }


def _profile_datetime(series: pd.Series) -> Dict[str, Any]:
    s = pd.to_datetime(series, errors="coerce").dropna()
    if s.empty:
        return {}
    return {
        "datetime_min": s.min().isoformat(),
        "datetime_max": s.max().isoformat(),
    }


def _profile_text(series: pd.Series) -> Dict[str, Any]:
    s = series.dropna().astype(str)
    if s.empty:
        return {}
    token_lens = s.str.split().str.len()
    return {
        "mean_token_len": float(token_lens.mean()),
        "avg_char_len": float(s.str.len().mean()),
    }


def _profile_column(col_name: str, df: pd.DataFrame, col_type: ColumnType) -> ColumnProfile:
    series = df[col_name]
    n_total = len(series)
    n_missing = int(series.isna().sum())
    n_unique = int(series.nunique(dropna=True))
    base = {
        "name": col_name,
        "type": col_type.value,
        "n_total": n_total,
        "n_missing": n_missing,
        "n_unique": n_unique,
        "missing_pct": round(100.0 * n_missing / max(n_total, 1), 4),
        "unique_pct": round(100.0 * n_unique / max(n_total, 1), 4),
        "memory_bytes": int(series.memory_usage(deep=True)),
    }
    extra: Dict[str, Any] = {}
    if col_type == ColumnType.NUMERIC:
        extra = _profile_numeric(series)
    elif col_type == ColumnType.CATEGORICAL:
        extra = _profile_categorical(series)
    elif col_type == ColumnType.BOOLEAN:
        extra = _profile_categorical(series)  # reuse value_counts
    elif col_type == ColumnType.DATETIME:
        extra = _profile_datetime(series)
    elif col_type == ColumnType.TEXT:
        extra = _profile_text(series)
    # ColumnType.EMPTY -> no extras
    return ColumnProfile(**{**base, **extra})


# ---------------------------------------------------------------------------
# Numeric correlation
# ---------------------------------------------------------------------------
def _numeric_correlation(df: pd.DataFrame, types: Dict[str, ColumnType]) -> Optional[Dict[str, Any]]:
    numeric_cols = [c for c, t in types.items() if t == ColumnType.NUMERIC and c in df.columns]
    if len(numeric_cols) < 2:
        return None
    sub = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
    corr = sub.corr(method="pearson")
    if corr.empty:
        return None
    return {
        "columns": numeric_cols,
        "matrix": corr.round(4).values.tolist(),
    }


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
class ProfileBuilder:
    """Build a ``DatasetProfile`` from a ``Dataset``."""

    def __init__(self, dataset: Dataset):
        self.dataset = dataset
        self.df: pd.DataFrame = dataset.df
        self.types: Dict[str, ColumnType] = infer_column_types(self.df)

    def build(self) -> DatasetProfile:
        df = self.df
        types = self.types
        n_rows, n_cols = df.shape

        # Missingness matrix as int8 (1 = missing, 0 = present)
        miss_matrix = df.isna().astype(np.int8).values

        n_duplicate = int(df.duplicated().sum())
        total_cells = df.size
        missing_cells = int(df.isna().sum().sum())

        col_profiles: List[ColumnProfile] = []
        for col in df.columns:
            col_profiles.append(_profile_column(col, df, types[col]))

        corr = _numeric_correlation(df, types)

        # First 5 rows as list-of-dicts for the report's "preview" table.
        sample_rows = df.head(5).astype(object).where(pd.notna(df.head(5)), None).to_dict(orient="records")

        return DatasetProfile(
            name=self.dataset.name,
            n_rows=n_rows,
            n_cols=n_cols,
            n_duplicate_rows=n_duplicate,
            duplicate_pct=round(100.0 * n_duplicate / max(n_rows, 1), 4),
            total_cells=total_cells,
            missing_cells=missing_cells,
            missing_pct=round(100.0 * missing_cells / max(total_cells, 1), 4),
            memory_mb=round(df.memory_usage(deep=True).sum() / 1024 / 1024, 4),
            columns=col_profiles,
            column_types={k: v.value for k, v in types.items()},
            missingness_matrix=miss_matrix,
            correlation_numeric=corr,
            sample_rows=sample_rows,
        )


# ---------------------------------------------------------------------------
# Drift detection
# ---------------------------------------------------------------------------
@dataclass
class ColumnDrift:
    """Per-column drift metrics comparing current vs reference."""

    name: str
    type: str
    psi: float
    psi_label: str
    ks_stat: Optional[float]
    ks_pvalue: Optional[float]
    n_current: int
    n_reference: int


@dataclass
class DriftReport:
    """Roll-up drift report."""

    current_name: str
    reference_name: str
    columns: List[ColumnDrift]
    n_severe: int
    n_moderate: int
    n_no_drift: int
    max_psi: float

    def to_dict(self) -> Dict[str, Any]:
        return {
            **asdict(self),
            "columns": [asdict(c) for c in self.columns],
        }


def _psi_numeric(current: pd.Series, reference: pd.Series, n_bins: int = 10) -> float:
    """PSI for numeric columns using reference quantiles as bin edges.

    Binning on the reference (not the union) is the canonical formulation —
    it ensures PSI stays comparable across multiple current snapshots taken
    against the same reference.
    """
    cur = pd.to_numeric(current, errors="coerce").dropna().values
    ref = pd.to_numeric(reference, errors="coerce").dropna().values
    if len(cur) == 0 or len(ref) == 0:
        return 0.0
    # Bin edges from reference quantiles, with -inf / +inf at the ends so
    # every current observation falls into a bin (no overflow).
    quantiles = np.linspace(0, 1, n_bins + 1)
    edges = np.quantile(ref, quantiles)
    edges[0] = -np.inf
    edges[-1] = np.inf
    # Deduplicate edges (happens if reference has very few unique values).
    edges = np.unique(edges)
    if len(edges) < 3:
        return 0.0
    ref_counts, _ = np.histogram(ref, bins=edges)
    cur_counts, _ = np.histogram(cur, bins=edges)
    ref_prop = ref_counts / max(ref_counts.sum(), 1)
    cur_prop = cur_counts / max(cur_counts.sum(), 1)
    # Avoid log(0) — floor at epsilon.
    eps = 1e-6
    ref_prop = np.clip(ref_prop, eps, None)
    cur_prop = np.clip(cur_prop, eps, None)
    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    return max(psi, 0.0)


def _psi_categorical(current: pd.Series, reference: pd.Series) -> float:
    """PSI for categorical columns using the union of categories as bins."""
    cur = current.dropna().astype(str)
    ref = reference.dropna().astype(str)
    if cur.empty or ref.empty:
        return 0.0
    all_cats = list(set(cur.unique()).union(set(ref.unique())))
    ref_counts = ref.value_counts().reindex(all_cats, fill_value=0).values
    cur_counts = cur.value_counts().reindex(all_cats, fill_value=0).values
    ref_prop = ref_counts / max(ref_counts.sum(), 1)
    cur_prop = cur_counts / max(cur_counts.sum(), 1)
    eps = 1e-6
    ref_prop = np.clip(ref_prop, eps, None)
    cur_prop = np.clip(cur_prop, eps, None)
    psi = float(np.sum((cur_prop - ref_prop) * np.log(cur_prop / ref_prop)))
    return max(psi, 0.0)


def compute_drift(current: DatasetProfile, current_df: pd.DataFrame,
                  reference: DatasetProfile, reference_df: pd.DataFrame) -> DriftReport:
    """Compute a ``DriftReport`` between two datasets.

    Parameters
    ----------
    current, reference : DatasetProfile
        Profiles of the two snapshots.
    current_df, reference_df : pd.DataFrame
        The underlying DataFrames (the profiles don't hold raw values).

    Returns
    -------
    DriftReport
    """
    column_drifts: List[ColumnDrift] = []
    n_severe = n_moderate = n_no_drift = 0
    max_psi = 0.0

    # Build lookup of reference column profiles by name.
    ref_cols = {c.name: c for c in reference.columns}

    for cur_col in current.columns:
        name = cur_col.name
        if name not in ref_cols:
            continue
        ref_col = ref_cols[name]
        if cur_col.type != ref_col.type:
            # Type changed between snapshots — flag as severe drift.
            psi = 1.0
            ks_stat = None
            ks_pvalue = None
        else:
            cur_series = current_df[name]
            ref_series = reference_df[name]
            if cur_col.type == ColumnType.NUMERIC.value:
                psi = _psi_numeric(cur_series, ref_series)
                try:
                    ks_stat, ks_pvalue = stats.ks_2samp(
                        pd.to_numeric(cur_series, errors="coerce").dropna().values,
                        pd.to_numeric(ref_series, errors="coerce").dropna().values,
                    )
                    ks_stat = float(ks_stat)
                    ks_pvalue = float(ks_pvalue)
                except Exception:
                    ks_stat = None
                    ks_pvalue = None
            elif cur_col.type in (ColumnType.CATEGORICAL.value, ColumnType.BOOLEAN.value):
                psi = _psi_categorical(cur_series, ref_series)
                ks_stat = None
                ks_pvalue = None
            elif cur_col.type == ColumnType.DATETIME.value:
                # Convert to ordinal (nanoseconds since epoch) then PSI.
                # Use .astype("int64") on dropna'd series (Series.view is
                # deprecated in pandas 2.x). NaT can't convert to int64
                # directly, so dropna() first.
                cur_ts = pd.to_datetime(cur_series, errors="coerce").dropna().astype("int64").astype(float)
                ref_ts = pd.to_datetime(ref_series, errors="coerce").dropna().astype("int64").astype(float)
                psi = _psi_numeric(cur_ts, ref_ts)
                ks_stat = None
                ks_pvalue = None
            else:
                # Text columns: drift based on length distribution.
                cur_len = cur_series.dropna().astype(str).str.len()
                ref_len = ref_series.dropna().astype(str).str.len()
                psi = _psi_numeric(cur_len, ref_len)
                ks_stat = None
                ks_pvalue = None

        psi = max(psi, 0.0)
        label = _psi_bucket_label(psi)
        if label == "no_drift":
            n_no_drift += 1
        elif label == "moderate_drift":
            n_moderate += 1
        else:
            n_severe += 1
        max_psi = max(max_psi, psi)

        column_drifts.append(ColumnDrift(
            name=name,
            type=cur_col.type,
            psi=round(psi, 4),
            psi_label=label,
            ks_stat=round(ks_stat, 4) if ks_stat is not None else None,
            ks_pvalue=f"{ks_pvalue:.2e}" if ks_pvalue is not None else None,
            n_current=int(cur_col.n_total),
            n_reference=int(ref_col.n_total),
        ))

    return DriftReport(
        current_name=current.name,
        reference_name=reference.name,
        columns=column_drifts,
        n_severe=n_severe,
        n_moderate=n_moderate,
        n_no_drift=n_no_drift,
        max_psi=round(max_psi, 4),
    )


__all__ = [
    "PSI_NO_DRIFT",
    "PSI_MODERATE_DRIFT",
    "ColumnProfile",
    "DatasetProfile",
    "ProfileBuilder",
    "ColumnDrift",
    "DriftReport",
    "compute_drift",
]
