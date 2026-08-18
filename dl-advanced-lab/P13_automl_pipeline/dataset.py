"""
dataset
=======

Dataset profiler and automated schema inference engine with synthetic
dataset generators for regression and classification tasks.

Public surface
--------------
- ``ColumnType``            : enum (NUMERIC, CATEGORICAL, DATETIME, TEXT, BINARY, ID).
- ``ColumnProfile``         : per-column profile (type, stats, missingness).
- ``DatasetProfile``         : full dataset profile (columns, target, task type).
- ``infer_schema``          : automated schema inference from a DataFrame.
- ``generate_classification_data`` : synthetic classification dataset.
- ``generate_regression_data``      : synthetic regression dataset.
- ``load_automl_dataset``    : one-call loader (CSV | synthetic).

Design notes
------------
1. **Schema inference is heuristic** — we classify each column by
   checking dtype, cardinality, and parse success rates. A column with
   < 50 unique values and dtype=object is likely categorical; one that
   parses as datetime in >80% of rows is datetime.

2. **Synthetic generators** — we provide both classification (n-class
   with informative features) and regression (linear + nonlinear target)
   generators so the pipeline can be smoke-tested without real data.
"""

from __future__ import annotations

import hashlib
import logging
import sys
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("automl_dataset")


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------
class ColumnType(str, Enum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BINARY = "binary"
    ID = "id"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


class TaskType(str, Enum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass
class ColumnProfile:
    """Profile of a single column."""

    name: str
    type: str  # ColumnType.value
    dtype: str
    n_unique: int
    n_missing: int
    missing_pct: float
    # Numeric stats.
    mean: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    # Categorical stats.
    top_values: Optional[List[Tuple[str, int]]] = None
    # Datetime stats.
    datetime_min: Optional[str] = None
    datetime_max: Optional[str] = None

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class DatasetProfile:
    """Full dataset profile."""

    n_rows: int
    n_cols: int
    columns: List[ColumnProfile]
    target_column: Optional[str]
    task_type: Optional[str]
    memory_mb: float
    n_numeric: int = 0
    n_categorical: int = 0
    n_datetime: int = 0
    n_binary: int = 0
    n_text: int = 0
    n_id: int = 0

    def to_dict(self) -> dict:
        return {
            **{k: v for k, v in asdict(self).items() if k != "columns"},
            "columns": [c.to_dict() for c in self.columns],
        }


# ---------------------------------------------------------------------------
# Schema inference
# ---------------------------------------------------------------------------
def _infer_column_type(series: pd.Series, name: str) -> str:
    """Infer the semantic type of a single column."""
    if series.dtype == bool:
        return ColumnType.BINARY.value

    # Try numeric first.
    if pd.api.types.is_numeric_dtype(series):
        # Check if it's an ID (sequential integers, high cardinality).
        if pd.api.types.is_integer_dtype(series):
            n_unique = series.nunique()
            if n_unique == len(series) and (series == series.iloc[0] + np.arange(len(series))).all():
                return ColumnType.ID.value
        # Check binary (only 0 and 1).
        unique_vals = series.dropna().unique()
        if len(unique_vals) <= 2 and set(unique_vals).issubset({0, 1, 0.0, 1.0}):
            return ColumnType.BINARY.value
        return ColumnType.NUMERIC.value

    # Try datetime.
    if pd.api.types.is_datetime64_any_dtype(series):
        return ColumnType.DATETIME.value

    # Object/string columns.
    non_null = series.dropna()
    if len(non_null) == 0:
        return ColumnType.CATEGORICAL.value

    n_unique = non_null.nunique()
    # Low cardinality → categorical.
    if n_unique <= 50:
        return ColumnType.CATEGORICAL.value

    # Try parsing as datetime.
    try:
        parsed = pd.to_datetime(non_null.head(min(100, len(non_null))), errors="coerce")
        if parsed.notna().sum() / len(parsed) > 0.8:
            return ColumnType.DATETIME.value
    except Exception:
        pass

    # High cardinality string → text (or ID if fully unique).
    if n_unique == len(series):
        return ColumnType.ID.value

    return ColumnType.TEXT.value


def infer_schema(
    df: pd.DataFrame,
    target_column: Optional[str] = None,
) -> DatasetProfile:
    """Infer the schema of a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
    target_column : str, optional
        If provided, the task type (classification/regression) is inferred
        from the target's column type.

    Returns
    -------
    DatasetProfile
    """
    columns: List[ColumnProfile] = []
    type_counts: Dict[str, int] = {}

    for col in df.columns:
        series = df[col]
        col_type = _infer_column_type(series, col)
        type_counts[col_type] = type_counts.get(col_type, 0) + 1

        profile = ColumnProfile(
            name=col,
            type=col_type,
            dtype=str(series.dtype),
            n_unique=int(series.nunique(dropna=True)),
            n_missing=int(series.isna().sum()),
            missing_pct=round(100.0 * series.isna().sum() / max(len(series), 1), 4),
        )

        if col_type in (ColumnType.NUMERIC.value, ColumnType.BINARY.value):
            numeric = pd.to_numeric(series, errors="coerce")
            profile.mean = float(numeric.mean()) if not numeric.isna().all() else None
            profile.std = float(numeric.std()) if not numeric.isna().all() else None
            profile.min = float(numeric.min()) if not numeric.isna().all() else None
            profile.max = float(numeric.max()) if not numeric.isna().all() else None
        elif col_type == ColumnType.CATEGORICAL.value:
            vc = series.value_counts(dropna=True).head(5)
            profile.top_values = [(str(k), int(v)) for k, v in vc.items()]
        elif col_type == ColumnType.DATETIME.value:
            try:
                dt = pd.to_datetime(series, errors="coerce").dropna()
                if len(dt) > 0:
                    profile.datetime_min = str(dt.min())
                    profile.datetime_max = str(dt.max())
            except Exception:
                pass

        columns.append(profile)

    # Infer task type from target.
    task_type = None
    if target_column and target_column in df.columns:
        target_profile = next(c for c in columns if c.name == target_column)
        # If the target has <= 20 unique values, treat as classification
        # (even if numeric — integer targets with few levels are class labels).
        if target_profile.n_unique <= 20:
            task_type = TaskType.CLASSIFICATION.value
        elif target_profile.type in (ColumnType.CATEGORICAL.value, ColumnType.BINARY.value):
            task_type = TaskType.CLASSIFICATION.value
        else:
            task_type = TaskType.REGRESSION.value

    memory_mb = round(df.memory_usage(deep=True).sum() / 1024 / 1024, 4)

    return DatasetProfile(
        n_rows=len(df),
        n_cols=len(df.columns),
        columns=columns,
        target_column=target_column,
        task_type=task_type,
        memory_mb=memory_mb,
        n_numeric=type_counts.get(ColumnType.NUMERIC.value, 0),
        n_categorical=type_counts.get(ColumnType.CATEGORICAL.value, 0),
        n_datetime=type_counts.get(ColumnType.DATETIME.value, 0),
        n_binary=type_counts.get(ColumnType.BINARY.value, 0),
        n_text=type_counts.get(ColumnType.TEXT.value, 0),
        n_id=type_counts.get(ColumnType.ID.value, 0),
    )


# ---------------------------------------------------------------------------
# Synthetic dataset generators
# ---------------------------------------------------------------------------
def generate_classification_data(
    n_samples: int = 1000,
    n_features: int = 10,
    n_classes: int = 3,
    n_categorical: int = 3,
    n_datetime: int = 1,
    noise: float = 0.5,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic classification dataset with mixed-type features.

    Features:
        * ``num_0`` .. ``num_{n_features-1}`` : numeric (float64) from a
          make_classification-style generator.
        * ``cat_0`` .. ``cat_{n_categorical-1}`` : categorical (string)
          with 5-10 levels.
        * ``dt_0`` .. ``dt_{n_datetime-1}`` : datetime columns.
        * ``target`` : integer class label (0..n_classes-1).

    Some columns have injected missingness (~5%).
    """
    rng = np.random.default_rng(seed)
    from sklearn.datasets import make_classification
    X_num, y = make_classification(
        n_samples=n_samples, n_features=n_features,
        n_informative=max(n_features // 2, 2),
        n_redundant=max(n_features // 4, 1),
        n_classes=n_classes, n_clusters_per_class=1,
        flip_y=noise * 0.1,  # fraction of label noise.
        random_state=seed,
    )

    df = pd.DataFrame()
    for i in range(n_features):
        col = X_num[:, i].astype(float)
        # Inject ~5% missingness.
        mask = rng.random(n_samples) < 0.05
        col[mask] = np.nan
        df[f"num_{i}"] = col

    for i in range(n_categorical):
        n_levels = rng.integers(3, 10)
        levels = [f"category_{j}" for j in range(n_levels)]
        cat = rng.choice(levels, size=n_samples)
        mask = rng.random(n_samples) < 0.03
        cat[mask] = None
        df[f"cat_{i}"] = cat

    for i in range(n_datetime):
        base = pd.Timestamp("2020-01-01")
        offsets = rng.integers(0, 365 * 3, size=n_samples)
        df[f"dt_{i}"] = [base + pd.Timedelta(days=int(o)) for o in offsets]

    df["target"] = y
    return df


def generate_regression_data(
    n_samples: int = 1000,
    n_features: int = 8,
    n_categorical: int = 2,
    noise: float = 0.3,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic regression dataset with mixed-type features.

    The target is a linear combination of the numeric features + one-hot
    encoded categorical features + noise.

    Target: ``target`` (float64, continuous).
    """
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n_samples, n_features))

    # Linear coefficients.
    coef = rng.uniform(-2, 2, size=n_features)
    y = X @ coef + rng.normal(0, noise, size=n_samples)

    df = pd.DataFrame()
    for i in range(n_features):
        col = X[:, i].astype(float)
        mask = rng.random(n_samples) < 0.05
        col[mask] = np.nan
        df[f"num_{i}"] = col

    for i in range(n_categorical):
        n_levels = rng.integers(3, 8)
        levels = [f"level_{j}" for j in range(n_levels)]
        cat = rng.choice(levels, size=n_samples)
        mask = rng.random(n_samples) < 0.03
        cat[mask] = None
        df[f"cat_{i}"] = cat

    df["target"] = y.astype(float)
    return df


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_automl_dataset(
    csv_path: Optional[Path | str] = None,
    target_column: Optional[str] = None,
    task: str = "classification",
    n_samples: int = 500,
    n_features: int = 8,
    seed: int = 42,
) -> Tuple[pd.DataFrame, DatasetProfile]:
    """One-call loader: returns (DataFrame, DatasetProfile).

    Resolution:
        1. CSV path → load + infer schema.
        2. Synthetic generator based on ``task``.
    """
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")
        df = pd.read_csv(path)
        source = str(path)
    else:
        log.info("Generating synthetic %s data (n=%d, features=%d, seed=%d)",
                 task, n_samples, n_features, seed)
        if task == "classification":
            df = generate_classification_data(
                n_samples=n_samples, n_features=n_features, seed=seed,
            )
        else:
            df = generate_regression_data(
                n_samples=n_samples, n_features=n_features, seed=seed,
            )
        source = "synthetic"
        if target_column is None:
            target_column = "target"

    profile = infer_schema(df, target_column=target_column)
    log.info("Schema inferred: %d rows × %d cols, %d numeric, %d categorical, %d datetime",
             profile.n_rows, profile.n_cols, profile.n_numeric,
             profile.n_categorical, profile.n_datetime)
    if profile.task_type:
        log.info("  Task type: %s (target=%s)", profile.task_type, profile.target_column)

    return df, profile


__all__ = [
    "ColumnType",
    "TaskType",
    "ColumnProfile",
    "DatasetProfile",
    "infer_schema",
    "generate_classification_data",
    "generate_regression_data",
    "load_automl_dataset",
]
