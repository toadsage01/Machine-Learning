"""
dataset
=======

Data acquisition, type inference and ETL for the AutoInsight EDA pipeline.

This module is the *only* place in P1 that knows how to talk to the outside
world (local files, HTTP URLs, data.gov.in resources). Everything downstream
(``model.py``, ``report.py``) operates on the pure-Python ``Dataset`` value
object returned here, which makes the profiling logic trivially unit-testable.

Public surface
--------------
- ``Dataset``              : frozen value object holding a DataFrame + provenance.
- ``ColumnType``           : enum of inferred column kinds.
- ``infer_column_types``   : heuristic type inference for a pandas DataFrame.
- ``clean_dataframe``      : idempotent ETL pass (column rename, dedup, NA norm).
- ``load_csv``             : load a local path OR http(s) URL into a ``Dataset``.
- ``DataGovLoader``        : thin client for data.gov.in CSV resources.

Design notes
------------
- Type inference is intentionally conservative: a column is only labelled
  ``NUMERIC`` if a non-trivial fraction of non-null values parse as floats,
  ``DATETIME`` only if >80 % parse under ``pd.to_datetime``. This avoids the
  classic "ID column inferred as numeric" foot-gun.
- ``clean_dataframe`` is **idempotent** — running it twice yields the same
  output as running it once, which is critical for caching and re-runs.
- HTTP downloads stream into ``~/.autoinsight_cache`` so re-runs of the same
  data.gov.in URL are instant and offline-friendly.
"""

from __future__ import annotations

import hashlib
import io
import os
import re
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Tuple
from urllib.parse import urlparse

import pandas as pd
import requests

# ---------------------------------------------------------------------------
# Repo-root bootstrap so that `from shared import apply_style` works from
# any project sub-directory without manual sys.path hacking.
# ---------------------------------------------------------------------------
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_CACHE_DIR = Path(os.environ.get("AUTOINSIGHT_CACHE", Path.home() / ".autoinsight_cache"))
DATETIME_PARSE_FRACTION = 0.80    # >=80 % of non-null rows must parse as datetime
NUMERIC_PARSE_FRACTION = 0.90     # >=90 % of non-null rows must parse as numeric
CARDINALITY_CATEGORICAL_THRESHOLD = 0.50  # if unique/total > this AND dtype is object -> TEXT
MAX_CATEGORY_CARDINALITY = 50     # above this, an object column is "high-card cat" (still CATEGORICAL but flagged)


class ColumnType(str, Enum):
    """Inferred semantic type for a column.

    The string values are also the CSS class names used in the HTML report,
    so changing them is a breaking change for ``report.py``.
    """

    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    TEXT = "text"
    BOOLEAN = "boolean"
    EMPTY = "empty"

    def __str__(self) -> str:  # pragma: no cover - trivial
        return self.value


# ---------------------------------------------------------------------------
# Value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Dataset:
    """Immutable container holding the loaded data and its provenance.

    Parameters
    ----------
    name : str
        Human-readable name shown in the report header.
    df : pd.DataFrame
        The (already cleaned) DataFrame.
    source : str
        Original source string (path or URL) used to load the data.
    source_kind : str
        One of "local", "http", "data_gov_in".
    sha256 : str
        SHA-256 hex digest of the raw bytes — used for caching and provenance.
    loaded_at : str
        ISO-8601 timestamp of when ``load_csv`` finished.
    """

    name: str
    df: pd.DataFrame
    source: str
    source_kind: str
    sha256: str
    loaded_at: str
    meta: Dict[str, str] = field(default_factory=dict)

    @property
    def shape(self) -> Tuple[int, int]:
        return self.df.shape


# ---------------------------------------------------------------------------
# Type inference
# ---------------------------------------------------------------------------
def _coerce_to_numeric(series: pd.Series) -> pd.Series:
    """Return numeric-coerced series WITHOUT mutating the input."""
    return pd.to_numeric(series, errors="coerce")


def _coerce_to_datetime(series: pd.Series) -> pd.Series:
    """Return datetime-coerced series WITHOUT mutating the input.

    Uses ``format='mixed'`` so that heterogeneous date strings do not emit
    a noisy UserWarning — AutoInsight is *intentionally* probing whether
    a column happens to be datetime-typed, so the warning is meaningless
    here and clutters stderr on every object column.
    """
    return pd.to_datetime(series, errors="coerce", format="mixed")


def infer_column_types(df: pd.DataFrame) -> Dict[str, ColumnType]:
    """Infer the semantic ``ColumnType`` for each column in ``df``.

    Heuristics
    ----------
    * Empty columns (all-NA) -> ``EMPTY``.
    * ``bool`` dtype -> ``BOOLEAN``.
    * Numeric dtype -> ``NUMERIC``.
    * Object/string columns are inspected:
        - If >= ``NUMERIC_PARSE_FRACTION`` of non-null values parse as
          numeric, label ``NUMERIC`` (catches string-encoded numerics).
        - Else if >= ``DATETIME_PARSE_FRACTION`` parse as datetime, label
          ``DATETIME``.
        - Else if the column has only 2 unique values (case-insensitive,
          e.g. yes/no, true/false, 0/1), label ``BOOLEAN``.
        - Else if cardinality / row_count > ``CARDINALITY_CATEGORICAL_THRESHOLD``
          AND mean token length > 4, label ``TEXT`` (free-form reviews).
        - Else ``CATEGORICAL``.
    """
    types: Dict[str, ColumnType] = {}
    n_rows = len(df)
    for col in df.columns:
        series = df[col]
        non_null = series.dropna()
        if len(non_null) == 0:
            types[col] = ColumnType.EMPTY
            continue

        # Already-typed numerics & booleans
        if pd.api.types.is_bool_dtype(series):
            types[col] = ColumnType.BOOLEAN
            continue
        if pd.api.types.is_numeric_dtype(series):
            types[col] = ColumnType.NUMERIC
            continue
        if pd.api.types.is_datetime64_any_dtype(series):
            types[col] = ColumnType.DATETIME
            continue

        # Object / category / string columns: try numeric, then datetime.
        n_non_null = len(non_null)
        if n_non_null == 0:
            types[col] = ColumnType.EMPTY
            continue

        numeric_coerced = _coerce_to_numeric(non_null)
        numeric_ok = numeric_coerced.notna().sum() / n_non_null
        if numeric_ok >= NUMERIC_PARSE_FRACTION:
            types[col] = ColumnType.NUMERIC
            continue

        datetime_coerced = _coerce_to_datetime(non_null)
        datetime_ok = datetime_coerced.notna().sum() / n_non_null
        if datetime_ok >= DATETIME_PARSE_FRACTION:
            types[col] = ColumnType.DATETIME
            continue

        # Boolean if only 2 unique values.
        uniq = non_null.astype(str).str.lower().str.strip().nunique()
        if uniq == 2:
            types[col] = ColumnType.BOOLEAN
            continue

        # Text vs Categorical — based on cardinality ratio and mean token length.
        cardinality = non_null.nunique()
        if n_rows > 0 and cardinality / n_rows > CARDINALITY_CATEGORICAL_THRESHOLD:
            mean_tokens = non_null.astype(str).str.split().str.len().mean()
            if mean_tokens is not None and mean_tokens > 4:
                types[col] = ColumnType.TEXT
                continue

        types[col] = ColumnType.CATEGORICAL

    return types


# ---------------------------------------------------------------------------
# Cleaning
# ---------------------------------------------------------------------------
_SNAKE_RE_1 = re.compile(r"(.)([A-Z][a-z]+)")
_SNAKE_RE_2 = re.compile(r"([a-z0-9])([A-Z])")


def _to_snake_case(name: str) -> str:
    """Convert ``"Avg Household Income"`` -> ``"avg_household_income"``.

    Also strips, lowercases, and replaces non-alphanumeric runs with ``_``.
    """
    name = str(name).strip()
    name = _SNAKE_RE_1.sub(r"\1_\2", name)
    name = _SNAKE_RE_2.sub(r"\1_\2", name)
    name = re.sub(r"[^0-9a-zA-Z]+", "_", name).strip("_").lower()
    return name or "unnamed_column"


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Idempotent ETL pass over a raw DataFrame.

    Operations (each is independent & order-tolerant):
        1. Normalize column names to ``snake_case``.
        2. Strip whitespace from string cells.
        3. Replace common NA tokens (``"n/a"``, ``"-"``, ``""``) with NaN.
        4. Drop fully-duplicate rows.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input frame.

    Returns
    -------
    pd.DataFrame
        Cleaned frame (a copy; the input is never mutated).
    """
    if df is None or df.empty:
        return df.copy() if df is not None else df

    out = df.copy()
    out.columns = [_to_snake_case(c) for c in out.columns]

    # Strip whitespace from object columns.
    obj_cols = out.select_dtypes(include=["object"]).columns
    for col in obj_cols:
        out[col] = out[col].astype(str).str.strip().replace(
            {"nan": pd.NA, "None": pd.NA, "": pd.NA, "n/a": pd.NA, "N/A": pd.NA, "-": pd.NA, "NA": pd.NA}
        )

    # Drop fully-duplicate rows (cheap and safe).
    out = out.drop_duplicates().reset_index(drop=True)
    return out


# ---------------------------------------------------------------------------
# SHA-256 helper (for provenance & cache keying)
# ---------------------------------------------------------------------------
def _sha256_bytes(payload: bytes) -> str:
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _detect_source_kind(source: str) -> str:
    """Classify a source string as local | http | data_gov_in."""
    if source.startswith("http://") or source.startswith("https://"):
        if "data.gov.in" in source:
            return "data_gov_in"
        return "http"
    return "local"


def _http_get(url: str, timeout: int = 30) -> bytes:
    """Stream an HTTP GET, raising on non-2xx."""
    response = requests.get(url, timeout=timeout, stream=True)
    response.raise_for_status()
    return response.content


def _cache_path_for(source: str, source_kind: str) -> Path:
    """Deterministic on-disk cache path for a given source URL/path."""
    DEFAULT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(source.encode("utf-8")).hexdigest()[:16]
    suffix = ".csv" if source_kind != "data_gov_in" else "_datagov.csv"
    return DEFAULT_CACHE_DIR / f"{source_kind}_{key}{suffix}"


def _read_csv_bytes(payload: bytes, name: str) -> pd.DataFrame:
    """Read CSV bytes into a DataFrame, tolerating common Indian-data quirks."""
    return pd.read_csv(
        io.BytesIO(payload),
        encoding="utf-8",
        encoding_errors="replace",
        low_memory=False,
        na_values=["", "NA", "N/A", "n/a", "-", "null", "NULL", "None"],
    )


def load_csv(
    source: str,
    name: Optional[str] = None,
    cache_dir: Optional[Path] = None,
    timeout: int = 30,
) -> Dataset:
    """Load a CSV from a local path, http(s) URL, or data.gov.in resource.

    Parameters
    ----------
    source : str
        Local file path OR http(s) URL.
    name : str, optional
        Display name for the dataset. Defaults to the file stem or URL path.
    cache_dir : Path, optional
        Override the default cache directory.
    timeout : int
        HTTP timeout in seconds (per request).

    Returns
    -------
    Dataset
        Cleaned dataset value object.

    Raises
    ------
    FileNotFoundError
        If ``source`` is a local path that does not exist.
    ValueError
        If the source cannot be parsed as CSV.
    """
    source_kind = _detect_source_kind(source)
    if name is None:
        if source_kind == "local":
            name = Path(source).stem
        else:
            parsed = urlparse(source)
            name = Path(parsed.path).stem or "remote_dataset"

    cache_target = (cache_dir or DEFAULT_CACHE_DIR) / f"{hashlib.sha256(source.encode()).hexdigest()[:16]}.csv"

    if source_kind == "local":
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"Local CSV not found: {path}")
        with path.open("rb") as fh:
            payload = fh.read()
    else:
        # HTTP / data.gov.in — use cache if present & fresh.
        cache_target.parent.mkdir(parents=True, exist_ok=True)
        if cache_target.exists() and cache_target.stat().st_size > 0:
            payload = cache_target.read_bytes()
        else:
            payload = _http_get(source, timeout=timeout)
            cache_target.write_bytes(payload)

    sha = _sha256_bytes(payload)
    df = _read_csv_bytes(payload, name)
    df = clean_dataframe(df)

    return Dataset(
        name=name,
        df=df,
        source=source,
        source_kind=source_kind,
        sha256=sha,
        loaded_at=pd.Timestamp.utcnow().isoformat(),
        meta={"cache_target": str(cache_target) if source_kind != "local" else ""},
    )


# ---------------------------------------------------------------------------
# data.gov.in helper
# ---------------------------------------------------------------------------
class DataGovLoader:
    """Thin wrapper around data.gov.in CSV resources.

    data.gov.in resources follow the URL pattern:
        https://data.gov.in/files/ogdp/v2/.../csv/<resource_id>_<serial>.csv
    or the older:
        https://data.gov.in/sites/default/files/<resource>.csv

    This loader does NOT scrape the catalog; it accepts a direct CSV URL
    (the kind exposed in the "Download" button on a resource page).
    """

    @staticmethod
    def load(resource_url: str, name: Optional[str] = None) -> Dataset:
        if "data.gov.in" not in resource_url:
            raise ValueError(
                f"DataGovLoader expected a data.gov.in URL, got: {resource_url}"
            )
        return load_csv(resource_url, name=name)


__all__ = [
    "Dataset",
    "ColumnType",
    "infer_column_types",
    "clean_dataframe",
    "load_csv",
    "DataGovLoader",
    "DEFAULT_CACHE_DIR",
]
