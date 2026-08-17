"""
dataset
=======

Iris dataset loader, schema definition, and reproducible train/test split
for the P2 production pipeline.

Public surface
--------------
- ``FeatureSchema``     : dataclass describing column names & roles.
- ``IrisDataset``       : frozen value object bundling X/y/train/test arrays.
- ``load_iris_data``     : load Iris from sklearn (or a local CSV fallback).
- ``build_train_test``   : stratified train/test split with explicit seed.
- ``FEATURE_NAMES``      : canonical feature name list (order matters!).
- ``TARGET_NAMES``       : canonical class name list (matches sklearn order).

Design notes
------------
- The split is **stratified** on the target so the 50/50/50 class balance
  is preserved in both train and test folds. This is critical for fair
  comparison of the four candidate models in ``train.py``.
- Feature scaling lives *inside* the sklearn Pipeline (see ``model.py``),
  not here. The dataset module exposes **raw** X so that we can compare
  scaled (LogReg, SVM) vs tree-based (RF, LightGBM) pipelines in a single
  ColumnTransformer.
- We pin a default ``random_state=42`` for reproducibility but expose it
  as a parameter so hyperparam search can sweep over it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple, Optional

import numpy as np
import pandas as pd
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split


# ---------------------------------------------------------------------------
# Canonical schema — single source of truth for feature & target names
# ---------------------------------------------------------------------------
FEATURE_NAMES: tuple[str, ...] = (
    "sepal_length_cm",
    "sepal_width_cm",
    "petal_length_cm",
    "petal_width_cm",
)

TARGET_NAMES: tuple[str, ...] = (
    "setosa",
    "versicolor",
    "virginica",
)


@dataclass(frozen=True)
class FeatureSchema:
    """Declarative schema used by the ColumnTransformer in ``model.py``."""

    numeric_features: tuple[str, ...] = FEATURE_NAMES
    categorical_features: tuple[str, ...] = ()  # Iris has none; kept for symmetry
    target_names: tuple[str, ...] = TARGET_NAMES

    @property
    def n_features(self) -> int:
        return len(self.numeric_features) + len(self.categorical_features)


# ---------------------------------------------------------------------------
# Dataset value object
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class IrisDataset:
    """Immutable bundle of train/test arrays + metadata.

    Attributes
    ----------
    X_train, X_test : np.ndarray
        Raw (unscaled) feature matrices, shape (n_samples, 4).
    y_train, y_test : np.ndarray
        Integer class labels in {0, 1, 2}.
    feature_names : tuple[str, ...]
        Ordered feature names; X columns are aligned with this tuple.
    target_names : tuple[str, ...]
        Class names indexed by integer label.
    random_state : int
        Seed used for the split (recorded for provenance).
    test_size : float
        Fraction held out for testing.
    source : str
        Where the data came from ("sklearn" or path to CSV).
    """

    X_train: np.ndarray
    X_test: np.ndarray
    y_train: np.ndarray
    y_test: np.ndarray
    feature_names: tuple[str, ...]
    target_names: tuple[str, ...]
    random_state: int
    test_size: float
    source: str
    n_train: int = field(init=False)
    n_test: int = field(init=False)

    def __post_init__(self) -> None:
        # Use object.__setattr__ because the dataclass is frozen.
        object.__setattr__(self, "n_train", int(self.X_train.shape[0]))
        object.__setattr__(self, "n_test", int(self.X_test.shape[0]))

    def as_dataframe(self, split: str = "train") -> pd.DataFrame:
        """Return the train or test split as a labelled DataFrame."""
        X = self.X_train if split == "train" else self.X_test
        y = self.y_train if split == "train" else self.y_test
        df = pd.DataFrame(X, columns=list(self.feature_names))
        df["target"] = y
        df["target_name"] = [self.target_names[i] for i in y]
        return df


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def load_iris_data(csv_path: Optional[Path | str] = None) -> Tuple[np.ndarray, np.ndarray]:
    """Load the Iris dataset, optionally from a local CSV.

    Parameters
    ----------
    csv_path : str or Path, optional
        If provided, loads from a CSV with columns matching
        ``FEATURE_NAMES`` plus a ``target`` column (integer-coded 0/1/2).
        If ``None`` (default), loads from ``sklearn.datasets.load_iris``.

    Returns
    -------
    (X, y) : tuple[np.ndarray, np.ndarray]
        X has shape (150, 4) float64; y has shape (150,) int64 in {0,1,2}.

    Raises
    ------
    FileNotFoundError
        If ``csv_path`` is given but does not exist.
    ValueError
        If the CSV does not contain the expected columns.
    """
    if csv_path is None:
        bunch = load_iris()
        X = bunch.data.astype(np.float64)
        y = bunch.target.astype(np.int64)
        return X, y

    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"Iris CSV not found: {path}")

    df = pd.read_csv(path)
    expected = set(FEATURE_NAMES) | {"target"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(
            f"CSV at {path} is missing required columns: {sorted(missing)}"
        )
    X = df[list(FEATURE_NAMES)].to_numpy(dtype=np.float64)
    y = df["target"].to_numpy(dtype=np.int64)
    return X, y


# ---------------------------------------------------------------------------
# Train/test split
# ---------------------------------------------------------------------------
def build_train_test(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.2,
    random_state: int = 42,
    stratify: Optional[np.ndarray] = None,
    source: str = "sklearn",
    feature_names: tuple[str, ...] = FEATURE_NAMES,
    target_names: tuple[str, ...] = TARGET_NAMES,
) -> IrisDataset:
    """Stratified train/test split returning an immutable ``IrisDataset``.

    Parameters
    ----------
    X, y : np.ndarray
        Features and integer labels.
    test_size : float
        Fraction of data to hold out for testing (default 0.20 = 30 of 150).
    random_state : int
        Seed for reproducibility.
    stratify : np.ndarray, optional
        Defaults to ``y`` to enforce stratification on the target. Pass
        ``None`` to disable.
    source : str
        Provenance label (e.g. "sklearn" or a CSV path).
    feature_names, target_names : tuple[str, ...]
        Schema overrides (rarely needed; the Iris defaults are canonical).
    """
    if stratify is None:
        stratify = y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    return IrisDataset(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_names=feature_names,
        target_names=target_names,
        random_state=random_state,
        test_size=test_size,
        source=source,
    )


# ---------------------------------------------------------------------------
# Convenience: one-call loader that returns the split
# ---------------------------------------------------------------------------
def load_iris_split(
    csv_path: Optional[Path | str] = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> IrisDataset:
    """One-call helper: load Iris and split in a single step."""
    X, y = load_iris_data(csv_path)
    source = "sklearn" if csv_path is None else str(csv_path)
    return build_train_test(
        X, y,
        test_size=test_size,
        random_state=random_state,
        source=source,
    )


__all__ = [
    "FeatureSchema",
    "IrisDataset",
    "FEATURE_NAMES",
    "TARGET_NAMES",
    "load_iris_data",
    "build_train_test",
    "load_iris_split",
]
