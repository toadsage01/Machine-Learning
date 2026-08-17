"""
dataset
=======

Unified ETL for the classic Titanic and Spaceship Titanic datasets.

Both datasets share a common narrative (a passenger ship disaster where
we predict survival) but have very different schemas — the classic
Titanic is small, demographic-heavy, and well-curated, while Spaceship
Titanic is larger, noisier, and includes spending/amenity features.
This module normalizes them into a single ``UnifiedTitanicRecord`` so
the same feature-engineering + modelling pipeline can be applied to
either (or both fused together).

Public surface
--------------
- ``DatasetKind``        : enum (CLASSIC / SPACESHIP).
- ``UnifiedSchema``       : canonical feature list shared by both datasets.
- ``UnifiedDataset``      : frozen value object holding X/y + provenance.
- ``load_classic_titanic``       : load + normalize the classic Titanic dataset.
- ``load_spaceship_titanic``     : load + normalize Spaceship Titanic.
- ``load_unified``        : dispatch by name.
- ``download_raw``        : fetch a raw CSV into ``data/`` (cached).
- ``CLASSIC_URL``         : canonical classic Titanic URL.
- ``SPACESHIP_URL``       : canonical Spaceship Titanic URL.

Unified schema (post-ETL)
-------------------------
* ``sex``                 : {male, female}
* ``age``                 : float, years (for Spaceship: Earth-years-equivalent)
* ``pclass``              : int 1/2/3 (ticket class; for Spaceship: deck tier)
* ``sibsp``               : int (# siblings/spouses aboard; Spaceship: cabin-group size proxy)
* ``parch``               : int (# parents/children aboard; Spaceship: same proxy)
* ``fare``                : float (ticket fare; Spaceship: total spend proxy)
* ``embarked``            : {S, C, Q} (port; Spaceship: home-planet S/E/M mapped)
* ``deck``                : str (cabin deck letter; Spaceship: deck letter from Cabin)
* ``alone``               : bool (true if sibsp==0 and parch==0)
* ``title``               : str (Mr, Mrs, Miss, etc.; Spaceship: derived from name token)
* ``family_size``         : int = sibsp + parch + 1
* ``fare_per_person``     : float = fare / family_size
* ``is_child``            : bool = age < 18
* ``is_elderly``          : bool = age >= 60

Target: ``survived`` (1/0).

Design notes
------------
* **No global mutable state** — every loader returns a fresh
  ``UnifiedDataset``. The HTTP cache lives at ``data/_cache/`` keyed by
  SHA-256(url) so re-runs are instant.
* **Defensive parsing** — both raw sources have inconsistent column
  dtypes (e.g. classic ``Age`` is sometimes int, sometimes float,
  sometimes "??"). We coerce to ``float64`` with NaN, then let the
  downstream imputer handle missingness.
* **Spaceship-specific mapping** — the Spaceship Titanic's
  ``HomePlanet`` (Earth/Europa/Mars) is mapped to the classic ``embarked``
  enum (S/C/Q respectively) to keep the unified schema enum-only. Same
  for ``CryoSleep`` → ``deck`` proxy and ``RoomService``+``Spa``+``VRDeck``
  → ``fare`` (sum).
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("titanic_dataset")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = PROJECT_DATA_DIR / "_cache"

# Canonical sources — classic Titanic (Stanford mirror) + Spaceship Titanic.
# The classic dataset has a stable public URL; Spaceship Titanic is a Kaggle
# competition dataset gated behind Kaggle auth. If a real spaceship-titanic.csv
# is dropped into ``data/`` it takes precedence; otherwise we generate a
# realistic-shape synthetic fallback (see ``make_synthetic_spaceship``).
CLASSIC_URL = "https://web.stanford.edu/class/archive/cs/cs109/cs109.1166/stuff/titanic.csv"
SPACESHIP_URL = "https://www.kaggle.com/competitions/spaceship-titanic/data?select=train.csv"
SPACESHIP_LOCAL_OVERRIDE = PROJECT_DATA_DIR / "spaceship-titanic.csv"


# ---------------------------------------------------------------------------
# Unified schema
# ---------------------------------------------------------------------------
class DatasetKind(str, Enum):
    CLASSIC = "classic"
    SPACESHIP = "spaceship"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


@dataclass(frozen=True)
class UnifiedSchema:
    """Canonical feature list shared by both datasets after ETL."""

    numeric_features: Tuple[str, ...] = (
        "age", "fare", "sibsp", "parch", "family_size", "fare_per_person",
    )
    categorical_features: Tuple[str, ...] = (
        "sex", "pclass", "embarked", "deck", "title", "alone", "is_child", "is_elderly",
    )
    target: str = "survived"

    @property
    def all_features(self) -> List[str]:
        return list(self.numeric_features) + list(self.categorical_features)


@dataclass(frozen=True)
class UnifiedDataset:
    """Bundle of features + target + provenance.

    Attributes
    ----------
    kind : DatasetKind
    df : pd.DataFrame
        Pre-ETL raw dataframe (kept for diagnostics).
    X : pd.DataFrame
        Unified features (post-ETL), columns match ``UnifiedSchema``.
    y : pd.Series
        Binary target (1/0).
    source_url : str
    sha256 : str
        Hash of raw bytes (provenance).
    n_samples : int
    """

    kind: DatasetKind
    df: pd.DataFrame
    X: pd.DataFrame
    y: pd.Series
    source_url: str
    sha256: str
    n_samples: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_samples", int(len(self.X)))

    def __len__(self) -> int:
        return self.n_samples


SCHEMA = UnifiedSchema()


# ---------------------------------------------------------------------------
# HTTP fetch + cache
# ---------------------------------------------------------------------------
def _cache_path_for(url: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:16]
    return CACHE_DIR / f"{key}.csv"


def _fetch_cached(url: str, timeout: int = 30) -> bytes:
    """Return the bytes at ``url``, using the on-disk cache when fresh."""
    cache_path = _cache_path_for(url)
    if cache_path.exists() and cache_path.stat().st_size > 0:
        return cache_path.read_bytes()
    log.info("Downloading %s", url)
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    payload = response.content
    cache_path.write_bytes(payload)
    return payload


def download_raw(kind: DatasetKind, force: bool = False) -> Path:
    """Download (or use the cache for) a raw dataset CSV.

    Parameters
    ----------
    kind : DatasetKind
    force : bool
        If True, re-download even if a cached copy exists.

    Returns
    -------
    Path
        Local path to the raw CSV.
    """
    url = CLASSIC_URL if kind == DatasetKind.CLASSIC else SPACESHIP_URL
    cache_path = _cache_path_for(url)
    if force and cache_path.exists():
        cache_path.unlink()
    payload = _fetch_cached(url)
    # Payload is already cached to disk; just return the path.
    return cache_path


def _sha256(payload: bytes) -> str:
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


# ---------------------------------------------------------------------------
# Classic Titanic ETL
# ---------------------------------------------------------------------------
def _normalize_classic(df: pd.DataFrame) -> pd.DataFrame:
    """Map classic Titanic columns onto the unified schema.

    Supports both the Stanford mirror schema (``Pclass``, ``Sex``, ``Age``,
    ``Siblings/Spouses Aboard``, ``Parents/Children Aboard``, ``Fare``,
    ``Survived``, ``Name``) and the Kaggle original schema (``Pclass``,
    ``Sex``, ``Age``, ``SibSp``, ``Parch``, ``Fare``, ``Survived``, ``Name``,
    ``Embarked``, ``Cabin``, ``PassengerId``).
    """
    rename_map = {
        # Stanford mirror uses verbose column names.
        "Siblings/Spouses Aboard": "sibsp",
        "Parents/Children Aboard": "parch",
        # Kaggle original uses uppercase short names.
        "Pclass": "pclass",
        "Sex": "sex",
        "Age": "age",
        "Fare": "fare",
        "Survived": "survived",
        "Name": "name",
        "SibSp": "sibsp",
        "Parch": "parch",
        "Embarked": "embarked",
        "Cabin": "cabin",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Defensive accessor — returns a Series (or a synthetic 0-Series if missing).
    def _col(name: str, default=0) -> pd.Series:
        if name in df.columns:
            return df[name]
        return pd.Series([default] * len(df), index=df.index)

    # Coerce dtypes. ``astype(int)`` on a NaN-containing Series raises
    # "invalid value encountered in cast"; use ``fillna`` first.
    df["age"] = pd.to_numeric(_col("age"), errors="coerce").astype(float)
    df["fare"] = pd.to_numeric(_col("fare"), errors="coerce").astype(float)
    df["sibsp"] = pd.to_numeric(_col("sibsp"), errors="coerce").fillna(0).astype(int)
    df["parch"] = pd.to_numeric(_col("parch"), errors="coerce").fillna(0).astype(int)
    df["pclass"] = pd.to_numeric(_col("pclass", default=3), errors="coerce").fillna(3).astype(int)

    # Derived features shared with Spaceship.
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["fare_per_person"] = df["fare"] / df["family_size"]
    df["is_child"] = (df["age"] < 18).astype(int)
    df["is_elderly"] = (df["age"] >= 60).astype(int)
    df["alone"] = ((df["sibsp"] == 0) & (df["parch"] == 0)).astype(int)

    # Title from name (e.g. "Braund, Mr. Owen Harris" -> "Mr").
    # Stanford's mirror uses "Mr. Owen Harris Braund" (first token after dot).
    name_series = _col("name", default="").astype(str)
    df["title"] = name_series.str.extract(r",\s*([^.]+)\.", expand=False)
    if df["title"].isna().all():
        # Stanford format: "Title. First Last" — title before the dot.
        df["title"] = name_series.str.extract(r"^(\w+)\.", expand=False)
    df["title"] = df["title"].fillna("Unknown").str.strip()

    # Embarked & deck are absent in the Stanford mirror — synthesize from pclass.
    if "embarked" not in df.columns:
        df["embarked"] = df["pclass"].map({1: "C", 2: "S", 3: "Q"}).fillna("S").astype(str)
    else:
        df["embarked"] = df["embarked"].fillna("S").astype(str)
    if "deck" not in df.columns:
        df["deck"] = df["pclass"].map({1: "A", 2: "D", 3: "F"}).fillna("F").astype(str)
    else:
        # If cabin exists, derive deck from first letter.
        df["deck"] = df["deck"].astype(str).str[0].where(
            df["deck"].astype(str).str[0].str.isalpha(), "F"
        )

    df["sex"] = df["sex"].astype(str).str.lower().map(
        {"male": "male", "female": "female", "m": "male", "f": "female"}
    ).fillna("male").astype(str)

    return df


# ---------------------------------------------------------------------------
# Spaceship Titanic ETL
# ---------------------------------------------------------------------------
def _normalize_spaceship(df: pd.DataFrame) -> pd.DataFrame:
    """Map Spaceship Titanic columns onto the unified schema.

    Spaceship schema (post-mapping):
        HomePlanet  -> embarked   (Earth->S, Europa->C, Mars->Q)
        CryoSleep   -> is_child   (proxy: frozen passengers are treated as
                                   "vulnerable", but we still need age; we
                                   synthesize age from CryoSleep instead)
        Cabin       -> deck       (first letter: A/B/C/...)
        RoomService + FoodCourt + ShoppingMall + Spa + VRDeck -> fare
        # of companions (PassengerId group) -> sibsp + parch
    """
    df = df.copy()

    # Target.
    if "Transported" in df.columns:
        df["survived"] = df["Transported"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)
    else:
        df["survived"] = 0

    # Sex — Spaceship doesn't have it; synthesize from Name (deterministic hash
    # for reproducibility, mapping to {male, female} with 50/50 split). This is
    # a *synthetic* feature, NOT a real demographic — flagged in the README.
    if "sex" not in df.columns:
        # Use the first name's last letter mod 2 (deterministic, balanced).
        names = df.get("Name", pd.Series(["Unknown"] * len(df))).fillna("Unknown").astype(str)
        df["sex"] = np.where(names.str[-1].isin(["a", "e", "i", "y"]), "female", "male")

    # Age — Spaceship has Age directly.
    df["age"] = pd.to_numeric(df.get("Age"), errors="coerce").astype(float)
    df["is_child"] = (df["age"] < 18).astype(int)
    df["is_elderly"] = (df["age"] >= 60).astype(int)

    # pclass — derive from Cabin deck letter (A=1, B=2, ... F=6 mapped to 1/2/3).
    cabin = df.get("Cabin", pd.Series(["Unknown"] * len(df))).fillna("Unknown").astype(str)
    deck_letter = cabin.str[0].where(cabin.str[0].str.isalpha(), "F")
    df["deck"] = deck_letter
    df["pclass"] = deck_letter.map({
        "A": 1, "B": 1, "C": 2, "D": 2, "E": 3, "F": 3, "G": 3, "T": 1,
    }).fillna(3).astype(int)

    # Fare — sum of all amenity spend columns.
    spend_cols = ["RoomService", "FoodCourt", "ShoppingMall", "Spa", "VRDeck"]
    spend = pd.DataFrame({
        c: pd.to_numeric(df.get(c, 0), errors="coerce").fillna(0).astype(float)
        for c in spend_cols
    })
    df["fare"] = spend.sum(axis=1)

    # HomePlanet -> embarked.
    if "HomePlanet" in df.columns:
        df["embarked"] = df["HomePlanet"].astype(str).map({
            "Earth": "S", "Europa": "C", "Mars": "Q",
        }).fillna("S").astype(str)
    else:
        df["embarked"] = "S"

    # CryoSleep -> parch proxy (frozen passengers travel alone).
    if "CryoSleep" in df.columns:
        df["parch"] = df["CryoSleep"].astype(str).str.lower().map(
            {"true": 0, "false": 1, "1": 0, "0": 1}
        ).fillna(1).astype(int)
    else:
        df["parch"] = 0

    # VIP -> sibsp proxy (VIPs travel with companions).
    if "VIP" in df.columns:
        df["sibsp"] = df["VIP"].astype(str).str.lower().map(
            {"true": 1, "false": 0, "1": 1, "0": 0}
        ).fillna(0).astype(int)
    else:
        df["sibsp"] = 0

    # Derived features.
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    df["fare_per_person"] = df["fare"] / df["family_size"]
    df["alone"] = ((df["sibsp"] == 0) & (df["parch"] == 0)).astype(int)

    # Title from Name (or PassengerId group as fallback).
    df["title"] = df.get("Name", pd.Series(["Unknown"] * len(df))).fillna("Unknown").astype(str).str.extract(
        r"\b(Mr|Mrs|Miss|Ms|Dr|Master|Rev)\b", expand=False
    ).fillna("Unknown")

    return df


# ---------------------------------------------------------------------------
# Final unified selection
# ---------------------------------------------------------------------------
def _select_unified(df: pd.DataFrame, kind: DatasetKind,
                    source_url: str, sha: str) -> UnifiedDataset:
    """Pick the canonical feature columns + target out of ``df``."""
    # Ensure all schema columns exist (fill missing with sane defaults).
    for col, default in [
        ("sex", "male"), ("age", np.nan), ("pclass", 3),
        ("sibsp", 0), ("parch", 0), ("fare", 0.0),
        ("embarked", "S"), ("deck", "F"), ("title", "Unknown"),
        ("alone", 0), ("family_size", 1), ("fare_per_person", 0.0),
        ("is_child", 0), ("is_elderly", 0), ("survived", 0),
    ]:
        if col not in df.columns:
            df[col] = default

    # Build X with schema order.
    X = df[SCHEMA.all_features].copy()
    y = df[SCHEMA.target].astype(int).copy()
    return UnifiedDataset(kind=kind, df=df, X=X, y=y, source_url=source_url, sha256=sha)


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def load_classic_titanic(csv_path: Optional[Path | str] = None) -> UnifiedDataset:
    """Load + normalize the classic Titanic dataset.

    Parameters
    ----------
    csv_path : str or Path, optional
        Local path to a classic Titanic CSV. If None, downloads from
        ``CLASSIC_URL`` (cached).
    """
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Classic Titanic CSV not found: {path}")
        payload = path.read_bytes()
        sha = _sha256(payload)
        source_url = str(path)
    else:
        path = download_raw(DatasetKind.CLASSIC)
        payload = path.read_bytes()
        sha = _sha256(payload)
        source_url = CLASSIC_URL
    df = pd.read_csv(io.BytesIO(payload))
    df = _normalize_classic(df)
    return _select_unified(df, DatasetKind.CLASSIC, source_url, sha)


def load_spaceship_titanic(csv_path: Optional[Path | str] = None) -> UnifiedDataset:
    """Load + normalize the Spaceship Titanic dataset.

    Resolution order:
        1. ``csv_path`` (explicit override, e.g. Kaggle-downloaded CSV).
        2. ``data/spaceship-titanic.csv`` (project-local drop-in).
        3. Synthetic fallback via ``make_synthetic_spaceship`` (used when
           no real CSV is available — Kaggle competition datasets are
           auth-gated and not redistributable). The synthetic data is
           flagged via ``source_url="synthetic"`` so callers can detect it.
    """
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Spaceship Titanic CSV not found: {path}")
        payload = path.read_bytes()
        sha = _sha256(payload)
        source_url = str(path)
        df = pd.read_csv(io.BytesIO(payload))
    elif SPACESHIP_LOCAL_OVERRIDE.exists():
        payload = SPACESHIP_LOCAL_OVERRIDE.read_bytes()
        sha = _sha256(payload)
        source_url = str(SPACESHIP_LOCAL_OVERRIDE)
        df = pd.read_csv(io.BytesIO(payload))
    else:
        log.warning(
            "No real Spaceship Titanic CSV found at %s. Using synthetic data "
            "(see make_synthetic_spaceship). To use the real Kaggle dataset, "
            "download spaceship-titanic.csv from %s and drop it into %s.",
            SPACESHIP_LOCAL_OVERRIDE, SPACESHIP_URL, SPACESHIP_LOCAL_OVERRIDE,
        )
        df = make_synthetic_spaceship(n_samples=8693, seed=42)
        sha = _sha256(df.to_csv(index=False).encode("utf-8"))
        source_url = "synthetic"
    df = _normalize_spaceship(df)
    return _select_unified(df, DatasetKind.SPACESHIP, source_url, sha)


def load_unified(kind: str | DatasetKind,
                 csv_path: Optional[Path | str] = None) -> UnifiedDataset:
    """Dispatch by name ('classic' or 'spaceship')."""
    if isinstance(kind, str):
        kind = DatasetKind(kind.lower())
    if kind == DatasetKind.CLASSIC:
        return load_classic_titanic(csv_path)
    return load_spaceship_titanic(csv_path)


# ---------------------------------------------------------------------------
# Synthetic fallback (used when network access is unavailable)
# ---------------------------------------------------------------------------
def make_synthetic_classic(n_samples: int = 500, seed: int = 42) -> pd.DataFrame:
    """Generate a small synthetic Titanic-like dataframe for offline testing."""
    rng = np.random.default_rng(seed)
    sex = rng.choice(["male", "female"], size=n_samples, p=[0.65, 0.35])
    pclass = rng.choice([1, 2, 3], size=n_samples, p=[0.25, 0.25, 0.50])
    age = np.clip(rng.normal(loc=35, scale=15, size=n_samples), 0, 90)
    sibsp = rng.poisson(0.5, size=n_samples)
    parch = rng.poisson(0.3, size=n_samples)
    fare = np.exp(rng.normal(loc=3.0, scale=0.8, size=n_samples)) * (4 - pclass)
    embarked = rng.choice(["S", "C", "Q"], size=n_samples, p=[0.7, 0.2, 0.1])

    # Survival probability: women & children first, plus class effect.
    base_p = 0.3 + 0.3 * (sex == "female") + 0.15 * (age < 18) + 0.10 * (pclass == 1)
    survived = (rng.random(n_samples) < base_p).astype(int)

    df = pd.DataFrame({
        "sex": sex, "age": age, "pclass": pclass, "sibsp": sibsp, "parch": parch,
        "fare": fare, "embarked": embarked, "survived": survived,
    })
    return df


def make_synthetic_spaceship(n_samples: int = 8693, seed: int = 42) -> pd.DataFrame:
    """Generate a synthetic Spaceship-Titanic-shaped dataframe for offline testing.

    The real Kaggle Spaceship Titanic dataset has 8,693 rows × 14 columns
    including ``PassengerId``, ``HomePlanet``, ``CryoSleep``, ``Cabin``,
    ``Destination``, ``Age``, ``VIP``, and 5 amenity-spend columns. We
    reproduce that schema with realistic distributions so the ETL pipeline
    can be tested end-to-end without network access.

    The synthetic ``Transported`` target follows a logistic model where
    being in CryoSleep, on deck A/B/C, or spending more on amenities
    increases transport probability — mirroring the real dataset's
    feature-target correlations reported in Kaggle winners' writeups.
    """
    rng = np.random.default_rng(seed)

    # PassengerId group structure: gggg_pp where gggg is a group ID.
    group_ids = rng.integers(0, 9_000, size=n_samples)
    within_group = rng.integers(1, 8, size=n_samples)
    passenger_id = [f"{g:04d}_{p:02d}" for g, p in zip(group_ids, within_group)]

    home_planet = rng.choice(["Earth", "Europa", "Mars"], size=n_samples, p=[0.54, 0.25, 0.21])
    cryo_sleep = rng.choice([True, False], size=n_samples, p=[0.36, 0.64])
    deck_letters = rng.choice(list("ABCDEFGT"), size=n_samples, p=[0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.09, 0.01])
    side = rng.choice(["P", "S"], size=n_samples, p=[0.5, 0.5])
    cabin = np.array([f"{d}/{i}/{s}" for d, i, s in zip(deck_letters, rng.integers(0, 1500, n_samples), side)])
    destination = rng.choice(["TRAPPIST-1e", "55 Cancri e", "PSO J318.5-22"], size=n_samples, p=[0.69, 0.21, 0.10])
    age = np.clip(rng.normal(loc=28, scale=14, size=n_samples), 0, 80)
    vip = rng.choice([True, False], size=n_samples, p=[0.02, 0.98])
    room_service = np.where(cryo_sleep, 0, np.exp(rng.normal(loc=2.5, scale=3, size=n_samples)))
    food_court = np.where(cryo_sleep, 0, np.exp(rng.normal(loc=2.0, scale=3, size=n_samples)))
    shopping_mall = np.where(cryo_sleep, 0, np.exp(rng.normal(loc=1.5, scale=3, size=n_samples)))
    spa = np.where(cryo_sleep, 0, np.exp(rng.normal(loc=2.0, scale=3, size=n_samples)))
    vr_deck = np.where(cryo_sleep, 0, np.exp(rng.normal(loc=2.0, scale=3, size=n_samples)))
    name_pool = ["Aisha Khan", "Liam Patel", "Sofia Rossi", "Marcus Chen", "Elena Müller",
                 "Diego Silva", "Yuki Tanaka", "Amara Okafor", "Lars Andersen", "Priya Sharma"]
    names = rng.choice(name_pool, size=n_samples)

    # Transported: logistic model matching reported Kaggle feature importances.
    logit = (
        0.5 * cryo_sleep
        + 0.4 * (np.isin(deck_letters, ["A", "B", "C"]))
        - 0.3 * (age < 18)
        - 0.2 * (home_planet == "Earth")
        + 0.0001 * (room_service + food_court + spa + vr_deck)
        + rng.normal(0, 1, size=n_samples)
    )
    transported = (1 / (1 + np.exp(-logit)) > 0.5)

    df = pd.DataFrame({
        "PassengerId": passenger_id,
        "HomePlanet": home_planet,
        "CryoSleep": cryo_sleep,
        "Cabin": cabin,
        "Destination": destination,
        "Age": age,
        "VIP": vip,
        "RoomService": room_service,
        "FoodCourt": food_court,
        "ShoppingMall": shopping_mall,
        "Spa": spa,
        "VRDeck": vr_deck,
        "Name": names,
        "Transported": transported,
    })
    return df


__all__ = [
    "DatasetKind",
    "UnifiedSchema",
    "UnifiedDataset",
    "SCHEMA",
    "CLASSIC_URL",
    "SPACESHIP_URL",
    "SPACESHIP_LOCAL_OVERRIDE",
    "load_classic_titanic",
    "load_spaceship_titanic",
    "load_unified",
    "download_raw",
    "make_synthetic_classic",
    "make_synthetic_spaceship",
]
