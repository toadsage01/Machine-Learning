"""
dataset
=======

Subscription churn ETL pipeline with tenure, event indicators, and
contract/demographic features. Supports a real-world dataset download
(Telco churn) with a synthetic survival-generator fallback for offline
testing.

Public surface
--------------
- ``ChurnSchema``                  : dataclass of feature names + roles.
- ``ChurnDataset``                : frozen value object bundling X/y + provenance.
- ``generate_synthetic_churn``     : synthetic subscription-churn generator
                                    with realistic hazard function.
- ``load_churn_dataset``           : one-call loader (CSV | synthetic).
- ``build_train_test_split``       : stratified split on the event indicator.

Design notes
------------
1. **Survival data needs (duration, event)** — every churn example is
   characterized by:
     * ``tenure_months``  : how long the customer has been subscribed
       (the survival *duration*).
     * ``churned``        : 1 if the customer has cancelled, 0 if they
       are still active (censored at the observation window). This is
       the *event indicator*.
   Standard classifiers predict ``churned`` alone; survival models use
   both columns to estimate the *time-to-churn* distribution.

2. **Synthetic generator with proportional hazards** — the synthetic
   generator uses a Cox-style hazard function:
       h(t | x) = h0(t) * exp(β·x)
   where ``h0(t) = λ * ρ * t^(ρ-1)`` is a Weibull baseline hazard. This
   gives us:
     * Realistic churn-time distributions (early churners churn faster)
     * A known ground-truth coefficient vector ``β`` for verification
     * Censoring at the observation window ``T_obs`` (customers who
       haven't churned by ``T_obs`` are right-censored)

3. **Both classifier + survival views** — the loader returns a
   ``ChurnDataset`` that contains BOTH the classifier view
   (``X``, ``y_churned``) AND the survival view (``durations``,
   ``events``). This lets ``model.py`` train either model family
   from the same data without re-loading.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests
from sklearn.model_selection import train_test_split

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("churn_dataset")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = PROJECT_DATA_DIR / "_cache"

# Canonical Telco churn dataset (IBM Watson sample, freely available
# via the tidyverse mirror on GitHub).
TELCO_URL = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ChurnSchema:
    """Canonical schema for the subscription-churn task.

    Features are split into:
      * ``numeric_features``  : continuous (tenure, monthly_charges, total_charges, ...)
      * ``categorical_features`` : contract / demographic (contract_type, internet_service, ...)
      * ``survival_columns``  : (duration, event) for the survival view
      * ``target``            : binary churn indicator (same as event but
                                semantically a classifier target)
    """

    numeric_features: Tuple[str, ...] = (
        "tenure_months", "monthly_charges", "total_charges", "age",
        "household_size",
    )
    categorical_features: Tuple[str, ...] = (
        "contract_type", "internet_service", "payment_method", "gender",
        "partner", "dependents", "phone_service", "paperless_billing",
    )
    duration_column: str = "tenure_months"
    event_column: str = "churned"
    target: str = "churned"

    @property
    def all_features(self) -> List[str]:
        return list(self.numeric_features) + list(self.categorical_features)


SCHEMA = ChurnSchema()


@dataclass(frozen=True)
class ChurnDataset:
    """Bundle of features + survival targets + provenance.

    Attributes
    ----------
    df : pd.DataFrame
        Full ETL'd dataframe (features + targets).
    X : pd.DataFrame
        Feature matrix (columns match ``ChurnSchema.all_features``).
    y_churned : pd.Series
        Binary churn indicator (1=churned, 0=active).
    durations : pd.Series
        Tenure in months (the survival duration).
    events : pd.Series
        Same as ``y_churned`` (1=observed churn, 0=right-censored).
    source : str
        "synthetic" | CSV path | URL.
    sha256 : str
    n_samples : int
    """

    df: pd.DataFrame
    X: pd.DataFrame
    y_churned: pd.Series
    durations: pd.Series
    events: pd.Series
    source: str
    sha256: str
    n_samples: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_samples", int(len(self.X)))


# ---------------------------------------------------------------------------
# Synthetic churn generator
# ---------------------------------------------------------------------------
def generate_synthetic_churn(
    n_samples: int = 2000,
    seed: int = 42,
    observation_window_months: int = 72,
    baseline_lambda: float = 0.0012,
    baseline_rho: float = 1.4,
) -> pd.DataFrame:
    """Generate a synthetic subscription-churn dataset with proportional hazards.

    The data-generating process is a Cox model with a Weibull baseline:
        h(t | x) = λ * ρ * t^(ρ-1) * exp(β·x)

    where ``x`` includes contract/demographic features and ``β`` is the
    ground-truth coefficient vector (returned via the dataset for
    verification).

    Parameters
    ----------
    n_samples : int
        Number of customers to generate.
    seed : int
        Reproducibility seed.
    observation_window_months : int
        Customers whose simulated churn time exceeds this are right-censored
        (their ``tenure_months`` is set to the window and ``churned`` to 0).
    baseline_lambda : float
        Weibull baseline rate (higher = faster average churn). Default 0.003
        yields ~30-35% observed churn rate over a 72-month window, matching
        the IBM Telco dataset's real churn rate.
    baseline_rho : float
        Weibull shape (>1 = increasing hazard over time, typical for churn).

    Returns
    -------
    pd.DataFrame
        Columns: numeric_features + categorical_features + tenure_months +
        churned + true_log_hazard.
    """
    rng = np.random.default_rng(seed)

    # ---- Demographic features --------------------------------------------
    gender = rng.choice(["Male", "Female"], size=n_samples, p=[0.5, 0.5])
    age = np.clip(rng.normal(loc=42, scale=12, size=n_samples), 18, 90).astype(int)
    household_size = np.clip(rng.poisson(2.5, size=n_samples), 1, 6)
    partner = np.where(rng.random(n_samples) < 0.5, "Yes", "No")
    dependents = np.where(
        (rng.random(n_samples) < 0.3) & (household_size > 1), "Yes", "No"
    )

    # ---- Contract features ------------------------------------------------
    contract_type = rng.choice(["Month-to-month", "One year", "Two year"],
                              size=n_samples, p=[0.55, 0.25, 0.20])
    internet_service = rng.choice(["DSL", "Fiber optic", "No"],
                                  size=n_samples, p=[0.35, 0.45, 0.20])
    phone_service = rng.choice(["Yes", "No"], size=n_samples, p=[0.90, 0.10])
    payment_method = rng.choice(
        ["Electronic check", "Mailed check", "Bank transfer (auto)", "Credit card (auto)"],
        size=n_samples, p=[0.35, 0.22, 0.22, 0.21],
    )
    paperless_billing = rng.choice(["Yes", "No"], size=n_samples, p=[0.6, 0.4])

    # Monthly charges — fiber-optic + month-to-month customers pay more.
    base_charges = 25 + 30 * (internet_service == "Fiber optic") + 15 * (phone_service == "Yes")
    monthly_charges = base_charges + rng.normal(0, 8, size=n_samples)

    # ---- Compute ground-truth log-hazard via the Cox linear predictor ----
    # β coefficients (ground truth for tests/verification).
    beta = {
        "contract_type_One year":      -0.7,
        "contract_type_Two year":      -1.4,
        "internet_service_Fiber optic": 0.6,
        "internet_service_No":         -0.3,
        "payment_method_Electronic check": 0.5,
        "paperless_billing_Yes":       0.2,
        "partner_Yes":                -0.2,
        "dependents_Yes":             -0.25,
        "age_centered":                0.005,  # older → slightly lower churn
        "monthly_charges_centered":    0.015,  # higher charges → higher churn
    }

    # Center age and monthly_charges for stability.
    age_centered = age - 42
    monthly_charges_centered = monthly_charges - 50

    log_hazard = (
        beta["contract_type_One year"] * (contract_type == "One year")
        + beta["contract_type_Two year"] * (contract_type == "Two year")
        + beta["internet_service_Fiber optic"] * (internet_service == "Fiber optic")
        + beta["internet_service_No"] * (internet_service == "No")
        + beta["payment_method_Electronic check"] * (payment_method == "Electronic check")
        + beta["paperless_billing_Yes"] * (paperless_billing == "Yes")
        + beta["partner_Yes"] * (partner == "Yes")
        + beta["dependents_Yes"] * (dependents == "Yes")
        + beta["age_centered"] * age_centered
        + beta["monthly_charges_centered"] * monthly_charges_centered
    )

    # ---- Sample churn time from Weibull baseline × exp(β·x) --------------
    # Inverse-CDF sampling for Weibull(λ, ρ) baseline under proportional
    # hazards: T = (−log(U) / (λ * exp(β·x)))^(1/ρ), U ~ Uniform(0, 1).
    U = rng.random(n_samples)
    # Floor exp(log_hazard) at a small positive value to avoid div-by-zero.
    hazard_scale = np.maximum(np.exp(log_hazard), 1e-6)
    # Inverse Weibull CDF: t = (-log(U) / (λ * scale))^(1/ρ)
    t_churn = (-np.log(U + 1e-12) / (baseline_lambda * hazard_scale)) ** (1.0 / baseline_rho)

    # Apply censoring at the observation window.
    tenure_months = np.minimum(t_churn, observation_window_months)
    # Round tenure to integer months (churn is observed monthly).
    tenure_months = np.clip(np.round(tenure_months), 1, observation_window_months).astype(int)
    churned = (t_churn <= observation_window_months).astype(int)

    # Total charges = monthly * tenure.
    total_charges = monthly_charges * tenure_months

    df = pd.DataFrame({
        "tenure_months": tenure_months,
        "monthly_charges": np.round(monthly_charges, 2),
        "total_charges": np.round(total_charges, 2),
        "age": age,
        "household_size": household_size,
        "contract_type": contract_type,
        "internet_service": internet_service,
        "payment_method": payment_method,
        "gender": gender,
        "partner": partner,
        "dependents": dependents,
        "phone_service": phone_service,
        "paperless_billing": paperless_billing,
        "churned": churned,
        "true_log_hazard": log_hazard,
    })
    return df


# ---------------------------------------------------------------------------
# Real-world downloader (Telco churn)
# ---------------------------------------------------------------------------
def _download_telco(timeout: int = 30) -> bytes:
    """Download the IBM Telco churn CSV."""
    log.info("Downloading Telco churn dataset from %s", TELCO_URL)
    response = requests.get(TELCO_URL, timeout=timeout)
    response.raise_for_status()
    return response.content


def _normalize_telco(df: pd.DataFrame) -> pd.DataFrame:
    """Map the IBM Telco churn schema to the unified schema.

    The IBM Telco dataset has these columns (sample):
        customerID, gender, SeniorCitizen, Partner, Dependents, tenure,
        PhoneService, MultipleLines, InternetService, OnlineSecurity, ...,
        Contract, PaperlessBilling, PaymentMethod, MonthlyCharges,
        TotalCharges, Churn
    """
    rename_map = {
        "tenure": "tenure_months",
        "MonthlyCharges": "monthly_charges",
        "TotalCharges": "total_charges",
        "SeniorCitizen": "age",  # proxy; we'll synthesize a proper age below
        "gender": "gender",
        "Partner": "partner",
        "Dependents": "dependents",
        "PhoneService": "phone_service",
        "InternetService": "internet_service",
        "Contract": "contract_type",
        "PaperlessBilling": "paperless_billing",
        "PaymentMethod": "payment_method",
        "Churn": "churned",
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})

    # Coerce dtypes.
    df["tenure_months"] = pd.to_numeric(df.get("tenure_months", 0), errors="coerce").fillna(0).astype(int)
    df["monthly_charges"] = pd.to_numeric(df.get("monthly_charges", 0), errors="coerce").fillna(0).astype(float)
    # TotalCharges has empty strings for new customers — coerce to 0.
    df["total_charges"] = pd.to_numeric(df.get("total_charges", 0).replace(" ", "0"), errors="coerce").fillna(0).astype(float)

    # Synthesize age (the IBM dataset only has SeniorCitizen 0/1).
    # Use SeniorCitizen to bias the age distribution: seniors → 65+, others → 25-55.
    if "age" in df.columns:
        # age was the rename of SeniorCitizen — convert to a real age.
        rng = np.random.default_rng(42)
        senior = df["age"].fillna(0).astype(int)
        df["age"] = np.where(senior == 1, rng.integers(65, 85, size=len(df)),
                            rng.integers(25, 55, size=len(df)))

    # IBM Telco lacks household_size — synthesize.
    rng = np.random.default_rng(42)
    df["household_size"] = np.where(df.get("dependents", "No") == "Yes",
                                    rng.integers(3, 6, size=len(df)),
                                    rng.integers(1, 3, size=len(df)))

    # Churn column: "Yes" → 1, "No" → 0.
    if "churned" in df.columns:
        df["churned"] = df["churned"].astype(str).str.lower().map(
            {"yes": 1, "no": 0, "1": 1, "0": 0, "true": 1, "false": 0}
        ).fillna(0).astype(int)
    else:
        df["churned"] = 0

    # Keep only schema columns.
    keep_cols = [c for c in SCHEMA.all_features + ["churned"] if c in df.columns]
    return df[keep_cols]


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _sha256(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


def load_churn_dataset(
    csv_path: Optional[Path | str] = None,
    use_real: bool = False,
    n_samples: int = 2000,
    seed: int = 42,
) -> ChurnDataset:
    """One-call loader for the churn dataset.

    Resolution order:
        1. ``csv_path`` (explicit override — must contain the unified schema).
        2. ``data/telco_churn.csv`` (project-local drop-in).
        3. Real-world download (if ``use_real=True``) via ``_download_telco``.
        4. Synthetic fallback via ``generate_synthetic_churn``.
    """
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Churn CSV not found: {path}")
        df = pd.read_csv(path)
        source = str(path)
        sha = _sha256(path.read_bytes())
    elif (PROJECT_DATA_DIR / "telco_churn.csv").exists():
        path = PROJECT_DATA_DIR / "telco_churn.csv"
        df = pd.read_csv(path)
        df = _normalize_telco(df)
        source = str(path)
        sha = _sha256(path.read_bytes())
    elif use_real:
        try:
            payload = _download_telco()
            df = pd.read_csv(io.BytesIO(payload))
            df = _normalize_telco(df)
            # Cache for re-runs.
            cache_path = PROJECT_DATA_DIR / "telco_churn.csv"
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            cache_path.write_bytes(payload)
            source = TELCO_URL
            sha = _sha256(payload)
        except Exception as exc:
            log.warning("Real Telco download failed (%s); falling back to synthetic.", exc)
            df = generate_synthetic_churn(n_samples=n_samples, seed=seed)
            source = "synthetic"
            sha = _sha256(df.to_csv(index=False))
    else:
        log.info("Using synthetic churn data (n=%d, seed=%d)", n_samples, seed)
        df = generate_synthetic_churn(n_samples=n_samples, seed=seed)
        source = "synthetic"
        sha = _sha256(df.to_csv(index=False))

    # Drop the auxiliary ``true_log_hazard`` column from features (it's a
    # generator diagnostic, not a real feature).
    feature_cols = [c for c in SCHEMA.all_features if c in df.columns]
    X = df[feature_cols].copy()
    y_churned = df[SCHEMA.event_column].astype(int).copy()
    durations = df[SCHEMA.duration_column].astype(int).copy()
    events = y_churned.copy()  # event == churn indicator for our setup
    return ChurnDataset(
        df=df, X=X, y_churned=y_churned, durations=durations, events=events,
        source=source, sha256=sha,
    )


# ---------------------------------------------------------------------------
# Stratified split on the event indicator
# ---------------------------------------------------------------------------
def build_train_test_split(
    ds: ChurnDataset,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[ChurnDataset, ChurnDataset]:
    """Stratified split on the churn event indicator.

    Returns two ``ChurnDataset`` value objects (train + test) that share
    the source/sha provenance of the input. The split is stratified on
    ``churned`` so train/test have the same churn rate.
    """
    train_idx, test_idx = train_test_split(
        ds.df.index, test_size=test_size, random_state=random_state,
        stratify=ds.y_churned,
    )
    def _make_subset(idx) -> ChurnDataset:
        sub_df = ds.df.loc[idx].reset_index(drop=True)
        feature_cols = [c for c in SCHEMA.all_features if c in sub_df.columns]
        return ChurnDataset(
            df=sub_df,
            X=sub_df[feature_cols].copy(),
            y_churned=sub_df[SCHEMA.event_column].astype(int).copy(),
            durations=sub_df[SCHEMA.duration_column].astype(int).copy(),
            events=sub_df[SCHEMA.event_column].astype(int).copy(),
            source=ds.source, sha256=ds.sha256,
        )
    return _make_subset(train_idx), _make_subset(test_idx)


__all__ = [
    "ChurnSchema",
    "ChurnDataset",
    "SCHEMA",
    "TELCO_URL",
    "generate_synthetic_churn",
    "load_churn_dataset",
    "build_train_test_split",
]
