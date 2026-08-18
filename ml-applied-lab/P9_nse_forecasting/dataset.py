"""
dataset
=======

NSE equity time-series ETL pipeline: fetches historical OHLCV data via
``yfinance`` (with a deterministic synthetic fallback for offline use),
computes lag/rolling-window technical indicators, performs Augmented
Dickey-Fuller stationarity checks, and exposes a sliding walk-forward
train/test split generator.

Public surface
--------------
- ``TickerConfig``                  : dataclass holding symbol + date range.
- ``DEFAULT_TICKERS``               : canonical NSE tickers list.
- ``EquityDataset``                 : frozen value object bundling OHLCV df
                                       + features + stationarity report.
- ``StationarityReport``             : ADF test result for a series.
- ``generate_synthetic_ohlcv``      : synthetic GBM-based OHLCV generator.
- ``download_ohlcv``                : yfinance downloader (with cache).
- ``add_technical_indicators``       : compute lag/rolling features.
- ``adf_stationarity``              : run Augmented Dickey-Fuller.
- ``build_walk_forward_splits``    : sliding-window split generator
                                       (returns ``(train_df, test_df)`` tuples
                                       with strict temporal ordering).
- ``load_equity_dataset``            : one-call loader.

Design notes
------------
1. **Walk-forward validation** is the gold standard for time-series
   evaluation: at each step, the model trains on the past ``train_window``
   days and forecasts the next ``test_window`` days. The window slides
   forward by ``step`` days, producing multiple (train, test) pairs.
   This is fundamentally different from k-fold CV because:
     * Train data always precedes test data temporally (no leakage).
     * Each test fold is a contiguous block (not random points).

2. **Lag features must respect time** — when computing ``lag_1`` (price
   shifted by 1 day), the shift must be **backward in time** (`.shift(1)`
   in pandas). A common bug is `.shift(-1)`, which leaks future prices
   into the current row's features. The test suite verifies that lag
   features never reference future data.

3. **Rolling windows are right-aligned** — `df.rolling(window=5).mean()`
   in pandas produces, for row ``t``, the mean of ``[t-4, t-3, t-2, t-1, t]``.
   This is correct (no future leakage) but means the first ``window-1``
   rows have NaN values. The loader drops these rows.

4. **Stationarity (ADF) is informational, not gating** — equity prices
   are typically non-stationary (random walk + drift), but **returns**
   are usually stationary. The stationarity report lets the user verify
   this; the loader doesn't force differencing because some forecasters
   (e.g. Chronos zero-shot) work on raw prices.
"""

from __future__ import annotations

import hashlib
import io
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import requests

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

log = logging.getLogger("nse_dataset")

# Lazy imports — yfinance may be unavailable in some contexts.
try:
    import yfinance as yf
    HAVE_YFINANCE = True
except Exception:  # pragma: no cover
    HAVE_YFINANCE = False

try:
    from statsmodels.tsa.stattools import adfuller
    HAVE_ADF = True
except Exception:  # pragma: no cover
    HAVE_ADF = False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"
CACHE_DIR = PROJECT_DATA_DIR / "_cache"

# Canonical NSE tickers — a mix of large-cap + IT + banking.
DEFAULT_TICKERS: Tuple[str, ...] = (
    "^NSEI",     # Nifty 50 index
    "RELIANCE.NS",
    "TCS.NS",
    "INFY.NS",
    "HDFCBANK.NS",
)

# Default train/test window sizes (in trading days; ~252 trading days/year).
DEFAULT_TRAIN_WINDOW = 252   # 1 year
DEFAULT_TEST_WINDOW = 21    # 1 month
DEFAULT_STEP = 21            # slide by 1 month


# ---------------------------------------------------------------------------
# Config & value objects
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TickerConfig:
    """Configuration for a single ticker download."""

    symbol: str
    start_date: str = "2020-01-01"
    end_date: str = "2024-12-31"
    interval: str = "1d"


@dataclass(frozen=True)
class StationarityReport:
    """Augmented Dickey-Fuller test result for a single series."""

    column: str
    adf_statistic: float
    p_value: float
    n_lags: int
    n_observations: int
    critical_values: Dict[str, float]
    is_stationary: bool  # True iff p_value < 0.05


@dataclass(frozen=True)
class EquityDataset:
    """Bundle of OHLCV + features + stationarity + provenance."""

    symbol: str
    df: pd.DataFrame           # raw OHLCV (Date-indexed)
    features_df: pd.DataFrame   # OHLCV + technical indicators (no NaNs)
    target_column: str
    feature_columns: List[str]
    stationarity: Dict[str, StationarityReport]
    source: str                 # "yfinance" | "synthetic" | path
    sha256: str
    n_samples: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_samples", int(len(self.features_df)))


# ---------------------------------------------------------------------------
# Synthetic OHLCV generator (offline fallback)
# ---------------------------------------------------------------------------
def generate_synthetic_ohlcv(
    symbol: str = "SYNTH.NS",
    n_days: int = 1000,
    start_price: float = 1000.0,
    annual_drift: float = 0.08,
    annual_volatility: float = 0.20,
    seed: int = 42,
) -> pd.DataFrame:
    """Generate a synthetic OHLCV dataframe using Geometric Brownian Motion.

    The synthetic data mimics real equity OHLCV:
        * Close prices follow GBM: S_{t+1} = S_t * exp((μ − σ²/2) dt + σ √dt ε)
        * Open = previous Close (gap = 0 for simplicity)
        * High = max(Open, Close) + intraday noise
        * Low  = min(Open, Close) − intraday noise
        * Volume = base + AR(1) noise

    Returns
    -------
    pd.DataFrame
        Index = DatetimeIndex (business days), columns = [Open, High, Low, Close, Volume].
    """
    rng = np.random.default_rng(seed)
    # Trading-day timestamps (skip weekends).
    dates = pd.bdate_range(start="2020-01-01", periods=n_days)

    # Annualized drift + volatility → daily.
    dt = 1.0 / 252
    mu_daily = annual_drift * dt
    sigma_daily = annual_volatility * np.sqrt(dt)

    # GBM close prices.
    log_returns = rng.normal(mu_daily - 0.5 * sigma_daily ** 2, sigma_daily, size=n_days)
    log_prices = np.cumsum(log_returns)
    close = start_price * np.exp(log_prices)

    # Open = previous close (with a tiny overnight gap).
    open_ = np.empty_like(close)
    open_[0] = start_price
    open_[1:] = close[:-1] * (1 + rng.normal(0, 0.005, size=n_days - 1))

    # High / Low: intraday extremes.
    intraday_range = close * sigma_daily * np.abs(rng.normal(0, 1, size=n_days))
    high = np.maximum(open_, close) + intraday_range * 0.5
    low = np.minimum(open_, close) - intraday_range * 0.5

    # Volume: base 1M + AR(1) noise.
    base_volume = 1_000_000
    vol_noise = np.zeros(n_days)
    for t in range(1, n_days):
        vol_noise[t] = 0.7 * vol_noise[t - 1] + rng.normal(0, 0.3)
    volume = (base_volume * (1 + vol_noise)).astype(int)

    df = pd.DataFrame({
        "Open": open_,
        "High": high,
        "Low": low,
        "Close": close,
        "Volume": volume,
    }, index=dates)
    df.index.name = "Date"
    return df


# ---------------------------------------------------------------------------
# yfinance downloader
# ---------------------------------------------------------------------------
def download_ohlcv(
    config: TickerConfig,
    cache: bool = True,
    timeout: int = 30,
) -> pd.DataFrame:
    """Download OHLCV data via yfinance (with on-disk cache).

    Returns a DataFrame with columns [Open, High, Low, Close, Volume]
    and a DatetimeIndex named ``Date``.
    """
    if not HAVE_YFINANCE:
        raise RuntimeError("yfinance is not installed.")

    cache_path = CACHE_DIR / f"{config.symbol.replace('^', 'idx_')}_{config.start_date}_{config.end_date}.csv"
    if cache and cache_path.exists() and cache_path.stat().st_size > 0:
        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        return df

    log.info("Downloading %s from yfinance (%s to %s)",
             config.symbol, config.start_date, config.end_date)
    df = yf.download(
        config.symbol,
        start=config.start_date,
        end=config.end_date,
        interval=config.interval,
        progress=False,
        auto_adjust=False,
    )
    if df is None or df.empty:
        raise RuntimeError(f"yfinance returned no data for {config.symbol}")

    # Flatten any MultiIndex columns (yfinance sometimes returns them).
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df.index.name = "Date"
    if cache:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(cache_path)
    return df


# ---------------------------------------------------------------------------
# Technical indicators
# ---------------------------------------------------------------------------
def add_technical_indicators(
    df: pd.DataFrame,
    lags: Tuple[int, ...] = (1, 2, 3, 5, 10),
    rolling_windows: Tuple[int, ...] = (5, 10, 20, 50),
    target_column: str = "Close",
) -> pd.DataFrame:
    """Add lag features + rolling-window indicators to an OHLCV dataframe.

    All features are computed with backward-looking windows only — no
    future data leakage.

    Lag features (per ``target_column``):
        * ``lag_{n}``  : target shifted by ``n`` days back.

    Rolling-window features (per ``target_column``):
        * ``roll_mean_{w}``   : mean over the past ``w`` days.
        * ``roll_std_{w}``    : std-dev over the past ``w`` days.
        * ``roll_min_{w}``    : min over the past ``w`` days.
        * ``roll_max_{w}``    : max over the past ``w`` days.

    Returns:
        * ``daily_return``    : pct change of close (1-day).
        * ``log_return``      : log pct change of close.
        * ``volatility_{w}``  : rolling std of log returns over window ``w``.
        * ``ema_{w}``         : exponential moving average.
        * ``rsi_{w}``         : Relative Strength Index (window ``w``).

    All rolling windows are right-aligned (i.e. they include the current
    observation but exclude any future data), so the first ``max(w)-1``
    rows will have NaN values. The caller is responsible for dropping
    these rows.
    """
    out = df.copy()
    target = out[target_column]

    # Lag features (always backward).
    for n in lags:
        out[f"lag_{n}"] = target.shift(n)

    # Rolling-window features (right-aligned, no future leakage).
    for w in rolling_windows:
        out[f"roll_mean_{w}"] = target.rolling(w).mean()
        out[f"roll_std_{w}"] = target.rolling(w).std()
        out[f"roll_min_{w}"] = target.rolling(w).min()
        out[f"roll_max_{w}"] = target.rolling(w).max()

    # Returns.
    out["daily_return"] = target.pct_change(periods=1)
    out["log_return"] = np.log(target / target.shift(1))

    # Rolling volatility (std of log returns).
    for w in rolling_windows:
        out[f"volatility_{w}"] = out["log_return"].rolling(w).std()

    # Exponential moving average.
    for w in rolling_windows:
        out[f"ema_{w}"] = target.ewm(span=w, adjust=False).mean()

    # RSI (Relative Strength Index).
    for w in rolling_windows:
        delta = target.diff()
        gain = delta.where(delta > 0, 0.0)
        loss = -delta.where(delta < 0, 0.0)
        avg_gain = gain.rolling(w).mean()
        avg_loss = loss.rolling(w).mean()
        # RSI = 100 - 100 / (1 + RS), where RS = avg_gain / avg_loss.
        # When avg_loss == 0 (no losses in the window), RS = inf and RSI = 100.
        # When avg_gain == 0 (no gains), RS = 0 and RSI = 0.
        # When both are 0 (no movement), RSI is undefined — set to 50 (neutral).
        rs = avg_gain / avg_loss
        rs = rs.replace([np.inf, -np.inf], np.nan)
        rsi = 100 - (100 / (1 + rs))
        # When avg_loss == 0 but avg_gain > 0, RSI should be 100.
        rsi[avg_loss.eq(0) & avg_gain.gt(0)] = 100.0
        # When both avg_gain and avg_loss are 0 (flat period), RSI is undefined.
        # Set to 50 (neutral) to avoid NaN.
        rsi[avg_gain.eq(0) & avg_loss.eq(0)] = 50.0
        out[f"rsi_{w}"] = rsi

    # Forward target: next-day close (what we want to predict).
    out["target_next_close"] = target.shift(-1)

    return out


# ---------------------------------------------------------------------------
# Stationarity check
# ---------------------------------------------------------------------------
def adf_stationarity(
    series: pd.Series,
    column_name: str = "",
    max_lags: Optional[int] = None,
    significance: float = 0.05,
) -> StationarityReport:
    """Run the Augmented Dickey-Fuller test on a series.

    H0: the series has a unit root (non-stationary).
    H1: the series is stationary.

    Returns a ``StationarityReport`` with the ADF statistic, p-value,
    number of lags used, and critical values. ``is_stationary`` is True
    iff ``p_value < significance``.

    Notes
    -----
    A constant series is treated as stationary (p_value=0) — ADF raises
    a ValueError on constants, so we catch it and return a synthetic
    "perfectly stationary" report.
    """
    if not HAVE_ADF:
        raise RuntimeError("statsmodels is not installed.")
    series = series.dropna().astype(float)
    if len(series) < 20:
        return StationarityReport(
            column=column_name, adf_statistic=float("nan"), p_value=1.0,
            n_lags=0, n_observations=len(series), critical_values={},
            is_stationary=False,
        )
    # Handle constant series — ADF raises ValueError on these.
    if series.nunique() <= 1:
        return StationarityReport(
            column=column_name, adf_statistic=float("-inf"), p_value=0.0,
            n_lags=0, n_observations=len(series),
            critical_values={"1%": float("nan"), "5%": float("nan"), "10%": float("nan")},
            is_stationary=True,
        )
    result = adfuller(series, maxlag=max_lags, autolag="AIC")
    adf_stat, p_val, n_lags, n_obs, crit_vals, _icbest = result
    return StationarityReport(
        column=column_name,
        adf_statistic=float(adf_stat),
        p_value=float(p_val),
        n_lags=int(n_lags),
        n_observations=int(n_obs),
        critical_values={k: float(v) for k, v in crit_vals.items()},
        is_stationary=bool(p_val < significance),
    )


# ---------------------------------------------------------------------------
# Walk-forward splits
# ---------------------------------------------------------------------------
def build_walk_forward_splits(
    df: pd.DataFrame,
    train_window: int = DEFAULT_TRAIN_WINDOW,
    test_window: int = DEFAULT_TEST_WINDOW,
    step: int = DEFAULT_STEP,
) -> Iterator[Tuple[pd.DataFrame, pd.DataFrame]]:
    """Yield ``(train_df, test_df)`` pairs for walk-forward validation.

    The dataframe is split into overlapping windows:
        * First fold:   train = df[0 : train_window],  test = df[train_window : train_window + test_window]
        * Second fold:  train = df[step : step + train_window],  test = df[step + train_window : step + train_window + test_window]
        * ...

    Parameters
    ----------
    df : pd.DataFrame
        Must be sorted by date ascending (caller's responsibility).
    train_window : int
        Number of consecutive rows in each training fold.
    test_window : int
        Number of consecutive rows in each test fold (immediately
        following the train fold).
    step : int
        Slide distance between consecutive folds.

    Yields
    ------
    (train_df, test_df)
        Each is a contiguous slice of the original dataframe, sorted
        ascending by date. Train always precedes test temporally.

    Notes
    -----
    The split guarantees temporal integrity:
        * Train data is always at earlier timestamps than test data.
        * No row appears in both train and test (the windows are
          non-overlapping within each fold).
    """
    if train_window <= 0 or test_window <= 0 or step <= 0:
        raise ValueError(f"Windows must be positive: train={train_window}, test={test_window}, step={step}")

    n = len(df)
    fold_size = train_window + test_window
    if n < fold_size:
        log.warning("DataFrame has %d rows, need at least %d for one fold.", n, fold_size)
        return

    # Slide the window until the test set would exceed the dataframe.
    start = 0
    while start + fold_size <= n:
        train_df = df.iloc[start : start + train_window]
        test_df = df.iloc[start + train_window : start + fold_size]
        # Critical: verify train ends BEFORE test starts.
        if train_df.index[-1] >= test_df.index[0]:
            raise RuntimeError(
                f"Walk-forward temporal violation: train ends at {train_df.index[-1]} "
                f"but test starts at {test_df.index[0]}"
            )
        yield train_df, test_df
        start += step


# ---------------------------------------------------------------------------
# Loader
# ---------------------------------------------------------------------------
def _sha256(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


def load_equity_dataset(
    symbol: str = "^NSEI",
    csv_path: Optional[Path | str] = None,
    use_yfinance: bool = False,
    n_days_synthetic: int = 1000,
    seed: int = 42,
    start_date: str = "2020-01-01",
    end_date: str = "2024-12-31",
    lags: Tuple[int, ...] = (1, 2, 3, 5, 10),
    rolling_windows: Tuple[int, ...] = (5, 10, 20, 50),
    run_stationarity: bool = True,
) -> EquityDataset:
    """One-call loader for an equity dataset with technical indicators.

    Resolution order:
        1. ``csv_path`` (explicit CSV override — must have OHLCV columns).
        2. yfinance download (if ``use_yfinance=True`` and network available).
        3. ``data/<symbol>_ohlcv.csv`` (project-local drop-in).
        4. Synthetic GBM generator (default fallback).
    """
    # Step 1 — load raw OHLCV.
    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"OHLCV CSV not found: {path}")
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        source = str(path)
        sha = _sha256(path.read_bytes())
    elif use_yfinance:
        try:
            config = TickerConfig(symbol=symbol, start_date=start_date, end_date=end_date)
            df = download_ohlcv(config)
            source = f"yfinance:{symbol}"
            sha = _sha256(df.to_csv().encode("utf-8"))
        except Exception as exc:
            log.warning("yfinance download failed (%s); using synthetic data.", exc)
            df = generate_synthetic_ohlcv(symbol, n_days=n_days_synthetic, seed=seed)
            source = "synthetic"
            sha = _sha256(df.to_csv().encode("utf-8"))
    elif (PROJECT_DATA_DIR / f"{symbol.replace('^', 'idx_')}_ohlcv.csv").exists():
        path = PROJECT_DATA_DIR / f"{symbol.replace('^', 'idx_')}_ohlcv.csv"
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        source = str(path)
        sha = _sha256(path.read_bytes())
    else:
        log.info("Using synthetic OHLCV data for %s (n_days=%d, seed=%d)",
                 symbol, n_days_synthetic, seed)
        df = generate_synthetic_ohlcv(symbol, n_days=n_days_synthetic, seed=seed)
        source = "synthetic"
        sha = _sha256(df.to_csv().encode("utf-8"))

    # Step 2 — ensure standard column names.
    df.columns = [c.capitalize() for c in df.columns]
    required = ["Open", "High", "Low", "Close", "Volume"]
    for col in required:
        if col not in df.columns:
            raise ValueError(f"Missing required OHLCV column: {col}")

    # Step 3 — add technical indicators.
    features_df = add_technical_indicators(
        df, lags=lags, rolling_windows=rolling_windows, target_column="Close",
    )
    # Drop rows with NaN (caused by lag/rolling windows).
    initial_len = len(features_df)
    features_df = features_df.dropna()
    final_len = len(features_df)
    log.info("Dropped %d rows with NaN (from lag/rolling windows)", initial_len - final_len)

    # Step 4 — run stationarity checks.
    stationarity: Dict[str, StationarityReport] = {}
    if run_stationarity:
        for col in ["Close", "daily_return", "log_return"]:
            if col in features_df.columns:
                stationarity[col] = adf_stationarity(features_df[col], column_name=col)
                log.info("  ADF %s: p=%.4f, stationary=%s",
                         col, stationarity[col].p_value, stationarity[col].is_stationary)

    # Step 5 — feature columns (everything except the forward target).
    feature_columns = [c for c in features_df.columns if c != "target_next_close"]

    return EquityDataset(
        symbol=symbol,
        df=df,
        features_df=features_df,
        target_column="target_next_close",
        feature_columns=feature_columns,
        stationarity=stationarity,
        source=source,
        sha256=sha,
    )


__all__ = [
    "TickerConfig",
    "DEFAULT_TICKERS",
    "EquityDataset",
    "StationarityReport",
    "DEFAULT_TRAIN_WINDOW",
    "DEFAULT_TEST_WINDOW",
    "DEFAULT_STEP",
    "generate_synthetic_ohlcv",
    "download_ohlcv",
    "add_technical_indicators",
    "adf_stationarity",
    "build_walk_forward_splits",
    "load_equity_dataset",
]
