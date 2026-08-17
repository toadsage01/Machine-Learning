"""
make_sample_data
================

Generate a small synthetic CSV used for smoke-testing the pipeline and
for the README's "quickstart" example. Also generates a "reference"
variant to exercise drift detection.

The dataset mimics an Indian data.gov.in-style schema: rain fall,
district, date, and a free-text "remarks" field. It is intentionally
imperfect (missing values, mixed types, duplicates) so that every code
path in AutoInsight is exercised.
"""

from __future__ import annotations

from pathlib import Path
import random
import sys

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _build(n_rows: int = 800, seed: int = 42, drift: bool = False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    districts = [
        "Pune", "Nashik", "Aurangabad", "Nagpur", "Kolhapur",
        "Thane", "Solapur", "Amravati", "Latur", "Ratnagiri",
    ]
    seasons = ["Monsoon", "Winter", "Summer", "Post-Monsoon"]

    def _remarks(i: int) -> str:
        pool = [
            "Normal rainfall observed",
            "Heavy downpour caused local flooding",
            "Dry spell — deficit rainfall",
            "Hailstorm damage reported",
            "Cyclonic impact on coastal belt",
            "Heatwave conditions prevailed",
            None, None, None,  # ~30% missing
        ]
        return rng.choice(pool)

    df = pd.DataFrame({
        "Date": pd.date_range("2023-01-01", periods=n_rows, freq="D"),
        "District": rng.choice(districts, size=n_rows),
        "State": "Maharashtra",
        "Season": rng.choice(seasons, size=n_rows),
        "Rainfall_mm": np.where(
            rng.random(n_rows) < 0.05,
            np.nan,
            np.clip(rng.normal(loc=85 if not drift else 110, scale=35, size=n_rows), 0, None),
        ),
        "Temperature_C": np.where(
            rng.random(n_rows) < 0.03,
            np.nan,
            rng.normal(loc=27 if not drift else 29, scale=4.5, size=n_rows),
        ),
        "Humidity_pct": rng.integers(20, 100, size=n_rows),
        "Station_ID": rng.choice([f"ST-{i:03d}" for i in range(1, 25)], size=n_rows),
        "Is_Coastal": rng.choice(["Yes", "No"], size=n_rows, p=[0.3, 0.7]),
        "Remarks": [_remarks(i) for i in range(n_rows)],
        "FreeText_Notes": [
            # Truly free-form text — high cardinality, multi-token sentences.
            f"Station log entry #{i}: observed weather pattern {'A' if i % 3 == 0 else 'B'} with wind speed {rng.uniform(2, 25):.1f} km/h and visibility {rng.uniform(0.5, 10):.1f} km."
            for i in range(n_rows)
        ],
        "Recorded_By": rng.choice(["IMD", "State Met Dept", None], size=n_rows, p=[0.6, 0.3, 0.1]),
    })
    # Inject ~1% duplicate rows.
    dup_idx = rng.choice(df.index, size=max(1, n_rows // 100), replace=False)
    df = pd.concat([df, df.loc[dup_idx]], ignore_index=True)
    # Inject a couple of "weird" cells that test the type-inference path.
    # Cast to object first to avoid pandas FutureWarning about dtype mismatch.
    df["Temperature_C"] = df["Temperature_C"].astype(object)
    df.loc[df.index[0], "Temperature_C"] = "thirty"
    return df


def main() -> None:
    out_dir = Path(__file__).resolve().parent
    out_dir.mkdir(parents=True, exist_ok=True)
    current = _build(n_rows=800, seed=42, drift=False)
    reference = _build(n_rows=800, seed=99, drift=True)  # shifted distributions

    current_path = out_dir / "sample_current.csv"
    reference_path = out_dir / "sample_reference.csv"
    current.to_csv(current_path, index=False)
    reference.to_csv(reference_path, index=False)
    print(f"Wrote {current_path}  ({len(current)} rows)")
    print(f"Wrote {reference_path}  ({len(reference)} rows)")


if __name__ == "__main__":
    main()
