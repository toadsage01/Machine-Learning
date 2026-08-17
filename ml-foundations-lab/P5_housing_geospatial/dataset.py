"""
dataset
=======

Indian metro housing ETL with OSMnx / geopy proximity features for the
P5 quantile-regression benchmark.

Public surface
--------------
- ``Metro``                       : enum of supported Indian metros.
- ``HousingSchema``               : canonical feature list.
- ``HousingDataset``              : frozen value object bundling X/y + provenance.
- ``PROXIMITY_POIS``               : OSM POI tags we query for each metro.
- ``generate_synthetic_mumbai``   : synthetic Mumbai housing generator.
- ``load_housing``                : one-call loader (synthetic OR CSV OR fresh-OSM).
- ``enrich_with_osm_features``     : add proximity-to-POI columns via OSMnx.
- ``haversine_distance_km``        : vectorized great-circle distance (km).

Synthetic vs real data
----------------------
Mumbai housing data with lat/lon is not freely available as a single
authoritative CSV. This module therefore exposes a *deterministic
synthetic generator* (``generate_synthetic_mumbai``) that produces a
realistic-shape dataset — Mumbai bounding box, plausible price-per-sqft
gradient (south = expensive; near transit/Marine Drive premium), and
geographically-meaningful coordinates. The downstream pipeline is
identical whether the data is synthetic or real; users can drop a real
CSV into ``data/mumbai_housing.csv`` to override.

OSMnx enrichment
----------------
``enrich_with_osm_features`` queries OpenStreetMap for each property's
nearest point-of-interest (POI) of the requested kinds (metro stations,
hospitals, schools, parks, malls). It uses ``osmnx.nearest_nodes`` /
``features_from_place`` under the hood and caches the downloaded graph
to ``data/_osm_cache/`` so re-runs are instant. Network is required
only on the first run; subsequent runs use the cache.

If OSMnx is unavailable (network blocked, package not installed), the
function gracefully falls back to a synthetic proximity estimate based
on the property's lat/lon (south Mumbai is treated as "close to
everything"; northern suburbs get larger distances). The synthetic
fallback is flagged via the ``proximity_source`` column.
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

log = logging.getLogger("housing_dataset")


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_DATA_DIR = Path(__file__).resolve().parent / "data"
OSM_CACHE_DIR = PROJECT_DATA_DIR / "_osm_cache"


# ---------------------------------------------------------------------------
# Indian metros with bounding boxes + OSM place names
# ---------------------------------------------------------------------------
class Metro(str, Enum):
    MUMBAI = "mumbai"
    DELHI = "delhi"
    BANGALORE = "bangalore"

    def __str__(self) -> str:  # pragma: no cover
        return self.value


# (lat_min, lat_max, lon_min, lon_max) bounding box per metro.
METRO_BBOXES: Dict[str, Tuple[float, float, float, float]] = {
    "mumbai":    (18.89, 19.27, 72.77, 73.10),    # Greater Mumbai
    "delhi":     (28.40, 28.88, 76.84, 77.35),    # NCT of Delhi
    "bangalore": (12.81, 13.10, 77.45, 77.80),    # BBMP
}

METRO_OSM_PLACES: Dict[str, str] = {
    "mumbai": "Mumbai, Maharashtra, India",
    "delhi": "Delhi, India",
    "bangalore": "Bengaluru, Karnataka, India",
}

# OSM POI tags we query for every property.
# Each tuple: (column_name, OSM_tags_dict)
PROXIMITY_POIS: List[Tuple[str, Dict[str, str]]] = [
    ("dist_metro_station_km",   {"railway": "station"}),
    ("dist_hospital_km",        {"amenity": "hospital"}),
    ("dist_school_km",          {"amenity": "school"}),
    ("dist_park_km",            {"leisure": "park"}),
    ("dist_mall_km",            {"shop": "mall"}),
    ("dist_bus_stop_km",        {"highway": "bus_stop"}),
    ("dist_commercial_hub_km",  {"shop": "supermarket"}),
]


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class HousingSchema:
    """Canonical feature list for the housing dataset."""

    numeric_features: Tuple[str, ...] = (
        "latitude", "longitude", "area_sqft", "bedrooms", "bathrooms",
        "age_years", "floor", "total_floors",
        "dist_metro_station_km", "dist_hospital_km", "dist_school_km",
        "dist_park_km", "dist_mall_km", "dist_bus_stop_km", "dist_commercial_hub_km",
    )
    categorical_features: Tuple[str, ...] = (
        "metro", "locality", "property_type", "furnishing",
    )
    target: str = "price_lakh"  # Indian convention: price in lakh INR (1 lakh = 100,000)

    @property
    def all_features(self) -> List[str]:
        return list(self.numeric_features) + list(self.categorical_features)


SCHEMA = HousingSchema()


@dataclass(frozen=True)
class HousingDataset:
    """Bundle of features + target + provenance."""

    metro: Metro
    df: pd.DataFrame           # raw + enriched dataframe
    X: pd.DataFrame             # post-ETL features (post-OSM enrichment)
    y: pd.Series                # target (price in lakh INR)
    source: str                 # "synthetic" / CSV path / "osmnx_enriched"
    sha256: str
    proximity_source: str       # "osmnx" | "synthetic_fallback"
    n_samples: int = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "n_samples", int(len(self.X)))

    def __len__(self) -> int:
        return self.n_samples


# ---------------------------------------------------------------------------
# Haversine distance
# ---------------------------------------------------------------------------
def haversine_distance_km(
    lat1: np.ndarray, lon1: np.ndarray,
    lat2: np.ndarray, lon2: np.ndarray,
) -> np.ndarray:
    """Vectorized great-circle distance in km using the haversine formula.

    All inputs are broadcastable. Earth radius = 6371 km.
    """
    R = 6371.0
    lat1 = np.radians(np.asarray(lat1, dtype=float))
    lat2 = np.radians(np.asarray(lat2, dtype=float))
    dlat = lat2 - lat1
    dlon = np.radians(np.asarray(lon2, dtype=float) - np.asarray(lon1, dtype=float))
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return 2 * R * np.arcsin(np.sqrt(a))


# ---------------------------------------------------------------------------
# Synthetic Mumbai housing generator
# ---------------------------------------------------------------------------
MUMBAI_LOCALITIES = [
    "Andheri West", "Bandra West", "Borivali", "Colaba", "Dadar",
    "Goregaon", "Juhu", "Khar", "Malad", "Powai",
    "Lower Parel", "Worli", "Goregaon East", "Vile Parle", "Kurla",
    "Chembur", "Ghatkopar", "Mulund", "Dombivli", "Thane West",
]

# Approximate centroid (lat, lon) for each locality — used by the synthetic
# generator to give each property a realistic coordinate.
MUMBAI_LOCALITY_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "Andheri West":   (19.1197, 72.8468),
    "Bandra West":    (19.0596, 72.8295),
    "Borivali":       (19.2307, 72.8567),
    "Colaba":         (18.9067, 72.8147),
    "Dadar":          (19.0176, 72.8418),
    "Goregaon":       (19.1656, 72.8493),
    "Juhu":           (19.0883, 72.8265),
    "Khar":           (19.0758, 72.8385),
    "Malad":          (19.1867, 72.8488),
    "Powai":          (19.1176, 72.9057),
    "Lower Parel":    (18.9989, 72.8305),
    "Worli":          (19.0150, 72.8175),
    "Goregaon East":  (19.1656, 72.8610),
    "Vile Parle":     (19.0996, 72.8400),
    "Kurla":          (19.0720, 72.8775),
    "Chembur":        (19.0596, 72.8925),
    "Ghatkopar":      (19.0819, 72.8875),
    "Mulund":         (19.1717, 72.9447),
    "Dombivli":       (19.2167, 73.0833),
    "Thane West":     (19.2183, 72.9760),
}


def generate_synthetic_mumbai(
    n_samples: int = 1500,
    seed: int = 42,
    bbox: Optional[Tuple[float, float, float, float]] = None,
) -> pd.DataFrame:
    """Generate a synthetic Mumbai housing dataset with realistic price gradients.

    Pricing model (matches qualitative Mumbai market behaviour):
        base_price_lakh = 50 + 4 * area_sqft / 100 + 25 * bedrooms
                        + 15 * bathrooms
                        + 80 * (1 / (1 + age_years / 10))         # newer = pricier
                        + 5 * (floor / max(total_floors, 1))       # higher floor premium
                        - 8 * dist_metro_station_km               # transit proximity
                        - 4 * dist_school_km
                        - 6 * dist_hospital_km
                        - 3 * dist_park_km
                        - 5 * dist_commercial_hub_km
                        + locality_premium                        # Bandra/Colaba/Juhu expensive
                        + noise

    The synthetic generator deliberately injects heteroscedastic noise
    (noise_std scales with area_sqft) so the quantile-regression
    intervals have meaningful width variation across price levels.
    """
    rng = np.random.default_rng(seed)
    bbox = bbox or METRO_BBOXES["mumbai"]
    lat_min, lat_max, lon_min, lon_max = bbox

    localities = rng.choice(MUMBAI_LOCALITIES, size=n_samples)
    # Property coords: jittered around the locality centroid.
    centroids_lat = np.array([MUMBAI_LOCALITY_CENTROIDS[l][0] for l in localities])
    centroids_lon = np.array([MUMBAI_LOCALITY_CENTROIDS[l][1] for l in localities])
    latitude = centroids_lat + rng.normal(0, 0.012, size=n_samples)
    longitude = centroids_lon + rng.normal(0, 0.012, size=n_samples)
    # Clip to bbox.
    latitude = np.clip(latitude, lat_min, lat_max)
    longitude = np.clip(longitude, lon_min, lon_max)

    area_sqft = np.clip(rng.lognormal(mean=7.0, sigma=0.45, size=n_samples), 250, 5000).astype(int)
    bedrooms = np.clip(rng.poisson(2.2, size=n_samples), 1, 6)
    bathrooms = np.clip(bedrooms - rng.integers(0, 2, size=n_samples), 1, 5)
    age_years = np.clip(rng.exponential(scale=12.0, size=n_samples), 0, 60).astype(int)
    total_floors = rng.choice([4, 7, 12, 22, 35, 50], size=n_samples, p=[0.25, 0.20, 0.20, 0.15, 0.12, 0.08])
    floor = np.clip(rng.integers(1, total_floors + 1), 1, total_floors)
    property_type = rng.choice(["Apartment", "Villa", "Independent House", "Penthouse"],
                               size=n_samples, p=[0.75, 0.10, 0.10, 0.05])
    furnishing = rng.choice(["Furnished", "Semi-Furnished", "Unfurnished"],
                            size=n_samples, p=[0.20, 0.45, 0.35])

    # Locality premium (Bandra/Colaba/Juhu/Worli are expensive).
    locality_premium_map = {
        "Colaba": 280, "Juhu": 250, "Bandra West": 240, "Worli": 220,
        "Lower Parel": 180, "Khar": 170, "Vile Parle": 120, "Andheri West": 110,
        "Powai": 100, "Dadar": 90, "Goregaon": 80, "Malad": 70, "Borivali": 60,
        "Andheri East": 70, "Goregaon East": 70, "Kurla": 50, "Chembur": 50,
        "Ghatkopar": 40, "Mulund": 40, "Dombivli": 30, "Thane West": 35,
    }
    locality_premium = np.array([locality_premium_map.get(l, 50) for l in localities])

    # Synthetic proximity features (will be overwritten by OSM enrichment if available).
    # South Mumbai (lower latitude) is closer to CBD/Marine Drive.
    south_proximity = (lat_max - latitude) / (lat_max - lat_min)  # 1=south, 0=north
    dist_metro_station_km = np.maximum(0.05, rng.normal(loc=2.5 - 1.5 * south_proximity, scale=1.2, size=n_samples))
    dist_hospital_km = np.maximum(0.1, rng.normal(loc=1.5 - 0.5 * south_proximity, scale=0.8, size=n_samples))
    dist_school_km = np.maximum(0.05, rng.normal(loc=1.0 - 0.3 * south_proximity, scale=0.6, size=n_samples))
    dist_park_km = np.maximum(0.05, rng.normal(loc=0.8, scale=0.5, size=n_samples))
    dist_mall_km = np.maximum(0.1, rng.normal(loc=2.0 - 0.6 * south_proximity, scale=1.0, size=n_samples))
    dist_bus_stop_km = np.maximum(0.02, rng.normal(loc=0.4, scale=0.3, size=n_samples))
    dist_commercial_hub_km = np.maximum(0.1, rng.normal(loc=2.5 - 1.5 * south_proximity, scale=1.3, size=n_samples))

    # Pricing model.
    base = (
        50
        + 4.0 * area_sqft / 100
        + 25 * bedrooms
        + 15 * bathrooms
        + 80 * (1 / (1 + age_years / 10))
        + 5 * (floor / np.maximum(total_floors, 1))
        - 8 * dist_metro_station_km
        - 4 * dist_school_km
        - 6 * dist_hospital_km
        - 3 * dist_park_km
        - 5 * dist_commercial_hub_km
        + locality_premium
    )
    # Heteroscedastic noise — bigger homes have wider price distributions.
    noise_std = 20 + area_sqft / 50
    price_lakh = base + rng.normal(0, noise_std)
    # Clip to a sane range.
    price_lakh = np.maximum(price_lakh, 5.0)

    df = pd.DataFrame({
        "latitude": latitude, "longitude": longitude,
        "area_sqft": area_sqft, "bedrooms": bedrooms, "bathrooms": bathrooms,
        "age_years": age_years, "floor": floor, "total_floors": total_floors,
        "dist_metro_station_km": dist_metro_station_km,
        "dist_hospital_km": dist_hospital_km,
        "dist_school_km": dist_school_km,
        "dist_park_km": dist_park_km,
        "dist_mall_km": dist_mall_km,
        "dist_bus_stop_km": dist_bus_stop_km,
        "dist_commercial_hub_km": dist_commercial_hub_km,
        "metro": "mumbai",
        "locality": localities,
        "property_type": property_type,
        "furnishing": furnishing,
        "price_lakh": price_lakh,
    })
    return df


# ---------------------------------------------------------------------------
# OSMnx enrichment
# ---------------------------------------------------------------------------
def _try_import_osmnx():
    """Lazy osmnx import — degrade gracefully if missing."""
    try:
        import osmnx as ox
        return ox
    except Exception:
        log.warning("osmnx not available — falling back to synthetic proximity features.")
        return None


def enrich_with_osm_features(
    df: pd.DataFrame,
    metro: Metro,
    pois: Optional[List[Tuple[str, Dict[str, str]]]] = None,
    cache: bool = True,
    network_timeout: int = 60,
) -> Tuple[pd.DataFrame, str]:
    """Add nearest-POI distance columns to ``df`` via OSMnx.

    Parameters
    ----------
    df : pd.DataFrame
        Must contain ``latitude`` and ``longitude`` columns.
    metro : Metro
        Which city's OSM graph to query.
    pois : list of (column_name, tag_dict), optional
        Defaults to ``PROXIMITY_POIS``.
    cache : bool
        If True, persist the OSM graph + POI GeoDataFrames to
        ``data/_osm_cache/`` so re-runs are instant.
    network_timeout : int
        Per-request timeout for OSMnx downloads (seconds).

    Returns
    -------
    (enriched_df, proximity_source)
        ``proximity_source`` is ``"osmnx"`` if real OSM data was used,
        or ``"synthetic_fallback"`` if OSMnx failed and we kept the
        existing proximity columns unchanged.
    """
    pois = pois or PROXIMITY_POIS
    ox = _try_import_osmnx()

    if ox is None:
        # osmnx unavailable — keep existing proximity columns (they were
        # set by the synthetic generator if the caller used it).
        return df.copy(), "synthetic_fallback"

    # Try to download POIs; if network is blocked, fall back.
    try:
        OSM_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        place = METRO_OSM_PLACES[metro.value]

        # Configure osmnx for caching + sensible rate-limit.
        ox.settings.timeout = network_timeout
        ox.settings.cache_folder = str(OSM_CACHE_DIR)
        ox.settings.use_cache = cache

        # Download POIs for each tag set.
        all_pois_gdf = []
        for col_name, tags in pois:
            try:
                gdf = ox.features_from_place(place, tags=tags)
                gdf = gdf.reset_index()[["geometry"]].copy()
                gdf["poi_kind"] = col_name
                # Keep only Point geometries (Polygon/LineString would need
                # nearest-edge matching which is more complex).
                gdf = gdf[gdf.geometry.geom_type == "Point"]
                all_pois_gdf.append(gdf)
            except Exception as exc:
                log.warning("Failed to fetch POIs for %s (%s): %s", col_name, tags, exc)

        if not all_pois_gdf:
            log.warning("OSMnx returned no POIs — keeping synthetic proximity features.")
            return df.copy(), "synthetic_fallback"

        all_pois = pd.concat(all_pois_gdf, ignore_index=True)

        # For each property, find the nearest POI of each kind.
        from geopy.distance import geodesic
        enriched = df.copy()
        for col_name, _ in pois:
            poi_subset = all_pois[all_pois["poi_kind"] == col_name]
            if len(poi_subset) == 0:
                continue
            poi_lats = poi_subset.geometry.y.values
            poi_lons = poi_subset.geometry.x.values

            # Vectorized nearest-POI search via haversine on the full grid.
            prop_lats = enriched["latitude"].values[:, None]
            prop_lons = enriched["longitude"].values[:, None]
            poi_lats_b = poi_lats[None, :]
            poi_lons_b = poi_lons[None, :]
            dists = haversine_distance_km(prop_lats, prop_lons, poi_lats_b, poi_lons_b)
            min_dists = dists.min(axis=1)
            enriched[col_name] = min_dists

        return enriched, "osmnx"

    except Exception as exc:
        log.warning("OSMnx enrichment failed (%s); using synthetic proximity features.", exc)
        return df.copy(), "synthetic_fallback"


# ---------------------------------------------------------------------------
# Final unified selection
# ---------------------------------------------------------------------------
def _select_unified(df: pd.DataFrame, metro: Metro, source: str,
                    sha: str, proximity_source: str) -> HousingDataset:
    """Pick canonical feature columns + target out of ``df``."""
    # Defensive defaults for any missing column.
    defaults = {
        "latitude": 0.0, "longitude": 0.0, "area_sqft": 800, "bedrooms": 2,
        "bathrooms": 2, "age_years": 10, "floor": 1, "total_floors": 7,
        "dist_metro_station_km": 2.0, "dist_hospital_km": 1.5,
        "dist_school_km": 1.0, "dist_park_km": 0.8, "dist_mall_km": 2.0,
        "dist_bus_stop_km": 0.4, "dist_commercial_hub_km": 2.5,
        "metro": metro.value, "locality": "unknown",
        "property_type": "Apartment", "furnishing": "Unfurnished",
        "price_lakh": 50.0,
    }
    for col, default in defaults.items():
        if col not in df.columns:
            df[col] = default

    X = df[SCHEMA.all_features].copy()
    y = df[SCHEMA.target].astype(float).copy()
    return HousingDataset(
        metro=metro, df=df, X=X, y=y,
        source=source, sha256=sha, proximity_source=proximity_source,
    )


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------
def _sha256(payload: bytes | str) -> str:
    if isinstance(payload, str):
        payload = payload.encode("utf-8")
    h = hashlib.sha256()
    h.update(payload)
    return h.hexdigest()


def load_housing(
    metro: str | Metro = Metro.MUMBAI,
    csv_path: Optional[Path | str] = None,
    n_samples: int = 1500,
    seed: int = 42,
    use_osm: bool = False,
) -> HousingDataset:
    """One-call loader for the housing dataset.

    Resolution order:
        1. ``csv_path`` (explicit override — must contain lat/lon + price_lakh).
        2. ``data/<metro>_housing.csv`` (project-local drop-in).
        3. ``generate_synthetic_<metro>`` fallback.

    Parameters
    ----------
    metro : Metro
        Which Indian metro to load.
    csv_path : str or Path, optional
        Explicit CSV override.
    n_samples, seed : int
        Used only by the synthetic fallback.
    use_osm : bool
        If True, attempt OSMnx enrichment. Default False because OSMnx
        requires network access on the first run.
    """
    if isinstance(metro, str):
        metro = Metro(metro.lower())

    if csv_path is not None:
        path = Path(csv_path)
        if not path.exists():
            raise FileNotFoundError(f"Housing CSV not found: {path}")
        df = pd.read_csv(path)
        source = str(path)
        sha = _sha256(path.read_bytes())
        proximity_source = "csv_provided"
    elif (PROJECT_DATA_DIR / f"{metro.value}_housing.csv").exists():
        path = PROJECT_DATA_DIR / f"{metro.value}_housing.csv"
        df = pd.read_csv(path)
        source = str(path)
        sha = _sha256(path.read_bytes())
        proximity_source = "csv_provided"
    else:
        log.info("Using synthetic %s housing data (n=%d, seed=%d)", metro.value, n_samples, seed)
        if metro == Metro.MUMBAI:
            df = generate_synthetic_mumbai(n_samples=n_samples, seed=seed)
        else:
            # For Delhi/Bangalore we use the Mumbai generator with a bbox swap
            # — keeps the pipeline testable. A future revision would add
            # city-specific generators.
            df = generate_synthetic_mumbai(n_samples=n_samples, seed=seed,
                                            bbox=METRO_BBOXES[metro.value])
            df["metro"] = metro.value
            # Adjust locality names for Delhi/Bangalore (cosmetic).
            if metro == Metro.DELHI:
                df["locality"] = df["locality"].replace({
                    "Bandra West": "Vasant Vihar", "Colaba": "Connaught Place",
                    "Juhu": "Lodhi Colony", "Worli": "India Gate",
                })
            elif metro == Metro.BANGALORE:
                df["locality"] = df["locality"].replace({
                    "Bandra West": "Indiranagar", "Colaba": "MG Road",
                    "Juhu": "Koramangala", "Worli": "Jayanagar",
                })
        source = "synthetic"
        sha = _sha256(df.to_csv(index=False))
        proximity_source = "synthetic_fallback"

    # Optionally enrich with real OSM proximity features.
    if use_osm:
        df, proximity_source = enrich_with_osm_features(df, metro)

    return _select_unified(df, metro, source, sha, proximity_source)


__all__ = [
    "Metro",
    "HousingSchema",
    "HousingDataset",
    "SCHEMA",
    "PROXIMITY_POIS",
    "METRO_BBOXES",
    "METRO_OSM_PLACES",
    "generate_synthetic_mumbai",
    "load_housing",
    "enrich_with_osm_features",
    "haversine_distance_km",
]
