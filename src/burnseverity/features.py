"""Spectral indices and USGS burn-severity classification.

Bands follow Sentinel-2 / Landsat naming (surface reflectance, 0-1):
  red, nir, swir2 are used for the Normalized Burn Ratio and NDVI.

NBR   = (NIR - SWIR2) / (NIR + SWIR2)
NDVI  = (NIR - RED)   / (NIR + RED)
dNBR  = NBR_prefire - NBR_postfire   (higher => more vegetation loss)

Severity thresholds are the USGS / Key & Benson (2006) composite breakpoints,
collapsed to four operational classes.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd

SEVERITY_CLASSES = ["unburned", "low", "moderate", "high"]
# dNBR lower bounds for each class (inclusive of lower bound).
_DNBR_BREAKS = [(-np.inf, "unburned"), (0.10, "low"), (0.27, "moderate"), (0.66, "high")]


def _safe_ratio(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    denom = np.where(np.abs(b) < 1e-12, np.nan, b)
    return a / denom


def nbr(nir, swir2):
    """Normalized Burn Ratio."""
    return _safe_ratio(np.asarray(nir) - np.asarray(swir2),
                       np.asarray(nir) + np.asarray(swir2))


def ndvi(nir, red):
    """Normalized Difference Vegetation Index."""
    return _safe_ratio(np.asarray(nir) - np.asarray(red),
                       np.asarray(nir) + np.asarray(red))


def dnbr(nbr_pre, nbr_post):
    """Differenced NBR (pre minus post)."""
    return np.asarray(nbr_pre, dtype=float) - np.asarray(nbr_post, dtype=float)


def classify_dnbr(values) -> np.ndarray:
    """Map dNBR values to severity-class labels via USGS breakpoints."""
    values = np.atleast_1d(np.asarray(values, dtype=float))
    out = np.empty(values.shape, dtype=object)
    out[:] = "unburned"
    for lower, label in _DNBR_BREAKS:
        out[values >= lower] = label
    return out


def add_indices(df: pd.DataFrame) -> pd.DataFrame:
    """Add pre/post NBR, NDVI, and dNBR feature columns from band columns.

    Expects columns: red_pre, nir_pre, swir2_pre, red_post, nir_post, swir2_post.
    """
    required = ["red_pre", "nir_pre", "swir2_pre", "red_post", "nir_post", "swir2_post"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise KeyError(f"Missing band columns: {missing}")
    out = df.copy()
    out["nbr_pre"] = nbr(out["nir_pre"], out["swir2_pre"])
    out["nbr_post"] = nbr(out["nir_post"], out["swir2_post"])
    out["ndvi_pre"] = ndvi(out["nir_pre"], out["red_pre"])
    out["ndvi_post"] = ndvi(out["nir_post"], out["red_post"])
    out["dnbr"] = dnbr(out["nbr_pre"], out["nbr_post"])
    out["dndvi"] = out["ndvi_pre"] - out["ndvi_post"]
    return out


FEATURE_COLUMNS: Sequence[str] = (
    "nbr_pre", "nbr_post", "ndvi_pre", "ndvi_post", "dnbr", "dndvi",
)
