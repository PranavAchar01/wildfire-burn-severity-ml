"""Synthetic and real multispectral fire data for burn-severity modeling.

The generator draws a latent severity per sample, then synthesizes physically
plausible pre/post Sentinel-2 reflectances: fire lowers NIR (vegetation loss)
and raises SWIR2 (char/ash, exposed soil), scaled by severity, with sensor
noise. dNBR computed from the noisy bands therefore correlates with — but does
not perfectly determine — the latent label, making this a realistic supervised
classification task rather than a thresholding identity.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .features import SEVERITY_CLASSES, add_indices

# Mean post-fire change in (NIR, SWIR2) reflectance per severity class.
_SEVERITY_EFFECT = {
    "unburned": (0.00, 0.00),
    "low": (-0.06, 0.05),
    "moderate": (-0.14, 0.11),
    "high": (-0.24, 0.20),
}


def make_synthetic_fire(n: int = 2000, seed: int = 0,
                        class_weights=(0.4, 0.25, 0.2, 0.15)) -> pd.DataFrame:
    """Return a DataFrame of band columns + a ``severity`` label column."""
    rng = np.random.default_rng(seed)
    labels = rng.choice(SEVERITY_CLASSES, size=n, p=class_weights)

    # Pre-fire healthy vegetation: high NIR, low SWIR2, moderate red.
    red_pre = np.clip(rng.normal(0.05, 0.015, n), 0.005, None)
    nir_pre = np.clip(rng.normal(0.40, 0.05, n), 0.05, None)
    swir2_pre = np.clip(rng.normal(0.12, 0.03, n), 0.01, None)

    d_nir = np.array([_SEVERITY_EFFECT[l][0] for l in labels])
    d_swir = np.array([_SEVERITY_EFFECT[l][1] for l in labels])

    noise = lambda s: rng.normal(0, s, n)
    nir_post = np.clip(nir_pre + d_nir + noise(0.03), 0.01, None)
    swir2_post = np.clip(swir2_pre + d_swir + noise(0.025), 0.01, None)
    # Burned pixels also brighten slightly in red.
    red_post = np.clip(red_pre + 0.4 * d_swir + noise(0.01), 0.005, None)

    df = pd.DataFrame({
        "red_pre": red_pre, "nir_pre": nir_pre, "swir2_pre": swir2_pre,
        "red_post": red_post, "nir_post": nir_post, "swir2_post": swir2_post,
        "severity": labels,
    })
    return df


def load_bands_csv(path: str) -> pd.DataFrame:
    """Load a CSV of band columns (+ optional severity) and validate it."""
    df = pd.read_csv(path)
    # Trigger validation of required band columns.
    add_indices(df)
    return df
