"""Wildfire burn-severity classification from multispectral reflectance."""
from .features import nbr, ndvi, dnbr, classify_dnbr, add_indices, SEVERITY_CLASSES
from .data import make_synthetic_fire, load_bands_csv
from .pipeline import build_pipeline, train_and_evaluate, compare_models

__all__ = [
    "nbr", "ndvi", "dnbr", "classify_dnbr", "add_indices", "SEVERITY_CLASSES",
    "make_synthetic_fire", "load_bands_csv",
    "build_pipeline", "train_and_evaluate", "compare_models",
]
__version__ = "0.1.0"
