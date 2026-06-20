import numpy as np
from burnseverity.features import nbr, ndvi, dnbr, classify_dnbr, SEVERITY_CLASSES


def test_nbr_formula():
    # NIR=0.4, SWIR2=0.1 -> (0.3)/(0.5) = 0.6
    assert abs(float(nbr(0.4, 0.1)) - 0.6) < 1e-9


def test_ndvi_range():
    v = ndvi(np.array([0.4, 0.5]), np.array([0.05, 0.1]))
    assert np.all(v > 0) and np.all(v < 1)


def test_dnbr_sign():
    assert dnbr(0.6, 0.1) > 0  # vegetation loss -> positive dNBR


def test_classify_thresholds():
    vals = [-0.1, 0.0, 0.15, 0.30, 0.70]
    out = classify_dnbr(vals)
    assert list(out) == ["unburned", "unburned", "low", "moderate", "high"]


def test_classify_all_known_classes():
    out = classify_dnbr([0.05, 0.2, 0.5, 0.9])
    assert set(out).issubset(set(SEVERITY_CLASSES))
