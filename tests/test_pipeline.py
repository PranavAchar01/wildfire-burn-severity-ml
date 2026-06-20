import numpy as np
from burnseverity.data import make_synthetic_fire
from burnseverity.features import add_indices, FEATURE_COLUMNS
from burnseverity.pipeline import train_and_evaluate, compare_models, build_pipeline


def test_add_indices_columns():
    df = make_synthetic_fire(n=50)
    feat = add_indices(df)
    for c in FEATURE_COLUMNS:
        assert c in feat.columns


def test_pipeline_beats_majority_baseline():
    df = make_synthetic_fire(n=1500, seed=1)
    baseline = df["severity"].value_counts(normalize=True).max()
    _, res = train_and_evaluate(df, model="random_forest")
    assert res.accuracy > baseline + 0.1
    assert res.macro_f1 > 0.4


def test_cross_val_stable():
    df = make_synthetic_fire(n=1200, seed=2)
    _, res = train_and_evaluate(df, model="gboost", cv=4)
    assert 0.0 <= res.cv_mean <= 1.0 and res.cv_std < 0.15


def test_compare_models_ranks():
    df = make_synthetic_fire(n=800, seed=3)
    table = compare_models(df, cv=3)
    assert list(table.columns) == ["model", "cv_accuracy", "cv_std"]
    assert table["cv_accuracy"].is_monotonic_decreasing


def test_feature_importance_present_for_rf():
    df = make_synthetic_fire(n=600, seed=4)
    _, res = train_and_evaluate(df, model="random_forest")
    assert "dnbr" in res.feature_importance
    assert abs(sum(res.feature_importance.values()) - 1.0) < 1e-6
