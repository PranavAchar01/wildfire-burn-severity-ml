"""scikit-learn pipeline: features -> scaling -> classifier, with evaluation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .features import FEATURE_COLUMNS, SEVERITY_CLASSES, add_indices

_MODELS = {
    "logreg": lambda: LogisticRegression(max_iter=1000),
    "random_forest": lambda: RandomForestClassifier(n_estimators=200, random_state=0),
    "gboost": lambda: GradientBoostingClassifier(random_state=0),
}


def _xy(df: pd.DataFrame):
    feat = add_indices(df)
    X = feat[list(FEATURE_COLUMNS)].to_numpy()
    y = feat["severity"].to_numpy()
    return X, y


def build_pipeline(model: str = "random_forest") -> Pipeline:
    if model not in _MODELS:
        raise ValueError(f"Unknown model {model!r}; choose from {list(_MODELS)}")
    return Pipeline([("scaler", StandardScaler()), ("clf", _MODELS[model]())])


@dataclass
class EvalResult:
    model: str
    accuracy: float
    macro_f1: float
    cv_mean: float
    cv_std: float
    report: str
    confusion: list = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)


def train_and_evaluate(df: pd.DataFrame, model: str = "random_forest",
                       test_size: float = 0.25, cv: int = 5, seed: int = 0):
    X, y = _xy(df)
    Xtr, Xte, ytr, yte = train_test_split(
        X, y, test_size=test_size, random_state=seed, stratify=y)
    pipe = build_pipeline(model)
    pipe.fit(Xtr, ytr)
    pred = pipe.predict(Xte)

    acc = float((pred == yte).mean())
    macro_f1 = float(f1_score(yte, pred, average="macro"))
    cv_scores = cross_val_score(build_pipeline(model), X, y, cv=cv)
    labels = [c for c in SEVERITY_CLASSES if c in set(y)]
    cm = confusion_matrix(yte, pred, labels=labels).tolist()

    importance: Dict[str, float] = {}
    clf = pipe.named_steps["clf"]
    if hasattr(clf, "feature_importances_"):
        importance = {f: float(v) for f, v in
                      zip(FEATURE_COLUMNS, clf.feature_importances_)}

    result = EvalResult(
        model=model, accuracy=acc, macro_f1=macro_f1,
        cv_mean=float(cv_scores.mean()), cv_std=float(cv_scores.std()),
        report=classification_report(yte, pred, labels=labels, zero_division=0),
        confusion=cm, feature_importance=importance,
    )
    return pipe, result


def compare_models(df: pd.DataFrame, cv: int = 5) -> pd.DataFrame:
    """Cross-validated accuracy for every model; sorted best-first."""
    X, y = _xy(df)
    rows = []
    for name in _MODELS:
        scores = cross_val_score(build_pipeline(name), X, y, cv=cv)
        rows.append({"model": name, "cv_accuracy": scores.mean(),
                     "cv_std": scores.std()})
    return pd.DataFrame(rows).sort_values("cv_accuracy", ascending=False).reset_index(drop=True)
