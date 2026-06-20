"""CLI: train a burn-severity classifier and report cross-validated metrics."""
from __future__ import annotations

import argparse
import json

import joblib

from .data import make_synthetic_fire, load_bands_csv
from .pipeline import train_and_evaluate, compare_models


def main():
    p = argparse.ArgumentParser(description="Wildfire burn-severity classifier")
    p.add_argument("--csv", default=None, help="CSV of band columns + severity")
    p.add_argument("--model", default="random_forest",
                   choices=["logreg", "random_forest", "gboost"])
    p.add_argument("--compare", action="store_true", help="Cross-validate all models")
    p.add_argument("--n", type=int, default=2000, help="Synthetic sample count")
    p.add_argument("--out", default=None, help="Save fitted pipeline (.joblib)")
    args = p.parse_args()

    df = load_bands_csv(args.csv) if args.csv else make_synthetic_fire(n=args.n)

    if args.compare:
        print(compare_models(df).to_string(index=False))
        return

    pipe, res = train_and_evaluate(df, model=args.model)
    print(json.dumps({
        "model": res.model, "accuracy": round(res.accuracy, 4),
        "macro_f1": round(res.macro_f1, 4),
        "cv_accuracy": f"{res.cv_mean:.4f} +/- {res.cv_std:.4f}",
        "feature_importance": {k: round(v, 4) for k, v in res.feature_importance.items()},
    }, indent=2))
    print("\nClassification report:\n" + res.report)
    if args.out:
        joblib.dump(pipe, args.out)
        print(f"Saved pipeline -> {args.out}")


if __name__ == "__main__":
    main()
