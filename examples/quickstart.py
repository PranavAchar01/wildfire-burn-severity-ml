"""Generate fire data, compare models, and report the best classifier."""
from burnseverity.data import make_synthetic_fire
from burnseverity.pipeline import compare_models, train_and_evaluate

df = make_synthetic_fire(n=2000)
print("Cross-validated model comparison:")
print(compare_models(df).to_string(index=False))

best = compare_models(df).iloc[0]["model"]
_, res = train_and_evaluate(df, model=best)
print(f"\nBest model: {best}  accuracy={res.accuracy:.3f}  macroF1={res.macro_f1:.3f}")
print("Feature importance:", {k: round(v, 3) for k, v in res.feature_importance.items()})
