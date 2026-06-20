# Wildfire Burn-Severity ML

An end-to-end **machine-learning / data-science** pipeline that classifies wildfire **burn severity** (unburned / low / moderate / high) from pre- and post-fire multispectral satellite reflectance. It demonstrates the full applied-ML workflow on a remote-sensing problem: feature engineering from raw bands, scaling, model comparison with cross-validation, evaluation, and feature-importance analysis.

## The problem

After a fire, burned vegetation drops in near-infrared reflectance and rises in shortwave-infrared (char, ash, exposed soil). The **Normalized Burn Ratio** captures this, and the **differenced NBR (dNBR = NBR_pre − NBR_post)** is the operational standard for mapping severity (Key & Benson, 2006). This project treats severity mapping as a supervised classification task, learning from engineered spectral features rather than a single threshold.

## What it does

- **Feature engineering** (`features.py`): computes NBR, NDVI, dNBR, and dNDVI from Sentinel-2/Landsat band columns, plus USGS dNBR severity breakpoints.
- **Modeling** (`pipeline.py`): a scikit-learn `Pipeline` (StandardScaler → classifier) supporting Logistic Regression, Random Forest, and Gradient Boosting, with stratified train/test split and k-fold cross-validation.
- **Evaluation**: accuracy, macro-F1, confusion matrix, classification report, and feature importances.
- **Realistic synthetic data** (`data.py`): draws a latent severity per sample, then synthesizes physically plausible pre/post reflectances with sensor noise — so dNBR correlates with, but doesn't trivially determine, the label.

## Quickstart

```bash
pip install -e ".[dev]"

# Train + evaluate on synthetic data
burnseverity --model random_forest

# Cross-validate all three models
burnseverity --compare

# Train on your own CSV and save the fitted pipeline
burnseverity --csv fire_samples.csv --model gboost --out model.joblib
```

Run the demo:

```bash
python examples/quickstart.py
```

## Input format

A CSV with six band columns (surface reflectance, 0–1) and an optional `severity` label:

```
red_pre,nir_pre,swir2_pre,red_post,nir_post,swir2_post,severity
0.05,0.41,0.12,0.07,0.20,0.30,high
```

## Example output

```json
{
  "model": "random_forest",
  "accuracy": 0.83,
  "macro_f1": 0.80,
  "cv_accuracy": "0.82 +/- 0.01",
  "feature_importance": {"dnbr": 0.41, "nbr_post": 0.22, "dndvi": 0.13}
}
```

## Tests

```bash
pytest -q
```

Tests verify index formulas, USGS threshold mapping, that the classifier beats the majority-class baseline, cross-validation stability, and feature-importance sanity.

## License

MIT — see [LICENSE](LICENSE).
