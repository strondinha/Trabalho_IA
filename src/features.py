from __future__ import annotations

from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split


def stratified_sample(X: pd.DataFrame, y: pd.Series, sample_frac: float, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Return a stratified sample pair (X, y)."""
    if sample_frac >= 1:
        return X, y
    if sample_frac <= 0:
        raise ValueError("sample_frac deve estar no intervalo (0, 1].")

    X_sample, _, y_sample, _ = train_test_split(
        X,
        y,
        train_size=sample_frac,
        stratify=y,
        random_state=random_state,
    )
    return X_sample, y_sample


def stratified_sample_train(X: pd.DataFrame, y: pd.Series, sample_frac: float, random_state: int = 42) -> tuple[pd.DataFrame, pd.Series]:
    """Backward-compatible alias used by existing notebook code."""
    return stratified_sample(X, y, sample_frac=sample_frac, random_state=random_state)


def get_rf_top_features(estimator: Any, top_n: int = 15) -> pd.DataFrame:
    """Return top-N RF importances from a fitted Pipeline or SearchCV.

    Expected input is either:
      - a fitted sklearn Pipeline with `preprocessor`, `selector`, `classifier` steps; or
      - a fitted SearchCV with `.best_estimator_` following the same structure.
    Args:
        estimator: Fitted Pipeline or SearchCV estimator.
        top_n: Number of top features to keep.

    Returns:
        DataFrame with columns:
          - `feature`: transformed feature name from preprocessing/selection.
          - `importance`: feature importance value from RandomForest.
        Returns an empty DataFrame when classifier importances are unavailable.
    """
    fitted = getattr(estimator, "best_estimator_", estimator)
    if not hasattr(fitted, "named_steps"):
        return pd.DataFrame(columns=["feature", "importance"])

    preprocessor = fitted.named_steps.get("preprocessor")
    classifier = fitted.named_steps.get("classifier")
    selector = fitted.named_steps.get("selector")

    if preprocessor is None or classifier is None or not hasattr(classifier, "feature_importances_"):
        return pd.DataFrame(columns=["feature", "importance"])

    feature_names = preprocessor.get_feature_names_out()
    if selector is not None and selector != "passthrough" and hasattr(selector, "get_support"):
        feature_names = feature_names[selector.get_support()]

    importances = classifier.feature_importances_
    n = min(len(feature_names), len(importances))
    if n == 0:
        return pd.DataFrame(columns=["feature", "importance"])

    return (
        pd.DataFrame({"feature": feature_names[:n], "importance": importances[:n]})
        .sort_values("importance", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
