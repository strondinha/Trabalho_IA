from __future__ import annotations

from typing import Any

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC


def build_preprocessor(categorical_cols: list[str], numeric_cols: list[str], scale_numeric: bool = False) -> ColumnTransformer:
    numeric_steps: list[tuple[str, Any]] = [("imputer", SimpleImputer(strategy="median"))]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("cat", categorical_pipeline, categorical_cols),
            ("num", Pipeline(steps=numeric_steps), numeric_cols),
        ]
    )


def build_rf_pipeline(categorical_cols: list[str], numeric_cols: list[str], random_state: int = 42) -> Pipeline:
    preprocessor = build_preprocessor(categorical_cols, numeric_cols, scale_numeric=False)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", "passthrough"),
            (
                "classifier",
                RandomForestClassifier(
                    random_state=random_state,
                    n_jobs=-1,
                    class_weight="balanced",
                ),
            ),
        ]
    )


def build_svm_pipeline(categorical_cols: list[str], numeric_cols: list[str], random_state: int = 42) -> Pipeline:
    preprocessor = build_preprocessor(categorical_cols, numeric_cols, scale_numeric=True)
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("selector", "passthrough"),
            (
                "classifier",
                SVC(
                    random_state=random_state,
                    class_weight="balanced",
                    probability=False,
                ),
            ),
        ]
    )


def rf_param_grid() -> list[dict[str, Any]]:
    return [
        {
            "selector": ["passthrough", SelectKBest(score_func=mutual_info_classif, k=20), SelectKBest(score_func=mutual_info_classif, k=40)],
            "classifier__n_estimators": [200, 350],
            "classifier__max_depth": [None, 20],
            "classifier__min_samples_split": [2, 5],
            "classifier__class_weight": ["balanced", "balanced_subsample"],
            "classifier__max_features": ["sqrt", "log2"],
        }
    ]


def svm_param_grid() -> list[dict[str, Any]]:
    return [
        {
            "selector": ["passthrough", SelectKBest(score_func=mutual_info_classif, k=20), SelectKBest(score_func=mutual_info_classif, k=40)],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__C": [0.5, 1.0, 5.0],
            "classifier__gamma": ["scale", "auto"],
            "classifier__class_weight": ["balanced"],
        }
    ]


def rf_param_grid_compact() -> list[dict[str, Any]]:
    return [
        {
            "selector": [
                "passthrough",
                SelectKBest(score_func=mutual_info_classif, k=20),
                SelectKBest(score_func=mutual_info_classif, k=40),
            ],
            "classifier__n_estimators": [150, 250, 350],
            "classifier__max_depth": [None, 20, 30],
            "classifier__min_samples_split": [2, 5],
            "classifier__class_weight": ["balanced", "balanced_subsample"],
            "classifier__max_features": ["sqrt", "log2"],
        }
    ]


def svm_param_grid_compact() -> list[dict[str, Any]]:
    return [
        {
            "selector": [
                "passthrough",
                SelectKBest(score_func=mutual_info_classif, k=20),
                SelectKBest(score_func=mutual_info_classif, k=40),
            ],
            "classifier__kernel": ["linear", "rbf"],
            "classifier__C": [0.5, 1.0, 2.0],
            "classifier__gamma": ["scale", "auto"],
            "classifier__class_weight": ["balanced"],
        }
    ]


def build_search(
    pipeline: Pipeline,
    param_grid: list[dict[str, Any]],
    *,
    cv: int = 3,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
    random_state: int = 42,
    use_randomized: bool = False,
    n_iter: int = 12,
):
    if use_randomized:
        return RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=param_grid,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            verbose=1,
            random_state=random_state,
        )

    return GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )


def build_experiment_estimators(
    rf_pipeline: Pipeline,
    svm_pipeline: Pipeline,
    *,
    mode: str = "MODO_RAPIDO",
    cv: int = 2,
    scoring: str = "f1_weighted",
    n_jobs: int = -1,
    random_state: int = 42,
    rf_n_iter: int = 10,
    svm_n_iter: int = 8,
):
    mode_key = mode.strip().upper()
    if mode_key not in {"MODO_RAPIDO", "MODO_COMPLETO"}:
        raise ValueError("mode deve ser 'MODO_RAPIDO' ou 'MODO_COMPLETO'.")

    rf_estimator = build_search(
        rf_pipeline,
        rf_param_grid_compact(),
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        use_randomized=True,
        n_iter=rf_n_iter,
    )

    if mode_key == "MODO_RAPIDO":
        svm_estimator = svm_pipeline
    else:
        svm_estimator = build_search(
            svm_pipeline,
            svm_param_grid_compact(),
            cv=cv,
            scoring=scoring,
            n_jobs=n_jobs,
            random_state=random_state,
            use_randomized=True,
            n_iter=svm_n_iter,
        )

    return rf_estimator, svm_estimator


def optional_smote(random_state: int = 42):
    """Return SMOTE instance if imbalanced-learn is installed, otherwise None."""
    try:
        from imblearn.over_sampling import SMOTE

        return SMOTE(random_state=random_state)
    except ImportError:
        return None
