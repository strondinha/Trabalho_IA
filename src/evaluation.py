from __future__ import annotations

from time import perf_counter
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix


def time_fit_and_predict(estimator: Any, X_train, y_train, X_test):
    t0 = perf_counter()
    estimator.fit(X_train, y_train)
    train_time = perf_counter() - t0

    t1 = perf_counter()
    y_pred = estimator.predict(X_test)
    inference_time = perf_counter() - t1

    return y_pred, train_time, inference_time


def classification_metrics(y_true, y_pred, labels: list[str] | None = None) -> dict[str, Any]:
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    report = classification_report(y_true, y_pred, labels=labels, output_dict=True, zero_division=0)
    text_report = classification_report(y_true, y_pred, labels=labels, zero_division=0)
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "confusion_matrix": cm,
        "classification_report": report,
        "classification_report_text": text_report,
    }


def plot_confusion(cm: np.ndarray, labels: list[str], title: str = "Matriz de Confusão", figsize: tuple[int, int] = (8, 6)):
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predito")
    ax.set_ylabel("Real")
    fig.tight_layout()
    return fig, ax


def summarize_result(model_name: str, metrics: dict[str, Any], train_time: float, inference_time: float) -> dict[str, Any]:
    weighted_f1 = metrics["classification_report"].get("weighted avg", {}).get("f1-score", np.nan)
    weighted_precision = metrics["classification_report"].get("weighted avg", {}).get("precision", np.nan)
    weighted_recall = metrics["classification_report"].get("weighted avg", {}).get("recall", np.nan)
    return {
        "model": model_name,
        "accuracy": metrics["accuracy"],
        "precision_weighted": weighted_precision,
        "recall_weighted": weighted_recall,
        "f1_weighted": weighted_f1,
        "train_time_s": train_time,
        "inference_time_s": inference_time,
    }
