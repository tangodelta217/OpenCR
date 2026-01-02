"""
Evaluation Metrics.

Provides metrics for classification and regression tasks.
"""

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opencr.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ClassificationMetrics:
    """Classification evaluation metrics."""

    accuracy: float
    precision: float
    recall: float
    f1: float
    auc_roc: float | None  # Only for binary/multi-class with probabilities
    confusion_matrix: NDArray[np.intp]
    per_class_f1: dict[str, float]

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auc_roc": self.auc_roc,
            "confusion_matrix": self.confusion_matrix.tolist(),
            "per_class_f1": self.per_class_f1,
        }


@dataclass
class RegressionMetrics:
    """Regression evaluation metrics."""

    mse: float
    rmse: float
    mae: float
    r2: float
    pearson_r: float

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "mse": self.mse,
            "rmse": self.rmse,
            "mae": self.mae,
            "r2": self.r2,
            "pearson_r": self.pearson_r,
        }


def compute_classification_metrics(
    y_true: NDArray,
    y_pred: NDArray,
    y_proba: NDArray[np.floating] | None = None,
) -> ClassificationMetrics:
    """
    Compute classification metrics.

    Args:
        y_true: True labels.
        y_pred: Predicted labels.
        y_proba: Predicted probabilities (optional).

    Returns:
        ClassificationMetrics object.
    """
    from sklearn.metrics import (
        accuracy_score,
        confusion_matrix,
        f1_score,
        precision_score,
        recall_score,
        roc_auc_score,
    )

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_true, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    cm = confusion_matrix(y_true, y_pred)

    # Per-class F1
    classes = np.unique(np.concatenate([y_true, y_pred]))
    per_class = {}
    for cls in classes:
        cls_mask = y_true == cls
        if cls_mask.sum() > 0:
            cls_f1 = f1_score(y_true == cls, y_pred == cls, zero_division=0)
            per_class[str(cls)] = float(cls_f1)

    # AUC-ROC
    auc = None
    if y_proba is not None:
        try:
            if len(classes) == 2:
                # Binary classification
                auc = roc_auc_score(y_true, y_proba[:, 1])
            else:
                # Multi-class
                auc = roc_auc_score(y_true, y_proba, multi_class="ovr", average="weighted")
        except Exception as e:
            logger.warning(f"Could not compute AUC: {e}")
            auc = None

    return ClassificationMetrics(
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
        auc_roc=float(auc) if auc is not None else None,
        confusion_matrix=cm,
        per_class_f1=per_class,
    )


def compute_regression_metrics(
    y_true: NDArray[np.floating],
    y_pred: NDArray[np.floating],
) -> RegressionMetrics:
    """
    Compute regression metrics.

    Args:
        y_true: True values.
        y_pred: Predicted values.

    Returns:
        RegressionMetrics object.
    """
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    # Pearson correlation
    if len(y_true) > 1 and np.std(y_true) > 0 and np.std(y_pred) > 0:
        pearson_r = float(np.corrcoef(y_true, y_pred)[0, 1])
    else:
        pearson_r = 0.0

    return RegressionMetrics(
        mse=float(mse),
        rmse=float(rmse),
        mae=float(mae),
        r2=float(r2),
        pearson_r=pearson_r,
    )


def aggregate_fold_metrics(
    fold_metrics: list[ClassificationMetrics | RegressionMetrics],
) -> dict[str, dict[str, float]]:
    """
    Aggregate metrics across folds.

    Args:
        fold_metrics: List of metrics from each fold.

    Returns:
        Dictionary with mean and std for each metric.
    """
    if not fold_metrics:
        return {}

    # Get all numeric metric names
    first = fold_metrics[0]
    metric_dict = first.to_dict()
    numeric_keys = [
        k for k, v in metric_dict.items() if isinstance(v, (int, float)) and v is not None
    ]

    result = {}
    for key in numeric_keys:
        values = []
        for fm in fold_metrics:
            val = fm.to_dict().get(key)
            if val is not None:
                values.append(val)

        if values:
            result[key] = {
                "mean": float(np.mean(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
            }

    return result
