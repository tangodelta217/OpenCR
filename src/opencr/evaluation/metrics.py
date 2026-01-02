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


def align_proba_to_classes(
    y_proba: NDArray[np.floating] | None,
    source_classes: NDArray | list,
    target_classes: NDArray | list,
) -> NDArray[np.floating] | None:
    """
    Align predicted probabilities to a target class order.

    Missing classes in the source are filled with 0.0.
    """
    if y_proba is None:
        return None

    y_proba_arr = np.asarray(y_proba)
    if y_proba_arr.ndim != 2:
        return y_proba_arr

    source = np.asarray(source_classes)
    target = np.asarray(target_classes)

    if source.size != y_proba_arr.shape[1]:
        raise ValueError("y_proba shape does not match source_classes")

    aligned = np.zeros((y_proba_arr.shape[0], target.size), dtype=y_proba_arr.dtype)
    for idx, cls in enumerate(target):
        matches = np.where(source == cls)[0]
        if matches.size > 0:
            aligned[:, idx] = y_proba_arr[:, matches[0]]

    return aligned


def compute_classification_metrics(
    y_true: NDArray,
    y_pred: NDArray,
    y_proba: NDArray[np.floating] | None = None,
    *,
    class_order: NDArray | list | None = None,
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
            y_proba_arr = np.asarray(y_proba)
            class_order_arr = None
            if class_order is not None:
                class_order_arr = np.asarray(class_order)
                if y_proba_arr.ndim == 2 and class_order_arr.size != y_proba_arr.shape[1]:
                    raise ValueError("y_proba shape does not match class_order")
                missing = [cls for cls in classes if cls not in class_order_arr]
                if missing:
                    raise ValueError("class_order does not cover classes in y_true")
                indices = [int(np.where(class_order_arr == cls)[0][0]) for cls in classes]
                if y_proba_arr.ndim == 2:
                    y_proba_arr = y_proba_arr[:, indices]

            if len(classes) == 2:
                # Binary classification
                if len(np.unique(y_true)) < 2:
                    raise ValueError("AUC undefined for single-class y_true")
                pos_class = classes[-1]
                y_binary = (y_true == pos_class).astype(int)
                if y_proba_arr.ndim == 1:
                    y_score = y_proba_arr
                else:
                    if y_proba_arr.shape[1] != len(classes):
                        raise ValueError("y_proba shape does not match number of classes")
                    pos_idx = int(np.where(classes == pos_class)[0][0])
                    y_score = y_proba_arr[:, pos_idx]
                auc = roc_auc_score(y_binary, y_score)
            else:
                # Multi-class
                if y_proba_arr.ndim != 2 or y_proba_arr.shape[1] != len(classes):
                    raise ValueError("y_proba shape does not match number of classes")
                auc = roc_auc_score(y_true, y_proba_arr, multi_class="ovr", average="macro")
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


def _bootstrap_subject_indices(
    subject_ids: NDArray,
    rng: np.random.Generator,
) -> NDArray[np.intp]:
    unique_subjects = np.unique(subject_ids)
    if unique_subjects.size == 0:
        return np.array([], dtype=np.intp)

    subject_to_indices = {
        subject: np.where(subject_ids == subject)[0] for subject in unique_subjects
    }
    sampled = rng.choice(unique_subjects, size=unique_subjects.size, replace=True)
    return np.concatenate([subject_to_indices[subject] for subject in sampled])


def compute_metrics_with_ci(
    y_true: NDArray,
    y_pred: NDArray,
    y_proba: NDArray[np.floating] | None,
    subject_ids: NDArray | list[str],
    task_type: str,
    *,
    n_boot: int = 200,
    seed: int = 42,
    class_order: NDArray | list | None = None,
) -> tuple[ClassificationMetrics | RegressionMetrics, dict[str, dict[str, float]]]:
    """
    Compute metrics and bootstrap confidence intervals by subject.

    Args:
        y_true: True labels/targets.
        y_pred: Predicted labels/targets.
        y_proba: Predicted probabilities (classification only).
        subject_ids: Subject ID per window.
        task_type: "classification" or "regression".
        n_boot: Number of bootstrap resamples.
        seed: RNG seed for reproducibility.

    Returns:
        Tuple of (metrics, ci_dict).
    """
    task = task_type.lower()
    subject_ids_arr = np.asarray(subject_ids)

    if len(y_true) != len(subject_ids_arr):
        raise ValueError("subject_ids length must match y_true length")
    if len(y_pred) != len(y_true):
        raise ValueError("y_pred length must match y_true length")

    if task == "regression":
        metrics = compute_regression_metrics(y_true, y_pred)
        metric_names = ["rmse", "mae", "r2"]
    else:
        metrics = compute_classification_metrics(y_true, y_pred, y_proba, class_order=class_order)
        metric_names = ["accuracy", "f1", "auc_roc"]
        if metrics.auc_roc is None:
            metric_names = ["accuracy", "f1"]

    if n_boot <= 0:
        return metrics, {}

    rng = np.random.default_rng(seed)
    values: dict[str, list[float]] = {name: [] for name in metric_names}

    for _ in range(n_boot):
        indices = _bootstrap_subject_indices(subject_ids_arr, rng)
        if indices.size == 0:
            continue

        if task == "regression":
            boot_metrics = compute_regression_metrics(y_true[indices], y_pred[indices])
        else:
            if y_proba is None or len(y_proba) == 0:
                boot_proba = None
            else:
                boot_proba = y_proba[indices]
            boot_metrics = compute_classification_metrics(
                y_true[indices], y_pred[indices], boot_proba, class_order=class_order
            )

        for name in metric_names:
            value = getattr(boot_metrics, name)
            if value is not None:
                values[name].append(float(value))

    ci = {}
    for name, vals in values.items():
        if not vals:
            continue
        arr = np.asarray(vals, dtype=np.float64)
        ci[name] = {
            "mean": float(np.mean(arr)),
            "low": float(np.percentile(arr, 2.5)),
            "high": float(np.percentile(arr, 97.5)),
        }

    return metrics, ci
