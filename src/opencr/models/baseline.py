"""
Baseline ML Models.

Provides scikit-learn based models for classification and regression.
"""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opencr import __version__
from opencr.logging import get_logger

logger = get_logger(__name__)


class TaskType(str, Enum):
    """Type of prediction task."""

    CLASSIFICATION = "classification"
    REGRESSION = "regression"


class ModelType(str, Enum):
    """Type of model to use."""

    RANDOM_FOREST = "random_forest"
    GRADIENT_BOOSTING = "gradient_boosting"
    XGBOOST = "xgboost"


@dataclass
class ModelConfig:
    """Configuration for baseline model."""

    task_type: TaskType = TaskType.CLASSIFICATION
    model_type: ModelType = ModelType.RANDOM_FOREST
    n_estimators: int = 100
    max_depth: int | None = 10
    random_state: int = 42
    n_jobs: int = -1
    extra_params: dict[str, Any] = field(default_factory=dict)


class BaselineModel:
    """
    Baseline ML model wrapper.

    Supports classification and regression with RandomForest or GradientBoosting.
    Optionally uses XGBoost if installed.
    """

    def __init__(self, config: ModelConfig | None = None) -> None:
        """
        Initialize model with configuration.

        Args:
            config: Model configuration.
        """
        self.config = config or ModelConfig()
        self.model: Any = None
        self.feature_names: list[str] | None = None
        self._create_model()

    def _create_model(self) -> None:
        """Create the underlying sklearn model."""
        if self.config.model_type == ModelType.XGBOOST:
            self.model = self._create_xgboost()
        elif self.config.model_type == ModelType.GRADIENT_BOOSTING:
            self.model = self._create_gradient_boosting()
        else:
            self.model = self._create_random_forest()

    def _create_random_forest(self) -> Any:
        """Create RandomForest model."""
        from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

        params = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
            **self.config.extra_params,
        }

        if self.config.task_type == TaskType.CLASSIFICATION:
            return RandomForestClassifier(**params)
        return RandomForestRegressor(**params)

    def _create_gradient_boosting(self) -> Any:
        """Create GradientBoosting model."""
        from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor

        params = {
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth or 3,
            "random_state": self.config.random_state,
            **self.config.extra_params,
        }

        if self.config.task_type == TaskType.CLASSIFICATION:
            return GradientBoostingClassifier(**params)
        return GradientBoostingRegressor(**params)

    def _create_xgboost(self) -> Any:
        """Create XGBoost model if available."""
        try:
            import xgboost as xgb

            params = {
                "n_estimators": self.config.n_estimators,
                "max_depth": self.config.max_depth or 6,
                "random_state": self.config.random_state,
                "n_jobs": self.config.n_jobs,
                **self.config.extra_params,
            }

            if self.config.task_type == TaskType.CLASSIFICATION:
                return xgb.XGBClassifier(**params)
            return xgb.XGBRegressor(**params)

        except ImportError:
            logger.warning("XGBoost not installed, falling back to GradientBoosting")
            return self._create_gradient_boosting()

    def fit(
        self,
        X: NDArray[np.floating],
        y: NDArray,
        feature_names: list[str] | None = None,
    ) -> "BaselineModel":
        """
        Fit model to training data.

        Args:
            X: Feature matrix, shape (n_samples, n_features).
            y: Target array, shape (n_samples,).
            feature_names: Optional feature names for interpretability.

        Returns:
            Self for chaining.
        """
        self.feature_names = feature_names
        logger.info(f"Training {self.config.model_type.value} on {X.shape[0]} samples")

        self.model.fit(X, y)

        return self

    def predict(self, X: NDArray[np.floating]) -> NDArray:
        """
        Make predictions.

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Predictions, shape (n_samples,).
        """
        return self.model.predict(X)

    def predict_proba(self, X: NDArray[np.floating]) -> NDArray[np.floating] | None:
        """
        Predict probabilities (classification only).

        Args:
            X: Feature matrix, shape (n_samples, n_features).

        Returns:
            Probabilities, shape (n_samples, n_classes) or None for regression.
        """
        if self.config.task_type != TaskType.CLASSIFICATION:
            return None

        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def get_feature_importance(self) -> dict[str, float] | None:
        """
        Get feature importances.

        Returns:
            Dictionary of feature names to importance scores, or None.
        """
        if not hasattr(self.model, "feature_importances_"):
            return None

        importances = self.model.feature_importances_

        if self.feature_names and len(self.feature_names) == len(importances):
            return dict(zip(self.feature_names, importances, strict=False))

        return {f"feature_{i}": imp for i, imp in enumerate(importances)}

    def save(self, path: Path) -> None:
        """Save model artifacts to disk."""
        import json

        import joblib

        path = Path(path)
        model_path = path if path.suffix else path / "model.joblib"
        model_path.parent.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.model, model_path)

        metadata = {
            "opencr_version": __version__,
            "task_type": self.config.task_type.value,
            "model_type": self.config.model_type.value,
            "n_estimators": self.config.n_estimators,
            "max_depth": self.config.max_depth,
            "random_state": self.config.random_state,
            "n_jobs": self.config.n_jobs,
            "extra_params": self.config.extra_params,
            "feature_names": self.feature_names,
        }

        metadata_path = model_path.parent / "metadata.json"
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2)

        if self.feature_names:
            schema_path = model_path.parent / "feature_schema.json"
            with open(schema_path, "w", encoding="utf-8") as f:
                json.dump({"feature_names": self.feature_names}, f, indent=2)

        logger.info(f"Model saved to {model_path}")

    @classmethod
    def load(cls, path: Path) -> "BaselineModel":
        """Load model artifacts from disk."""
        import json

        import joblib

        path = Path(path)
        model_path = path if path.is_file() else path / "model.joblib"

        data = joblib.load(model_path)
        if isinstance(data, dict) and "model" in data:
            instance = cls(config=data.get("config", ModelConfig()))
            instance.model = data["model"]
            instance.feature_names = data.get("feature_names")
            logger.info(f"Loaded legacy model bundle from {model_path}")
            return instance

        metadata_path = model_path.parent / "metadata.json"
        feature_schema_path = model_path.parent / "feature_schema.json"
        if metadata_path.exists():
            with open(metadata_path, encoding="utf-8") as f:
                metadata = json.load(f)
            config = ModelConfig(
                task_type=TaskType(metadata.get("task_type", TaskType.CLASSIFICATION.value)),
                model_type=ModelType(metadata.get("model_type", ModelType.RANDOM_FOREST.value)),
                n_estimators=metadata.get("n_estimators", 100),
                max_depth=metadata.get("max_depth"),
                random_state=metadata.get("random_state", 42),
                n_jobs=metadata.get("n_jobs", -1),
                extra_params=metadata.get("extra_params", {}),
            )
            feature_names = metadata.get("feature_names")
        else:
            logger.warning(f"metadata.json not found in {model_path.parent}")
            config = ModelConfig()
            feature_names = None

        if feature_names is None and feature_schema_path.exists():
            with open(feature_schema_path, encoding="utf-8") as f:
                schema = json.load(f)
            feature_names = schema.get("feature_names")

        instance = cls(config=config)
        instance.model = data
        instance.feature_names = feature_names

        logger.info(f"Model loaded from {model_path}")
        return instance
