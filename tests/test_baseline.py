"""
Tests for baseline ML pipeline.

Tests features, models, LOSO CV, and CLI commands.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from opencr.evaluation.cv import loso_split
from opencr.evaluation.metrics import (
    compute_classification_metrics,
    compute_regression_metrics,
)
from opencr.features.bioz import extract_bioz_features
from opencr.features.fusion import extract_all_features
from opencr.features.ppg import extract_ppg_features
from opencr.models.baseline import BaselineModel, ModelConfig, ModelType, TaskType


@pytest.fixture
def synthetic_preprocessed_dataset(tmp_path: Path) -> Path:
    """Create a synthetic preprocessed dataset with 3 subjects."""
    np.random.seed(42)
    fs = 100.0
    window_samples = 3000  # 30 seconds

    for subject_id in ["subj001", "subj002", "subj003"]:
        n_windows = 10

        # Create windows with different class distributions per subject
        X = np.random.randn(n_windows, 2, window_samples)

        # Add signal pattern based on label
        y = np.array([1, 1, 1, 2, 2, 2, 3, 3, 4, 4])
        for w in range(n_windows):
            # Add frequency signature based on label
            freq = 0.5 + y[w] * 0.3
            t = np.arange(window_samples) / fs
            X[w, 0] += np.sin(2 * np.pi * freq * t)
            X[w, 1] += np.sin(2 * np.pi * freq * 0.5 * t)

        sqi = np.random.uniform(0.6, 1.0, (n_windows, 2))
        valid_mask = np.ones(n_windows, dtype=bool)
        timestamps = np.arange(n_windows) * 30.0

        np.savez(
            tmp_path / f"{subject_id}.npz",
            X=X,
            y=y,
            sqi=sqi,
            valid_mask=valid_mask,
            timestamps=timestamps,
            metadata={"fs": fs, "n_windows": n_windows},
        )

    return tmp_path


class TestFeatures:
    """Tests for feature extraction."""

    def test_ppg_features(self) -> None:
        """Test PPG feature extraction."""
        fs = 100.0
        t = np.arange(3000) / fs
        window = np.sin(2 * np.pi * 1.2 * t) + np.random.normal(0, 0.1, len(t))

        features = extract_ppg_features(window, fs)

        assert len(features) > 10
        assert "ppg_mean" in features
        assert "ppg_hr_band_power" in features
        assert all(isinstance(v, float) for v in features.values())

    def test_bioz_features(self) -> None:
        """Test BioZ feature extraction."""
        fs = 100.0
        t = np.arange(3000) / fs
        window = np.sin(2 * np.pi * 0.25 * t) + np.random.normal(0, 0.05, len(t))

        features = extract_bioz_features(window, fs)

        assert len(features) > 10
        assert "bioz_mean" in features
        assert "bioz_resp_rate_est" in features

    def test_all_features(self) -> None:
        """Test combined feature extraction."""
        fs = 100.0
        X = np.random.randn(5, 2, 3000)
        sqi = np.random.uniform(0.5, 1.0, (5, 2))

        features, names = extract_all_features(X, fs, sqi)

        assert features.shape[0] == 5
        assert features.shape[1] == len(names)
        assert not np.any(np.isnan(features))
        assert not np.any(np.isinf(features))


class TestLOSO:
    """Tests for LOSO cross-validation."""

    def test_loso_split(self) -> None:
        """Test LOSO produces correct splits."""
        subjects = np.array(["A", "A", "A", "B", "B", "C", "C", "C", "C"])

        folds = list(loso_split(subjects))

        assert len(folds) == 3  # 3 unique subjects

        # Check fold 0 (test on A)
        fold_a = folds[0]
        assert fold_a.test_subject == "A"
        assert len(fold_a.test_indices) == 3
        assert len(fold_a.train_indices) == 6

        # Verify no overlap
        for fold in folds:
            train_set = set(fold.train_indices)
            test_set = set(fold.test_indices)
            assert train_set.isdisjoint(test_set)

    def test_loso_no_leakage(self) -> None:
        """Test that LOSO prevents subject leakage."""
        subjects = np.array(["A"] * 5 + ["B"] * 5 + ["C"] * 5)

        for fold in loso_split(subjects):
            train_subjects = subjects[fold.train_indices]
            test_subjects = subjects[fold.test_indices]

            # All test samples should be from ONE subject
            assert len(np.unique(test_subjects)) == 1

            # That subject should NOT appear in training
            test_subject = np.unique(test_subjects)[0]
            assert test_subject not in train_subjects


class TestMetrics:
    """Tests for evaluation metrics."""

    def test_classification_metrics(self) -> None:
        """Test classification metrics computation."""
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 0, 2, 2])

        metrics = compute_classification_metrics(y_true, y_pred)

        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1 <= 1
        assert metrics.confusion_matrix.shape == (3, 3)

    def test_regression_metrics(self) -> None:
        """Test regression metrics computation."""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])

        metrics = compute_regression_metrics(y_true, y_pred)

        assert metrics.rmse > 0
        assert metrics.mae > 0
        assert metrics.r2 > 0.9  # Should be good fit


class TestBaselineModel:
    """Tests for baseline model."""

    def test_model_train_predict(self) -> None:
        """Test model training and prediction."""
        X_train = np.random.randn(100, 10)
        y_train = np.random.randint(0, 3, 100)
        X_test = np.random.randn(20, 10)

        config = ModelConfig(
            task_type=TaskType.CLASSIFICATION,
            model_type=ModelType.RANDOM_FOREST,
            n_estimators=10,
        )
        model = BaselineModel(config)
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        assert y_pred.shape == (20,)
        assert y_proba is not None
        assert y_proba.shape == (20, 3)

    def test_model_save_load(self, tmp_path: Path) -> None:
        """Test model serialization."""
        X = np.random.randn(50, 5)
        y = np.random.randint(0, 2, 50)

        model = BaselineModel()
        model.fit(X, y)

        model_path = tmp_path / "model.joblib"
        model.save(model_path)

        loaded = BaselineModel.load(model_path)
        y_pred_original = model.predict(X[:10])
        y_pred_loaded = loaded.predict(X[:10])

        np.testing.assert_array_equal(y_pred_original, y_pred_loaded)


class TestBaselineCLI:
    """Tests for baseline CLI commands."""

    def test_baseline_train_help(self) -> None:
        """Test baseline train --help."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "baseline", "train", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "loso" in result.stdout.lower()
        assert "cv" in result.stdout.lower()

    def test_baseline_evaluate_help(self) -> None:
        """Test baseline evaluate --help."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "baseline", "evaluate", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0

    def test_full_pipeline(self, synthetic_preprocessed_dataset: Path) -> None:
        """Test full train + evaluate pipeline."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "baseline_run"

            # Train
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "opencr",
                    "baseline",
                    "train",
                    str(synthetic_preprocessed_dataset),
                    "--output",
                    str(output_dir),
                    "--cv",
                    "loso",
                    "--n-estimators",
                    "10",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0, f"Train failed: {result.stderr}"
            assert (output_dir / "results.json").exists()
            assert (output_dir / "results.csv").exists()

            # Verify results.json structure
            with open(output_dir / "results.json") as f:
                results = json.load(f)

            assert results["n_subjects"] == 3
            assert len(results["folds"]) == 3
            assert "aggregated_metrics" in results

            # Verify LOSO: each fold tests on one subject
            test_subjects = {f["test_subject"] for f in results["folds"]}
            assert test_subjects == {"subj001", "subj002", "subj003"}

            # Evaluate
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "opencr",
                    "baseline",
                    "evaluate",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert (output_dir / "summary.json").exists()
