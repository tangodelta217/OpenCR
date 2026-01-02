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
    align_proba_to_classes,
    compute_classification_metrics,
    compute_regression_metrics,
)
from opencr.features.bioz import extract_bioz_features
from opencr.features.fusion import FeatureConfig, extract_all_features
from opencr.features.ppg import extract_ppg_features
from opencr.models.baseline import BaselineModel, ModelConfig, ModelType, TaskType
from opencr.targets.opencr import normalize_steps_to_opencr, steps_to_ordinal


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

        y_opencr = normalize_steps_to_opencr(y, direction="auto")
        y_ord = steps_to_ordinal(y, n_bins=4, direction="auto")

        np.savez(
            tmp_path / f"{subject_id}.npz",
            X=X,
            y=y,
            y_step=y,
            y_opencr=y_opencr,
            y_ord=y_ord,
            sqi=sqi,
            valid_mask=valid_mask,
            timestamps=timestamps,
            metadata={
                "fs": fs,
                "n_windows": n_windows,
                "target_direction": "auto",
                "ordinal_bins": 4,
            },
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

    def test_ablation_feature_dimensions(self) -> None:
        """Feature dimensions should differ across ablation configs."""
        fs = 100.0
        X = np.random.randn(4, 2, 1000)
        sqi = np.random.uniform(0.5, 1.0, (4, 2))

        ppg_only = FeatureConfig(
            include_ppg=True,
            include_bioz=False,
            include_sqi=True,
            include_cross=False,
        )
        fusion = FeatureConfig(
            include_ppg=True,
            include_bioz=True,
            include_sqi=True,
            include_cross=True,
        )

        feats_ppg, names_ppg = extract_all_features(X, fs, sqi, config=ppg_only)
        feats_fusion, names_fusion = extract_all_features(X, fs, sqi, config=fusion)

        assert feats_ppg.shape[1] == len(names_ppg)
        assert feats_fusion.shape[1] == len(names_fusion)
        assert feats_fusion.shape[1] > feats_ppg.shape[1]

    def test_sqi_feature_toggle_changes_dimensions(self) -> None:
        """SQI feature toggle should change feature dimensions."""
        fs = 100.0
        X = np.random.randn(3, 2, 800)
        sqi = np.random.uniform(0.2, 1.0, (3, 2))

        config_on = FeatureConfig(
            include_ppg=True,
            include_bioz=True,
            include_sqi=True,
            include_cross=True,
        )
        config_off = FeatureConfig(
            include_ppg=True,
            include_bioz=True,
            include_sqi=False,
            include_cross=True,
        )

        feats_on, names_on = extract_all_features(X, fs, sqi, config=config_on)
        feats_off, names_off = extract_all_features(X, fs, sqi, config=config_off)

        assert feats_on.shape[1] == len(names_on)
        assert feats_off.shape[1] == len(names_off)
        assert feats_on.shape[1] > feats_off.shape[1]


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

    def test_multiclass_auc_alignment_missing_class(self) -> None:
        """Aligned y_proba should handle missing classes without error."""
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 1, 1, 0, 1, 0])

        y_proba_local = np.array(
            [
                [0.9, 0.1],
                [0.2, 0.8],
                [0.6, 0.4],
                [0.85, 0.15],
                [0.3, 0.7],
                [0.7, 0.3],
            ],
            dtype=np.float64,
        )
        source_classes = np.array([0, 1])
        class_order = np.array([0, 1, 2])
        y_proba_aligned = align_proba_to_classes(y_proba_local, source_classes, class_order)

        assert y_proba_aligned.shape == (6, 3)
        assert np.allclose(y_proba_aligned[:, 2], 0.0)

        metrics = compute_classification_metrics(
            y_true,
            y_pred,
            y_proba_aligned,
            class_order=class_order,
        )
        assert metrics.auc_roc is not None
        assert 0.0 <= metrics.auc_roc <= 1.0


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
        feature_names = [f"f{i}" for i in range(X.shape[1])]

        model = BaselineModel()
        model.fit(X, y, feature_names=feature_names)

        model_path = tmp_path / "model.joblib"
        model.save(model_path)

        assert (tmp_path / "metadata.json").exists()
        assert (tmp_path / "feature_schema.json").exists()

        loaded = BaselineModel.load(model_path)
        y_pred_original = model.predict(X[:10])
        y_pred_loaded = loaded.predict(X[:10])

        np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
        assert loaded.feature_names == feature_names


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

    def test_baseline_train_invalid_cv(self, synthetic_preprocessed_dataset: Path) -> None:
        """Test baseline train rejects invalid CV strategy."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "baseline_run"

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
                    "foo",
                ],
                capture_output=True,
                text=True,
            )

            combined = (result.stdout + result.stderr).lower()
            assert result.returncode == 1
            assert "cv must be" in combined
            assert "loso" in combined

    def test_baseline_train_invalid_modalities(self, synthetic_preprocessed_dataset: Path) -> None:
        """Test baseline train rejects empty modality selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "baseline_run"

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
                    "--no-use-ppg",
                    "--no-use-bioz",
                ],
                capture_output=True,
                text=True,
            )

            combined = (result.stdout + result.stderr).lower()
            assert result.returncode == 1
            assert "use-ppg" in combined
            assert "use-bioz" in combined

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
            assert (output_dir / "splits.json").exists()
            assert (output_dir / "manifest.json").exists()

            # Verify results.json structure
            with open(output_dir / "results.json") as f:
                results = json.load(f)

            assert results["n_subjects"] == 3
            assert results["target"] == "opencr"
            assert "coverage" in results
            assert "metrics_ci" in results
            assert len(results["folds"]) == 3
            assert "aggregated_metrics" in results
            assert results["ablation"] == {
                "use_ppg": True,
                "use_bioz": True,
                "use_sqi_gating": True,
                "include_sqi_features": True,
                "sqi_gating_scope": "auto",
                "sqi_gating_scope_effective": "min",
            }

            with open(output_dir / "manifest.json", encoding="utf-8") as f:
                manifest = json.load(f)
            assert manifest["stage"] == "baseline_train"
            assert len(manifest.get("splits", [])) == 3
            assert manifest["config"]["ablation"]["use_sqi_gating"] is True

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

    def test_gating_off_uses_all_windows(self, tmp_path: Path) -> None:
        """Gating off should not filter any windows."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        rng = np.random.default_rng(7)

        def write_subject(subject_id: str, valid_mask: np.ndarray) -> None:
            n_windows = len(valid_mask)
            window_samples = 200
            X = rng.standard_normal((n_windows, 2, window_samples))
            y_step = np.tile([1, 2, 3, 4], n_windows // 4 + 1)[:n_windows]
            y_opencr = normalize_steps_to_opencr(y_step, direction="auto")
            y_ord = steps_to_ordinal(y_step, n_bins=4, direction="auto")
            sqi = rng.uniform(0.2, 1.0, (n_windows, 2))
            timestamps = np.arange(n_windows) * 30.0
            metadata = {"fs": 100.0, "target_direction": "auto", "ordinal_bins": 4}

            np.savez(
                processed_dir / f"{subject_id}.npz",
                X=X,
                y=y_step,
                y_step=y_step,
                y_opencr=y_opencr,
                y_ord=y_ord,
                sqi=sqi,
                valid_mask=valid_mask,
                timestamps=timestamps,
                metadata=metadata,
            )

        write_subject("subjA", np.array([True, False, True, False, True]))
        write_subject("subjB", np.array([False, True, True, False, True]))

        output_dir = tmp_path / "baseline_run"

        from opencr.cli import baseline_train

        baseline_train(
            processed_dir,
            output_dir=output_dir,
            cv="loso",
            use_ppg=True,
            use_bioz=True,
            use_sqi_gating=False,
            include_sqi_features=True,
            sqi_gating_scope="auto",
            target="opencr",
            model_type="random_forest",
            n_estimators=5,
            n_boot=0,
            seed=13,
            data_card=None,
        )

        with open(output_dir / "results.json") as f:
            results = json.load(f)

        assert results["coverage"] == pytest.approx(1.0)

    def test_gating_scope_auto_ppg_ignores_bioz(self, tmp_path: Path) -> None:
        """PPG-only + auto should not gate on BioZ."""
        processed_dir = tmp_path / "processed"
        processed_dir.mkdir()

        def write_subject(subject_id: str) -> None:
            n_windows = 4
            window_samples = 100
            X = np.random.randn(n_windows, 2, window_samples)
            y_step = np.array([1, 2, 3, 4])
            y_opencr = normalize_steps_to_opencr(y_step, direction="auto")
            y_ord = steps_to_ordinal(y_step, n_bins=4, direction="auto")
            sqi = np.column_stack([np.full(n_windows, 0.9), np.full(n_windows, 0.1)])
            valid_mask_ppg = np.ones(n_windows, dtype=bool)
            valid_mask_bioz = np.array([False, False, True, False])
            valid_mask_min = valid_mask_ppg & valid_mask_bioz
            timestamps = np.arange(n_windows) * 30.0
            metadata = {
                "fs": 100.0,
                "target_direction": "auto",
                "ordinal_bins": 4,
                "channels": ["ppg", "bioz"],
                "sqi_threshold": 0.5,
            }

            np.savez(
                processed_dir / f"{subject_id}.npz",
                X=X,
                y=y_step,
                y_step=y_step,
                y_opencr=y_opencr,
                y_ord=y_ord,
                sqi=sqi,
                valid_mask=valid_mask_min,
                valid_mask_min=valid_mask_min,
                valid_mask_ppg=valid_mask_ppg,
                valid_mask_bioz=valid_mask_bioz,
                timestamps=timestamps,
                metadata=metadata,
            )

        write_subject("subjA")
        write_subject("subjB")

        from opencr.cli import baseline_train

        output_ppg = tmp_path / "run_ppg"
        baseline_train(
            processed_dir,
            output_dir=output_ppg,
            cv="loso",
            use_ppg=True,
            use_bioz=False,
            use_sqi_gating=True,
            include_sqi_features=True,
            sqi_gating_scope="auto",
            target="opencr",
            model_type="random_forest",
            n_estimators=5,
            n_boot=0,
            seed=5,
            data_card=None,
        )

        with open(output_ppg / "results.json") as f:
            results_ppg = json.load(f)
        assert results_ppg["coverage"] == pytest.approx(1.0)

        output_fusion = tmp_path / "run_fusion"
        baseline_train(
            processed_dir,
            output_dir=output_fusion,
            cv="loso",
            use_ppg=True,
            use_bioz=True,
            use_sqi_gating=True,
            include_sqi_features=True,
            sqi_gating_scope="min",
            target="opencr",
            model_type="random_forest",
            n_estimators=5,
            n_boot=0,
            seed=5,
            data_card=None,
        )

        with open(output_fusion / "results.json") as f:
            results_fusion = json.load(f)
        assert results_fusion["coverage"] < 1.0
