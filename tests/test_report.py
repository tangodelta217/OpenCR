"""
Tests for report generation.

Tests Annex A report generation with tables and figures.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest


@pytest.fixture
def synthetic_baseline_run(tmp_path: Path) -> Path:
    """Create a synthetic baseline run for report testing."""
    np.random.seed(42)

    # Create run directory with results
    run_dir = tmp_path / "baseline_run"
    run_dir.mkdir()

    # Create fold directories with predictions
    fold_results = []
    for i in range(3):
        fold_dir = run_dir / f"fold_{i:02d}"
        fold_dir.mkdir()

        # Synthetic predictions
        n_samples = 20
        y_true = np.random.randint(0, 3, n_samples)
        y_pred = y_true.copy()
        # Introduce some errors
        error_mask = np.random.rand(n_samples) < 0.2
        y_pred[error_mask] = (y_pred[error_mask] + 1) % 3
        y_proba = np.random.rand(n_samples, 3)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        np.savez(
            fold_dir / "predictions.npz",
            y_true=y_true,
            y_pred=y_pred,
            y_proba=y_proba,
        )

        # Metrics
        accuracy = (y_true == y_pred).mean()
        fold_results.append(
            {
                "fold_idx": i,
                "test_subject": f"subj00{i+1}",
                "n_train": 40,
                "n_test": n_samples,
                "metrics": {
                    "accuracy": float(accuracy),
                    "precision": float(np.random.uniform(0.6, 0.9)),
                    "recall": float(np.random.uniform(0.6, 0.9)),
                    "f1": float(np.random.uniform(0.6, 0.9)),
                    "auc_roc": float(np.random.uniform(0.7, 0.95)),
                    "confusion_matrix": [[5, 2], [1, 12]],
                    "per_class_f1": {"0": 0.7, "1": 0.8, "2": 0.75},
                },
            }
        )

    # Create results.json
    results = {
        "model_type": "random_forest",
        "cv_strategy": "loso",
        "target": "step",
        "task_type": "classification",
        "n_subjects": 3,
        "n_samples": 60,
        "n_features": 45,
        "feature_names": [f"feature_{i}" for i in range(45)],
        "folds": fold_results,
        "aggregated_metrics": {
            "accuracy": {"mean": 0.75, "std": 0.05, "min": 0.70, "max": 0.80},
            "f1": {"mean": 0.72, "std": 0.04, "min": 0.68, "max": 0.76},
            "auc_roc": {"mean": 0.82, "std": 0.03, "min": 0.79, "max": 0.85},
        },
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return run_dir


@pytest.fixture
def synthetic_baseline_run_regression(tmp_path: Path) -> Path:
    """Create a synthetic regression run for report testing."""
    run_dir = tmp_path / "baseline_run_reg"
    run_dir.mkdir()

    fold_dir = run_dir / "fold_00"
    fold_dir.mkdir()

    n_samples = 30
    y_true = np.linspace(0, 100, n_samples)
    y_pred = y_true + np.random.normal(0, 5, n_samples)

    np.savez(
        fold_dir / "predictions.npz",
        y_true=y_true,
        y_pred=y_pred,
        y_proba=np.array([]),
    )

    results = {
        "model_type": "random_forest",
        "cv_strategy": "loso",
        "target": "opencr",
        "task_type": "regression",
        "n_subjects": 1,
        "n_samples": n_samples,
        "n_features": 45,
        "feature_names": [f"feature_{i}" for i in range(45)],
        "folds": [
            {
                "fold_idx": 0,
                "test_subject": "subj001",
                "n_train": 50,
                "n_test": n_samples,
                "metrics": {
                    "rmse": 5.0,
                    "mae": 4.0,
                    "r2": 0.85,
                },
            }
        ],
        "aggregated_metrics": {
            "rmse": {"mean": 5.0, "std": 0.0, "min": 5.0, "max": 5.0},
            "mae": {"mean": 4.0, "std": 0.0, "min": 4.0, "max": 4.0},
            "r2": {"mean": 0.85, "std": 0.0, "min": 0.85, "max": 0.85},
        },
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    return run_dir


class TestAnnexA:
    """Tests for Annex A report generation."""

    def test_generate_annex_a(self, synthetic_baseline_run: Path) -> None:
        """Test generating Annex A report."""
        from opencr.report.annexA import generate_annex_a

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = generate_annex_a(synthetic_baseline_run, output_dir)

            # Check at least 1 table and 2 figures
            assert len(result["tables"]) >= 1
            assert len(result["figures"]) >= 2

            # Check files exist
            for table_path in result["tables"]:
                assert Path(table_path).exists()

            for fig_path in result["figures"]:
                assert Path(fig_path).exists()

    def test_summary_table_content(self, synthetic_baseline_run: Path) -> None:
        """Test summary table has correct content."""
        from opencr.report.annexA import generate_annex_a

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = generate_annex_a(synthetic_baseline_run, output_dir)

            # Check CSV table
            csv_tables = [p for p in result["tables"] if p.endswith(".csv")]
            assert len(csv_tables) >= 1

            with open(csv_tables[0]) as f:
                content = f.read()
                assert "Fold" in content
                assert "Subject" in content
                assert "Accuracy" in content

    def test_markdown_table_content(self, synthetic_baseline_run: Path) -> None:
        """Test markdown table has correct content."""
        from opencr.report.annexA import generate_annex_a

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = generate_annex_a(synthetic_baseline_run, output_dir)

            # Check MD table
            md_tables = [p for p in result["tables"] if p.endswith(".md")]
            assert len(md_tables) >= 1

            with open(md_tables[0]) as f:
                content = f.read()
                assert "Annex A" in content
                assert "|" in content  # Table formatting

    def test_multiclass_auc_range(self) -> None:
        """Test multiclass AUC computation returns a valid value."""
        from opencr.report.annexA import _compute_auc

        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_proba = np.random.rand(6, 3)
        y_proba = y_proba / y_proba.sum(axis=1, keepdims=True)

        auc_val, roc_inputs = _compute_auc(y_true, y_proba)

        assert auc_val is not None
        assert 0.0 <= auc_val <= 1.0
        assert roc_inputs is None

    def test_regression_figures(self, synthetic_baseline_run_regression: Path) -> None:
        """Test regression report generates scatter and error figures."""
        from opencr.report.annexA import generate_annex_a

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            result = generate_annex_a(synthetic_baseline_run_regression, output_dir)

            figure_names = {Path(p).name for p in result["figures"]}
            assert "opencr_scatter.png" in figure_names
            assert "opencr_error.png" in figure_names


class TestReportCLI:
    """Tests for report CLI commands."""

    def test_report_annexa_help(self) -> None:
        """Test report annexA --help."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "report", "annexA", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "output" in result.stdout.lower()

    def test_report_annexa_runs(self, synthetic_baseline_run: Path) -> None:
        """Test report annexA command runs successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "opencr",
                    "report",
                    "annexA",
                    str(synthetic_baseline_run),
                    "--output",
                    str(output_dir),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0

            # Check outputs exist
            figures_dir = output_dir / "figures" / "annexA"
            tables_dir = output_dir / "tables" / "annexA"

            assert figures_dir.exists()
            assert tables_dir.exists()

            # Check at least some files
            assert len(list(figures_dir.glob("*.png"))) >= 2
            assert len(list(tables_dir.glob("*"))) >= 1

    def test_report_metrics_runs(self, synthetic_baseline_run: Path) -> None:
        """Test report metrics command runs successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "metrics.json"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "opencr",
                    "report",
                    "metrics",
                    str(synthetic_baseline_run),
                    "--output",
                    str(output_path),
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert output_path.exists()
            assert (Path(tmpdir) / "summary.csv").exists()
            assert (Path(tmpdir) / "summary.md").exists()
