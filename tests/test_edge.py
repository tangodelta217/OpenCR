"""
Tests for edge export module.

Tests artifact generation and budget estimation.
"""

import json
import sys
import tempfile
from pathlib import Path

import joblib
import pytest
from sklearn.ensemble import RandomForestClassifier

from opencr.edge.export import EdgeConfig, export_edge_model


@pytest.fixture
def synthetic_baseline_run_for_edge(tmp_path: Path) -> Path:
    """Create a synthetic baseline run with a real trained model."""
    run_dir = tmp_path / "baseline_run_edge"
    run_dir.mkdir()

    # Create a small trained model
    clf = RandomForestClassifier(n_estimators=5, random_state=42)
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    clf.fit(X, y)

    # Save model in fold_00
    fold_dir = run_dir / "fold_00"
    fold_dir.mkdir()
    joblib.dump(clf, fold_dir / "model.joblib")

    # Create results.json
    results = {
        "model_type": "random_forest",
        "cv_strategy": "loso",
        "n_features": 2,
        "n_subjects": 1,
        "folds": [{"fold_idx": 0, "test_subject": "S01", "metrics": {}}],
    }

    with open(run_dir / "results.json", "w") as f:
        json.dump(results, f)

    return run_dir


class TestEdgeExport:
    """Tests for edge export functionality."""

    def test_export_pickle_stub(self, synthetic_baseline_run_for_edge: Path) -> None:
        """Test exporting as pickle stub (fallback)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "edge_out"

            # Force pickle format
            config = EdgeConfig(target_format="pickle", target_latency_ms=50.0)

            result = export_edge_model(synthetic_baseline_run_for_edge, output_dir, config)

            # Check files
            assert (output_dir / "model_stub.joblib").exists()
            assert (output_dir / "edge_budget.json").exists()
            assert (output_dir / "export_manifest.json").exists()
            assert (output_dir / "tflite_future" / "README.md").exists()

            # Check contents
            budget = result["budget"]
            assert budget["format"] == "pickle_stub"
            assert budget["model_size"]["bytes"] > 0
            assert budget["latency"]["target_ms"] == 50.0

            # Check logic
            assert "plan_b" in budget

    def test_export_auto_selection(self, synthetic_baseline_run_for_edge: Path) -> None:
        """Test auto format selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "edge_auto"

            result = export_edge_model(
                synthetic_baseline_run_for_edge,
                output_dir,
                EdgeConfig(target_format="auto")
            )

            # Should be pickle unless skl2onnx is installed in the test env
            # We check that it ran successfully regardless of result format
            assert result["format"] in ["onnx", "pickle_stub"]
            assert Path(result["edge_model_path"]).exists()

    def test_cli_export_command(self, synthetic_baseline_run_for_edge: Path) -> None:
        """Test the CLI command for edge export."""
        import subprocess

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "cli_out"

            cmd = [
                sys.executable, "-m", "opencr", "edge", "export",
                "--run", str(synthetic_baseline_run_for_edge),
                "--out", str(output_dir),
                "--format", "pickle"
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode == 0, f"Command failed: {result.stderr}\\nOutput: {result.stdout}"
            assert "Export Complete" in result.stdout
            assert (output_dir / "edge_budget.json").exists()

