"""
Tests for edge export module.

Tests artifact generation and budget estimation.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from opencr.edge.export import EdgeConfig, export_edge_model
from opencr.models.baseline import BaselineModel, ModelConfig, ModelType, TaskType


@pytest.fixture
def synthetic_baseline_run_for_edge(tmp_path: Path) -> Path:
    """Create a synthetic baseline run with a real trained model."""
    run_dir = tmp_path / "baseline_run_edge"
    run_dir.mkdir()

    # Create a small trained model
    X = [[0, 0], [1, 1]]
    y = [0, 1]
    config = ModelConfig(
        task_type=TaskType.CLASSIFICATION,
        model_type=ModelType.RANDOM_FOREST,
        n_estimators=5,
        random_state=42,
    )
    clf = BaselineModel(config)
    clf.fit(np.array(X), np.array(y), feature_names=["f0", "f1"])

    # Save model in fold_00
    fold_dir = run_dir / "fold_00"
    fold_dir.mkdir()
    clf.save(fold_dir / "model.joblib")

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

    def test_budget_flash_exceeds(self, synthetic_baseline_run_for_edge: Path) -> None:
        """Flash budget should fail when artifacts exceed target."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "edge_out"

            config = EdgeConfig(
                target_format="pickle",
                target_latency_ms=1000.0,
                target_flash_kb=0.001,
            )
            result = export_edge_model(synthetic_baseline_run_for_edge, output_dir, config)
            budget = result["budget"]

            assert budget["flash"]["status"] == "fail"
            assert budget["meets_budget"] is False

    def test_budget_ram_unknown(self, synthetic_baseline_run_for_edge: Path) -> None:
        """RAM unknown should force overall budget to unknown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "edge_out"

            config = EdgeConfig(
                target_format="pickle",
                target_latency_ms=1000.0,
                target_flash_kb=100000.0,
            )
            result = export_edge_model(synthetic_baseline_run_for_edge, output_dir, config)
            budget = result["budget"]

            assert budget["ram"]["status"] == "unknown"
            assert budget["meets_budget"] is None

    def test_export_auto_selection(self, synthetic_baseline_run_for_edge: Path) -> None:
        """Test auto format selection."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "edge_auto"

            result = export_edge_model(
                synthetic_baseline_run_for_edge, output_dir, EdgeConfig(target_format="auto")
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
                sys.executable,
                "-m",
                "opencr",
                "edge",
                "export",
                "--run",
                str(synthetic_baseline_run_for_edge),
                "--out",
                str(output_dir),
                "--format",
                "pickle",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert (
                result.returncode == 0
            ), f"Command failed: {result.stderr}\\nOutput: {result.stdout}"
            assert "Export Complete" in result.stdout
            assert (output_dir / "edge_budget.json").exists()

    def test_cli_benchmark_command(self, synthetic_baseline_run_for_edge: Path) -> None:
        """Test the CLI command for edge benchmark."""
        model_path = synthetic_baseline_run_for_edge / "fold_00" / "model.joblib"
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "benchmark.json"

            cmd = [
                sys.executable,
                "-m",
                "opencr",
                "edge",
                "benchmark",
                str(model_path),
                "--output",
                str(output_path),
                "--iterations",
                "5",
                "--warmup",
                "1",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert (
                result.returncode == 0
            ), f"Command failed: {result.stderr}\\nOutput: {result.stdout}"
            assert output_path.exists()
            assert (Path(tmpdir) / "benchmark_summary.csv").exists()
            assert (Path(tmpdir) / "benchmark_summary.md").exists()

            with open(output_path, encoding="utf-8") as f:
                benchmark = json.load(f)
            assert benchmark["inference"]["status"] in {"ok", "skipped"}

    def test_cli_export_quantize_errors(self, synthetic_baseline_run_for_edge: Path) -> None:
        """Quantize flag should error for baseline sklearn exports."""
        model_run = synthetic_baseline_run_for_edge
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "cli_quantize"

            cmd = [
                sys.executable,
                "-m",
                "opencr",
                "edge",
                "export",
                "--run",
                str(model_run),
                "--out",
                str(output_dir),
                "--quantize",
            ]

            result = subprocess.run(cmd, capture_output=True, text=True)

            assert result.returncode != 0
            assert "Quantization" in result.stdout or "Quantization" in result.stderr
