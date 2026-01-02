"""
Smoke tests for OpenCR package.

These tests verify that the package can be imported and the CLI is functional.
"""

import subprocess
import sys


class TestPackageImport:
    """Test that the package can be imported correctly."""

    def test_import_opencr(self) -> None:
        """Test that opencr package can be imported."""
        import opencr

        assert opencr.__version__ is not None
        assert opencr.__version__ == "0.1.0"

    def test_import_cli(self) -> None:
        """Test that CLI module can be imported."""
        from opencr.cli import app

        assert app is not None

    def test_import_logging(self) -> None:
        """Test that logging module can be imported."""
        from opencr.logging import get_logger, setup_logging

        assert get_logger is not None
        assert setup_logging is not None

    def test_get_logger(self) -> None:
        """Test that get_logger returns a logger instance."""
        from opencr.logging import get_logger

        logger = get_logger("test")
        assert logger is not None
        assert logger.name == "test"


class TestCLIMain:
    """Test main CLI functionality."""

    def test_cli_help(self) -> None:
        """Test that 'opencr --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "OpenCR" in result.stdout or "opencr" in result.stdout.lower()

    def test_cli_version(self) -> None:
        """Test that 'opencr --version' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "--version"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "0.1.0" in result.stdout


class TestCLIDataCommands:
    """Test data subcommands."""

    def test_data_help(self) -> None:
        """Test that 'opencr data --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "data", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "fetch" in result.stdout.lower()
        assert "preprocess" in result.stdout.lower()

    def test_data_fetch_help(self) -> None:
        """Test that 'opencr data fetch --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "data", "fetch", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "dataset" in result.stdout.lower()

    def test_data_preprocess_help(self) -> None:
        """Test that 'opencr data preprocess --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "data", "preprocess", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "anti-leakage" in result.stdout.lower() or "subject" in result.stdout.lower()


class TestCLIBaselineCommands:
    """Test baseline subcommands."""

    def test_baseline_help(self) -> None:
        """Test that 'opencr baseline --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "baseline", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "train" in result.stdout.lower()
        assert "evaluate" in result.stdout.lower()

    def test_baseline_train_help(self) -> None:
        """Test that 'opencr baseline train --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "baseline", "train", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "cv" in result.stdout.lower() or "loso" in result.stdout.lower()
        assert "model" in result.stdout.lower()

    def test_baseline_evaluate_help(self) -> None:
        """Test that 'opencr baseline evaluate --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "baseline", "evaluate", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "run" in result.stdout.lower() or "output" in result.stdout.lower()


class TestCLIReportCommands:
    """Test report subcommands."""

    def test_report_help(self) -> None:
        """Test that 'opencr report --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "report", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "annexa" in result.stdout.lower() or "annex" in result.stdout.lower()

    def test_report_annexa_help(self) -> None:
        """Test that 'opencr report annexA --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "report", "annexA", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "output" in result.stdout.lower() or "run" in result.stdout.lower()

    def test_report_metrics_help(self) -> None:
        """Test that 'opencr report metrics --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "report", "metrics", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "output" in result.stdout.lower()


class TestCLIEdgeCommands:
    """Test edge subcommands."""

    def test_edge_help(self) -> None:
        """Test that 'opencr edge --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "edge", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "export" in result.stdout.lower()
        assert "benchmark" in result.stdout.lower()

    def test_edge_export_help(self) -> None:
        """Test that 'opencr edge export --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "edge", "export", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "run" in result.stdout.lower()
        assert "format" in result.stdout.lower()

    def test_edge_benchmark_help(self) -> None:
        """Test that 'opencr edge benchmark --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "edge", "benchmark", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "iterations" in result.stdout.lower()


class TestCLIDemoCommands:
    """Test demo subcommands."""

    def test_demo_help(self) -> None:
        """Test that 'opencr demo --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "demo", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "run" in result.stdout.lower()
        assert "fuel" in result.stdout.lower()

    def test_demo_run_help(self) -> None:
        """Test that 'opencr demo run --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "demo", "run", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "subject" in result.stdout.lower() or "delay" in result.stdout.lower()

    def test_demo_fuel_gauge_help(self) -> None:
        """Test that 'opencr demo fuel-gauge --help' works."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "demo", "fuel-gauge", "--help"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        assert "threshold" in result.stdout.lower()
