"""
Tests for dataset adapters.

Creates synthetic .npz files and validates adapter functionality.
"""

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from opencr.data.adapters import LocalNpzAdapter
from opencr.data.adapters.base import DataCard, SubjectData


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a synthetic dataset with two subjects."""
    # Parameters
    fs_ppg = 100.0
    fs_bioz = 50.0
    duration = 10  # seconds

    for subject_id in ["subject001", "subject002"]:
        # Generate synthetic signals
        t_ppg = np.arange(int(fs_ppg * duration)) / fs_ppg
        t_bioz = np.arange(int(fs_bioz * duration)) / fs_bioz

        ppg = np.sin(2 * np.pi * 1.2 * t_ppg) + np.random.normal(0, 0.1, len(t_ppg))
        bioz = np.sin(2 * np.pi * 0.3 * t_bioz) + np.random.normal(0, 0.05, len(t_bioz))

        # Protocol steps (4 levels)
        step = np.repeat([1, 2, 3, 4], len(t_ppg) // 4)
        if len(step) < len(t_ppg):
            step = np.concatenate([step, np.full(len(t_ppg) - len(step), 4)])

        # Save .npz file
        np.savez(
            tmp_path / f"{subject_id}.npz",
            ppg=ppg,
            bioz=bioz,
            fs_ppg=fs_ppg,
            fs_bioz=fs_bioz,
            t=t_ppg,
            step=step,
            metadata={"subject_id": subject_id, "age": 25},
        )

    return tmp_path


@pytest.fixture
def minimal_dataset(tmp_path: Path) -> Path:
    """Create a minimal dataset with required fields only."""
    fs_ppg = 100.0
    fs_bioz = 50.0
    duration = 5

    ppg = np.random.randn(int(fs_ppg * duration))
    bioz = np.random.randn(int(fs_bioz * duration))

    np.savez(
        tmp_path / "subject001.npz",
        ppg=ppg,
        bioz=bioz,
        fs_ppg=fs_ppg,
        fs_bioz=fs_bioz,
    )

    return tmp_path


@pytest.fixture
def mixed_steps_dataset(tmp_path: Path) -> Path:
    """Create a dataset with mixed protocol level availability."""
    fs_ppg = 100.0
    fs_bioz = 50.0
    duration = 10  # seconds

    for subject_id, include_steps in [
        ("subject001", True),
        ("subject002", True),
        ("subject003", False),
    ]:
        t_ppg = np.arange(int(fs_ppg * duration)) / fs_ppg
        t_bioz = np.arange(int(fs_bioz * duration)) / fs_bioz

        ppg = np.sin(2 * np.pi * 1.2 * t_ppg) + np.random.normal(0, 0.1, len(t_ppg))
        bioz = np.sin(2 * np.pi * 0.3 * t_bioz) + np.random.normal(0, 0.05, len(t_bioz))

        payload = {
            "ppg": ppg,
            "bioz": bioz,
            "fs_ppg": fs_ppg,
            "fs_bioz": fs_bioz,
            "t": t_ppg,
            "metadata": {"subject_id": subject_id, "age": 25},
        }

        if include_steps:
            step = np.repeat([1, 2, 3, 4], len(t_ppg) // 4)
            if len(step) < len(t_ppg):
                step = np.concatenate([step, np.full(len(t_ppg) - len(step), 4)])
            payload["step"] = step

        np.savez(tmp_path / f"{subject_id}.npz", **payload)

    return tmp_path


class TestLocalNpzAdapter:
    """Tests for LocalNpzAdapter."""

    def test_list_subjects(self, synthetic_dataset: Path) -> None:
        """Test listing subjects."""
        adapter = LocalNpzAdapter(synthetic_dataset)
        subjects = adapter.list_subjects()

        assert len(subjects) == 2
        assert "subject001" in subjects
        assert "subject002" in subjects

    def test_load_subject(self, synthetic_dataset: Path) -> None:
        """Test loading a subject."""
        adapter = LocalNpzAdapter(synthetic_dataset)
        data = adapter.load_subject("subject001")

        assert isinstance(data, SubjectData)
        assert data.subject_id == "subject001"
        assert "ppg" in data.signals
        assert "bioz" in data.signals
        assert data.sampling_rates["ppg"] == 100.0
        assert data.sampling_rates["bioz"] == 50.0
        assert data.protocol_levels is not None
        assert len(data.protocol_levels) == 1000  # 10s * 100Hz

    def test_load_subject_single_channel_2d(self, tmp_path: Path) -> None:
        """Accept single-channel signals with shape (1, N)."""
        fs = 100.0
        n_samples = 200
        ppg = np.random.randn(1, n_samples)
        bioz = np.random.randn(1, n_samples)

        np.savez(
            tmp_path / "subject001.npz",
            ppg=ppg,
            bioz=bioz,
            fs_ppg=fs,
            fs_bioz=fs,
        )

        adapter = LocalNpzAdapter(tmp_path)
        data = adapter.load_subject("subject001")

        assert data.signals["ppg"].shape == (n_samples,)
        assert data.signals["bioz"].shape == (n_samples,)

    def test_load_subject_not_found(self, synthetic_dataset: Path) -> None:
        """Test loading a non-existent subject."""
        adapter = LocalNpzAdapter(synthetic_dataset)

        with pytest.raises(KeyError, match="not found"):
            adapter.load_subject("nonexistent")

    def test_protocol_levels(self, synthetic_dataset: Path) -> None:
        """Test getting protocol levels."""
        adapter = LocalNpzAdapter(synthetic_dataset)
        levels = adapter.protocol_levels("subject001")

        assert levels is not None
        assert len(np.unique(levels)) == 4  # 4 protocol levels

    def test_describe(self, synthetic_dataset: Path) -> None:
        """Test generating data card."""
        adapter = LocalNpzAdapter(synthetic_dataset)
        card = adapter.describe()

        assert isinstance(card, DataCard)
        assert card.num_subjects == 2
        assert card.adapter_type == "LocalNpzAdapter"
        assert "ppg" in card.signals
        assert "bioz" in card.signals
        assert card.has_protocol_levels is True
        assert set(card.subjects_with_steps) == {"subject001", "subject002"}
        assert card.subjects_missing_steps == []

    def test_describe_mixed_steps(self, mixed_steps_dataset: Path) -> None:
        """Test data card captures mixed protocol availability."""
        adapter = LocalNpzAdapter(mixed_steps_dataset)
        card = adapter.describe()

        assert card.num_subjects == 3
        assert card.has_protocol_levels is True
        assert set(card.subjects_with_steps) == {"subject001", "subject002"}
        assert card.subjects_missing_steps == ["subject003"]

    def test_validate(self, synthetic_dataset: Path) -> None:
        """Test dataset validation."""
        adapter = LocalNpzAdapter(synthetic_dataset)
        is_valid, errors = adapter.validate()

        assert is_valid is True
        assert len(errors) == 0

    def test_minimal_dataset(self, minimal_dataset: Path) -> None:
        """Test loading dataset with only required fields."""
        adapter = LocalNpzAdapter(minimal_dataset)
        data = adapter.load_subject("subject001")

        assert data.protocol_levels is None
        assert data.timestamps is None

    def test_to_dict(self, synthetic_dataset: Path) -> None:
        """Test DataCard serialization."""
        adapter = LocalNpzAdapter(synthetic_dataset)
        card = adapter.describe()
        card_dict = card.to_dict()

        assert isinstance(card_dict, dict)
        assert card_dict["num_subjects"] == 2
        assert json.dumps(card_dict)  # Should be JSON serializable


class TestAdapterErrors:
    """Tests for error handling in adapters."""

    def test_missing_directory(self) -> None:
        """Test error when directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            LocalNpzAdapter(Path("/nonexistent/path"))

    def test_empty_directory(self, tmp_path: Path) -> None:
        """Test error when no .npz files exist."""
        with pytest.raises(ValueError, match="No .npz files found"):
            LocalNpzAdapter(tmp_path)

    def test_missing_required_fields(self, tmp_path: Path) -> None:
        """Test error when required fields are missing."""
        # Create .npz with missing fields
        np.savez(tmp_path / "subject001.npz", ppg=np.zeros(100))

        adapter = LocalNpzAdapter(tmp_path)
        with pytest.raises(ValueError, match="missing required fields"):
            adapter.load_subject("subject001")

    def test_multichannel_rejected(self, tmp_path: Path) -> None:
        """Test error when signals have multiple channels."""
        fs = 100.0
        n_samples = 200
        ppg = np.random.randn(2, n_samples)
        bioz = np.random.randn(n_samples)

        np.savez(
            tmp_path / "subject001.npz",
            ppg=ppg,
            bioz=bioz,
            fs_ppg=fs,
            fs_bioz=fs,
        )

        adapter = LocalNpzAdapter(tmp_path)
        with pytest.raises(ValueError, match="Multichannel not supported"):
            adapter.load_subject("subject001")


class TestDataFetchCLI:
    """Tests for the CLI data fetch command."""

    def test_fetch_generates_data_card(self, synthetic_dataset: Path, tmp_path: Path) -> None:
        """Test that fetch generates data_card.json."""
        output_dir = tmp_path / "output"

        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "opencr",
                "data",
                "fetch",
                str(synthetic_dataset),
                "--output",
                str(output_dir),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert (output_dir / "data_card.json").exists()

        # Validate JSON content
        with open(output_dir / "data_card.json") as f:
            card = json.load(f)

        assert card["num_subjects"] == 2
        assert "ppg" in card["signals"]
        assert card["adapter_type"] == "LocalNpzAdapter"
        assert card["subjects_with_steps"] == ["subject001", "subject002"]
        assert card["subjects_missing_steps"] == []

    def test_fetch_invalid_directory(self, tmp_path: Path) -> None:
        """Test fetch with invalid directory."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "opencr",
                "data",
                "fetch",
                str(tmp_path / "nonexistent"),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
        assert "not found" in result.stdout.lower() or "error" in result.stdout.lower()

    def test_fetch_empty_directory(self, tmp_path: Path) -> None:
        """Test fetch with empty directory."""
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "opencr",
                "data",
                "fetch",
                str(tmp_path),
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 1
