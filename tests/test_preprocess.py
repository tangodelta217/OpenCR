"""
Tests for preprocessing pipeline.

Creates synthetic signals and validates preprocessing functionality.
"""

import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

from opencr.data.adapters.base import SubjectData
from opencr.preprocess.filters import bandpass_filter, highpass_filter, lowpass_filter
from opencr.preprocess.pipeline import PreprocessingConfig, PreprocessingPipeline
from opencr.preprocess.sqi import SQIConfig, compute_sqi, compute_sqi_batch
from opencr.preprocess.windowing import WindowConfig, segment_windows


@pytest.fixture
def synthetic_signal() -> tuple[np.ndarray, float]:
    """Create a synthetic signal for testing."""
    fs = 100.0  # 100 Hz
    duration = 60  # 60 seconds
    t = np.arange(int(fs * duration)) / fs

    # Combination of frequencies: 1 Hz (target) + 10 Hz (noise)
    signal = np.sin(2 * np.pi * 1.0 * t) + 0.3 * np.sin(2 * np.pi * 10.0 * t)
    signal += np.random.normal(0, 0.1, len(t))

    return signal.astype(np.float64), fs


@pytest.fixture
def synthetic_dataset(tmp_path: Path) -> Path:
    """Create a synthetic dataset for pipeline testing."""
    fs_ppg = 100.0
    fs_bioz = 50.0
    duration = 120  # 2 minutes to have enough windows

    for subject_id in ["subj001", "subj002"]:
        t_ppg = np.arange(int(fs_ppg * duration)) / fs_ppg

        # PPG: heart rate ~1.2 Hz
        ppg = np.sin(2 * np.pi * 1.2 * t_ppg) + np.random.normal(0, 0.1, len(t_ppg))

        # BioZ: breathing rate ~0.3 Hz
        t_bioz = np.arange(int(fs_bioz * duration)) / fs_bioz
        bioz = np.sin(2 * np.pi * 0.3 * t_bioz) + np.random.normal(0, 0.05, len(t_bioz))

        # Protocol steps
        step = np.repeat([1, 2, 3, 4], len(t_ppg) // 4)
        if len(step) < len(t_ppg):
            step = np.concatenate([step, np.full(len(t_ppg) - len(step), 4)])

        np.savez(
            tmp_path / f"{subject_id}.npz",
            ppg=ppg.astype(np.float64),
            bioz=bioz.astype(np.float64),
            fs_ppg=fs_ppg,
            fs_bioz=fs_bioz,
            step=step,
        )

    return tmp_path


class TestFilters:
    """Tests for signal filters."""

    def test_bandpass_filter(self, synthetic_signal: tuple[np.ndarray, float]) -> None:
        """Test bandpass filter removes out-of-band noise."""
        signal, fs = synthetic_signal

        # Filter to keep 0.5-2 Hz (should keep 1 Hz, remove 10 Hz)
        filtered = bandpass_filter(signal, fs, 0.5, 2.0)

        assert filtered.shape == signal.shape
        assert np.std(filtered) < np.std(signal)  # Less noisy

    def test_highpass_filter(self, synthetic_signal: tuple[np.ndarray, float]) -> None:
        """Test highpass filter."""
        signal, fs = synthetic_signal
        filtered = highpass_filter(signal, fs, 0.1)

        assert filtered.shape == signal.shape

    def test_lowpass_filter(self, synthetic_signal: tuple[np.ndarray, float]) -> None:
        """Test lowpass filter."""
        signal, fs = synthetic_signal
        filtered = lowpass_filter(signal, fs, 5.0)

        assert filtered.shape == signal.shape

    def test_filter_invalid_params(self, synthetic_signal: tuple[np.ndarray, float]) -> None:
        """Test filter rejects invalid parameters."""
        signal, fs = synthetic_signal

        with pytest.raises(ValueError):
            bandpass_filter(signal, fs, -1, 10)  # Negative low

        with pytest.raises(ValueError):
            bandpass_filter(signal, fs, 1, 100)  # High > Nyquist

        with pytest.raises(ValueError):
            bandpass_filter(signal, fs, 10, 5)  # Low > High


class TestWindowing:
    """Tests for signal windowing."""

    def test_segment_windows_basic(self, synthetic_signal: tuple[np.ndarray, float]) -> None:
        """Test basic windowing."""
        signal, fs = synthetic_signal
        config = WindowConfig(window_sec=10.0, stride_sec=2.0)

        result, labels = segment_windows(signal, fs, config)

        # 60 seconds, 10s windows, 2s stride -> (60-10)/2 + 1 = 26 windows
        assert result.n_windows == 26
        assert result.window_samples == 1000  # 10s * 100 Hz
        assert result.windows.shape == (26, 1000)
        assert labels is None

    def test_segment_windows_with_labels(self) -> None:
        """Test windowing with label alignment."""
        fs = 100.0
        signal = np.random.randn(6000)  # 60 seconds
        labels = np.repeat([1, 2, 3], 2000)  # 3 labels, 20s each

        config = WindowConfig(window_sec=10.0, stride_sec=10.0)
        result, window_labels = segment_windows(signal, fs, config, labels)

        assert result.n_windows == 6
        assert window_labels is not None
        assert len(window_labels) == 6
        # First 2 windows should be label 1, next 2 label 2, etc.
        assert window_labels[0] == 1
        assert window_labels[2] == 2

    def test_window_too_short(self) -> None:
        """Test error when signal is too short."""
        signal = np.random.randn(100)
        config = WindowConfig(window_sec=10.0, stride_sec=2.0)

        with pytest.raises(ValueError, match="too short"):
            segment_windows(signal, 100.0, config)


class TestSQI:
    """Tests for SQI computation."""

    def test_compute_sqi_clean_signal(self) -> None:
        """Test SQI on clean signal."""
        fs = 100.0
        t = np.arange(1000) / fs
        signal = np.sin(2 * np.pi * 1.0 * t)  # Clean 1 Hz sine

        config = SQIConfig(energy_band_low=0.5, energy_band_high=2.0)
        result = compute_sqi(signal, fs, config)

        # Verify scores are in valid range
        assert 0 <= result.clip_score <= 1
        assert result.energy_score > 0.3  # Most energy in band
        assert result.combined_score > 0.2  # Overall quality (relaxed)

    def test_compute_sqi_clipped_signal(self) -> None:
        """Test SQI handles various signal types."""
        # Constant signal (edge case)
        signal = np.ones(1000)
        result = compute_sqi(signal, 100.0)
        assert 0 <= result.clip_score <= 1
        assert 0 <= result.combined_score <= 1

        # High frequency noise
        signal2 = np.random.randn(1000)
        result2 = compute_sqi(signal2, 100.0)
        assert 0 <= result2.clip_score <= 1
        assert 0 <= result2.combined_score <= 1

    def test_compute_sqi_batch(self) -> None:
        """Test batch SQI computation."""
        windows = np.random.randn(10, 1000)  # 10 windows
        sqi = compute_sqi_batch(windows, 100.0)

        assert sqi.shape == (10,)
        assert np.all((sqi >= 0) & (sqi <= 1))

    def test_compute_sqi_batch_multichannel(self) -> None:
        """Test batch SQI on multichannel data."""
        windows = np.random.randn(10, 2, 1000)  # 10 windows, 2 channels
        sqi = compute_sqi_batch(windows, 100.0)

        assert sqi.shape == (10, 2)

    def test_compute_sqi_bioz_respiration(self) -> None:
        """Test BioZ SQI accepts respiration-dominant signals."""
        fs = 100.0
        t = np.arange(3000) / fs
        signal = np.sin(2 * np.pi * 0.2 * t)  # 0.2 Hz respiration

        config = SQIConfig(bioz_resp_band_low=0.1, bioz_resp_band_high=0.7)
        result = compute_sqi(signal, fs, config, channel="bioz")

        assert result.energy_score > 0.2
        assert result.combined_score > 0.1


class TestPipeline:
    """Tests for preprocessing pipeline."""

    def test_pipeline_process_subject(self, synthetic_dataset: Path) -> None:
        """Test pipeline processes a subject correctly."""
        from opencr.data.adapters import LocalNpzAdapter

        adapter = LocalNpzAdapter(synthetic_dataset)
        config = PreprocessingConfig(window_sec=30.0, stride_sec=10.0)
        pipeline = PreprocessingPipeline(config)

        subject_data = adapter.load_subject("subj001")
        result = pipeline.process_subject(subject_data)

        # Check output shapes
        assert result.X.ndim == 3  # (n_windows, n_channels, window_samples)
        assert result.X.shape[1] == 2  # 2 channels (ppg, bioz)
        assert result.sqi.shape[0] == result.X.shape[0]
        assert result.valid_mask.shape[0] == result.X.shape[0]
        assert len(result.timestamps) == result.X.shape[0]

    def test_bioz_resp_preserved(self) -> None:
        """Test BioZ respiration content survives preprocessing filter."""
        fs = 100.0
        t = np.arange(int(fs * 60)) / fs
        bioz = np.sin(2 * np.pi * 0.2 * t)
        ppg = np.sin(2 * np.pi * 1.2 * t)

        subject = SubjectData(
            subject_id="s1",
            signals={"ppg": ppg, "bioz": bioz},
            sampling_rates={"ppg": fs, "bioz": fs},
        )

        config = PreprocessingConfig(window_sec=30.0, stride_sec=10.0)
        pipeline = PreprocessingPipeline(config)
        result = pipeline.process_subject(subject)

        bioz_window = result.X[0, 1]
        sqi_config = SQIConfig(bioz_resp_band_low=0.1, bioz_resp_band_high=0.7)
        sqi = compute_sqi(bioz_window, fs, sqi_config, channel="bioz")

        assert sqi.energy_score > 0.2

    def test_pipeline_process_dataset(self, synthetic_dataset: Path) -> None:
        """Test pipeline processes entire dataset."""
        from opencr.data.adapters import LocalNpzAdapter

        adapter = LocalNpzAdapter(synthetic_dataset)
        config = PreprocessingConfig(window_sec=30.0, stride_sec=10.0)
        pipeline = PreprocessingPipeline(config)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)
            stats = pipeline.process_dataset(adapter, output_dir)

            assert stats["processed"] == 2
            assert stats["failed"] == 0
            assert stats["total_windows"] > 0

            # Check output files
            assert (output_dir / "config.json").exists()
            assert (output_dir / "stats.json").exists()
            assert (output_dir / "processed" / "subj001.npz").exists()
            assert (output_dir / "processed" / "subj002.npz").exists()

            # Load and verify processed data
            with np.load(output_dir / "processed" / "subj001.npz", allow_pickle=True) as data:
                assert "X" in data.files
                assert "y_step" in data.files
                assert "y_opencr" in data.files
                assert "y_ord" in data.files
                assert "sqi" in data.files
                assert "valid_mask" in data.files
                assert data["y_opencr"].shape == data["y_step"].shape
                assert data["y_ord"].shape == data["y_step"].shape


class TestPreprocessCLI:
    """Tests for CLI preprocess command."""

    def test_preprocess_help(self) -> None:
        """Test preprocess --help."""
        result = subprocess.run(
            [sys.executable, "-m", "opencr", "data", "preprocess", "--help"],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        assert "window-sec" in result.stdout
        assert "stride-sec" in result.stdout
        assert "sqi-threshold" in result.stdout

    def test_preprocess_runs(self, synthetic_dataset: Path) -> None:
        """Test preprocess command runs successfully."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "output"

            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "opencr",
                    "data",
                    "preprocess",
                    str(synthetic_dataset),
                    "--output",
                    str(output_dir),
                    "--window-sec",
                    "30",
                    "--stride-sec",
                    "10",
                ],
                capture_output=True,
                text=True,
            )

            assert result.returncode == 0
            assert (output_dir / "processed").exists()
            assert (output_dir / "config.json").exists()
            manifest_path = output_dir / "manifest.json"
            assert manifest_path.exists()
            with open(manifest_path, encoding="utf-8") as f:
                manifest = json.load(f)
            assert manifest["stage"] == "preprocess"
            assert "dataset_hash" in manifest
            assert "config" in manifest
