"""
Preprocessing Pipeline Orchestrator.

Coordinates all preprocessing steps: filtering, resampling, windowing, SQI.
"""

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from opencr.data.adapters.base import DatasetAdapter, SubjectData
from opencr.logging import get_logger
from opencr.preprocess.filters import bandpass_filter, resample_signal
from opencr.preprocess.sqi import SQIConfig, compute_sqi_batch
from opencr.preprocess.windowing import WindowConfig, WindowResult, segment_windows

logger = get_logger(__name__)


@dataclass
class PreprocessingConfig:
    """Complete preprocessing configuration."""

    # Windowing
    window_sec: float = 30.0  # Window size in seconds
    stride_sec: float = 2.0  # Stride in seconds

    # Resampling (None = keep original)
    resample_hz: float | None = None

    # Filtering (None = no filtering)
    filter_low_hz: float | None = 0.5
    filter_high_hz: float | None = 4.0
    filter_order: int = 4

    # SQI
    sqi_threshold: float = 0.5  # Quality gating threshold
    sqi_config: SQIConfig = field(default_factory=SQIConfig)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        result["sqi_config"] = asdict(self.sqi_config)
        return result

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "PreprocessingConfig":
        """Create from dictionary."""
        if "sqi_config" in data and isinstance(data["sqi_config"], dict):
            data["sqi_config"] = SQIConfig(**data["sqi_config"])
        return cls(**data)


@dataclass
class ProcessedSubject:
    """Processed data for a single subject."""

    subject_id: str
    X: NDArray[np.floating]  # (n_windows, n_channels, window_samples)
    y: NDArray | None  # Labels per window (if available)
    sqi: NDArray[np.floating]  # (n_windows, n_channels)
    valid_mask: NDArray[np.bool_]  # Windows passing SQI threshold
    timestamps: NDArray[np.floating]  # Start time of each window
    metadata: dict[str, Any]


class PreprocessingPipeline:
    """
    Pipeline for preprocessing physiological signals.

    Coordinates:
    1. Optional resampling to common rate
    2. Optional bandpass filtering
    3. Sliding window segmentation
    4. SQI computation
    5. Quality gating
    """

    def __init__(self, config: PreprocessingConfig) -> None:
        """
        Initialize pipeline with configuration.

        Args:
            config: Preprocessing configuration.
        """
        self.config = config
        self._window_config = WindowConfig(
            window_sec=config.window_sec,
            stride_sec=config.stride_sec,
        )

    def _preprocess_signal(
        self,
        signal: NDArray[np.floating],
        fs: float,
    ) -> tuple[NDArray[np.floating], float]:
        """
        Apply resampling and filtering to a single signal.

        Returns:
            Tuple of (processed_signal, effective_fs).
        """
        effective_fs = fs

        # Resample if requested
        if self.config.resample_hz is not None:
            signal = resample_signal(signal, fs, self.config.resample_hz)
            effective_fs = self.config.resample_hz

        # Apply bandpass filter if configured
        if self.config.filter_low_hz is not None and self.config.filter_high_hz is not None:
            signal = bandpass_filter(
                signal,
                effective_fs,
                self.config.filter_low_hz,
                self.config.filter_high_hz,
                self.config.filter_order,
            )

        return signal, effective_fs

    def process_subject(self, subject_data: SubjectData) -> ProcessedSubject:
        """
        Process a single subject's data.

        Args:
            subject_data: Raw data from adapter.

        Returns:
            ProcessedSubject with segmented windows and SQI.
        """
        logger.info(f"Processing subject: {subject_data.subject_id}")

        # Determine target sampling rate
        if self.config.resample_hz is not None:
            target_fs = self.config.resample_hz
        else:
            # Use PPG sampling rate as reference
            target_fs = subject_data.sampling_rates.get("ppg", 100.0)

        # Process each signal channel
        processed_signals: dict[str, NDArray[np.floating]] = {}
        for signal_name, signal in subject_data.signals.items():
            original_fs = subject_data.sampling_rates[signal_name]
            processed, _ = self._preprocess_signal(signal, original_fs)

            # Ensure same length by resampling to target_fs if not already
            if self.config.resample_hz is None and original_fs != target_fs:
                processed = resample_signal(processed, original_fs, target_fs)

            processed_signals[signal_name] = processed

        # Ensure all signals have same length (use shortest)
        min_len = min(len(s) for s in processed_signals.values())
        for name in processed_signals:
            processed_signals[name] = processed_signals[name][:min_len]

        # Get labels if available
        labels = None
        if subject_data.protocol_levels is not None:
            labels = subject_data.protocol_levels
            # Resample labels to match signal length
            if len(labels) != min_len:
                # Simple nearest-neighbor resampling for labels
                indices = np.linspace(0, len(labels) - 1, min_len).astype(int)
                labels = labels[indices]

        # Segment each channel
        channel_windows: dict[str, WindowResult] = {}
        window_labels = None

        for signal_name, signal in processed_signals.items():
            result, w_labels = segment_windows(signal, target_fs, self._window_config, labels)
            channel_windows[signal_name] = result
            if w_labels is not None:
                window_labels = w_labels

        # Stack channels into (n_windows, n_channels, window_samples)
        channel_names = list(channel_windows.keys())
        first_result = channel_windows[channel_names[0]]
        n_windows = first_result.n_windows
        window_samples = first_result.window_samples
        n_channels = len(channel_names)

        X = np.zeros((n_windows, n_channels, window_samples), dtype=np.float64)
        for c, ch_name in enumerate(channel_names):
            X[:, c, :] = channel_windows[ch_name].windows

        # Compute SQI per channel
        sqi = compute_sqi_batch(X, target_fs, self.config.sqi_config)

        # Quality gating: window is valid if min(sqi across channels) >= threshold
        min_sqi_per_window = sqi.min(axis=1) if sqi.ndim > 1 else sqi
        valid_mask = min_sqi_per_window >= self.config.sqi_threshold

        logger.info(
            f"  Windows: {n_windows}, Valid: {valid_mask.sum()} ({valid_mask.mean()*100:.1f}%)"
        )

        return ProcessedSubject(
            subject_id=subject_data.subject_id,
            X=X,
            y=window_labels,
            sqi=sqi,
            valid_mask=valid_mask,
            timestamps=first_result.timestamps,
            metadata={
                "channels": channel_names,
                "fs": target_fs,
                "window_sec": self.config.window_sec,
                "stride_sec": self.config.stride_sec,
                "n_windows": n_windows,
                "n_valid": int(valid_mask.sum()),
                "sqi_threshold": self.config.sqi_threshold,
            },
        )

    def process_dataset(
        self,
        adapter: DatasetAdapter,
        output_dir: Path,
    ) -> dict[str, Any]:
        """
        Process entire dataset and save to output directory.

        Args:
            adapter: Dataset adapter.
            output_dir: Output directory for processed files.

        Returns:
            Summary statistics dictionary.
        """
        output_dir = Path(output_dir)
        processed_dir = output_dir / "processed"
        processed_dir.mkdir(parents=True, exist_ok=True)

        subjects = adapter.list_subjects()
        logger.info(f"Processing {len(subjects)} subjects...")

        stats = {
            "total_subjects": len(subjects),
            "processed": 0,
            "failed": 0,
            "total_windows": 0,
            "valid_windows": 0,
            "subjects_processed": [],
            "subjects_failed": [],
        }

        for subject_id in subjects:
            try:
                subject_data = adapter.load_subject(subject_id)
                processed = self.process_subject(subject_data)

                # Save processed data
                output_path = processed_dir / f"{subject_id}.npz"
                np.savez(
                    output_path,
                    X=processed.X,
                    y=processed.y if processed.y is not None else np.array([]),
                    sqi=processed.sqi,
                    valid_mask=processed.valid_mask,
                    timestamps=processed.timestamps,
                    metadata=processed.metadata,
                )

                stats["processed"] += 1
                stats["total_windows"] += processed.X.shape[0]
                stats["valid_windows"] += int(processed.valid_mask.sum())
                stats["subjects_processed"].append(subject_id)

            except Exception as e:
                logger.error(f"Failed to process {subject_id}: {e}")
                stats["failed"] += 1
                stats["subjects_failed"].append({"subject_id": subject_id, "error": str(e)})

        # Save configuration and stats
        config_path = output_dir / "config.json"
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(self.config.to_dict(), f, indent=2)

        stats_path = output_dir / "stats.json"
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            f"Processing complete: {stats['processed']}/{stats['total_subjects']} subjects, "
            f"{stats['valid_windows']}/{stats['total_windows']} valid windows "
            f"({stats['valid_windows']/max(stats['total_windows'],1)*100:.1f}%)"
        )

        return stats
