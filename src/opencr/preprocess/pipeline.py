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
from opencr.preprocess.filters import (
    bandpass_filter,
    highpass_filter,
    lowpass_filter,
    resample_signal,
)
from opencr.preprocess.sqi import SQIConfig, compute_sqi_batch
from opencr.preprocess.windowing import WindowConfig, WindowResult, segment_windows
from opencr.targets.opencr import (
    apply_target_map_opencr,
    apply_target_map_ordinal,
    build_target_map,
    normalize_steps_to_opencr,
    steps_to_ordinal,
)

logger = get_logger(__name__)


_SINGLE_CHANNEL_ERROR = (
    "Multichannel not supported in v0.x. Select one channel or average before loading."
)


def _ensure_single_channel_signal(
    signal: NDArray[np.floating] | NDArray,
    *,
    name: str,
    subject_id: str,
) -> NDArray[np.floating]:
    arr = np.asarray(signal)
    if arr.ndim == 1:
        return arr.astype(np.float64, copy=False)
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0].astype(np.float64, copy=False)
    raise ValueError(
        f"{_SINGLE_CHANNEL_ERROR} Signal '{name}' in '{subject_id}' has shape {arr.shape}."
    )


def _resample_time_axis(
    t: NDArray[np.floating] | NDArray,
    target_len: int,
) -> NDArray[np.floating]:
    t_arr = np.asarray(t, dtype=np.float64)
    if t_arr.ndim != 1:
        raise ValueError(f"Timestamps must be 1D, got shape {t_arr.shape}")
    if t_arr.size == 0:
        raise ValueError("Timestamps array is empty")
    if target_len <= 0:
        raise ValueError(f"target_len must be positive, got {target_len}")
    if t_arr.size == target_len:
        return t_arr
    if t_arr.size < 2:
        raise ValueError("Timestamps array must have at least 2 values to resample")
    if np.any(np.diff(t_arr) < 0):
        raise ValueError("Timestamps must be non-decreasing")
    src_idx = np.linspace(0, t_arr.size - 1, num=target_len)
    return np.interp(src_idx, np.arange(t_arr.size), t_arr).astype(np.float64)


def _resample_steps_by_time(
    steps: NDArray,
    t_step: NDArray[np.floating],
    t_new: NDArray[np.floating],
) -> NDArray:
    if t_step.ndim != 1 or t_new.ndim != 1:
        raise ValueError("t_step and t_new must be 1D arrays")
    if steps.shape[0] != t_step.shape[0]:
        raise ValueError("step and t_step must have the same length")
    if t_step.size == 0:
        raise ValueError("t_step is empty")
    if np.any(np.diff(t_step) < 0):
        raise ValueError("t_step must be non-decreasing")

    idx = np.searchsorted(t_step, t_new, side="left")
    idx = np.clip(idx, 0, t_step.size - 1)
    prev_idx = np.clip(idx - 1, 0, t_step.size - 1)
    next_idx = idx
    use_prev = (t_new - t_step[prev_idx]) <= (t_step[next_idx] - t_new)
    nearest_idx = np.where(use_prev, prev_idx, next_idx)
    return steps[nearest_idx]


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
    bioz_filter_low_hz: float | None = 0.05
    bioz_filter_high_hz: float | None = None

    # SQI
    sqi_threshold: float = 0.5  # Quality gating threshold
    sqi_config: SQIConfig = field(default_factory=SQIConfig)

    # Targets
    target_direction: str = "auto"
    ordinal_bins: int = 4

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
    y: NDArray | None  # Legacy labels per window (step-based)
    y_step: NDArray | None  # Step labels per window
    y_opencr: NDArray[np.floating] | None  # OpenCR target per window
    y_ord: NDArray[np.int_] | None  # Ordinal target per window
    sqi: NDArray[np.floating]  # (n_windows, n_channels)
    valid_mask: NDArray[np.bool_]  # Windows passing SQI threshold
    valid_mask_min: NDArray[np.bool_]  # Min SQI across channels
    valid_mask_ppg: NDArray[np.bool_] | None  # PPG-only SQI mask
    valid_mask_bioz: NDArray[np.bool_] | None  # BioZ-only SQI mask
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
        filter_low_hz: float | None,
        filter_high_hz: float | None,
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
        if filter_low_hz is not None or filter_high_hz is not None:
            if filter_low_hz is None:
                signal = lowpass_filter(
                    signal, effective_fs, filter_high_hz, self.config.filter_order
                )
            elif filter_high_hz is None:
                signal = highpass_filter(
                    signal, effective_fs, filter_low_hz, self.config.filter_order
                )
            else:
                signal = bandpass_filter(
                    signal,
                    effective_fs,
                    filter_low_hz,
                    filter_high_hz,
                    self.config.filter_order,
                )

        return signal, effective_fs

    def process_subject(
        self,
        subject_data: SubjectData,
        *,
        target_map: dict[str, Any] | None = None,
    ) -> ProcessedSubject:
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
        signals_single_channel: dict[str, NDArray[np.floating]] = {}
        raw_lengths: dict[str, int] = {}
        for signal_name, signal in subject_data.signals.items():
            signal_arr = _ensure_single_channel_signal(
                signal, name=signal_name, subject_id=subject_data.subject_id
            )
            signals_single_channel[signal_name] = signal_arr
            raw_lengths[signal_name] = signal_arr.shape[0]

        processed_signals: dict[str, NDArray[np.floating]] = {}
        for signal_name, signal in signals_single_channel.items():
            original_fs = subject_data.sampling_rates[signal_name]
            name_lower = signal_name.lower()
            if name_lower == "bioz":
                low_hz = self.config.bioz_filter_low_hz
                high_hz = self.config.bioz_filter_high_hz
            else:
                low_hz = self.config.filter_low_hz
                high_hz = self.config.filter_high_hz

            processed, _ = self._preprocess_signal(signal, original_fs, low_hz, high_hz)

            # Ensure same length by resampling to target_fs if not already
            if self.config.resample_hz is None and original_fs != target_fs:
                processed = resample_signal(processed, original_fs, target_fs)

            processed_signals[signal_name] = processed

        # Ensure all signals have same length (use shortest)
        min_len = min(len(s) for s in processed_signals.values())
        for name in processed_signals:
            processed_signals[name] = processed_signals[name][:min_len]

        timestamps = subject_data.timestamps or {}
        t_signal = None
        if timestamps:
            for signal_name in signals_single_channel:
                if signal_name in timestamps:
                    t_candidate = np.asarray(timestamps[signal_name], dtype=np.float64)
                    if t_candidate.shape[0] != raw_lengths[signal_name]:
                        raise ValueError(
                            "Timestamps length does not match signal "
                            f"'{signal_name}' for subject '{subject_data.subject_id}'"
                        )
                    t_signal = t_candidate
                    break
            if t_signal is None and "t" in timestamps:
                t_candidate = np.asarray(timestamps["t"], dtype=np.float64)
                if t_candidate.shape[0] not in raw_lengths.values():
                    raise ValueError(
                        "Timestamps length does not match any signal "
                        f"for subject '{subject_data.subject_id}'"
                    )
                t_signal = t_candidate

        if t_signal is not None:
            t_new = _resample_time_axis(t_signal, min_len)
        else:
            t_new = np.arange(min_len, dtype=np.float64) / target_fs

        # Get labels if available
        labels = None
        if subject_data.protocol_levels is not None:
            labels = np.asarray(subject_data.protocol_levels)
            t_step = None
            if timestamps:
                if "t_step" in timestamps:
                    t_step = np.asarray(timestamps["t_step"], dtype=np.float64)
                elif "t" in timestamps:
                    t_step = np.asarray(timestamps["t"], dtype=np.float64)

            if t_step is not None:
                if t_step.shape[0] != labels.shape[0]:
                    raise ValueError(
                        "step and t_step length mismatch for subject "
                        f"'{subject_data.subject_id}'"
                    )
                if t_signal is None and t_step.size > 0:
                    t_new = t_new + float(t_step[0])
                labels = _resample_steps_by_time(labels, t_step, t_new)
            elif labels.shape[0] != min_len:
                raise ValueError(
                    "step length does not match signal length and no timestamps provided "
                    f"for subject '{subject_data.subject_id}'"
                )

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
        sqi = compute_sqi_batch(X, target_fs, self.config.sqi_config, channel_names=channel_names)

        # Quality gating masks
        threshold = self.config.sqi_threshold
        if sqi.ndim > 1:
            min_sqi_per_window = sqi.min(axis=1)
            valid_mask_min = min_sqi_per_window >= threshold
        else:
            min_sqi_per_window = sqi
            valid_mask_min = sqi >= threshold

        valid_mask_ppg = None
        valid_mask_bioz = None
        channel_names_lower = [name.lower() for name in channel_names]
        if sqi.ndim > 1:
            if "ppg" in channel_names_lower:
                idx = channel_names_lower.index("ppg")
                valid_mask_ppg = sqi[:, idx] >= threshold
            if "bioz" in channel_names_lower:
                idx = channel_names_lower.index("bioz")
                valid_mask_bioz = sqi[:, idx] >= threshold
        elif channel_names_lower:
            if channel_names_lower[0] == "ppg":
                valid_mask_ppg = valid_mask_min
            elif channel_names_lower[0] == "bioz":
                valid_mask_bioz = valid_mask_min

        valid_mask = valid_mask_min

        logger.info(
            f"  Windows: {n_windows}, Valid: {valid_mask.sum()} ({valid_mask.mean()*100:.1f}%)"
        )

        y_step = window_labels
        y_opencr = None
        y_ord = None
        target_direction = self.config.target_direction
        if y_step is not None:
            if target_map is not None:
                y_opencr = apply_target_map_opencr(y_step, target_map)
                y_ord = apply_target_map_ordinal(y_step, target_map)
                target_direction = str(target_map.get("direction", target_direction))
            else:
                y_opencr = normalize_steps_to_opencr(y_step, direction=self.config.target_direction)
                y_ord = steps_to_ordinal(
                    y_step,
                    n_bins=self.config.ordinal_bins,
                    direction=self.config.target_direction,
                )

        return ProcessedSubject(
            subject_id=subject_data.subject_id,
            X=X,
            y=window_labels,
            y_step=y_step,
            y_opencr=y_opencr,
            y_ord=y_ord,
            sqi=sqi,
            valid_mask=valid_mask,
            valid_mask_min=valid_mask_min,
            valid_mask_ppg=valid_mask_ppg,
            valid_mask_bioz=valid_mask_bioz,
            timestamps=first_result.timestamps,
            metadata={
                "channels": channel_names,
                "fs": target_fs,
                "window_sec": self.config.window_sec,
                "stride_sec": self.config.stride_sec,
                "n_windows": n_windows,
                "n_valid": int(valid_mask.sum()),
                "coverage_min": float(valid_mask_min.mean()),
                "coverage_ppg": (
                    float(valid_mask_ppg.mean()) if valid_mask_ppg is not None else None
                ),
                "coverage_bioz": (
                    float(valid_mask_bioz.mean()) if valid_mask_bioz is not None else None
                ),
                "sqi_threshold": self.config.sqi_threshold,
                "target_direction": target_direction,
                "ordinal_bins": self.config.ordinal_bins,
            },
        )

    def process_dataset(
        self,
        adapter: DatasetAdapter,
        output_dir: Path,
        *,
        protocol: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Process entire dataset and save to output directory.

        Args:
            adapter: Dataset adapter.
            output_dir: Output directory for processed files.
            protocol: Optional protocol definition (levels + direction).

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
            "valid_windows_min": 0,
            "valid_windows_ppg": 0,
            "valid_windows_bioz": 0,
            "subjects_processed": [],
            "subjects_failed": [],
            "subjects_missing_steps": [],
        }

        steps_list: list[NDArray[np.floating]] = []
        subjects_missing_steps: list[str] = []
        for subject_id in subjects:
            try:
                protocol_levels = adapter.protocol_levels(subject_id)
            except Exception as e:
                logger.error(f"Failed to read protocol levels for {subject_id}: {e}")
                subjects_missing_steps.append(subject_id)
                continue
            if protocol_levels is None:
                subjects_missing_steps.append(subject_id)
                continue
            protocol_array = np.asarray(protocol_levels, dtype=np.float64)
            if protocol_array.size == 0:
                subjects_missing_steps.append(subject_id)
                continue
            steps_list.append(protocol_array)

        target_map = None
        target_map_payload: dict[str, Any] | None = None
        target_map_source = "dataset_global"
        protocol_levels = None
        protocol_direction = None
        target_map_warning = None

        if protocol is not None:
            protocol_levels = protocol.get("levels")
            protocol_direction = protocol.get("direction")
            target_map = build_target_map(
                [np.asarray(protocol_levels, dtype=np.float64)],
                direction=str(protocol_direction),
                n_bins=self.config.ordinal_bins,
            )
            target_map_source = "protocol"
            target_map_payload = dict(target_map)
            target_map_payload["source"] = target_map_source
            target_map_payload["protocol_levels"] = protocol_levels
            target_map_payload["protocol_direction"] = protocol_direction
        elif steps_list:
            target_map = build_target_map(
                steps_list,
                direction=self.config.target_direction,
                n_bins=self.config.ordinal_bins,
            )
            target_map_payload = dict(target_map)
            target_map_payload["source"] = target_map_source
            target_map_payload["protocol_levels"] = None
            target_map_payload["protocol_direction"] = None
            target_map_warning = (
                "Protocol inferred from dataset; for strict evaluation provide "
                "protocol.json or --protocol-levels."
            )
            logger.warning(target_map_warning)
        else:
            target_map_payload = {
                "direction": self.config.target_direction,
                "direction_source": (
                    "auto" if self.config.target_direction == "auto" else "explicit"
                ),
                "global_min_step": None,
                "global_max_step": None,
                "ordinal": {
                    "mode": "none",
                    "n_bins": self.config.ordinal_bins,
                },
                "has_steps": False,
                "source": target_map_source,
                "protocol_levels": None,
                "protocol_direction": None,
            }
            target_map_warning = (
                "No protocol steps found; targets will be empty. Provide protocol.json "
                "or --protocol-levels for strict evaluation."
            )
            logger.warning(target_map_warning)

        if target_map_payload is not None and target_map_source != "protocol":
            target_map_payload.setdefault("source", target_map_source)
            target_map_payload.setdefault("protocol_levels", protocol_levels)
            target_map_payload.setdefault("protocol_direction", protocol_direction)

        if target_map and target_map_source == "protocol":
            logger.info(
                "Target map (protocol): direction=%s, min_step=%.4f, max_step=%.4f, mode=%s",
                target_map["direction"],
                target_map["global_min_step"],
                target_map["global_max_step"],
                target_map.get("ordinal", {}).get("mode"),
            )
        elif target_map:
            logger.info(
                "Target map: direction=%s, min_step=%.4f, max_step=%.4f, mode=%s",
                target_map["direction"],
                target_map["global_min_step"],
                target_map["global_max_step"],
                target_map.get("ordinal", {}).get("mode"),
            )

        if subjects_missing_steps:
            logger.warning(
                "Subjects missing protocol steps: %s",
                ", ".join(subjects_missing_steps),
            )

        target_map_path = output_dir / "target_map.json"
        with open(target_map_path, "w", encoding="utf-8") as f:
            json.dump(target_map_payload, f, indent=2)

        stats["subjects_missing_steps"] = subjects_missing_steps
        stats["target_map_path"] = str(target_map_path)
        stats["target_map_source"] = target_map_source
        stats["protocol_levels"] = protocol_levels
        stats["protocol_direction"] = protocol_direction
        if target_map_warning:
            stats["target_map_warning"] = target_map_warning

        for subject_id in subjects:
            try:
                subject_data = adapter.load_subject(subject_id)
                processed = self.process_subject(subject_data, target_map=target_map)

                # Save processed data
                output_path = processed_dir / f"{subject_id}.npz"
                np.savez(
                    output_path,
                    X=processed.X,
                    y=processed.y if processed.y is not None else np.array([]),
                    y_step=processed.y_step if processed.y_step is not None else np.array([]),
                    y_opencr=processed.y_opencr if processed.y_opencr is not None else np.array([]),
                    y_ord=processed.y_ord if processed.y_ord is not None else np.array([]),
                    sqi=processed.sqi,
                    valid_mask=processed.valid_mask,
                    valid_mask_min=processed.valid_mask_min,
                    valid_mask_ppg=(
                        processed.valid_mask_ppg
                        if processed.valid_mask_ppg is not None
                        else np.array([])
                    ),
                    valid_mask_bioz=(
                        processed.valid_mask_bioz
                        if processed.valid_mask_bioz is not None
                        else np.array([])
                    ),
                    timestamps=processed.timestamps,
                    metadata=processed.metadata,
                )

                stats["processed"] += 1
                stats["total_windows"] += processed.X.shape[0]
                stats["valid_windows"] += int(processed.valid_mask.sum())
                stats["valid_windows_min"] += int(processed.valid_mask_min.sum())
                if processed.valid_mask_ppg is not None:
                    stats["valid_windows_ppg"] += int(processed.valid_mask_ppg.sum())
                if processed.valid_mask_bioz is not None:
                    stats["valid_windows_bioz"] += int(processed.valid_mask_bioz.sum())
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
        if stats["total_windows"] > 0:
            stats["coverage_min"] = stats["valid_windows_min"] / stats["total_windows"]
            stats["coverage_ppg"] = stats["valid_windows_ppg"] / stats["total_windows"]
            stats["coverage_bioz"] = stats["valid_windows_bioz"] / stats["total_windows"]
        with open(stats_path, "w", encoding="utf-8") as f:
            json.dump(stats, f, indent=2)

        logger.info(
            f"Processing complete: {stats['processed']}/{stats['total_subjects']} subjects, "
            f"{stats['valid_windows']}/{stats['total_windows']} valid windows "
            f"({stats['valid_windows']/max(stats['total_windows'],1)*100:.1f}%)"
        )

        return stats
