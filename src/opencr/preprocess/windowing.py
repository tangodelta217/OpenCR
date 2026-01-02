"""
Signal Windowing / Segmentation.

Provides sliding window segmentation for signal processing.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from opencr.logging import get_logger

logger = get_logger(__name__)


@dataclass
class WindowConfig:
    """Configuration for windowing."""

    window_sec: float  # Window size in seconds
    stride_sec: float  # Stride (hop) size in seconds
    min_windows: int = 1  # Minimum number of windows required

    def __post_init__(self) -> None:
        if self.window_sec <= 0:
            raise ValueError(f"window_sec must be positive, got {self.window_sec}")
        if self.stride_sec <= 0:
            raise ValueError(f"stride_sec must be positive, got {self.stride_sec}")
        if self.stride_sec > self.window_sec:
            raise ValueError(
                f"stride_sec ({self.stride_sec}) should be <= window_sec ({self.window_sec})"
            )


@dataclass
class WindowResult:
    """Result of windowing operation."""

    windows: NDArray[np.floating]  # Shape: (n_windows, window_samples)
    timestamps: NDArray[np.floating]  # Start time of each window
    n_windows: int
    window_samples: int
    fs: float


def segment_windows(
    signal: NDArray[np.floating],
    fs: float,
    config: WindowConfig,
    labels: NDArray | None = None,
) -> tuple[WindowResult, NDArray | None]:
    """
    Segment signal into overlapping windows.

    Args:
        signal: Input signal, shape (N,).
        fs: Sampling frequency in Hz.
        config: Windowing configuration.
        labels: Optional labels aligned with signal, shape (N,).
                Returns majority label per window.

    Returns:
        Tuple of (WindowResult, window_labels or None).

    Raises:
        ValueError: If signal is too short for even one window.
    """
    window_samples = int(config.window_sec * fs)
    stride_samples = int(config.stride_sec * fs)

    n_samples = signal.shape[0]
    is_multichannel = signal.ndim > 1

    if n_samples < window_samples:
        raise ValueError(
            f"Signal too short ({n_samples} samples) for window size "
            f"({window_samples} samples = {config.window_sec}s at {fs} Hz)"
        )

    n_windows = 1 + (n_samples - window_samples) // stride_samples

    if n_windows < config.min_windows:
        raise ValueError(f"Only {n_windows} windows possible, but min_windows={config.min_windows}")

    logger.debug(
        f"Segmenting: {n_samples} samples -> {n_windows} windows "
        f"(window={window_samples}, stride={stride_samples})"
    )

    if is_multichannel:
        n_channels = signal.shape[1]
        windows = np.zeros((n_windows, window_samples, n_channels), dtype=signal.dtype)
    else:
        windows = np.zeros((n_windows, window_samples), dtype=signal.dtype)

    timestamps = np.zeros(n_windows, dtype=np.float64)
    window_labels: NDArray | None = None

    if labels is not None:
        window_labels = np.zeros(n_windows, dtype=labels.dtype)

    for i in range(n_windows):
        start = i * stride_samples
        end = start + window_samples
        windows[i] = signal[start:end]
        timestamps[i] = start / fs

        if labels is not None and window_labels is not None:
            window_label_segment = labels[start:end]
            unique, counts = np.unique(window_label_segment, return_counts=True)
            window_labels[i] = unique[np.argmax(counts)]

    result = WindowResult(
        windows=windows,
        timestamps=timestamps,
        n_windows=n_windows,
        window_samples=window_samples,
        fs=fs,
    )

    return result, window_labels


def merge_channel_windows(
    windows_dict: dict[str, WindowResult],
) -> NDArray[np.floating]:
    """
    Merge windows from multiple channels into a single tensor.

    Args:
        windows_dict: Dictionary mapping channel names to WindowResult.
                      All must have same n_windows.

    Returns:
        Tensor of shape (n_windows, n_channels, window_samples).
    """
    channels = list(windows_dict.keys())

    if not channels:
        raise ValueError("No channels provided")

    first = windows_dict[channels[0]]
    n_windows = first.n_windows

    # Verify consistency
    for ch_name, wr in windows_dict.items():
        if wr.n_windows != n_windows:
            raise ValueError(
                f"Channel '{ch_name}' has {wr.n_windows} windows, " f"expected {n_windows}"
            )

    # Stack into (n_windows, n_channels, window_samples)
    stacked = np.stack([windows_dict[ch].windows for ch in channels], axis=1)

    logger.debug(f"Merged {len(channels)} channels: shape={stacked.shape}")

    return stacked
