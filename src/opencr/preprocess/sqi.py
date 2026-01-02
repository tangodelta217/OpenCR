"""
Signal Quality Index (SQI) Computation.

Provides basic SQI metrics for quality gating:
- Clipping detection
- In-band energy ratio
- Signal regularity (variance stability)
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from opencr.logging import get_logger

logger = get_logger(__name__)


@dataclass
class SQIConfig:
    """Configuration for SQI computation."""

    clip_threshold: float = 0.99  # Fraction of max for clipping detection
    energy_band_low: float = 0.5  # Hz - lower bound for in-band energy
    energy_band_high: float = 4.0  # Hz - upper bound for in-band energy
    regularity_window: int = 10  # Window size for regularity computation


@dataclass
class SQIResult:
    """SQI computation results per window."""

    clip_score: float  # 1.0 = no clipping, 0.0 = heavy clipping
    energy_score: float  # Ratio of in-band energy to total energy
    regularity_score: float  # Variance stability score
    combined_score: float  # Overall SQI (0-1)

    def is_valid(self, threshold: float = 0.5) -> bool:
        """Check if window passes quality threshold."""
        return self.combined_score >= threshold


def compute_clip_score(
    window: NDArray[np.floating],
    threshold: float = 0.99,
) -> float:
    """
    Compute clipping score for a window.

    Returns 1.0 if no clipping, 0.0 if heavily clipped.

    Args:
        window: Signal window, shape (N,).
        threshold: Fraction of absolute max to consider as clipped.

    Returns:
        Clipping score between 0 and 1.
    """
    if len(window) == 0:
        return 0.0

    abs_window = np.abs(window)
    max_val = np.max(abs_window)

    if max_val == 0:
        return 1.0  # Flat signal, no clipping

    clip_level = max_val * threshold
    n_clipped = np.sum(abs_window >= clip_level)
    clip_ratio = n_clipped / len(window)

    # Score: 1.0 if no clipping, decreases with clip ratio
    # Use exponential decay for smoother scoring
    return float(np.exp(-10 * clip_ratio))


def compute_energy_score(
    window: NDArray[np.floating],
    fs: float,
    low_hz: float = 0.5,
    high_hz: float = 4.0,
) -> float:
    """
    Compute in-band energy ratio.

    Returns ratio of energy in target band to total energy.

    Args:
        window: Signal window, shape (N,).
        fs: Sampling frequency in Hz.
        low_hz: Lower frequency bound.
        high_hz: Upper frequency bound.

    Returns:
        Energy ratio between 0 and 1.
    """
    if len(window) < 4:
        return 0.0

    # Compute FFT
    fft = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(len(window), 1.0 / fs)
    power = np.abs(fft) ** 2

    total_power = np.sum(power)
    if total_power == 0:
        return 0.0

    # In-band power
    in_band_mask = (freqs >= low_hz) & (freqs <= high_hz)
    in_band_power = np.sum(power[in_band_mask])

    return float(in_band_power / total_power)


def compute_regularity_score(
    window: NDArray[np.floating],
    chunk_size: int = 10,
) -> float:
    """
    Compute signal regularity score based on variance stability.

    Divides window into chunks and measures variance consistency.

    Args:
        window: Signal window, shape (N,).
        chunk_size: Number of chunks for variance computation.

    Returns:
        Regularity score between 0 and 1.
    """
    if len(window) < chunk_size * 2:
        return 0.5  # Not enough data, neutral score

    n_samples = len(window)
    samples_per_chunk = n_samples // chunk_size

    if samples_per_chunk < 2:
        return 0.5

    # Compute variance per chunk
    chunk_vars = []
    for i in range(chunk_size):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        chunk = window[start:end]
        chunk_vars.append(np.var(chunk))

    chunk_vars = np.array(chunk_vars)

    # Coefficient of variation of variances
    mean_var = np.mean(chunk_vars)
    if mean_var == 0:
        return 1.0  # Flat signal

    cv = np.std(chunk_vars) / mean_var

    # Score: 1.0 for uniform variance, decreases with CV
    return float(np.exp(-cv))


def compute_sqi(
    window: NDArray[np.floating],
    fs: float,
    config: SQIConfig | None = None,
) -> SQIResult:
    """
    Compute all SQI metrics for a single window.

    Args:
        window: Signal window, shape (N,).
        fs: Sampling frequency in Hz.
        config: SQI configuration (uses defaults if None).

    Returns:
        SQIResult with all computed metrics.
    """
    if config is None:
        config = SQIConfig()

    # Handle multi-channel by taking first channel or mean
    if window.ndim > 1:
        window = window.mean(axis=1)

    clip_score = compute_clip_score(window, config.clip_threshold)
    energy_score = compute_energy_score(window, fs, config.energy_band_low, config.energy_band_high)
    regularity_score = compute_regularity_score(window, config.regularity_window)

    # Combined score: geometric mean for balanced weighting
    # Clipping is most critical, so weight it higher
    combined = (clip_score**0.4) * (energy_score**0.3) * (regularity_score**0.3)

    return SQIResult(
        clip_score=clip_score,
        energy_score=energy_score,
        regularity_score=regularity_score,
        combined_score=float(combined),
    )


def compute_sqi_batch(
    windows: NDArray[np.floating],
    fs: float,
    config: SQIConfig | None = None,
) -> NDArray[np.floating]:
    """
    Compute SQI for batch of windows.

    Args:
        windows: Windows array, shape (n_windows, window_samples) or
                 (n_windows, n_channels, window_samples).
        fs: Sampling frequency in Hz.
        config: SQI configuration.

    Returns:
        SQI scores array, shape (n_windows,) or (n_windows, n_channels).
    """
    if config is None:
        config = SQIConfig()

    n_windows = windows.shape[0]

    if windows.ndim == 2:
        # Single channel: (n_windows, window_samples)
        sqi_scores = np.zeros(n_windows, dtype=np.float64)
        for i in range(n_windows):
            result = compute_sqi(windows[i], fs, config)
            sqi_scores[i] = result.combined_score
    elif windows.ndim == 3:
        # Multi-channel: (n_windows, n_channels, window_samples)
        n_channels = windows.shape[1]
        sqi_scores = np.zeros((n_windows, n_channels), dtype=np.float64)
        for i in range(n_windows):
            for c in range(n_channels):
                result = compute_sqi(windows[i, c], fs, config)
                sqi_scores[i, c] = result.combined_score
    else:
        raise ValueError(f"Unexpected windows shape: {windows.shape}")

    logger.debug(f"Computed SQI for {n_windows} windows, mean={sqi_scores.mean():.3f}")

    return sqi_scores
