"""
Signal Filters for Preprocessing.

Provides bandpass, highpass, and lowpass filters using scipy.
All filters use Butterworth design for smooth frequency response.
"""

import numpy as np
from numpy.typing import NDArray
from scipy.signal import butter, sosfiltfilt

from opencr.logging import get_logger

logger = get_logger(__name__)


def bandpass_filter(
    signal: NDArray[np.floating],
    fs: float,
    low_hz: float,
    high_hz: float,
    order: int = 4,
) -> NDArray[np.floating]:
    """
    Apply a zero-phase Butterworth bandpass filter.

    Args:
        signal: Input signal array, shape (N,) or (N, channels).
        fs: Sampling frequency in Hz.
        low_hz: Lower cutoff frequency in Hz.
        high_hz: Upper cutoff frequency in Hz.
        order: Filter order (default: 4).

    Returns:
        Filtered signal with same shape as input.

    Raises:
        ValueError: If filter parameters are invalid.
    """
    nyq = fs / 2.0

    if low_hz <= 0:
        raise ValueError(f"low_hz must be positive, got {low_hz}")
    if high_hz >= nyq:
        raise ValueError(f"high_hz must be < Nyquist ({nyq} Hz), got {high_hz}")
    if low_hz >= high_hz:
        raise ValueError(f"low_hz ({low_hz}) must be < high_hz ({high_hz})")

    low_norm = low_hz / nyq
    high_norm = high_hz / nyq

    sos = butter(order, [low_norm, high_norm], btype="band", output="sos")

    logger.debug(f"Bandpass filter: {low_hz}-{high_hz} Hz, order={order}")

    return sosfiltfilt(sos, signal, axis=0)


def highpass_filter(
    signal: NDArray[np.floating],
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> NDArray[np.floating]:
    """
    Apply a zero-phase Butterworth highpass filter.

    Args:
        signal: Input signal array, shape (N,) or (N, channels).
        fs: Sampling frequency in Hz.
        cutoff_hz: Cutoff frequency in Hz.
        order: Filter order (default: 4).

    Returns:
        Filtered signal with same shape as input.
    """
    nyq = fs / 2.0

    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
    if cutoff_hz >= nyq:
        raise ValueError(f"cutoff_hz must be < Nyquist ({nyq} Hz), got {cutoff_hz}")

    cutoff_norm = cutoff_hz / nyq
    sos = butter(order, cutoff_norm, btype="high", output="sos")

    logger.debug(f"Highpass filter: {cutoff_hz} Hz, order={order}")

    return sosfiltfilt(sos, signal, axis=0)


def lowpass_filter(
    signal: NDArray[np.floating],
    fs: float,
    cutoff_hz: float,
    order: int = 4,
) -> NDArray[np.floating]:
    """
    Apply a zero-phase Butterworth lowpass filter.

    Args:
        signal: Input signal array, shape (N,) or (N, channels).
        fs: Sampling frequency in Hz.
        cutoff_hz: Cutoff frequency in Hz.
        order: Filter order (default: 4).

    Returns:
        Filtered signal with same shape as input.
    """
    nyq = fs / 2.0

    if cutoff_hz <= 0:
        raise ValueError(f"cutoff_hz must be positive, got {cutoff_hz}")
    if cutoff_hz >= nyq:
        raise ValueError(f"cutoff_hz must be < Nyquist ({nyq} Hz), got {cutoff_hz}")

    cutoff_norm = cutoff_hz / nyq
    sos = butter(order, cutoff_norm, btype="low", output="sos")

    logger.debug(f"Lowpass filter: {cutoff_hz} Hz, order={order}")

    return sosfiltfilt(sos, signal, axis=0)


def resample_signal(
    signal: NDArray[np.floating],
    fs_original: float,
    fs_target: float,
) -> NDArray[np.floating]:
    """
    Resample signal to target sampling rate.

    Args:
        signal: Input signal array, shape (N,) or (N, channels).
        fs_original: Original sampling frequency in Hz.
        fs_target: Target sampling frequency in Hz.

    Returns:
        Resampled signal.
    """
    from scipy.signal import resample

    if fs_original == fs_target:
        return signal

    n_original = len(signal)
    n_target = int(n_original * fs_target / fs_original)

    logger.debug(
        f"Resampling: {fs_original} Hz -> {fs_target} Hz ({n_original} -> {n_target} samples)"
    )

    return resample(signal, n_target, axis=0)
