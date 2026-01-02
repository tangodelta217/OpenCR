"""
PPG Feature Extraction.

Extracts time and frequency domain features from PPG windows.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from opencr.logging import get_logger

logger = get_logger(__name__)


def extract_ppg_features(
    window: NDArray[np.floating],
    fs: float,
) -> dict[str, float]:
    """
    Extract features from a PPG window.

    Args:
        window: PPG signal window, shape (N,).
        fs: Sampling frequency in Hz.

    Returns:
        Dictionary of feature names to values.
    """
    features: dict[str, float] = {}

    # Handle edge cases
    if len(window) < 10:
        return _empty_ppg_features()

    # Time domain features
    features["ppg_mean"] = float(np.mean(window))
    features["ppg_std"] = float(np.std(window))
    features["ppg_min"] = float(np.min(window))
    features["ppg_max"] = float(np.max(window))
    features["ppg_range"] = features["ppg_max"] - features["ppg_min"]
    features["ppg_median"] = float(np.median(window))

    # Statistical moments
    features["ppg_skewness"] = float(stats.skew(window))
    features["ppg_kurtosis"] = float(stats.kurtosis(window))

    # Slope features (first derivative)
    diff = np.diff(window)
    features["ppg_slope_mean"] = float(np.mean(diff))
    features["ppg_slope_std"] = float(np.std(diff))
    features["ppg_slope_max"] = float(np.max(diff))
    features["ppg_slope_min"] = float(np.min(diff))

    # Zero crossings (approximation of frequency)
    zero_mean = window - np.mean(window)
    zero_crossings = np.sum(np.diff(np.sign(zero_mean)) != 0)
    features["ppg_zero_crossings"] = float(zero_crossings)
    features["ppg_zero_cross_rate"] = float(zero_crossings / len(window))

    # Frequency domain features
    fft = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(len(window), 1.0 / fs)
    power = np.abs(fft) ** 2
    total_power = np.sum(power) + 1e-10

    # Heart rate band (0.5-4 Hz)
    hr_mask = (freqs >= 0.5) & (freqs <= 4.0)
    hr_power = np.sum(power[hr_mask])
    features["ppg_hr_band_power"] = float(hr_power)
    features["ppg_hr_band_ratio"] = float(hr_power / total_power)

    # Low frequency band (0.04-0.15 Hz)
    lf_mask = (freqs >= 0.04) & (freqs <= 0.15)
    lf_power = np.sum(power[lf_mask])
    features["ppg_lf_power"] = float(lf_power)

    # High frequency band (0.15-0.4 Hz)
    hf_mask = (freqs >= 0.15) & (freqs <= 0.4)
    hf_power = np.sum(power[hf_mask])
    features["ppg_hf_power"] = float(hf_power)

    # LF/HF ratio
    features["ppg_lf_hf_ratio"] = float(lf_power / (hf_power + 1e-10))

    # Dominant frequency in HR band
    if np.any(hr_mask):
        hr_freqs = freqs[hr_mask]
        hr_powers = power[hr_mask]
        if len(hr_powers) > 0:
            dom_idx = np.argmax(hr_powers)
            features["ppg_dominant_freq"] = float(hr_freqs[dom_idx])
            features["ppg_dominant_power"] = float(hr_powers[dom_idx])
        else:
            features["ppg_dominant_freq"] = 0.0
            features["ppg_dominant_power"] = 0.0
    else:
        features["ppg_dominant_freq"] = 0.0
        features["ppg_dominant_power"] = 0.0

    return features


def _empty_ppg_features() -> dict[str, float]:
    """Return empty features dict for invalid windows."""
    return {
        "ppg_mean": 0.0,
        "ppg_std": 0.0,
        "ppg_min": 0.0,
        "ppg_max": 0.0,
        "ppg_range": 0.0,
        "ppg_median": 0.0,
        "ppg_skewness": 0.0,
        "ppg_kurtosis": 0.0,
        "ppg_slope_mean": 0.0,
        "ppg_slope_std": 0.0,
        "ppg_slope_max": 0.0,
        "ppg_slope_min": 0.0,
        "ppg_zero_crossings": 0.0,
        "ppg_zero_cross_rate": 0.0,
        "ppg_hr_band_power": 0.0,
        "ppg_hr_band_ratio": 0.0,
        "ppg_lf_power": 0.0,
        "ppg_hf_power": 0.0,
        "ppg_lf_hf_ratio": 0.0,
        "ppg_dominant_freq": 0.0,
        "ppg_dominant_power": 0.0,
    }


def extract_ppg_features_batch(
    windows: NDArray[np.floating],
    fs: float,
) -> NDArray[np.floating]:
    """
    Extract PPG features for batch of windows.

    Args:
        windows: Windows array, shape (n_windows, window_samples).
        fs: Sampling frequency in Hz.

    Returns:
        Feature matrix, shape (n_windows, n_features).
    """
    n_windows = windows.shape[0]
    feature_list = []

    for i in range(n_windows):
        feat_dict = extract_ppg_features(windows[i], fs)
        feature_list.append(list(feat_dict.values()))

    return np.array(feature_list, dtype=np.float64)


def get_ppg_feature_names() -> list[str]:
    """Get list of PPG feature names in order."""
    return list(_empty_ppg_features().keys())
