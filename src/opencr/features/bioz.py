"""
BioZ Feature Extraction.

Extracts time and frequency domain features from Bioimpedance windows.
"""

import numpy as np
from numpy.typing import NDArray
from scipy import stats

from opencr.logging import get_logger

logger = get_logger(__name__)


def extract_bioz_features(
    window: NDArray[np.floating],
    fs: float,
) -> dict[str, float]:
    """
    Extract features from a BioZ window.

    Args:
        window: BioZ signal window, shape (N,).
        fs: Sampling frequency in Hz.

    Returns:
        Dictionary of feature names to values.
    """
    features: dict[str, float] = {}

    # Handle edge cases
    if len(window) < 10:
        return _empty_bioz_features()

    # Time domain features
    features["bioz_mean"] = float(np.mean(window))
    features["bioz_std"] = float(np.std(window))
    features["bioz_min"] = float(np.min(window))
    features["bioz_max"] = float(np.max(window))
    features["bioz_range"] = features["bioz_max"] - features["bioz_min"]
    features["bioz_median"] = float(np.median(window))

    # Statistical moments
    features["bioz_skewness"] = float(stats.skew(window))
    features["bioz_kurtosis"] = float(stats.kurtosis(window))

    # Slope features (trend in respiration)
    diff = np.diff(window)
    features["bioz_slope_mean"] = float(np.mean(diff))
    features["bioz_slope_std"] = float(np.std(diff))
    features["bioz_slope_max"] = float(np.max(diff))
    features["bioz_slope_min"] = float(np.min(diff))

    # Zero crossings (respiration approximation)
    zero_mean = window - np.mean(window)
    zero_crossings = np.sum(np.diff(np.sign(zero_mean)) != 0)
    features["bioz_zero_crossings"] = float(zero_crossings)

    # Estimate respiration rate from zero crossings
    # Each breath cycle has 2 zero crossings
    duration_sec = len(window) / fs
    breaths_per_sec = (zero_crossings / 2) / duration_sec
    features["bioz_resp_rate_est"] = float(breaths_per_sec * 60)  # breaths/min

    # Frequency domain features
    fft = np.fft.rfft(window)
    freqs = np.fft.rfftfreq(len(window), 1.0 / fs)
    power = np.abs(fft) ** 2
    total_power = np.sum(power) + 1e-10

    # Respiratory band (0.1-0.5 Hz, ~6-30 breaths/min)
    resp_mask = (freqs >= 0.1) & (freqs <= 0.5)
    resp_power = np.sum(power[resp_mask])
    features["bioz_resp_band_power"] = float(resp_power)
    features["bioz_resp_band_ratio"] = float(resp_power / total_power)

    # Very low frequency (circulation, 0.003-0.04 Hz)
    vlf_mask = (freqs >= 0.003) & (freqs <= 0.04)
    vlf_power = np.sum(power[vlf_mask])
    features["bioz_vlf_power"] = float(vlf_power)

    # Dominant frequency in respiratory band
    if np.any(resp_mask):
        resp_freqs = freqs[resp_mask]
        resp_powers = power[resp_mask]
        if len(resp_powers) > 0:
            dom_idx = np.argmax(resp_powers)
            features["bioz_dominant_freq"] = float(resp_freqs[dom_idx])
            features["bioz_dominant_power"] = float(resp_powers[dom_idx])
        else:
            features["bioz_dominant_freq"] = 0.0
            features["bioz_dominant_power"] = 0.0
    else:
        features["bioz_dominant_freq"] = 0.0
        features["bioz_dominant_power"] = 0.0

    # Energy in different bands
    features["bioz_total_energy"] = float(total_power)

    return features


def _empty_bioz_features() -> dict[str, float]:
    """Return empty features dict for invalid windows."""
    return {
        "bioz_mean": 0.0,
        "bioz_std": 0.0,
        "bioz_min": 0.0,
        "bioz_max": 0.0,
        "bioz_range": 0.0,
        "bioz_median": 0.0,
        "bioz_skewness": 0.0,
        "bioz_kurtosis": 0.0,
        "bioz_slope_mean": 0.0,
        "bioz_slope_std": 0.0,
        "bioz_slope_max": 0.0,
        "bioz_slope_min": 0.0,
        "bioz_zero_crossings": 0.0,
        "bioz_resp_rate_est": 0.0,
        "bioz_resp_band_power": 0.0,
        "bioz_resp_band_ratio": 0.0,
        "bioz_vlf_power": 0.0,
        "bioz_dominant_freq": 0.0,
        "bioz_dominant_power": 0.0,
        "bioz_total_energy": 0.0,
    }


def extract_bioz_features_batch(
    windows: NDArray[np.floating],
    fs: float,
) -> NDArray[np.floating]:
    """
    Extract BioZ features for batch of windows.

    Args:
        windows: Windows array, shape (n_windows, window_samples).
        fs: Sampling frequency in Hz.

    Returns:
        Feature matrix, shape (n_windows, n_features).
    """
    n_windows = windows.shape[0]
    feature_list = []

    for i in range(n_windows):
        feat_dict = extract_bioz_features(windows[i], fs)
        feature_list.append(list(feat_dict.values()))

    return np.array(feature_list, dtype=np.float64)


def get_bioz_feature_names() -> list[str]:
    """Get list of BioZ feature names in order."""
    return list(_empty_bioz_features().keys())
