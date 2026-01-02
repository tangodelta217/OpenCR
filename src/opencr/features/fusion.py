"""
Feature Fusion.

Combines PPG and BioZ features with SQI statistics.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from opencr.features.bioz import extract_bioz_features, get_bioz_feature_names
from opencr.features.ppg import extract_ppg_features, get_ppg_feature_names
from opencr.logging import get_logger

logger = get_logger(__name__)


@dataclass
class FeatureConfig:
    """Configuration for feature extraction."""

    include_ppg: bool = True
    include_bioz: bool = True
    include_sqi: bool = True
    include_cross: bool = True  # Cross-channel features


def extract_all_features(
    X: NDArray[np.floating],
    fs: float,
    sqi: NDArray[np.floating] | None = None,
    valid_mask: NDArray[np.bool_] | None = None,
    config: FeatureConfig | None = None,
    channel_names: list[str] | None = None,
) -> tuple[NDArray[np.floating], list[str]]:
    """
    Extract all features from multi-channel windows.

    Args:
        X: Windows tensor, shape (n_windows, n_channels, window_samples).
        fs: Sampling frequency in Hz.
        sqi: SQI scores, shape (n_windows, n_channels) or None.
        valid_mask: Valid window mask, shape (n_windows,) or None.
        config: Feature configuration.
        channel_names: Channel names (default: ["ppg", "bioz"]).

    Returns:
        Tuple of (feature_matrix, feature_names).
        feature_matrix shape: (n_windows, n_features).
    """
    if config is None:
        config = FeatureConfig()

    if channel_names is None:
        channel_names = ["ppg", "bioz"]

    n_windows, n_channels, window_samples = X.shape

    all_features: list[list[float]] = []
    feature_names: list[str] = []
    names_initialized = False

    for w in range(n_windows):
        window_features: list[float] = []
        window_names: list[str] = []

        # Extract PPG features (first channel)
        if config.include_ppg and n_channels >= 1:
            ppg_feats = extract_ppg_features(X[w, 0], fs)
            window_features.extend(ppg_feats.values())
            if not names_initialized:
                window_names.extend(ppg_feats.keys())

        # Extract BioZ features (second channel)
        if config.include_bioz and n_channels >= 2:
            bioz_feats = extract_bioz_features(X[w, 1], fs)
            window_features.extend(bioz_feats.values())
            if not names_initialized:
                window_names.extend(bioz_feats.keys())

        # Add SQI features
        if config.include_sqi and sqi is not None:
            if sqi.ndim == 1:
                window_features.append(float(sqi[w]))
                if not names_initialized:
                    window_names.append("sqi_combined")
            else:
                for c in range(sqi.shape[1]):
                    window_features.append(float(sqi[w, c]))
                    if not names_initialized:
                        ch_name = channel_names[c] if c < len(channel_names) else f"ch{c}"
                        window_names.append(f"sqi_{ch_name}")

                # Min SQI across channels
                window_features.append(float(np.min(sqi[w])))
                if not names_initialized:
                    window_names.append("sqi_min")

        # Add cross-channel features
        if config.include_cross and n_channels >= 2:
            # Correlation between channels
            corr = np.corrcoef(X[w, 0], X[w, 1])[0, 1]
            if np.isnan(corr):
                corr = 0.0
            window_features.append(float(corr))
            if not names_initialized:
                window_names.append("cross_correlation")

            # Ratio of energies
            energy_0 = np.sum(X[w, 0] ** 2)
            energy_1 = np.sum(X[w, 1] ** 2)
            energy_ratio = energy_0 / (energy_1 + 1e-10)
            window_features.append(float(energy_ratio))
            if not names_initialized:
                window_names.append("cross_energy_ratio")

        all_features.append(window_features)

        if not names_initialized:
            feature_names = window_names
            names_initialized = True

    feature_matrix = np.array(all_features, dtype=np.float64)

    # Replace NaN/Inf with 0
    feature_matrix = np.nan_to_num(feature_matrix, nan=0.0, posinf=0.0, neginf=0.0)

    logger.debug(f"Extracted {feature_matrix.shape[1]} features from {n_windows} windows")

    return feature_matrix, feature_names


def get_all_feature_names(
    include_ppg: bool = True,
    include_bioz: bool = True,
    include_sqi: bool = True,
    n_channels: int = 2,
) -> list[str]:
    """Get list of all feature names."""
    names = []

    if include_ppg:
        names.extend(get_ppg_feature_names())

    if include_bioz:
        names.extend(get_bioz_feature_names())

    if include_sqi:
        names.extend([f"sqi_ch{i}" for i in range(n_channels)])
        names.append("sqi_min")

    names.extend(["cross_correlation", "cross_energy_ratio"])

    return names
