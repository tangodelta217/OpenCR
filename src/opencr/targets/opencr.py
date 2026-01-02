"""
OpenCR target mapping utilities.

Maps protocol steps to:
  - OpenCR continuous target in [0, 100]
  - Ordinal labels for classification
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

_DIRECTION_OPTIONS = {"auto", "increasing", "decreasing"}


def _as_float_array(steps: NDArray | list[float] | list[int]) -> NDArray[np.floating]:
    steps_array = np.asarray(steps, dtype=np.float64)
    if steps_array.size == 0:
        raise ValueError("steps must be non-empty")
    if np.isnan(steps_array).any():
        raise ValueError("steps contains NaN values")
    return steps_array


def _resolve_direction(steps: NDArray[np.floating], direction: str) -> str:
    if direction not in _DIRECTION_OPTIONS:
        raise ValueError(
            f"direction must be one of {sorted(_DIRECTION_OPTIONS)}, got '{direction}'"
        )
    if direction == "auto":
        return "decreasing" if np.any(steps < 0) else "increasing"
    return direction


def normalize_steps_to_opencr(
    steps: NDArray | list[float] | list[int],
    *,
    direction: str = "auto",
) -> NDArray[np.floating]:
    """
    Map protocol steps to OpenCR scores in [0, 100].

    Heuristic for direction="auto":
      - if any step is negative: assume more negative = more severe
      - otherwise: assume larger step = more severe

    Args:
        steps: Protocol step values per window.
        direction: "auto", "increasing" (larger step = more severe),
                   or "decreasing" (smaller/more negative step = more severe).

    Returns:
        OpenCR scores in [0, 100], float array with same shape as input.
    """
    steps_array = _as_float_array(steps)
    resolved = _resolve_direction(steps_array, direction)

    severity = steps_array if resolved == "increasing" else -steps_array
    min_val = float(np.min(severity))
    max_val = float(np.max(severity))

    if max_val == min_val:
        return np.full_like(severity, 100.0, dtype=np.float64)

    opencr = (max_val - severity) / (max_val - min_val) * 100.0
    return np.clip(opencr, 0.0, 100.0).astype(np.float64)


def steps_to_ordinal(
    steps: NDArray | list[float] | list[int],
    *,
    n_bins: int = 4,
    direction: str = "auto",
) -> NDArray[np.int_]:
    """
    Map protocol steps to ordinal labels.

    The output is ordered so that 0 corresponds to the least severe level.
    If the number of unique steps is greater than n_bins, adjacent levels
    are grouped into n_bins bins.

    Args:
        steps: Protocol step values per window.
        n_bins: Desired number of ordinal bins (>= 2).
        direction: "auto", "increasing", or "decreasing".

    Returns:
        Ordinal labels as integer array with same shape as input.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    steps_array = _as_float_array(steps)
    resolved = _resolve_direction(steps_array, direction)
    severity = steps_array if resolved == "increasing" else -steps_array

    unique_vals = np.unique(severity)
    if unique_vals.size == 1:
        return np.zeros_like(severity, dtype=int)

    sorted_unique = np.sort(unique_vals)
    ranks = np.searchsorted(sorted_unique, severity)

    if unique_vals.size <= n_bins:
        return ranks.astype(int)

    groups = np.array_split(np.arange(unique_vals.size), n_bins)
    bin_index = np.zeros(unique_vals.size, dtype=int)
    for idx, group in enumerate(groups):
        bin_index[group] = idx

    return bin_index[ranks].astype(int)
