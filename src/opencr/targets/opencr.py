"""
OpenCR target mapping utilities.

Maps protocol steps to:
  - OpenCR continuous target in [0, 100]
  - Ordinal labels for classification
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

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


def _severity_from_steps(
    steps: NDArray[np.floating],
    direction: str,
) -> NDArray[np.floating]:
    if direction not in {"increasing", "decreasing"}:
        raise ValueError(f"direction must be increasing or decreasing, got '{direction}'")
    return steps if direction == "increasing" else -steps


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


def build_target_map(
    steps_list: Iterable[NDArray | list[float] | list[int]],
    *,
    direction: str = "auto",
    n_bins: int = 4,
) -> dict[str, Any]:
    """
    Build a global target map from all protocol steps in a dataset.

    Args:
        steps_list: Iterable of step arrays (one per subject).
        direction: "auto", "increasing", or "decreasing".
        n_bins: Desired number of ordinal bins (>= 2).

    Returns:
        Dictionary describing the global mapping for OpenCR + ordinal targets.
    """
    if n_bins < 2:
        raise ValueError(f"n_bins must be >= 2, got {n_bins}")

    collected: list[NDArray[np.floating]] = []
    for steps in steps_list:
        if steps is None:
            continue
        arr = np.asarray(steps, dtype=np.float64)
        if arr.size == 0:
            continue
        if np.isnan(arr).any():
            raise ValueError("steps contains NaN values")
        collected.append(arr.reshape(-1))

    if not collected:
        raise ValueError("No protocol steps available to build target map")

    all_steps = np.concatenate(collected)
    resolved = _resolve_direction(all_steps, direction)
    severity = _severity_from_steps(all_steps, resolved)

    global_min_step = float(np.min(all_steps))
    global_max_step = float(np.max(all_steps))

    unique_severity = np.unique(severity)
    ordinal: dict[str, Any] = {"n_bins": int(n_bins)}

    if unique_severity.size <= n_bins:
        sorted_severity = np.sort(unique_severity)
        sorted_steps = (sorted_severity if resolved == "increasing" else -sorted_severity).astype(
            np.float64
        )
        ordinal.update(
            {
                "mode": "levels",
                "levels_unique_sorted": sorted_steps.tolist(),
            }
        )
    else:
        sorted_severity = np.sort(unique_severity)
        groups = np.array_split(sorted_severity, n_bins)
        upper_bounds = [float(group[-1]) for group in groups]
        ordinal.update(
            {
                "mode": "bins",
                "bins": upper_bounds,
                "bins_space": "severity",
            }
        )

    return {
        "direction": resolved,
        "direction_source": "auto" if direction == "auto" else "explicit",
        "global_min_step": global_min_step,
        "global_max_step": global_max_step,
        "ordinal": ordinal,
    }


def apply_target_map_opencr(
    steps: NDArray | list[float] | list[int],
    target_map: dict[str, Any],
) -> NDArray[np.floating]:
    """
    Apply a global target map to convert steps into OpenCR scores.
    """
    steps_array = _as_float_array(steps)
    direction = target_map.get("direction")
    if direction not in {"increasing", "decreasing"}:
        raise ValueError(f"target_map direction must be increasing or decreasing, got {direction}")

    min_step = target_map.get("global_min_step")
    max_step = target_map.get("global_max_step")
    if min_step is None or max_step is None:
        raise ValueError("target_map must include global_min_step and global_max_step")

    severity = _severity_from_steps(steps_array, direction)
    min_severity = min_step if direction == "increasing" else -max_step
    max_severity = max_step if direction == "increasing" else -min_step

    if max_severity == min_severity:
        return np.full_like(severity, 100.0, dtype=np.float64)

    opencr = (max_severity - severity) / (max_severity - min_severity) * 100.0
    return np.clip(opencr, 0.0, 100.0).astype(np.float64)


def apply_target_map_ordinal(
    steps: NDArray | list[float] | list[int],
    target_map: dict[str, Any],
) -> NDArray[np.int_]:
    """
    Apply a global target map to convert steps into ordinal labels.
    """
    steps_array = _as_float_array(steps)
    direction = target_map.get("direction")
    if direction not in {"increasing", "decreasing"}:
        raise ValueError(f"target_map direction must be increasing or decreasing, got {direction}")

    ordinal = target_map.get("ordinal") or {}
    mode = ordinal.get("mode")
    n_bins = ordinal.get("n_bins")
    if not isinstance(n_bins, int) or n_bins < 2:
        raise ValueError("target_map ordinal.n_bins must be an int >= 2")

    severity = _severity_from_steps(steps_array, direction)

    if mode == "levels":
        levels = np.asarray(ordinal.get("levels_unique_sorted"), dtype=np.float64)
        if levels.size == 0:
            raise ValueError("target_map ordinal.levels_unique_sorted is empty")
        severity_levels = _severity_from_steps(levels, direction)
        ranks = np.searchsorted(severity_levels, severity)
        if np.any(ranks >= severity_levels.size):
            raise ValueError("steps contain values outside target_map levels")
        if not np.allclose(severity_levels[ranks], severity):
            raise ValueError("steps contain values outside target_map levels")
        return ranks.astype(int)

    if mode == "bins":
        bins = np.asarray(ordinal.get("bins"), dtype=np.float64)
        if bins.size != n_bins:
            raise ValueError("target_map ordinal.bins must have length n_bins")
        if np.any(np.diff(bins) < 0):
            raise ValueError("target_map ordinal.bins must be sorted ascending")
        indices = np.digitize(severity, bins, right=True)
        if np.any(indices >= n_bins):
            raise ValueError("steps contain values outside target_map bins")
        return indices.astype(int)

    raise ValueError("target_map ordinal.mode must be 'levels' or 'bins'")
