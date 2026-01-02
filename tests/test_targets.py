"""
Tests for OpenCR target mappings.
"""

import numpy as np
import pytest

from opencr.targets.opencr import normalize_steps_to_opencr, steps_to_ordinal


def test_normalize_steps_to_opencr_negative_auto() -> None:
    """Auto direction: negative steps -> more negative is more severe."""
    steps = np.array([0, -10, -20, -30], dtype=float)
    opencr = normalize_steps_to_opencr(steps, direction="auto")

    assert opencr[0] == pytest.approx(100.0)
    assert opencr[-1] == pytest.approx(0.0)
    assert np.all((opencr >= 0.0) & (opencr <= 100.0))


def test_normalize_steps_to_opencr_positive_auto() -> None:
    """Auto direction: non-negative steps -> larger is more severe."""
    steps = np.array([0, 1, 2, 3], dtype=float)
    opencr = normalize_steps_to_opencr(steps, direction="auto")

    assert opencr[0] == pytest.approx(100.0)
    assert opencr[-1] == pytest.approx(0.0)
    assert np.all((opencr >= 0.0) & (opencr <= 100.0))


def test_steps_to_ordinal_grouping() -> None:
    """Ordinal mapping groups adjacent levels when unique steps > n_bins."""
    steps = np.array([0, 1, 2, 3, 4, 5], dtype=float)
    y_ord = steps_to_ordinal(steps, n_bins=3, direction="increasing")

    assert y_ord.tolist() == [0, 0, 1, 1, 2, 2]
    assert y_ord.dtype == int
