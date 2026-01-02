"""
Tests for subject-wise bootstrap confidence intervals.
"""

import numpy as np
import pytest

from opencr.evaluation.metrics import compute_metrics_with_ci


def test_bootstrap_by_subject_single_draw() -> None:
    """Bootstrap should resample whole subjects, not individual windows."""
    y_true = np.array([1, 1, 1, 0])
    y_pred = np.array([1, 1, 1, 1])
    subject_ids = np.array(["A", "A", "A", "B"])

    seed = 123
    rng = np.random.default_rng(seed)
    unique_subjects = np.unique(subject_ids)
    sampled = rng.choice(unique_subjects, size=len(unique_subjects), replace=True)
    indices = np.concatenate([np.where(subject_ids == sid)[0] for sid in sampled])

    expected_accuracy = float(np.mean(y_true[indices] == y_pred[indices]))

    _, ci = compute_metrics_with_ci(
        y_true,
        y_pred,
        None,
        subject_ids,
        "classification",
        n_boot=1,
        seed=seed,
    )

    assert ci["accuracy"]["mean"] == pytest.approx(expected_accuracy)
    assert ci["accuracy"]["low"] == pytest.approx(expected_accuracy)
    assert ci["accuracy"]["high"] == pytest.approx(expected_accuracy)


def test_bootstrap_deterministic_seed() -> None:
    """Fixed seed should yield deterministic CI values."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    subject_ids = np.array(["S1", "S1", "S2", "S2", "S3", "S3"])

    _, ci_1 = compute_metrics_with_ci(
        y_true,
        y_pred,
        None,
        subject_ids,
        "classification",
        n_boot=25,
        seed=7,
    )
    _, ci_2 = compute_metrics_with_ci(
        y_true,
        y_pred,
        None,
        subject_ids,
        "classification",
        n_boot=25,
        seed=7,
    )

    for metric in ci_1:
        assert ci_1[metric]["mean"] == pytest.approx(ci_2[metric]["mean"])
        assert ci_1[metric]["low"] == pytest.approx(ci_2[metric]["low"])
        assert ci_1[metric]["high"] == pytest.approx(ci_2[metric]["high"])
