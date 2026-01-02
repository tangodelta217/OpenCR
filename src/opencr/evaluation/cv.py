"""
Leave-One-Subject-Out (LOSO) Cross-Validation.

Provides anti-leakage CV splits at the subject level.
"""

from collections.abc import Generator
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from opencr.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LOSOResult:
    """Result of a single LOSO fold."""

    fold_idx: int
    test_subject: str
    train_subjects: list[str]
    train_indices: NDArray[np.intp]
    test_indices: NDArray[np.intp]


def loso_split(
    subject_ids: NDArray | list[str],
) -> Generator[LOSOResult, None, None]:
    """
    Generate Leave-One-Subject-Out splits.

    Args:
        subject_ids: Array of subject IDs for each sample.

    Yields:
        LOSOResult for each fold.
    """
    subject_ids = np.asarray(subject_ids)
    unique_subjects = np.unique(subject_ids)
    n_subjects = len(unique_subjects)

    logger.info(f"LOSO split: {n_subjects} subjects, {len(subject_ids)} total samples")

    for fold_idx, test_subject in enumerate(unique_subjects):
        test_mask = subject_ids == test_subject
        train_mask = ~test_mask

        train_indices = np.where(train_mask)[0]
        test_indices = np.where(test_mask)[0]

        train_subjects = [s for s in unique_subjects if s != test_subject]

        yield LOSOResult(
            fold_idx=fold_idx,
            test_subject=str(test_subject),
            train_subjects=[str(s) for s in train_subjects],
            train_indices=train_indices,
            test_indices=test_indices,
        )


def create_subject_array(
    subject_ids: list[str],
    windows_per_subject: dict[str, int],
) -> NDArray:
    """
    Create array of subject IDs expanded to window level.

    Args:
        subject_ids: List of unique subject IDs.
        windows_per_subject: Dict mapping subject ID to number of windows.

    Returns:
        Array of subject IDs for each window.
    """
    result = []
    for sid in subject_ids:
        n_windows = windows_per_subject.get(sid, 0)
        result.extend([sid] * n_windows)
    return np.array(result)
