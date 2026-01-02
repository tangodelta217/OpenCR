"""
Local NPZ Dataset Adapter.

Loads datasets stored as .npz files, one per subject.
Expected structure:
    data_dir/
        subject001.npz
        subject002.npz
        ...

Each .npz file must contain:
    - ppg: PPG signal array, shape (N,) or (1, N)
    - bioz: Bioimpedance signal array, shape (N,) or (1, N)
    - fs_ppg: PPG sampling rate (scalar)
    - fs_bioz: Bioimpedance sampling rate (scalar)

Optional fields:
    - t: Timestamps array aligned with the reference signal and step
    - step: Protocol step/level array (aligned in time)
    - t_step: Timestamps array aligned with step (if different from t)
    - metadata: Dictionary with additional info
"""

from pathlib import Path
from typing import Any

import numpy as np

from opencr.data.adapters.base import DataCard, DatasetAdapter, SubjectData
from opencr.logging import get_logger

logger = get_logger(__name__)

# Required fields in each .npz file
REQUIRED_FIELDS = {"ppg", "bioz", "fs_ppg", "fs_bioz"}

# Optional fields
OPTIONAL_FIELDS = {"t", "step", "t_step", "metadata"}


_SINGLE_CHANNEL_ERROR = (
    "Multichannel not supported in v0.x. Select one channel or average before loading."
)


def _ensure_single_channel(
    signal: np.ndarray,
    *,
    name: str,
    subject_id: str,
) -> np.ndarray:
    arr = np.asarray(signal)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 1:
        return arr[0]
    raise ValueError(
        f"{_SINGLE_CHANNEL_ERROR} Signal '{name}' in '{subject_id}' has shape {arr.shape}."
    )


class LocalNpzAdapter(DatasetAdapter):
    """
    Adapter for local .npz format datasets.

    Expects a directory containing one .npz file per subject.
    Each file must contain PPG and BioZ signals with their sampling rates.

    Example:
        adapter = LocalNpzAdapter(Path("data/raw"))
        subjects = adapter.list_subjects()  # ["subject001", "subject002", ...]
        data = adapter.load_subject("subject001")
        print(data.signals["ppg"].shape)  # (N,) or (1, N)
    """

    def __init__(self, data_path: Path) -> None:
        """
        Initialize adapter with path to .npz files directory.

        Args:
            data_path: Path to directory containing .npz files.

        Raises:
            FileNotFoundError: If data_path does not exist.
            ValueError: If no .npz files are found.
        """
        super().__init__(data_path)
        self._npz_files = sorted(self.data_path.glob("*.npz"))
        if not self._npz_files:
            raise ValueError(f"No .npz files found in: {self.data_path}")
        logger.debug(f"Found {len(self._npz_files)} .npz files in {self.data_path}")

    def list_subjects(self) -> list[str]:
        """
        List all subject IDs (derived from .npz filenames).

        Returns:
            List of subject IDs (filename stems).
        """
        return [f.stem for f in self._npz_files]

    def load_subject(self, subject_id: str) -> SubjectData:
        """
        Load data for a single subject.

        Args:
            subject_id: Subject ID (filename stem without .npz).

        Returns:
            SubjectData with signals, sampling rates, and metadata.

        Raises:
            KeyError: If subject not found.
            ValueError: If required fields are missing.
        """
        npz_path = self.data_path / f"{subject_id}.npz"
        if not npz_path.exists():
            available = self.list_subjects()
            raise KeyError(
                f"Subject '{subject_id}' not found. "
                f"Available subjects: {available[:5]}{'...' if len(available) > 5 else ''}"
            )

        logger.debug(f"Loading subject: {subject_id}")

        with np.load(npz_path, allow_pickle=True) as data:
            # Check required fields
            available_fields = set(data.files)
            missing = REQUIRED_FIELDS - available_fields
            if missing:
                raise ValueError(
                    f"Subject '{subject_id}' is missing required fields: {missing}. "
                    f"Required: {REQUIRED_FIELDS}. Found: {available_fields}"
                )

            # Extract signals
            signals = {
                "ppg": _ensure_single_channel(
                    np.asarray(data["ppg"]),
                    name="ppg",
                    subject_id=subject_id,
                ),
                "bioz": _ensure_single_channel(
                    np.asarray(data["bioz"]),
                    name="bioz",
                    subject_id=subject_id,
                ),
            }

            # Extract sampling rates (handle scalar or 0-d array)
            fs_ppg = float(data["fs_ppg"])
            fs_bioz = float(data["fs_bioz"])
            sampling_rates = {
                "ppg": fs_ppg,
                "bioz": fs_bioz,
            }

            # Extract optional timestamps
            timestamps = None
            if "t" in available_fields or "t_step" in available_fields:
                timestamps = {}
                if "t" in available_fields:
                    timestamps["t"] = np.asarray(data["t"])
                if "t_step" in available_fields:
                    timestamps["t_step"] = np.asarray(data["t_step"])

            # Extract optional protocol levels
            protocol_levels = None
            if "step" in available_fields:
                protocol_levels = np.asarray(data["step"])

            # Extract optional metadata
            metadata: dict[str, Any] = {}
            if "metadata" in available_fields:
                meta_item = data["metadata"]
                if hasattr(meta_item, "item"):
                    metadata = dict(meta_item.item())
                else:
                    metadata = dict(meta_item)

            # Add file info to metadata
            metadata["source_file"] = str(npz_path)
            metadata["ppg_shape"] = signals["ppg"].shape
            metadata["bioz_shape"] = signals["bioz"].shape

        return SubjectData(
            subject_id=subject_id,
            signals=signals,
            sampling_rates=sampling_rates,
            timestamps=timestamps,
            protocol_levels=protocol_levels,
            metadata=metadata,
        )

    def protocol_levels(self, subject_id: str) -> Any | None:
        """
        Get protocol levels for a subject.

        Args:
            subject_id: Subject ID.

        Returns:
            Protocol levels array or None if not available.
        """
        data = self.load_subject(subject_id)
        return data.protocol_levels

    def describe(self) -> DataCard:
        """
        Generate a data card for this dataset.

        Returns:
            DataCard with dataset metadata.
        """
        subjects = self.list_subjects()

        subjects_with_steps: list[str] = []
        subjects_missing_steps: list[str] = []
        sample = None
        for subject_id in subjects:
            subject = self.load_subject(subject_id)
            if sample is None:
                sample = subject
            levels = subject.protocol_levels
            if levels is None or np.asarray(levels).size == 0:
                subjects_missing_steps.append(subject_id)
            else:
                subjects_with_steps.append(subject_id)

        if sample is None:
            raise ValueError("No subjects available to describe dataset")

        has_protocol = len(subjects_with_steps) > 0

        return DataCard(
            name=self.data_path.name,
            description=f"Local NPZ dataset with {len(subjects)} subjects",
            adapter_type="LocalNpzAdapter",
            source_path=str(self.data_path.absolute()),
            num_subjects=len(subjects),
            signals=list(sample.signals.keys()),
            sampling_rates=sample.sampling_rates,
            has_protocol_levels=has_protocol,
            subjects=subjects,
            subjects_with_steps=subjects_with_steps,
            subjects_missing_steps=subjects_missing_steps,
            metadata={
                "format": "npz",
                "sample_shapes": {k: v.shape for k, v in sample.signals.items()},
            },
        )
