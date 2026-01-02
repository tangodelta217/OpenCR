"""
Local NPZ Dataset Adapter.

Loads datasets stored as .npz files, one per subject.
Expected structure:
    data_dir/
        subject001.npz
        subject002.npz
        ...

Each .npz file must contain:
    - ppg: PPG signal array
    - bioz: Bioimpedance signal array
    - fs_ppg: PPG sampling rate (scalar)
    - fs_bioz: Bioimpedance sampling rate (scalar)

Optional fields:
    - t: Timestamps array (aligned with longest signal)
    - step: Protocol step/level array (aligned in time)
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
OPTIONAL_FIELDS = {"t", "step", "metadata"}


class LocalNpzAdapter(DatasetAdapter):
    """
    Adapter for local .npz format datasets.

    Expects a directory containing one .npz file per subject.
    Each file must contain PPG and BioZ signals with their sampling rates.

    Example:
        adapter = LocalNpzAdapter(Path("data/raw"))
        subjects = adapter.list_subjects()  # ["subject001", "subject002", ...]
        data = adapter.load_subject("subject001")
        print(data.signals["ppg"].shape)  # (N,) or (N, channels)
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
                "ppg": np.asarray(data["ppg"]),
                "bioz": np.asarray(data["bioz"]),
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
            if "t" in available_fields:
                timestamps = {"t": np.asarray(data["t"])}

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

        # Sample first subject to get signal info
        sample = self.load_subject(subjects[0])

        # Check all subjects for protocol levels
        has_protocol = sample.protocol_levels is not None

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
            metadata={
                "format": "npz",
                "sample_shapes": {k: v.shape for k, v in sample.signals.items()},
            },
        )
