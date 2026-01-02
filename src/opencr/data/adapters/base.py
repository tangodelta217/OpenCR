"""
Base Dataset Adapter - Abstract interface for dataset loading.

All dataset adapters must implement this interface to ensure compatibility
with the OpenCR pipeline. This enables labs to use their own datasets without
modifying the core codebase.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class SubjectData:
    """
    Container for a single subject's data.

    Attributes:
        subject_id: Unique identifier for the subject.
        signals: Dictionary mapping signal names to numpy arrays.
                 Example: {"ppg": np.array(...), "bioz": np.array(...)}
        sampling_rates: Dictionary mapping signal names to their sampling rates (Hz).
                        Example: {"ppg": 100.0, "bioz": 50.0}
        timestamps: Optional dictionary mapping signal names or "t"/"t_step"
            to timestamp arrays.
        protocol_levels: Optional array of protocol levels/steps aligned in time.
        metadata: Additional metadata about the subject or recording.
    """

    subject_id: str
    signals: dict[str, Any]  # signal_name -> numpy array
    sampling_rates: dict[str, float]  # signal_name -> Hz
    timestamps: dict[str, Any] | None = None  # signal_name -> timestamp array
    protocol_levels: Any | None = None  # Protocol steps aligned in time
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class DataCard:
    """
    Data card describing a dataset.

    Provides metadata for reproducibility and documentation.
    """

    name: str
    description: str
    adapter_type: str
    source_path: str
    num_subjects: int
    signals: list[str]
    sampling_rates: dict[str, float]
    has_protocol_levels: bool
    subjects: list[str]
    subjects_with_steps: list[str] = field(default_factory=list)
    subjects_missing_steps: list[str] = field(default_factory=list)
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "adapter_type": self.adapter_type,
            "source_path": self.source_path,
            "num_subjects": self.num_subjects,
            "signals": self.signals,
            "sampling_rates": self.sampling_rates,
            "has_protocol_levels": self.has_protocol_levels,
            "subjects": self.subjects,
            "subjects_with_steps": self.subjects_with_steps,
            "subjects_missing_steps": self.subjects_missing_steps,
            "metadata": self.metadata,
        }


class DatasetAdapter(ABC):
    """
    Abstract base class for dataset adapters.

    Implement this interface to integrate a new dataset format with OpenCR.
    This enables any lab to use their own data without modifying core code.

    Example usage:
        adapter = LocalNpzAdapter(Path("data/raw"))
        subjects = adapter.list_subjects()
        data = adapter.load_subject(subjects[0])
        card = adapter.describe()
    """

    def __init__(self, data_path: Path) -> None:
        """
        Initialize adapter with path to data.

        Args:
            data_path: Path to the dataset directory.

        Raises:
            FileNotFoundError: If data_path does not exist.
            ValueError: If data_path is not a valid dataset.
        """
        self.data_path = Path(data_path)
        if not self.data_path.exists():
            raise FileNotFoundError(f"Data path not found: {self.data_path}")
        if not self.data_path.is_dir():
            raise ValueError(f"Data path must be a directory: {self.data_path}")

    @abstractmethod
    def list_subjects(self) -> list[str]:
        """
        List all available subject IDs in the dataset.

        Returns:
            List of subject identifiers as strings.

        Raises:
            ValueError: If no valid subjects are found.
        """
        ...

    @abstractmethod
    def load_subject(self, subject_id: str) -> SubjectData:
        """
        Load all data for a single subject.

        Args:
            subject_id: Unique identifier for the subject.

        Returns:
            SubjectData containing signals, sampling rates, and metadata.

        Raises:
            KeyError: If subject_id is not found.
            ValueError: If subject data is malformed.
        """
        ...

    @abstractmethod
    def protocol_levels(self, subject_id: str) -> Any | None:
        """
        Get protocol levels/steps for a subject (if available).

        Protocol levels represent different exercise intensities or experimental
        conditions aligned with the signal timestamps.

        Args:
            subject_id: Unique identifier for the subject.

        Returns:
            Array of protocol levels aligned in time, or None if not available.

        Raises:
            KeyError: If subject_id is not found.
        """
        ...

    @abstractmethod
    def describe(self) -> DataCard:
        """
        Generate a data card describing the dataset.

        Returns:
            DataCard with metadata for reproducibility and documentation.
        """
        ...

    def validate(self) -> tuple[bool, list[str]]:
        """
        Validate the dataset structure.

        Returns:
            Tuple of (is_valid, list of error messages).
        """
        errors: list[str] = []

        try:
            subjects = self.list_subjects()
            if not subjects:
                errors.append("No subjects found in dataset")
        except Exception as e:
            errors.append(f"Failed to list subjects: {e}")
            return False, errors

        # Validate first subject as sample
        try:
            sample = self.load_subject(subjects[0])
            if not sample.signals:
                errors.append(f"Subject {subjects[0]} has no signals")
            if not sample.sampling_rates:
                errors.append(f"Subject {subjects[0]} has no sampling rates")
        except Exception as e:
            errors.append(f"Failed to load sample subject {subjects[0]}: {e}")

        return len(errors) == 0, errors
