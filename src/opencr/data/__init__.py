"""OpenCR Data Module - Dataset adapters and preprocessing utilities."""

from opencr.data.adapters.base import DatasetAdapter
from opencr.data.adapters.local_npz import LocalNpzAdapter

__all__ = ["DatasetAdapter", "LocalNpzAdapter"]
