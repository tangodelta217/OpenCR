"""Dataset adapters for loading different data formats."""

from opencr.data.adapters.base import DatasetAdapter
from opencr.data.adapters.local_npz import LocalNpzAdapter

__all__ = ["DatasetAdapter", "LocalNpzAdapter"]
