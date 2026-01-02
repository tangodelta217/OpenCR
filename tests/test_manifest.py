"""
Tests for dataset hash computation.
"""

import os
import shutil
from pathlib import Path

import numpy as np

from opencr.repro.manifest import compute_dataset_hash


def test_dataset_hash_stable_ignores_mtime(tmp_path: Path) -> None:
    """Stable hash should ignore mtime differences."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()

    data = np.arange(10, dtype=np.float64)
    np.savez(root_a / "subject001.npz", ppg=data, bioz=data, fs_ppg=100.0, fs_bioz=50.0)

    shutil.copyfile(root_a / "subject001.npz", root_b / "subject001.npz")
    os.utime(root_b / "subject001.npz", (1, 1))

    hash_a = compute_dataset_hash(root_a, method="stable")
    hash_b = compute_dataset_hash(root_b, method="stable")

    assert hash_a is not None
    assert hash_b is not None
    assert hash_a["value"] == hash_b["value"]


def test_dataset_hash_stable_vs_content_same_size_diff_content(tmp_path: Path) -> None:
    """Stable hash matches for same size, content hash differs."""
    root_a = tmp_path / "a"
    root_b = tmp_path / "b"
    root_a.mkdir()
    root_b.mkdir()

    path_a = root_a / "subject001.npz"
    path_b = root_b / "subject001.npz"
    path_a.write_bytes(b"a" * 1024)
    path_b.write_bytes(b"b" * 1024)

    hash_a_stable = compute_dataset_hash(root_a, method="stable")
    hash_b_stable = compute_dataset_hash(root_b, method="stable")
    hash_a_content = compute_dataset_hash(root_a, method="content")
    hash_b_content = compute_dataset_hash(root_b, method="content")

    assert hash_a_stable["value"] == hash_b_stable["value"]
    assert hash_a_content["value"] != hash_b_content["value"]


def test_dataset_hash_hybrid_threshold(tmp_path: Path) -> None:
    """Hybrid uses content below threshold and stable above."""
    root = tmp_path / "data"
    root.mkdir()
    (root / "subject001.npz").write_bytes(b"x" * 10)

    hash_stable = compute_dataset_hash(root, method="stable")
    hash_content = compute_dataset_hash(root, method="content")

    hash_hybrid_stable = compute_dataset_hash(root, method="hybrid", threshold_bytes=0)
    assert hash_hybrid_stable["effective_method"] == "stable"
    assert hash_hybrid_stable["value"] == hash_stable["value"]

    hash_hybrid_content = compute_dataset_hash(root, method="hybrid", threshold_bytes=100)
    assert hash_hybrid_content["effective_method"] == "content"
    assert hash_hybrid_content["value"] == hash_content["value"]
