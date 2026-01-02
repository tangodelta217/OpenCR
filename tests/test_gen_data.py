"""Smoke test for synthetic dataset generation and preprocessing."""

import subprocess
import sys
from pathlib import Path

from gen_data import generate_dataset


def test_gen_data_fetch_preprocess_smoke(tmp_path: Path) -> None:
    data_dir = tmp_path / "synthetic"
    generate_dataset(
        data_dir,
        n_subjects=2,
        duration_sec=90.0,
        fs=50.0,
        levels=[0.0, -15.0, -30.0],
        direction="more_severe_lower",
        seed=7,
    )

    assert (data_dir / "protocol.json").exists()

    fetch_dir = tmp_path / "fetch"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opencr",
            "data",
            "fetch",
            str(data_dir),
            "--output",
            str(fetch_dir),
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (fetch_dir / "data_card.json").exists()

    preprocess_dir = tmp_path / "preprocess"
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "opencr",
            "data",
            "preprocess",
            str(data_dir),
            "--output",
            str(preprocess_dir),
            "--window-sec",
            "10",
            "--stride-sec",
            "5",
            "--filter-low",
            "0",
            "--filter-high",
            "0",
            "--bioz-filter-low",
            "0",
            "--bioz-filter-high",
            "0",
            "--data-hash",
            "none",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
    assert (preprocess_dir / "processed").exists()
    assert (preprocess_dir / "manifest.json").exists()
