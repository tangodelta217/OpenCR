"""Run manifest utilities for reproducible pipelines."""

from __future__ import annotations

import hashlib
import json
import shlex
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def get_git_commit_hash(cwd: Path | None = None) -> str | None:
    """Return the current git commit hash if available."""
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=cwd or Path.cwd(),
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    return result.stdout.strip() or None


def _format_command(argv: list[str]) -> str:
    """Render argv into a readable command string."""
    try:
        return shlex.join(argv)
    except AttributeError:
        return " ".join(argv)


def _hash_file_contents(path: Path, hasher: hashlib._Hash) -> None:
    """Hash file contents in chunks."""
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            hasher.update(chunk)


def compute_dataset_hash(
    root_dir: Path,
    method: str = "stable",
    pattern: str = "*.npz",
    *,
    threshold_bytes: int | None = None,
) -> dict[str, Any] | None:
    """
    Compute a reproducible dataset hash.

    Args:
        root_dir: Directory containing dataset files.
        method: "metadata", "stable", "content", "hybrid", or "none".
        pattern: File glob pattern to include.
        threshold_bytes: Size threshold for hybrid mode (bytes).

    Returns:
        Dictionary with hash metadata, or None if disabled.
    """
    method = method.lower()
    if method == "none":
        return None
    if method not in {"metadata", "stable", "content", "hybrid"}:
        raise ValueError(f"Unknown hash method: {method}")

    root_dir = Path(root_dir)
    files = sorted(root_dir.rglob(pattern))
    total_bytes = sum(path.stat().st_size for path in files)

    effective_method = method
    if method == "hybrid":
        if threshold_bytes is None:
            threshold_bytes = 200 * 1024 * 1024
        effective_method = "content" if total_bytes <= threshold_bytes else "stable"

    hasher = hashlib.sha256()
    for path in files:
        rel_path = str(path.relative_to(root_dir))
        if effective_method == "metadata":
            stat = path.stat()
            record = f"{rel_path}|{stat.st_size}|{stat.st_mtime_ns}"
            hasher.update(record.encode("utf-8"))
        elif effective_method == "stable":
            stat = path.stat()
            record = f"{rel_path}|{stat.st_size}"
            hasher.update(record.encode("utf-8"))
        else:
            hasher.update(rel_path.encode("utf-8"))
            _hash_file_contents(path, hasher)

    return {
        "method": method,
        "effective_method": effective_method,
        "threshold_bytes": threshold_bytes if method == "hybrid" else None,
        "total_bytes": total_bytes,
        "value": hasher.hexdigest(),
        "n_files": len(files),
        "pattern": pattern,
        "root": str(root_dir),
    }


def build_manifest(
    *,
    stage: str,
    seed: int | None,
    command_args: list[str] | None = None,
    data_card_path: Path | None = None,
    dataset_hash: dict[str, Any] | None = None,
    config: dict[str, Any] | None = None,
    splits: list[dict[str, Any]] | None = None,
    artifacts: dict[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a run manifest dictionary."""
    argv = command_args or sys.argv
    manifest: dict[str, Any] = {
        "schema_version": "1.0",
        "stage": stage,
        "timestamp": datetime.now(UTC).isoformat(),
        "git_commit": get_git_commit_hash(),
        "command": {
            "argv": list(argv),
            "string": _format_command(list(argv)),
            "cwd": str(Path.cwd()),
        },
        "seed": seed,
        "data_card_path": str(data_card_path) if data_card_path else None,
        "dataset_hash": dataset_hash,
        "config": config or {},
        "splits": splits or [],
        "artifacts": artifacts or {},
    }

    if extra:
        manifest.update(extra)

    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> Path:
    """Write manifest JSON to disk."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    return path


def load_manifest(path: Path) -> dict[str, Any] | None:
    """Load a manifest if it exists."""
    path = Path(path)
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)
