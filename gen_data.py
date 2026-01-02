
"""Generate a synthetic dataset compatible with the LocalNpzAdapter."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

DEFAULT_LEVELS = [0.0, -15.0, -30.0, -45.0, -60.0]
DEFAULT_DIRECTION = "more_severe_lower"


def _parse_levels(raw: str) -> list[float]:
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    if not parts:
        raise ValueError("levels must be a non-empty comma-separated list")
    return [float(p) for p in parts]


def _build_step_array(n_samples: int, levels: list[float]) -> np.ndarray:
    if n_samples <= 0:
        raise ValueError("n_samples must be positive")
    if not levels:
        raise ValueError("levels must be non-empty")
    step = np.empty(n_samples, dtype=np.float64)
    samples_per_level = n_samples // len(levels)
    start = 0
    for idx, level in enumerate(levels):
        end = start + samples_per_level
        if idx == len(levels) - 1:
            end = n_samples
        step[start:end] = level
        start = end
    return step


def _severity_from_steps(steps: np.ndarray, direction: str) -> np.ndarray:
    min_level = float(np.min(steps))
    max_level = float(np.max(steps))
    if max_level == min_level:
        return np.zeros_like(steps, dtype=np.float64)
    if direction == "more_severe_lower":
        return (max_level - steps) / (max_level - min_level)
    if direction == "more_severe_higher":
        return (steps - min_level) / (max_level - min_level)
    raise ValueError("direction must be more_severe_lower or more_severe_higher")


def _simulate_signals(
    t: np.ndarray,
    steps: np.ndarray,
    fs: float,
    *,
    direction: str,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    severity = _severity_from_steps(steps, direction)

    base_hr_hz = rng.uniform(1.0, 1.4)
    hr_gain = 0.5
    hr_hz = base_hr_hz + hr_gain * severity
    phase = np.cumsum(2.0 * np.pi * hr_hz / fs)

    ppg_amp = (1.0 - 0.3 * severity) * rng.uniform(0.9, 1.1)
    ppg = ppg_amp * (np.sin(phase) + 0.25 * np.sin(2.0 * phase + 0.2))
    ppg += 0.05 * np.sin(2.0 * np.pi * 0.05 * t + rng.uniform(0, 2.0 * np.pi))
    ppg += 0.05 * rng.standard_normal(t.size)

    resp_hz = rng.uniform(0.18, 0.25)
    resp_phase = 2.0 * np.pi * resp_hz * t + rng.uniform(0, 2.0 * np.pi)
    bioz_amp = (1.0 + 0.2 * severity) * rng.uniform(0.9, 1.1)
    bioz = bioz_amp * (0.7 * np.sin(resp_phase) + 0.1 * np.sin(phase + 0.3))
    bioz += 0.03 * np.sin(2.0 * np.pi * 0.01 * t + rng.uniform(0, 2.0 * np.pi))
    bioz += 0.03 * rng.standard_normal(t.size)

    return ppg.astype(np.float32), bioz.astype(np.float32)


def generate_dataset(
    output_dir: Path,
    *,
    n_subjects: int = 3,
    duration_sec: float = 600.0,
    fs: float = 100.0,
    levels: list[float] | None = None,
    direction: str = DEFAULT_DIRECTION,
    seed: int = 123,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if n_subjects <= 0:
        raise ValueError("n_subjects must be positive")
    if duration_sec <= 0:
        raise ValueError("duration_sec must be positive")
    if fs <= 0:
        raise ValueError("fs must be positive")

    levels = levels or DEFAULT_LEVELS
    if direction not in {"more_severe_lower", "more_severe_higher"}:
        raise ValueError("direction must be more_severe_lower or more_severe_higher")

    protocol = {"levels": levels, "direction": direction}
    with open(output_dir / "protocol.json", "w", encoding="utf-8") as f:
        json.dump(protocol, f, indent=2)

    n_samples = int(round(duration_sec * fs))
    t = np.arange(n_samples, dtype=np.float64) / fs
    steps = _build_step_array(n_samples, levels)

    for idx in range(n_subjects):
        subject_id = f"S{idx + 1:03d}"
        rng = np.random.default_rng(seed + idx)
        ppg, bioz = _simulate_signals(t, steps, fs, direction=direction, rng=rng)
        np.savez(
            output_dir / f"{subject_id}.npz",
            ppg=ppg,
            bioz=bioz,
            t=t,
            step=steps,
            fs_ppg=float(fs),
            fs_bioz=float(fs),
            metadata={"subject_id": subject_id, "seed": int(seed + idx)},
        )

    print(f"Created synthetic dataset in {output_dir}")
    print(f"Subjects: {n_subjects}, Duration: {duration_sec:.1f}s, Fs: {fs:.1f} Hz")


def create_synthetic_data(output_dir: Path) -> None:
    generate_dataset(output_dir)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate synthetic OpenCR dataset.")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/raw"),
        help="Output directory for synthetic dataset",
    )
    parser.add_argument(
        "--n-subjects",
        type=int,
        default=3,
        help="Number of subjects to generate",
    )
    parser.add_argument(
        "--duration-sec",
        type=float,
        default=600.0,
        help="Total duration per subject in seconds",
    )
    parser.add_argument(
        "--fs",
        type=float,
        default=100.0,
        help="Sampling rate for PPG and BioZ (Hz)",
    )
    parser.add_argument(
        "--levels",
        type=str,
        default="0,-15,-30,-45,-60",
        help="Comma-separated protocol levels",
    )
    parser.add_argument(
        "--direction",
        type=str,
        default=DEFAULT_DIRECTION,
        help="Protocol direction: more_severe_lower or more_severe_higher",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=123,
        help="Random seed for reproducibility",
    )

    args = parser.parse_args()
    levels = _parse_levels(args.levels)

    generate_dataset(
        args.output,
        n_subjects=args.n_subjects,
        duration_sec=args.duration_sec,
        fs=args.fs,
        levels=levels,
        direction=args.direction,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
