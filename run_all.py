"""Run the full OpenCR pipeline end-to-end."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_command(args: list[str]) -> None:
    """Run a command and raise on failure."""
    print(f"[run_all] {' '.join(args)}")
    subprocess.run(args, check=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run full OpenCR pipeline.")
    parser.add_argument("data_dir", help="Path to raw dataset directory")
    parser.add_argument(
        "--run-dir",
        default="runs/run_all",
        help="Base output directory for this run",
    )
    parser.add_argument("--adapter", default="npz", help="Dataset adapter (default: npz)")
    parser.add_argument("--target", default="opencr", help="Target: opencr, ordinal, step")
    parser.add_argument("--model", default="random_forest", help="Model type")
    parser.add_argument("--n-estimators", type=int, default=100, help="Model estimators")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--n-boot", type=int, default=200, help="Bootstrap resamples")
    parser.add_argument("--window-sec", type=float, default=30.0, help="Window size (s)")
    parser.add_argument("--stride-sec", type=float, default=2.0, help="Stride size (s)")
    parser.add_argument("--resample-hz", type=float, default=None, help="Resample Hz")
    parser.add_argument("--sqi-threshold", type=float, default=0.5, help="SQI threshold")
    parser.add_argument(
        "--report-dir",
        default=None,
        help="Report output directory (default: <run-dir>/report)",
    )
    parser.add_argument(
        "--data-hash",
        default="hybrid",
        help="Dataset hash method: stable, metadata, content, hybrid, none",
    )
    parser.add_argument(
        "--data-hash-threshold-mb",
        type=float,
        default=200.0,
        help="Hybrid hash threshold in MB",
    )

    args = parser.parse_args()
    data_dir = Path(args.data_dir)
    run_dir = Path(args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)

    fetch_dir = run_dir / "fetch"
    preprocess_dir = run_dir / "preprocess"
    baseline_dir = run_dir / "baseline"
    report_dir = Path(args.report_dir) if args.report_dir else run_dir / "report"

    data_card_path = fetch_dir / "data_card.json"
    processed_dir = preprocess_dir / "processed"

    base_cmd = [sys.executable, "-m", "opencr"]

    run_command(
        base_cmd
        + [
            "data",
            "fetch",
            str(data_dir),
            "--output",
            str(fetch_dir),
            "--adapter",
            args.adapter,
        ]
    )

    preprocess_cmd = base_cmd + [
        "data",
        "preprocess",
        str(data_dir),
        "--output",
        str(preprocess_dir),
        "--window-sec",
        str(args.window_sec),
        "--stride-sec",
        str(args.stride_sec),
        "--sqi-threshold",
        str(args.sqi_threshold),
        "--data-card",
        str(data_card_path),
        "--data-hash",
        args.data_hash,
        "--data-hash-threshold-mb",
        str(args.data_hash_threshold_mb),
    ]
    if args.resample_hz is not None:
        preprocess_cmd.extend(["--resample-hz", str(args.resample_hz)])
    run_command(preprocess_cmd)

    run_command(
        base_cmd
        + [
            "baseline",
            "train",
            str(processed_dir),
            "--output",
            str(baseline_dir),
            "--cv",
            "loso",
            "--target",
            args.target,
            "--model",
            args.model,
            "--n-estimators",
            str(args.n_estimators),
            "--seed",
            str(args.seed),
            "--n-boot",
            str(args.n_boot),
            "--data-card",
            str(data_card_path),
        ]
    )

    run_command(base_cmd + ["baseline", "evaluate", str(baseline_dir)])

    run_command(
        base_cmd
        + [
            "report",
            "annexA",
            str(baseline_dir),
            "--output",
            str(report_dir),
        ]
    )

    print(f"[run_all] Completed. Outputs in {run_dir}")


if __name__ == "__main__":
    main()
