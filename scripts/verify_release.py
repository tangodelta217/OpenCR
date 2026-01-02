#!/usr/bin/env python
"""
Run release verification gates in fail-fast mode.

Usage:
  python scripts/verify_release.py [--e2e]
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _run(title: str, command: list[str]) -> int:
    print(f"[RUN] {title}")
    result = subprocess.run(command, cwd=ROOT)
    if result.returncode != 0:
        print(f"[FAIL] {title} (exit {result.returncode})")
        return result.returncode
    print(f"[PASS] {title}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run release verification gates (fail-fast)."
    )
    parser.add_argument(
        "--e2e",
        action="store_true",
        help="Run the end-to-end demo and edge checks after core gates.",
    )
    args = parser.parse_args()

    python = sys.executable
    gates: list[tuple[str, list[str]]] = [
        ("install dev dependencies", [python, "-m", "pip", "install", "-e", ".[dev]"]),
        ("pytest", [python, "-m", "pytest"]),
        ("ruff", [python, "-m", "ruff", "check", "src", "tests"]),
        ("black", [python, "-m", "black", "--check", "src", "tests"]),
        ("repo audit", [python, "scripts/repo_audit.py", "--strict"]),
    ]

    for title, command in gates:
        rc = _run(title, command)
        if rc != 0:
            return rc

    if args.e2e:
        e2e_gates: list[tuple[str, list[str]]] = [
            (
                "gen_data demo",
                [
                    python,
                    "gen_data.py",
                    "--output",
                    "data/_demo",
                    "--n-subjects",
                    "4",
                    "--duration-sec",
                    "120",
                    "--fs",
                    "100",
                    "--seed",
                    "123",
                ],
            ),
            (
                "run_all demo",
                [
                    python,
                    "run_all.py",
                    "data/_demo",
                    "--run-dir",
                    "runs/_demo_run",
                    "--report-dir",
                    "runs/_demo_run/report",
                ],
            ),
            (
                "edge export",
                [
                    python,
                    "-m",
                    "opencr",
                    "edge",
                    "export",
                    "--run",
                    "runs/_demo_run/baseline",
                    "--out",
                    "runs/_demo_run/edge",
                    "--format",
                    "auto",
                ],
            ),
            (
                "edge benchmark",
                [
                    python,
                    "-m",
                    "opencr",
                    "edge",
                    "benchmark",
                    "--run",
                    "runs/_demo_run/baseline",
                    "--out",
                    "runs/_demo_run/edge",
                ],
            ),
        ]
        for title, command in e2e_gates:
            rc = _run(title, command)
            if rc != 0:
                return rc

    print("Release verification PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
