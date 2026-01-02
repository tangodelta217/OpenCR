"""Metrics summary report generation."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from opencr.logging import get_logger

logger = get_logger(__name__)


def generate_metrics_summary(run_dir: Path, output_dir: Path) -> dict[str, Any]:
    """Generate summary tables from results.json."""
    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}")

    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    aggregated = results.get("aggregated_metrics", {})
    metrics_ci = results.get("metrics_ci", {})
    target = results.get("target", "unknown")
    task_type = results.get("task_type", "unknown")
    folds = results.get("folds", [])

    csv_path = output_dir / "summary.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Target", target])
        writer.writerow(["Task", task_type])
        writer.writerow(["N_Folds", len(folds)])
        writer.writerow([])
        writer.writerow(["Metric", "Mean", "Std", "Min", "Max", "CI_Low", "CI_High"])
        for metric_name, stats in aggregated.items():
            if isinstance(stats, dict) and "mean" in stats:
                ci = metrics_ci.get(metric_name, {})
                ci_low = f"{ci['low']:.4f}" if "low" in ci else "N/A"
                ci_high = f"{ci['high']:.4f}" if "high" in ci else "N/A"
                writer.writerow(
                    [
                        metric_name,
                        f"{stats['mean']:.4f}",
                        f"{stats['std']:.4f}",
                        f"{stats['min']:.4f}",
                        f"{stats['max']:.4f}",
                        ci_low,
                        ci_high,
                    ]
                )

    md_path = output_dir / "summary.md"
    lines = [
        "# Metrics Summary",
        "",
        f"Target: {target}",
        f"Task: {task_type}",
        f"Folds: {len(folds)}",
        "",
        "| Metric | Mean | Std | Min | Max | CI Low | CI High |",
        "|--------|------|-----|-----|-----|--------|---------|",
    ]

    for metric_name, stats in aggregated.items():
        if isinstance(stats, dict) and "mean" in stats:
            ci = metrics_ci.get(metric_name, {})
            ci_low = f"{ci['low']:.4f}" if "low" in ci else "N/A"
            ci_high = f"{ci['high']:.4f}" if "high" in ci else "N/A"
            lines.append(
                f"| {metric_name} | {stats['mean']:.4f} | {stats['std']:.4f} | "
                f"{stats['min']:.4f} | {stats['max']:.4f} | {ci_low} | {ci_high} |"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Generated metrics summary: {csv_path}, {md_path}")
    return {
        "tables": [str(csv_path), str(md_path)],
        "output_dir": str(output_dir),
    }
