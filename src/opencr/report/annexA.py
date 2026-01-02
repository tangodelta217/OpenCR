"""
Annex A Report Generator.

Generates tables and figures for the TFM Annex A:
- Summary table (markdown and CSV)
- ROC/PR curves (classification)
- Temporal fuel gauge visualization
"""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from opencr.logging import get_logger

logger = get_logger(__name__)

# Optional matplotlib import
try:
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend for headless environments
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not installed, figure generation disabled")


@dataclass
class AnnexAConfig:
    """Configuration for Annex A report generation."""

    dpi: int = 150
    figsize: tuple[int, int] = (10, 6)
    style: str = "seaborn-v0_8-whitegrid"


def generate_annex_a(
    run_dir: Path,
    output_dir: Path,
    config: AnnexAConfig | None = None,
) -> dict[str, Any]:
    """
    Generate Annex A report from a baseline run.

    Args:
        run_dir: Path to baseline run directory.
        output_dir: Output directory for report.
        config: Report configuration.

    Returns:
        Dictionary with paths to generated files.
    """
    if config is None:
        config = AnnexAConfig()

    run_dir = Path(run_dir)
    output_dir = Path(output_dir)

    # Create output directories
    figures_dir = output_dir / "figures" / "annexA"
    tables_dir = output_dir / "tables" / "annexA"
    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)

    # Load results
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}")

    with open(results_path) as f:
        results = json.load(f)

    generated_files: dict[str, list[Path]] = {"tables": [], "figures": []}

    # Generate summary table
    table_path = _generate_summary_table(results, tables_dir)
    generated_files["tables"].append(table_path)

    # Generate markdown table
    md_path = _generate_markdown_table(results, tables_dir)
    generated_files["tables"].append(md_path)

    if HAS_MATPLOTLIB:
        # Generate ROC curve
        roc_path = _generate_roc_curve(results, run_dir, figures_dir, config)
        if roc_path:
            generated_files["figures"].append(roc_path)

        # Generate fuel gauge visualization
        gauge_path = _generate_fuel_gauge(results, run_dir, figures_dir, config)
        if gauge_path:
            generated_files["figures"].append(gauge_path)

        # Generate metrics bar plot
        bar_path = _generate_metrics_barplot(results, figures_dir, config)
        if bar_path:
            generated_files["figures"].append(bar_path)

    logger.info(
        f"Generated {len(generated_files['tables'])} tables, "
        f"{len(generated_files['figures'])} figures"
    )

    return {
        "tables": [str(p) for p in generated_files["tables"]],
        "figures": [str(p) for p in generated_files["figures"]],
        "output_dir": str(output_dir),
    }


def _generate_summary_table(results: dict, output_dir: Path) -> Path:
    """Generate CSV summary table."""
    import csv

    csv_path = output_dir / "summary.csv"

    folds = results.get("folds", [])
    aggregated = results.get("aggregated_metrics", {})

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        # Header
        writer.writerow(["Fold", "Subject", "N_Train", "N_Test", "Accuracy", "F1", "AUC"])

        # Fold rows
        for fold in folds:
            m = fold["metrics"]
            auc = m.get("auc_roc")
            writer.writerow(
                [
                    fold["fold_idx"],
                    fold["test_subject"],
                    fold["n_train"],
                    fold["n_test"],
                    f"{m['accuracy']:.4f}",
                    f"{m['f1']:.4f}",
                    f"{auc:.4f}" if auc else "N/A",
                ]
            )

        # Aggregated row
        writer.writerow([])
        writer.writerow(["Metric", "Mean", "Std", "Min", "Max"])
        for metric_name, stats in aggregated.items():
            if isinstance(stats, dict) and "mean" in stats:
                writer.writerow(
                    [
                        metric_name,
                        f"{stats['mean']:.4f}",
                        f"{stats['std']:.4f}",
                        f"{stats['min']:.4f}",
                        f"{stats['max']:.4f}",
                    ]
                )

    logger.info(f"Generated summary table: {csv_path}")
    return csv_path


def _generate_markdown_table(results: dict, output_dir: Path) -> Path:
    """Generate Markdown summary table."""
    md_path = output_dir / "summary.md"

    folds = results.get("folds", [])
    aggregated = results.get("aggregated_metrics", {})

    lines = [
        "# Annex A: LOSO Cross-Validation Results",
        "",
        "## Per-Fold Results",
        "",
        "| Fold | Subject | N_Train | N_Test | Accuracy | F1 | AUC |",
        "|------|---------|---------|--------|----------|-----|-----|",
    ]

    for fold in folds:
        m = fold["metrics"]
        auc = m.get("auc_roc")
        auc_str = f"{auc:.4f}" if auc else "N/A"
        lines.append(
            f"| {fold['fold_idx']} | {fold['test_subject']} | "
            f"{fold['n_train']} | {fold['n_test']} | "
            f"{m['accuracy']:.4f} | {m['f1']:.4f} | {auc_str} |"
        )

    lines.extend(
        [
            "",
            "## Aggregated Metrics",
            "",
            "| Metric | Mean | Std | Min | Max |",
            "|--------|------|-----|-----|-----|",
        ]
    )

    for metric_name, stats in aggregated.items():
        if isinstance(stats, dict) and "mean" in stats:
            lines.append(
                f"| {metric_name} | {stats['mean']:.4f} | "
                f"{stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} |"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Generated markdown table: {md_path}")
    return md_path


def _generate_roc_curve(
    results: dict,
    run_dir: Path,
    output_dir: Path,
    config: AnnexAConfig,
) -> Path | None:
    """Generate ROC curve from predictions."""
    from sklearn.metrics import auc, roc_curve

    try:
        plt.style.use(config.style)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=config.figsize)

    folds = results.get("folds", [])
    colors = plt.cm.tab10.colors

    all_tprs = []
    mean_fpr = np.linspace(0, 1, 100)
    aucs = []

    for i, fold in enumerate(folds):
        pred_path = run_dir / f"fold_{fold['fold_idx']:02d}" / "predictions.npz"

        if not pred_path.exists():
            continue

        with np.load(pred_path, allow_pickle=True) as data:
            y_true = data["y_true"]
            y_proba = data["y_proba"]

            if len(y_proba) == 0:
                continue

            # Handle multi-class by using one-vs-rest for first class
            if y_proba.ndim == 2 and y_proba.shape[1] > 1:
                # Use positive class probability
                y_score = y_proba[:, 1] if y_proba.shape[1] == 2 else y_proba.max(axis=1)
                y_binary = (y_true == y_true.max()).astype(int)
            else:
                y_score = y_proba.ravel()
                y_binary = y_true

            try:
                fpr, tpr, _ = roc_curve(y_binary, y_score)
                roc_auc = auc(fpr, tpr)
                aucs.append(roc_auc)

                # Interpolate for mean
                interp_tpr = np.interp(mean_fpr, fpr, tpr)
                interp_tpr[0] = 0.0
                all_tprs.append(interp_tpr)

                ax.plot(
                    fpr,
                    tpr,
                    color=colors[i % len(colors)],
                    alpha=0.3,
                    lw=1,
                    label=f"Fold {i} (AUC={roc_auc:.2f})",
                )
            except Exception as e:
                logger.warning(f"Could not compute ROC for fold {i}: {e}")

    if not all_tprs:
        plt.close(fig)
        return None

    # Mean ROC
    mean_tpr = np.mean(all_tprs, axis=0)
    mean_tpr[-1] = 1.0
    mean_auc = np.mean(aucs)
    std_auc = np.std(aucs)

    ax.plot(
        mean_fpr,
        mean_tpr,
        color="blue",
        lw=2,
        label=f"Mean ROC (AUC={mean_auc:.2f} ± {std_auc:.2f})",
    )

    # Confidence interval
    std_tpr = np.std(all_tprs, axis=0)
    ax.fill_between(
        mean_fpr,
        np.maximum(mean_tpr - std_tpr, 0),
        np.minimum(mean_tpr + std_tpr, 1),
        color="blue",
        alpha=0.2,
    )

    # Diagonal
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random")

    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curves - LOSO Cross-Validation", fontsize=14)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(True, alpha=0.3)

    roc_path = output_dir / "roc_curves.png"
    fig.tight_layout()
    fig.savefig(roc_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Generated ROC curve: {roc_path}")
    return roc_path


def _generate_fuel_gauge(
    results: dict,
    run_dir: Path,
    output_dir: Path,
    config: AnnexAConfig,
) -> Path | None:
    """Generate fuel gauge temporal visualization."""
    folds = results.get("folds", [])

    if not folds:
        return None

    # Use first fold as example
    fold = folds[0]
    pred_path = run_dir / f"fold_{fold['fold_idx']:02d}" / "predictions.npz"

    if not pred_path.exists():
        return None

    with np.load(pred_path, allow_pickle=True) as data:
        y_true = data["y_true"]
        y_pred = data["y_pred"]

    if len(y_true) == 0:
        return None

    try:
        plt.style.use(config.style)
    except Exception:
        pass

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)

    n_samples = len(y_true)
    x = np.arange(n_samples)

    # Top: True vs Predicted
    ax1 = axes[0]
    ax1.plot(x, y_true, "b-", lw=1.5, label="True", alpha=0.8)
    ax1.plot(x, y_pred, "r--", lw=1.5, label="Predicted", alpha=0.8)
    ax1.set_ylabel("Class / Level", fontsize=12)
    ax1.set_title(f"Temporal Prediction - Subject: {fold['test_subject']}", fontsize=14)
    ax1.legend(loc="upper right")
    ax1.grid(True, alpha=0.3)

    # Bottom: Fuel gauge style (agreement indicator)
    ax2 = axes[1]

    # Compute agreement
    agreement = (y_true == y_pred).astype(float)

    # Moving average for smooth gauge
    window = min(10, max(1, n_samples // 20))
    if window > 1:
        gauge = np.convolve(agreement, np.ones(window) / window, mode="same")
    else:
        gauge = agreement

    # Color by agreement level
    colors = plt.cm.RdYlGn(gauge)

    for i in range(n_samples - 1):
        ax2.fill_between(
            [x[i], x[i + 1]],
            0,
            gauge[i],
            color=colors[i],
            alpha=0.8,
        )

    ax2.axhline(y=0.5, color="orange", linestyle="--", lw=1, alpha=0.7, label="50% threshold")
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("Agreement Score", fontsize=12)
    ax2.set_xlabel("Window Index", fontsize=12)
    ax2.set_title("Prediction Agreement (Fuel Gauge)", fontsize=14)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.3)

    gauge_path = output_dir / "fuel_gauge.png"
    fig.tight_layout()
    fig.savefig(gauge_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Generated fuel gauge: {gauge_path}")
    return gauge_path


def _generate_metrics_barplot(
    results: dict,
    output_dir: Path,
    config: AnnexAConfig,
) -> Path | None:
    """Generate bar plot of aggregated metrics."""
    aggregated = results.get("aggregated_metrics", {})

    if not aggregated:
        return None

    try:
        plt.style.use(config.style)
    except Exception:
        pass

    metrics = []
    means = []
    stds = []

    for name, stats in aggregated.items():
        if isinstance(stats, dict) and "mean" in stats:
            metrics.append(name.replace("_", " ").title())
            means.append(stats["mean"])
            stds.append(stats["std"])

    if not metrics:
        return None

    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(metrics))
    bars = ax.bar(x, means, yerr=stds, capsize=5, color="steelblue", alpha=0.8)

    ax.set_ylabel("Score", fontsize=12)
    ax.set_xlabel("Metric", fontsize=12)
    ax.set_title("Aggregated Metrics - LOSO Cross-Validation", fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=45, ha="right")
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis="y", alpha=0.3)

    # Add value labels
    for bar, mean, std in zip(bars, means, stds, strict=False):
        height = bar.get_height()
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            height + std + 0.02,
            f"{mean:.3f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    bar_path = output_dir / "metrics_summary.png"
    fig.tight_layout()
    fig.savefig(bar_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Generated metrics bar plot: {bar_path}")
    return bar_path
