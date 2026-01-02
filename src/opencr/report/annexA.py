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
    task_type = results.get("task_type", "classification")

    # Generate summary table
    table_path = _generate_summary_table(results, tables_dir)
    generated_files["tables"].append(table_path)

    # Generate markdown table
    md_path = _generate_markdown_table(results, tables_dir)
    generated_files["tables"].append(md_path)

    if HAS_MATPLOTLIB:
        if task_type == "regression":
            scatter_path = _generate_regression_scatter(results, run_dir, figures_dir, config)
            if scatter_path:
                generated_files["figures"].append(scatter_path)

            error_path = _generate_regression_error(results, run_dir, figures_dir, config)
            if error_path:
                generated_files["figures"].append(error_path)
        else:
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
    metrics_ci = results.get("metrics_ci", {})
    target = results.get("target", "unknown")
    task_type = results.get("task_type", "unknown")

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)

        writer.writerow(["Target", target])
        writer.writerow(["Task", task_type])
        writer.writerow([])

        if task_type == "regression":
            writer.writerow(["Fold", "Subject", "N_Train", "N_Test", "RMSE", "MAE", "R2"])
        else:
            writer.writerow(["Fold", "Subject", "N_Train", "N_Test", "Accuracy", "F1", "AUC"])

        # Fold rows
        for fold in folds:
            m = fold["metrics"]
            if task_type == "regression":
                writer.writerow(
                    [
                        fold["fold_idx"],
                        fold["test_subject"],
                        fold["n_train"],
                        fold["n_test"],
                        f"{m['rmse']:.4f}",
                        f"{m['mae']:.4f}",
                        f"{m['r2']:.4f}",
                    ]
                )
            else:
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

    logger.info(f"Generated summary table: {csv_path}")
    return csv_path


def _generate_markdown_table(results: dict, output_dir: Path) -> Path:
    """Generate Markdown summary table."""
    md_path = output_dir / "summary.md"

    folds = results.get("folds", [])
    aggregated = results.get("aggregated_metrics", {})
    metrics_ci = results.get("metrics_ci", {})
    target = results.get("target", "unknown")
    task_type = results.get("task_type", "unknown")

    if task_type == "regression":
        header = "| Fold | Subject | N_Train | N_Test | RMSE | MAE | R2 |"
        divider = "|------|---------|---------|--------|------|-----|----|"
    else:
        header = "| Fold | Subject | N_Train | N_Test | Accuracy | F1 | AUC |"
        divider = "|------|---------|---------|--------|----------|-----|-----|"

    lines = [
        "# Annex A: LOSO Cross-Validation Results",
        "",
        f"Target: {target}",
        f"Task: {task_type}",
        "",
        "## Per-Fold Results",
        "",
        header,
        divider,
    ]

    for fold in folds:
        m = fold["metrics"]
        if task_type == "regression":
            lines.append(
                f"| {fold['fold_idx']} | {fold['test_subject']} | "
                f"{fold['n_train']} | {fold['n_test']} | "
                f"{m['rmse']:.4f} | {m['mae']:.4f} | {m['r2']:.4f} |"
            )
        else:
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
            "| Metric | Mean | Std | Min | Max | CI Low | CI High |",
            "|--------|------|-----|-----|-----|--------|---------|",
        ]
    )

    for metric_name, stats in aggregated.items():
        if isinstance(stats, dict) and "mean" in stats:
            ci = metrics_ci.get(metric_name, {})
            ci_low = f"{ci['low']:.4f}" if "low" in ci else "N/A"
            ci_high = f"{ci['high']:.4f}" if "high" in ci else "N/A"
            lines.append(
                f"| {metric_name} | {stats['mean']:.4f} | "
                f"{stats['std']:.4f} | {stats['min']:.4f} | {stats['max']:.4f} | "
                f"{ci_low} | {ci_high} |"
            )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info(f"Generated markdown table: {md_path}")
    return md_path


def _compute_auc(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    *,
    class_order: np.ndarray | list | None = None,
) -> tuple[float | None, tuple[np.ndarray, np.ndarray] | None]:
    """Compute AUC for binary or multiclass classification."""
    from sklearn.metrics import roc_auc_score

    if y_proba is None or len(y_proba) == 0:
        return None, None

    y_proba_arr = np.asarray(y_proba)
    classes = np.unique(y_true)

    if class_order is not None and y_proba_arr.ndim == 2:
        order = np.asarray(class_order)
        if y_proba_arr.shape[1] == order.size:
            try:
                indices = [int(np.where(order == cls)[0][0]) for cls in classes]
                y_proba_arr = y_proba_arr[:, indices]
            except Exception:
                return None, None

    if len(classes) == 2:
        pos_class = classes[-1]
        y_binary = (y_true == pos_class).astype(int)
        if y_proba_arr.ndim == 1:
            y_score = y_proba_arr
        else:
            if y_proba_arr.shape[1] != len(classes):
                return None, None
            pos_idx = int(np.where(classes == pos_class)[0][0])
            y_score = y_proba_arr[:, pos_idx]
        auc_val = roc_auc_score(y_binary, y_score)
        return auc_val, (y_binary, y_score)

    if y_proba_arr.ndim != 2 or y_proba_arr.shape[1] != len(classes):
        return None, None

    if y_proba_arr.ndim != 2 or y_proba_arr.shape[1] != len(classes):
        return None, None

    auc_val = roc_auc_score(y_true, y_proba_arr, multi_class="ovr", average="macro")
    return auc_val, None


def _generate_roc_curve(
    results: dict,
    run_dir: Path,
    output_dir: Path,
    config: AnnexAConfig,
) -> Path | None:
    """Generate ROC curve from predictions."""
    from sklearn.metrics import roc_curve

    try:
        plt.style.use(config.style)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=config.figsize)

    folds = results.get("folds", [])
    class_order = results.get("class_order")
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

            try:
                roc_auc, roc_inputs = _compute_auc(y_true, y_proba, class_order=class_order)
                if roc_auc is None:
                    continue
                aucs.append(roc_auc)

                if roc_inputs is None:
                    continue

                y_binary, y_score = roc_inputs
                fpr, tpr, _ = roc_curve(y_binary, y_score)

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
        label=f"Mean ROC (AUC={mean_auc:.2f} +/- {std_auc:.2f})",
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
    y_max = max((m + s) for m, s in zip(means, stds, strict=False))
    if y_max <= 1.1:
        ax.set_ylim(0, 1.1)
    else:
        ax.set_ylim(0, y_max * 1.1)
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


def _generate_regression_scatter(
    results: dict,
    run_dir: Path,
    output_dir: Path,
    config: AnnexAConfig,
) -> Path | None:
    """Generate scatter plot for regression predictions."""
    folds = results.get("folds", [])
    if not folds:
        return None

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

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.scatter(y_true, y_pred, alpha=0.6, edgecolor="none")

    min_val = float(min(np.min(y_true), np.min(y_pred)))
    max_val = float(max(np.max(y_true), np.max(y_pred)))
    ax.plot([min_val, max_val], [min_val, max_val], "k--", lw=1)

    ax.set_xlabel("True OpenCR", fontsize=12)
    ax.set_ylabel("Predicted OpenCR", fontsize=12)
    ax.set_title("OpenCR Regression: True vs Predicted", fontsize=14)
    ax.grid(True, alpha=0.3)

    scatter_path = output_dir / "opencr_scatter.png"
    fig.tight_layout()
    fig.savefig(scatter_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Generated regression scatter: {scatter_path}")
    return scatter_path


def _generate_regression_error(
    results: dict,
    run_dir: Path,
    output_dir: Path,
    config: AnnexAConfig,
) -> Path | None:
    """Generate temporal error plot for regression predictions."""
    folds = results.get("folds", [])
    if not folds:
        return None

    fold = folds[0]
    pred_path = run_dir / f"fold_{fold['fold_idx']:02d}" / "predictions.npz"
    if not pred_path.exists():
        return None

    with np.load(pred_path, allow_pickle=True) as data:
        y_true = data["y_true"]
        y_pred = data["y_pred"]

    if len(y_true) == 0:
        return None

    error = y_pred - y_true
    x = np.arange(len(error))

    try:
        plt.style.use(config.style)
    except Exception:
        pass

    fig, ax = plt.subplots(figsize=config.figsize)
    ax.plot(x, error, lw=1.5, color="steelblue")
    ax.axhline(0, color="black", lw=1, linestyle="--")
    ax.set_xlabel("Window Index", fontsize=12)
    ax.set_ylabel("Prediction Error", fontsize=12)
    ax.set_title("OpenCR Regression: Temporal Error", fontsize=14)
    ax.grid(True, alpha=0.3)

    error_path = output_dir / "opencr_error.png"
    fig.tight_layout()
    fig.savefig(error_path, dpi=config.dpi, bbox_inches="tight")
    plt.close(fig)

    logger.info(f"Generated regression error plot: {error_path}")
    return error_path
