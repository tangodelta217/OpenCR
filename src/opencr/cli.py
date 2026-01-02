"""
OpenCR CLI - Command Line Interface

Main entry point for all OpenCR operations.
Usage: opencr [COMMAND] [OPTIONS]
"""

from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel

from opencr import __version__
from opencr.logging import get_logger, setup_logging

# Initialize Typer app
app = typer.Typer(
    name="opencr",
    help="OpenCR: Open Cognitive Radio - ML/Signal Processing Pipeline",
    add_completion=False,
    rich_markup_mode="rich",
)

# Subcommand groups
data_app = typer.Typer(help="Data management: fetch, preprocess, split")
baseline_app = typer.Typer(help="Baseline model: train, evaluate")
report_app = typer.Typer(help="Report generation: annexA, metrics")
edge_app = typer.Typer(help="Edge deployment: export, quantize, benchmark")
demo_app = typer.Typer(help="Demo and visualization: run, fuel-gauge")

app.add_typer(data_app, name="data")
app.add_typer(baseline_app, name="baseline")
app.add_typer(report_app, name="report")
app.add_typer(edge_app, name="edge")
app.add_typer(demo_app, name="demo")

# Console for rich output
console = Console()
logger = get_logger(__name__)


def _ensure_dir(path: Path) -> Path:
    """Ensure directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    logger.debug(f"Ensured directory exists: {path}")
    return path


def _validate_file_exists(path: Path, name: str = "file") -> None:
    """Validate that a file exists."""
    if not path.exists():
        console.print(f"[red]Error:[/red] {name} not found: {path}")
        raise typer.Exit(code=1)
    if not path.is_file():
        console.print(f"[red]Error:[/red] {name} is not a file: {path}")
        raise typer.Exit(code=1)


def _validate_dir_exists(path: Path, name: str = "directory") -> None:
    """Validate that a directory exists."""
    if not path.exists():
        console.print(f"[red]Error:[/red] {name} not found: {path}")
        raise typer.Exit(code=1)
    if not path.is_dir():
        console.print(f"[red]Error:[/red] {name} is not a directory: {path}")
        raise typer.Exit(code=1)


def version_callback(value: bool) -> None:
    """Print version and exit."""
    if value:
        console.print(f"[bold blue]OpenCR[/bold blue] version [green]{__version__}[/green]")
        raise typer.Exit()


@app.callback()
def main(
    version: bool = typer.Option(
        None,
        "--version",
        "-v",
        help="Show version and exit.",
        callback=version_callback,
        is_eager=True,
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose",
        "-V",
        help="Enable verbose logging.",
    ),
) -> None:
    """
    OpenCR: Open Cognitive Radio - Professional ML/Signal Processing Pipeline.

    Use 'opencr COMMAND --help' for more information on a specific command.

    \b
    Available command groups:
      data      Data management: fetch, preprocess, split
      baseline  Baseline model: train, evaluate
      report    Report generation: annexA, metrics
      edge      Edge deployment: export, quantize, benchmark
      demo      Demo and visualization: run, fuel-gauge
    """
    setup_logging(verbose=verbose)


# =============================================================================
# DATA COMMANDS
# =============================================================================


@data_app.command("fetch")
def data_fetch(
    input_dir: Path = typer.Argument(
        ...,
        help="Path to local dataset directory",
    ),
    output_dir: Path = typer.Option(
        Path("runs/fetch"),
        "--output",
        "-o",
        help="Output directory for data card",
    ),
    adapter: str = typer.Option(
        "npz",
        "--adapter",
        "-a",
        help="Dataset adapter type: npz",
    ),
) -> None:
    """
    Validate local dataset and generate data card.

    This is a non-destructive fetch that validates the structure of a local
    dataset directory and generates a data_card.json for reproducibility.

    \b
    Supported adapters:
      npz - Local .npz files (one per subject)

    \b
    Example:
      opencr data fetch data/raw --output runs/fetch --adapter npz
    """
    import json

    from opencr.data.adapters import LocalNpzAdapter

    setup_logging()
    logger.info(f"Validating dataset: {input_dir}")

    # Validate input directory
    if not input_dir.exists():
        console.print(f"[red]Error:[/red] Input directory not found: {input_dir}")
        raise typer.Exit(code=1)

    if not input_dir.is_dir():
        console.print(f"[red]Error:[/red] Input path is not a directory: {input_dir}")
        raise typer.Exit(code=1)

    # Select adapter
    adapters = {"npz": LocalNpzAdapter}
    if adapter not in adapters:
        console.print(
            f"[red]Error:[/red] Unknown adapter '{adapter}'. Available: {list(adapters.keys())}"
        )
        raise typer.Exit(code=1)

    # Create adapter and validate
    try:
        dataset_adapter = adapters[adapter](input_dir)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e

    # Validate dataset structure
    is_valid, errors = dataset_adapter.validate()
    if not is_valid:
        console.print("[red]Dataset validation failed:[/red]")
        for error in errors:
            console.print(f"  - {error}")
        raise typer.Exit(code=1)

    # Generate data card
    data_card = dataset_adapter.describe()

    # Ensure output directory exists
    _ensure_dir(output_dir)

    # Write data card
    card_path = output_dir / "data_card.json"
    with open(card_path, "w", encoding="utf-8") as f:
        json.dump(data_card.to_dict(), f, indent=2, ensure_ascii=False)

    logger.info(f"Data card written to: {card_path}")

    console.print(
        Panel(
            f"[green]Dataset validated successfully![/green]\n\n"
            f"Adapter: [cyan]{adapter}[/cyan]\n"
            f"Source: [cyan]{input_dir}[/cyan]\n"
            f"Subjects: {data_card.num_subjects}\n"
            f"Signals: {', '.join(data_card.signals)}\n"
            f"Sampling rates: {data_card.sampling_rates}\n"
            f"Protocol levels: {'Yes' if data_card.has_protocol_levels else 'No'}\n\n"
            f"Data card: [cyan]{card_path}[/cyan]",
            title="[green]Fetch Complete[/green]",
        )
    )


@data_app.command("preprocess")
def data_preprocess(
    input_dir: Path = typer.Argument(
        ...,
        help="Input directory with raw data (npz files)",
    ),
    output_dir: Path = typer.Option(
        Path("data/processed"),
        "--output",
        "-o",
        help="Output directory for processed data",
    ),
    window_sec: float = typer.Option(
        30.0,
        "--window-sec",
        "-w",
        help="Window size in seconds",
    ),
    stride_sec: float = typer.Option(
        2.0,
        "--stride-sec",
        "-s",
        help="Stride (hop) size in seconds",
    ),
    resample_hz: float | None = typer.Option(
        None,
        "--resample-hz",
        help="Resample to this frequency (Hz). None = keep original.",
    ),
    sqi_threshold: float = typer.Option(
        0.5,
        "--sqi-threshold",
        "-q",
        help="SQI threshold for quality gating (0.0-1.0)",
    ),
    filter_low: float = typer.Option(
        0.5,
        "--filter-low",
        help="Bandpass filter low frequency (Hz). 0 = no filter.",
    ),
    filter_high: float = typer.Option(
        4.0,
        "--filter-high",
        help="Bandpass filter high frequency (Hz). 0 = no filter.",
    ),
) -> None:
    """
    Preprocess raw data for training.

    Applies the full preprocessing pipeline:
    1. Load data via adapter
    2. Optional resampling and bandpass filtering
    3. Sliding window segmentation
    4. SQI computation per window
    5. Quality gating (valid_mask)

    Output per subject: X (windows), y (labels), sqi, valid_mask, timestamps.

    \\b
    Example:
      opencr data preprocess data/raw --output data/processed --window-sec 30 --stride-sec 2
    """
    from opencr.data.adapters import LocalNpzAdapter
    from opencr.preprocess.pipeline import PreprocessingConfig, PreprocessingPipeline

    setup_logging()
    logger.info(f"Preprocessing data: {input_dir} -> {output_dir}")

    # Validate input
    _validate_dir_exists(input_dir, "Input directory")

    if window_sec <= 0:
        console.print(f"[red]Error:[/red] window-sec must be positive, got {window_sec}")
        raise typer.Exit(code=1)

    if stride_sec <= 0:
        console.print(f"[red]Error:[/red] stride-sec must be positive, got {stride_sec}")
        raise typer.Exit(code=1)

    if stride_sec > window_sec:
        console.print(
            f"[red]Error:[/red] stride-sec ({stride_sec}) must be <= window-sec ({window_sec})"
        )
        raise typer.Exit(code=1)

    if not 0 <= sqi_threshold <= 1:
        console.print(
            f"[red]Error:[/red] sqi-threshold must be between 0 and 1, got {sqi_threshold}"
        )
        raise typer.Exit(code=1)

    # Create adapter
    try:
        adapter = LocalNpzAdapter(input_dir)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e

    # Configure pipeline
    filter_low_val = filter_low if filter_low > 0 else None
    filter_high_val = filter_high if filter_high > 0 else None

    config = PreprocessingConfig(
        window_sec=window_sec,
        stride_sec=stride_sec,
        resample_hz=resample_hz,
        filter_low_hz=filter_low_val,
        filter_high_hz=filter_high_val,
        sqi_threshold=sqi_threshold,
    )

    pipeline = PreprocessingPipeline(config)

    # Show configuration
    console.print(
        Panel(
            f"[cyan]Preprocessing Configuration[/cyan]\\n\\n"
            f"Input: [green]{input_dir}[/green]\\n"
            f"Output: [green]{output_dir}[/green]\\n"
            f"Window: {window_sec}s, Stride: {stride_sec}s\\n"
            f"Resample: {resample_hz or 'original'} Hz\\n"
            f"Filter: {filter_low_val or 'none'}-{filter_high_val or 'none'} Hz\\n"
            f"SQI threshold: {sqi_threshold}\\n"
            f"Subjects: {len(adapter.list_subjects())}",
            title="[bold]Preprocessing[/bold]",
        )
    )

    # Run pipeline
    _ensure_dir(output_dir)

    try:
        stats = pipeline.process_dataset(adapter, output_dir)
    except Exception as e:
        console.print(f"[red]Error during processing:[/red] {e}")
        raise typer.Exit(code=1) from e

    # Show results
    valid_pct = stats["valid_windows"] / max(stats["total_windows"], 1) * 100
    console.print(
        Panel(
            f"[green]Preprocessing Complete![/green]\\n\\n"
            f"Subjects processed: {stats['processed']}/{stats['total_subjects']}\\n"
            f"Total windows: {stats['total_windows']}\\n"
            f"Valid windows: {stats['valid_windows']} ({valid_pct:.1f}%)\\n"
            f"Failed: {stats['failed']}\\n\\n"
            f"Output: [cyan]{output_dir}[/cyan]",
            title="[green]Complete[/green]",
        )
    )


# =============================================================================
# BASELINE COMMANDS
# =============================================================================


@baseline_app.command("train")
def baseline_train(
    processed_dir: Path = typer.Argument(
        ...,
        help="Directory with preprocessed data (from opencr data preprocess)",
    ),
    output_dir: Path = typer.Option(
        Path("runs/baseline"),
        "--output",
        "-o",
        help="Output directory for models and results",
    ),
    cv: str = typer.Option(
        "loso",
        "--cv",
        help="Cross-validation strategy: loso (leave-one-subject-out)",
    ),
    model_type: str = typer.Option(
        "random_forest",
        "--model",
        "-m",
        help="Model type: random_forest, gradient_boosting, xgboost",
    ),
    n_estimators: int = typer.Option(
        100,
        "--n-estimators",
        help="Number of trees/estimators",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed",
    ),
) -> None:
    """
    Train baseline model with LOSO cross-validation.

    Trains a baseline classifier using Leave-One-Subject-Out CV for anti-leakage.
    Saves models, predictions, and metrics for each fold.

    \\b
    Example:
      opencr baseline train runs/preprocess/processed --output runs/baseline --cv loso
    """
    import json

    import numpy as np

    from opencr.evaluation.cv import loso_split
    from opencr.evaluation.metrics import (
        aggregate_fold_metrics,
        compute_classification_metrics,
    )
    from opencr.features.fusion import extract_all_features
    from opencr.models.baseline import BaselineModel, ModelConfig, ModelType, TaskType

    setup_logging()
    logger.info(f"Training baseline: {processed_dir}")

    _validate_dir_exists(processed_dir, "Processed data directory")
    _ensure_dir(output_dir)

    # Find all processed subject files
    npz_files = sorted(processed_dir.glob("*.npz"))
    if not npz_files:
        console.print(f"[red]Error:[/red] No processed .npz files in {processed_dir}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Found {len(npz_files)} subjects[/cyan]")

    # Load all subjects and extract features
    all_features = []
    all_labels = []
    all_subjects = []
    subject_ids = []
    fs = 100.0  # Default, will be overwritten if available

    for npz_path in npz_files:
        subject_id = npz_path.stem
        subject_ids.append(subject_id)

        with np.load(npz_path, allow_pickle=True) as data:
            X = data["X"]
            y = data["y"]
            sqi = data["sqi"]
            valid_mask = data["valid_mask"]
            meta = data["metadata"].item() if "metadata" in data.files else {}
            fs = meta.get("fs", 100.0)

            # Skip subjects with no valid windows
            if valid_mask.sum() == 0:
                logger.warning(f"Subject {subject_id} has no valid windows, skipping")
                continue

            # Extract features from valid windows only
            X_valid = X[valid_mask]
            y_valid = y[valid_mask] if len(y) > 0 else np.zeros(X_valid.shape[0])
            sqi_valid = sqi[valid_mask]

            features, feature_names = extract_all_features(X_valid, fs, sqi_valid)

            n_windows = features.shape[0]
            all_features.append(features)
            all_labels.append(y_valid)
            all_subjects.extend([subject_id] * n_windows)

    if not all_features:
        console.print("[red]Error:[/red] No valid data found")
        raise typer.Exit(code=1)

    # Concatenate all data
    X_all = np.vstack(all_features)
    y_all = np.concatenate(all_labels)
    subjects_arr = np.array(all_subjects)

    console.print(f"[cyan]Total samples: {len(X_all)}, Features: {X_all.shape[1]}[/cyan]")

    # Configure model
    model_types = {
        "random_forest": ModelType.RANDOM_FOREST,
        "gradient_boosting": ModelType.GRADIENT_BOOSTING,
        "xgboost": ModelType.XGBOOST,
    }

    if model_type not in model_types:
        console.print(f"[red]Error:[/red] Unknown model type: {model_type}")
        raise typer.Exit(code=1)

    model_config = ModelConfig(
        task_type=TaskType.CLASSIFICATION,
        model_type=model_types[model_type],
        n_estimators=n_estimators,
        random_state=seed,
    )

    # LOSO cross-validation
    fold_results = []
    fold_metrics = []

    for fold in loso_split(subjects_arr):
        console.print(f"  Fold {fold.fold_idx + 1}: test={fold.test_subject}")

        X_train = X_all[fold.train_indices]
        y_train = y_all[fold.train_indices]
        X_test = X_all[fold.test_indices]
        y_test = y_all[fold.test_indices]

        # Train model
        model = BaselineModel(model_config)
        model.fit(X_train, y_train, feature_names)

        # Predict
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)

        # Compute metrics
        metrics = compute_classification_metrics(y_test, y_pred, y_proba)
        fold_metrics.append(metrics)

        # Save fold results
        fold_dir = output_dir / f"fold_{fold.fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        model.save(fold_dir / "model.joblib")

        np.savez(
            fold_dir / "predictions.npz",
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba if y_proba is not None else np.array([]),
        )

        fold_results.append(
            {
                "fold_idx": fold.fold_idx,
                "test_subject": fold.test_subject,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "metrics": metrics.to_dict(),
            }
        )

    # Aggregate metrics
    aggregated = aggregate_fold_metrics(fold_metrics)

    # Save overall results
    results = {
        "model_type": model_type,
        "cv_strategy": cv,
        "n_subjects": len(subject_ids),
        "n_samples": len(X_all),
        "n_features": X_all.shape[1],
        "feature_names": feature_names,
        "folds": fold_results,
        "aggregated_metrics": aggregated,
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create results.csv
    import csv

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["fold", "subject", "accuracy", "f1", "auc_roc"])
        for fr in fold_results:
            writer.writerow(
                [
                    fr["fold_idx"],
                    fr["test_subject"],
                    f"{fr['metrics']['accuracy']:.4f}",
                    f"{fr['metrics']['f1']:.4f}",
                    f"{fr['metrics']['auc_roc']:.4f}" if fr["metrics"]["auc_roc"] else "N/A",
                ]
            )

    console.print(
        Panel(
            f"[green]Training Complete![/green]\\n\\n"
            f"Subjects: {len(subject_ids)}\\n"
            f"Folds: {len(fold_results)}\\n"
            f"Mean Accuracy: {aggregated.get('accuracy', {}).get('mean', 0):.4f} "
            f"(+/- {aggregated.get('accuracy', {}).get('std', 0):.4f})\\n"
            f"Mean F1: {aggregated.get('f1', {}).get('mean', 0):.4f}\\n\\n"
            f"Output: [cyan]{output_dir}[/cyan]",
            title="[green]Training Complete[/green]",
        )
    )


@baseline_app.command("evaluate")
def baseline_evaluate(
    run_dir: Path = typer.Argument(
        ...,
        help="Path to training run directory",
    ),
    output_file: Path = typer.Option(
        None,
        "--output",
        "-o",
        help="Output file for aggregated results (default: run_dir/summary.json)",
    ),
) -> None:
    """
    Aggregate and display baseline evaluation results.

    Reads fold results from a training run and generates summary statistics.

    \\b
    Example:
      opencr baseline evaluate runs/baseline
    """
    import json

    setup_logging()
    logger.info(f"Evaluating results: {run_dir}")

    _validate_dir_exists(run_dir, "Run directory")

    results_path = run_dir / "results.json"
    if not results_path.exists():
        console.print(f"[red]Error:[/red] results.json not found in {run_dir}")
        raise typer.Exit(code=1)

    with open(results_path) as f:
        results = json.load(f)

    aggregated = results.get("aggregated_metrics", {})
    folds = results.get("folds", [])

    # Display results
    console.print("\\n[bold]Per-Fold Results:[/bold]")
    for fold in folds:
        m = fold["metrics"]
        auc_str = f"{m['auc_roc']:.3f}" if m["auc_roc"] else "N/A"
        console.print(
            f"  Fold {fold['fold_idx']:2d} ({fold['test_subject']:>10s}): "
            f"Acc={m['accuracy']:.3f}, F1={m['f1']:.3f}, "
            f"AUC={auc_str}"
        )

    console.print("\\n[bold]Aggregated Metrics:[/bold]")
    for metric, stats in aggregated.items():
        if isinstance(stats, dict) and "mean" in stats:
            console.print(
                f"  {metric}: {stats['mean']:.4f} (+/- {stats['std']:.4f}) "
                f"[{stats['min']:.4f}, {stats['max']:.4f}]"
            )

    # Save summary
    out_path = output_file or (run_dir / "summary.json")
    summary = {
        "n_folds": len(folds),
        "n_subjects": results.get("n_subjects"),
        "aggregated_metrics": aggregated,
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    console.print(f"\\n[green]Summary saved to:[/green] {out_path}")


# =============================================================================
# REPORT COMMANDS
# =============================================================================


@report_app.command("annexA")
def report_annex_a(
    run_dir: Path = typer.Argument(
        ...,
        help="Directory with baseline run results",
    ),
    output_dir: Path = typer.Option(
        Path("docs"),
        "--output",
        "-o",
        help="Output directory for report",
    ),
) -> None:
    """
    Generate Annex A report for TFM.

    Creates tables and figures for the Annex A:
    - Summary table (CSV and Markdown)
    - ROC curves
    - Fuel gauge visualization
    - Metrics bar plot

    \\b
    Example:
      opencr report annexA runs/baseline --output docs
    """
    from opencr.report.annexA import generate_annex_a

    setup_logging()
    logger.info(f"Generating Annex A report: {run_dir}")

    _validate_dir_exists(run_dir, "Run directory")

    results_path = run_dir / "results.json"
    if not results_path.exists():
        console.print(f"[red]Error:[/red] results.json not found in {run_dir}")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[cyan]Generating Annex A Report[/cyan]\\n\\n"
            f"Run: [green]{run_dir}[/green]\\n"
            f"Output: [green]{output_dir}[/green]",
            title="[bold]Report Generation[/bold]",
        )
    )

    try:
        result = generate_annex_a(run_dir, output_dir)
    except Exception as e:
        console.print(f"[red]Error during report generation:[/red] {e}")
        raise typer.Exit(code=1) from e

    tables = result.get("tables", [])
    figures = result.get("figures", [])

    console.print(
        Panel(
            f"[green]Report Generated![/green]\\n\\n"
            f"Tables: {len(tables)}\\n"
            + "".join(f"  - {t}\\n" for t in tables)
            + f"Figures: {len(figures)}\\n"
            + "".join(f"  - {f}\\n" for f in figures)
            + f"\\nOutput: [cyan]{output_dir}[/cyan]",
            title="[green]Complete[/green]",
        )
    )


@report_app.command("metrics")
def report_metrics(
    results_dir: Path = typer.Argument(
        ...,
        help="Directory with evaluation results",
    ),
    output_path: Path = typer.Option(
        Path("docs/metrics.json"),
        "--output",
        "-o",
        help="Output path for metrics JSON",
    ),
) -> None:
    """
    Export evaluation metrics to JSON.

    Creates a structured JSON file with all computed metrics.

    \b
    Example:
      opencr report metrics runs/eval --output docs/metrics.json
    """
    setup_logging()
    logger.info(f"Exporting metrics: {results_dir}")

    _validate_dir_exists(results_dir, "Results directory")
    _ensure_dir(output_path.parent)

    console.print(
        Panel(
            f"[yellow]Metrics Export[/yellow]\n\n"
            f"Results: [cyan]{results_dir}[/cyan]\n"
            f"Output: [cyan]{output_path}[/cyan]\n\n"
            f"[dim]Not implemented yet[/dim]",
            title="📈 Metrics",
        )
    )


# =============================================================================
# EDGE COMMANDS
# =============================================================================


@edge_app.command("export")
def edge_export(
    run_dir: Path = typer.Option(
        ...,
        "--run",
        "-r",
        help="Path to baseline run directory (containing results.json and models)",
    ),
    output_dir: Path = typer.Option(
        Path("runs/edge_export"),
        "--out",
        "-o",
        help="Output directory for edge artifacts",
    ),
    format: str = typer.Option(
        "auto",
        "--format",
        "-f",
        help="Target format: auto, onnx, pickle",
    ),
    quantize: bool = typer.Option(
        False,
        "--quantize/--no-quantize",
        "-q",
        help="Apply quantization (ONNX only)",
    ),
    latency_budget: float = typer.Option(
        100.0,
        "--latency",
        "-l",
        help="Target latency budget in ms",
    ),
) -> None:
    """
    Export model to edge-ready format.

    Generates deployment artifacts and a budget report.
    Supports ONNX export for sklearn models (if skl2onnx installed) or pickle stubs.

    \\b
    Example:
      opencr edge export --run runs/baseline --out runs/edge --format onnx
    """
    from opencr.edge.export import EdgeConfig, export_edge_model

    setup_logging()
    logger.info(f"Exporting edge model from: {run_dir}")

    _validate_dir_exists(run_dir, "Run directory")

    config = EdgeConfig(
        target_format=format,
        quantize=quantize,
        target_latency_ms=latency_budget,
    )

    try:
        result = export_edge_model(run_dir, output_dir, config)
    except Exception as e:
        console.print(f"[red]Error during export:[/red] {e}")
        # Print stack trace for debugging
        import traceback

        traceback.print_exc()
        raise typer.Exit(code=1) from e

    budget = result["budget"]
    manifest = result["manifest_path"]

    # Display summary
    console.print(
        Panel(
            f"[bold green]Export Complete![/bold green]\\n\\n"
            f"Format: [cyan]{result['format']}[/cyan]\\n"
            f"Path: [cyan]{result['edge_model_path']}[/cyan]\\n"
            f"Size: {budget['model_size']['kb']} KB\\n"
            f"Latency Est: {budget['latency']['estimated_ms']} ms "
            f"({'OK' if budget['latency']['meets_budget'] else 'WARNING'})\\n\\n"
            f"Manifest: {manifest}",
            title="Edge Artifacts",
        )
    )

    if budget["notes"]:
        console.print("\\n[bold]Notes:[/bold]")
        for note in budget["notes"]:
            console.print(f"- {note}")


@edge_app.command("benchmark")
def edge_benchmark(
    model_path: Path = typer.Argument(
        ...,
        help="Path to ONNX model",
    ),
    output_path: Path = typer.Option(
        Path("docs/benchmark.json"),
        "--output",
        "-o",
        help="Output path for benchmark results",
    ),
    device: str = typer.Option(
        "cpu",
        "--device",
        "-d",
        help="Device: cpu, cuda, tensorrt",
    ),
    iterations: int = typer.Option(
        100,
        "--iterations",
        "-n",
        help="Number of benchmark iterations",
    ),
    warmup: int = typer.Option(
        10,
        "--warmup",
        help="Number of warmup iterations",
    ),
) -> None:
    """
    Run inference benchmark on edge model.

    Measures latency, throughput, and memory usage.

    \b
    Example:
      opencr edge benchmark models/model.onnx --iterations 1000 --device cpu
    """
    setup_logging()
    logger.info(f"Running benchmark: {model_path}")

    _validate_file_exists(model_path, "ONNX model")
    _ensure_dir(output_path.parent)

    if iterations <= 0:
        console.print(f"[red]Error:[/red] iterations must be positive, got {iterations}")
        raise typer.Exit(code=1)

    if warmup < 0:
        console.print(f"[red]Error:[/red] warmup must be non-negative, got {warmup}")
        raise typer.Exit(code=1)

    console.print(
        Panel(
            f"[yellow]Edge Benchmark[/yellow]\n\n"
            f"Model: [cyan]{model_path}[/cyan]\n"
            f"Device: {device}\n"
            f"Iterations: {iterations}\n"
            f"Warmup: {warmup}\n"
            f"Output: [cyan]{output_path}[/cyan]\n\n"
            f"[dim]Not implemented yet[/dim]",
            title="⚡ Benchmark",
        )
    )


# =============================================================================
# DEMO COMMANDS
# =============================================================================


@demo_app.command("run")
def demo_run(
    run_dir: Path = typer.Argument(
        ...,
        help="Path to baseline run directory",
    ),
    subject: str = typer.Option(
        None,
        "--subject",
        "-s",
        help="Subject ID to visualize (uses first if not specified)",
    ),
    delay: int = typer.Option(
        500,
        "--delay",
        "-d",
        help="Delay between updates in milliseconds",
    ),
) -> None:
    """
    Run live fuel gauge demo.

    Streams predictions with a live visualization of accuracy and confidence.

    \\b
    Example:
      opencr demo run runs/baseline --subject S01 --delay 300
    """
    from opencr.demo.cli_demo import DemoConfig, run_demo

    setup_logging()
    logger.info(f"Running demo: {run_dir}")

    _validate_dir_exists(run_dir, "Run directory")

    if delay <= 0:
        console.print(f"[red]Error:[/red] delay must be positive, got {delay}")
        raise typer.Exit(code=1)

    config = DemoConfig(delay_ms=delay)
    run_demo(run_dir, subject, config)


@demo_app.command("fuel-gauge")
def demo_fuel_gauge(
    threshold: float = typer.Option(
        0.7,
        "--threshold",
        "-t",
        help="Confidence threshold for alerts",
    ),
    duration: int = typer.Option(
        10,
        "--duration",
        "-d",
        help="Demo duration in seconds",
    ),
) -> None:
    """
    Run simulated fuel gauge demo.

    Shows a real-time fuel gauge with simulated data.
    Anyone can understand this in 20 seconds!

    \\b
    Example:
      opencr demo fuel-gauge --threshold 0.8 --duration 15
    """
    from opencr.demo.cli_demo import run_fuel_gauge_demo

    setup_logging()
    logger.info("Running fuel-gauge demo")

    if not 0 <= threshold <= 1:
        console.print(f"[red]Error:[/red] threshold must be between 0 and 1, got {threshold}")
        raise typer.Exit(code=1)

    if duration <= 0:
        console.print(f"[red]Error:[/red] duration must be positive, got {duration}")
        raise typer.Exit(code=1)

    run_fuel_gauge_demo(threshold, duration)


# Entry point for `python -m opencr`
if __name__ == "__main__":
    app()
