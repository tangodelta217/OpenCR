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

app = typer.Typer(
    name="opencr",
    help="OpenCR: Open Compensatory Reserve - ML/Signal Processing Pipeline",
    add_completion=False,
    rich_markup_mode="rich",
)

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
    OpenCR: Open Compensatory Reserve - Professional ML/Signal Processing Pipeline.

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
    target_direction: str = typer.Option(
        "auto",
        "--target-direction",
        help="Target direction: auto, increasing, decreasing",
    ),
    ordinal_bins: int = typer.Option(
        4,
        "--ordinal-bins",
        help="Number of ordinal bins for y_ord",
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
    bioz_filter_low: float = typer.Option(
        0.05,
        "--bioz-filter-low",
        help="BioZ filter low frequency (Hz). 0 = no filter.",
    ),
    bioz_filter_high: float = typer.Option(
        0.0,
        "--bioz-filter-high",
        help="BioZ filter high frequency (Hz). 0 = no filter.",
    ),
    data_card: Path | None = typer.Option(
        None,
        "--data-card",
        help="Path to data_card.json for reproducibility",
    ),
    data_hash: str = typer.Option(
        "metadata",
        "--data-hash",
        help="Dataset hash method: metadata, content, none",
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
    import sys

    from opencr.data.adapters import LocalNpzAdapter
    from opencr.preprocess.pipeline import PreprocessingConfig, PreprocessingPipeline
    from opencr.repro.manifest import build_manifest, compute_dataset_hash, write_manifest

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

    target_direction = target_direction.lower()
    if target_direction not in {"auto", "increasing", "decreasing"}:
        console.print("[red]Error:[/red] target-direction must be auto, increasing, or decreasing")
        raise typer.Exit(code=1)

    if ordinal_bins < 2:
        console.print(f"[red]Error:[/red] ordinal-bins must be >= 2, got {ordinal_bins}")
        raise typer.Exit(code=1)

    data_hash = data_hash.lower()
    if data_hash not in {"metadata", "content", "none"}:
        console.print("[red]Error:[/red] data-hash must be metadata, content, or none")
        raise typer.Exit(code=1)

    if data_card is not None:
        _validate_file_exists(data_card, "data_card.json")
    else:
        candidate = input_dir / "data_card.json"
        if candidate.exists():
            data_card = candidate

    # Create adapter
    try:
        adapter = LocalNpzAdapter(input_dir)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(code=1) from e

    # Configure pipeline
    filter_low_val = filter_low if filter_low > 0 else None
    filter_high_val = filter_high if filter_high > 0 else None
    bioz_filter_low_val = bioz_filter_low if bioz_filter_low > 0 else None
    bioz_filter_high_val = bioz_filter_high if bioz_filter_high > 0 else None

    config = PreprocessingConfig(
        window_sec=window_sec,
        stride_sec=stride_sec,
        resample_hz=resample_hz,
        filter_low_hz=filter_low_val,
        filter_high_hz=filter_high_val,
        bioz_filter_low_hz=bioz_filter_low_val,
        bioz_filter_high_hz=bioz_filter_high_val,
        sqi_threshold=sqi_threshold,
        target_direction=target_direction,
        ordinal_bins=ordinal_bins,
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
            f"PPG filter: {filter_low_val or 'none'}-{filter_high_val or 'none'} Hz\\n"
            f"BioZ filter: {bioz_filter_low_val or 'none'}-{bioz_filter_high_val or 'none'} Hz\\n"
            f"SQI threshold: {sqi_threshold}\\n"
            f"Target direction: {target_direction}, Ordinal bins: {ordinal_bins}\\n"
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

    dataset_hash = compute_dataset_hash(output_dir / "processed", method=data_hash)
    manifest = build_manifest(
        stage="preprocess",
        seed=None,
        command_args=sys.argv,
        data_card_path=data_card,
        dataset_hash=dataset_hash,
        config={"preprocess": config.to_dict()},
        artifacts={
            "config_path": str(output_dir / "config.json"),
            "stats_path": str(output_dir / "stats.json"),
            "processed_dir": str(output_dir / "processed"),
        },
    )
    write_manifest(output_dir / "manifest.json", manifest)


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
    target: str = typer.Option(
        "opencr",
        "--target",
        "-t",
        help="Target: opencr (regression), ordinal (classification), step (classification)",
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
    n_boot: int = typer.Option(
        200,
        "--n-boot",
        help="Number of bootstrap resamples for CI",
    ),
    seed: int = typer.Option(
        42,
        "--seed",
        help="Random seed",
    ),
    data_card: Path | None = typer.Option(
        None,
        "--data-card",
        help="Path to data_card.json for reproducibility",
    ),
) -> None:
    """
    Train baseline model with LOSO cross-validation.

    Trains a baseline classifier using Leave-One-Subject-Out CV for anti-leakage.
    Saves models, predictions, and metrics for each fold.

    \\b
    Example:
      opencr baseline train runs/preprocess/processed --output runs/baseline --cv loso --target opencr
    """
    import json
    import sys
    from dataclasses import asdict

    import numpy as np

    from opencr.evaluation.cv import loso_split
    from opencr.evaluation.metrics import (
        aggregate_fold_metrics,
        compute_classification_metrics,
        compute_metrics_with_ci,
        compute_regression_metrics,
    )
    from opencr.features.fusion import FeatureConfig, extract_all_features
    from opencr.models.baseline import BaselineModel, ModelConfig, ModelType, TaskType
    from opencr.repro.manifest import (
        build_manifest,
        compute_dataset_hash,
        load_manifest,
        write_manifest,
    )

    setup_logging()
    logger.info(f"Training baseline: {processed_dir}")

    _validate_dir_exists(processed_dir, "Processed data directory")
    _ensure_dir(output_dir)

    target = target.lower()
    if target not in {"opencr", "ordinal", "step"}:
        console.print("[red]Error:[/red] target must be opencr, ordinal, or step")
        raise typer.Exit(code=1)

    # Find all processed subject files
    npz_files = sorted(processed_dir.glob("*.npz"))
    if not npz_files:
        console.print(f"[red]Error:[/red] No processed .npz files in {processed_dir}")
        raise typer.Exit(code=1)

    console.print(f"[cyan]Found {len(npz_files)} subjects[/cyan]")

    preprocess_manifest = load_manifest(processed_dir.parent / "manifest.json")
    preprocess_config = None
    if preprocess_manifest:
        preprocess_config = preprocess_manifest.get("config", {}).get("preprocess")
    if preprocess_config is None:
        config_path = processed_dir.parent / "config.json"
        if config_path.exists():
            with open(config_path) as f:
                preprocess_config = json.load(f)

    if data_card is not None:
        _validate_file_exists(data_card, "data_card.json")
    else:
        data_card_value = None
        if preprocess_manifest:
            data_card_value = preprocess_manifest.get("data_card_path")
        if data_card_value:
            data_card = Path(data_card_value)
        else:
            candidate = processed_dir.parent / "data_card.json"
            if candidate.exists():
                data_card = candidate

    if preprocess_manifest and "dataset_hash" in preprocess_manifest:
        dataset_hash = preprocess_manifest.get("dataset_hash")
    else:
        dataset_hash = compute_dataset_hash(processed_dir, method="metadata")

    # Load all subjects and extract features
    all_features = []
    all_labels = []
    all_subjects = []
    subject_ids = []
    target_direction = None
    ordinal_bins = None
    total_windows = 0
    total_valid = 0
    fs = 100.0  # Default, will be overwritten if available

    for npz_path in npz_files:
        subject_id = npz_path.stem
        subject_ids.append(subject_id)

        with np.load(npz_path, allow_pickle=True) as data:
            X = data["X"]
            sqi = data["sqi"]
            valid_mask = data["valid_mask"]
            meta = data["metadata"].item() if "metadata" in data.files else {}
            fs = meta.get("fs", 100.0)

            if target_direction is None:
                target_direction = meta.get("target_direction")
            if ordinal_bins is None:
                ordinal_bins = meta.get("ordinal_bins")

            if "y_step" in data.files:
                y_step = data["y_step"]
            elif "y" in data.files:
                y_step = data["y"]
            else:
                y_step = np.array([])
            y_opencr = data["y_opencr"] if "y_opencr" in data.files else np.array([])
            y_ord = data["y_ord"] if "y_ord" in data.files else np.array([])

            total_windows += X.shape[0]
            total_valid += int(valid_mask.sum())

            # Skip subjects with no valid windows
            if valid_mask.sum() == 0:
                logger.warning(f"Subject {subject_id} has no valid windows, skipping")
                continue

            if target == "opencr":
                y_target = y_opencr
                target_name = "y_opencr"
            elif target == "ordinal":
                y_target = y_ord
                target_name = "y_ord"
            else:
                y_target = y_step
                target_name = "y_step"

            if y_target.size == 0:
                console.print(
                    f"[red]Error:[/red] {target_name} missing in {npz_path}. "
                    "Re-run preprocessing with protocol steps or choose --target step."
                )
                raise typer.Exit(code=1)
            if y_target.shape[0] != X.shape[0]:
                console.print(
                    f"[red]Error:[/red] {target_name} length does not match X in {npz_path} "
                    f"({y_target.shape[0]} vs {X.shape[0]})"
                )
                raise typer.Exit(code=1)

            # Extract features from valid windows only
            X_valid = X[valid_mask]
            y_valid = y_target[valid_mask]
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

    task_type = TaskType.REGRESSION if target == "opencr" else TaskType.CLASSIFICATION

    model_config = ModelConfig(
        task_type=task_type,
        model_type=model_types[model_type],
        n_estimators=n_estimators,
        random_state=seed,
    )

    # LOSO cross-validation
    fold_results = []
    fold_metrics = []
    ci_y_true = []
    ci_y_pred = []
    ci_y_proba = []
    ci_subjects = []
    has_proba = True
    splits_info: list[dict[str, object]] = []

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
        if task_type == TaskType.REGRESSION:
            metrics = compute_regression_metrics(y_test, y_pred)
            y_proba = None
        else:
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

        ci_y_true.append(y_test)
        ci_y_pred.append(y_pred)
        ci_subjects.append(np.array([fold.test_subject] * len(y_test)))
        if task_type == TaskType.CLASSIFICATION:
            if y_proba is None or len(y_proba) == 0:
                has_proba = False
            else:
                ci_y_proba.append(y_proba)

        fold_results.append(
            {
                "fold_idx": fold.fold_idx,
                "test_subject": fold.test_subject,
                "n_train": len(X_train),
                "n_test": len(X_test),
                "metrics": metrics.to_dict(),
            }
        )
        splits_info.append(
            {
                "fold_idx": fold.fold_idx,
                "test_subject": fold.test_subject,
                "train_subjects": fold.train_subjects,
            }
        )

    # Aggregate metrics
    aggregated = aggregate_fold_metrics(fold_metrics)

    metrics_ci = {}
    if n_boot > 0:
        y_true_all = np.concatenate(ci_y_true)
        y_pred_all = np.concatenate(ci_y_pred)
        subject_ids_all = np.concatenate(ci_subjects)
        y_proba_all = None
        if task_type == TaskType.CLASSIFICATION and has_proba and ci_y_proba:
            y_proba_all = np.vstack(ci_y_proba)

        _, metrics_ci = compute_metrics_with_ci(
            y_true_all,
            y_pred_all,
            y_proba_all,
            subject_ids_all,
            task_type.value,
            n_boot=n_boot,
            seed=seed,
        )

    # Save overall results
    results = {
        "model_type": model_type,
        "cv_strategy": cv,
        "target": target,
        "task_type": task_type.value,
        "target_direction": target_direction,
        "ordinal_bins": ordinal_bins,
        "coverage": (total_valid / total_windows) if total_windows > 0 else 0.0,
        "n_subjects": len(subject_ids),
        "n_samples": len(X_all),
        "n_features": X_all.shape[1],
        "feature_names": feature_names,
        "folds": fold_results,
        "aggregated_metrics": aggregated,
        "metrics_ci": metrics_ci,
        "bootstrap": {"n_boot": n_boot, "seed": seed},
    }

    with open(output_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    splits_path = output_dir / "splits.json"
    with open(splits_path, "w", encoding="utf-8") as f:
        json.dump(splits_info, f, indent=2)

    feature_config = FeatureConfig()
    manifest = build_manifest(
        stage="baseline_train",
        seed=seed,
        command_args=sys.argv,
        data_card_path=data_card,
        dataset_hash=dataset_hash,
        config={
            "preprocess": preprocess_config or {},
            "features": asdict(feature_config),
            "model": asdict(model_config),
            "cv": {"strategy": cv, "n_folds": len(fold_results)},
        },
        splits=splits_info,
        artifacts={
            "results_path": str(output_dir / "results.json"),
            "results_csv": str(output_dir / "results.csv"),
            "splits_path": str(splits_path),
            "run_dir": str(output_dir),
        },
        extra={"target": target, "task_type": task_type.value},
    )
    write_manifest(output_dir / "manifest.json", manifest)

    # Create results.csv
    import csv

    csv_path = output_dir / "results.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        if task_type == TaskType.REGRESSION:
            writer.writerow(["fold", "subject", "rmse", "mae", "r2"])
            for fr in fold_results:
                writer.writerow(
                    [
                        fr["fold_idx"],
                        fr["test_subject"],
                        f"{fr['metrics']['rmse']:.4f}",
                        f"{fr['metrics']['mae']:.4f}",
                        f"{fr['metrics']['r2']:.4f}",
                    ]
                )
        else:
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

    if task_type == TaskType.REGRESSION:
        mean_rmse = aggregated.get("rmse", {}).get("mean", 0)
        mean_mae = aggregated.get("mae", {}).get("mean", 0)
        console.print(
            Panel(
                f"[green]Training Complete![/green]\\n\\n"
                f"Subjects: {len(subject_ids)}\\n"
                f"Folds: {len(fold_results)}\\n"
                f"Mean RMSE: {mean_rmse:.4f}\\n"
                f"Mean MAE: {mean_mae:.4f}\\n\\n"
                f"Coverage: {results['coverage']:.1%}\\n\\n"
                f"Output: [cyan]{output_dir}[/cyan]",
                title="[green]Training Complete[/green]",
            )
        )
    else:
        console.print(
            Panel(
                f"[green]Training Complete![/green]\\n\\n"
                f"Subjects: {len(subject_ids)}\\n"
                f"Folds: {len(fold_results)}\\n"
                f"Mean Accuracy: {aggregated.get('accuracy', {}).get('mean', 0):.4f} "
                f"(+/- {aggregated.get('accuracy', {}).get('std', 0):.4f})\\n"
                f"Mean F1: {aggregated.get('f1', {}).get('mean', 0):.4f}\\n\\n"
                f"Coverage: {results['coverage']:.1%}\\n\\n"
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
    task_type = results.get("task_type", "classification")
    target = results.get("target", "unknown")
    metrics_ci = results.get("metrics_ci", {})

    # Display results
    console.print(f"\\n[bold]Target:[/bold] {target} ({task_type})")
    console.print("\\n[bold]Per-Fold Results:[/bold]")
    if task_type == "regression":
        for fold in folds:
            m = fold["metrics"]
            console.print(
                f"  Fold {fold['fold_idx']:2d} ({fold['test_subject']:>10s}): "
                f"RMSE={m['rmse']:.3f}, MAE={m['mae']:.3f}, R2={m['r2']:.3f}"
            )
    else:
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
            ci = metrics_ci.get(metric)
            ci_str = ""
            if ci:
                ci_str = f" CI95 [{ci['low']:.4f}, {ci['high']:.4f}]"
            console.print(
                f"  {metric}: {stats['mean']:.4f} (+/- {stats['std']:.4f}) "
                f"[{stats['min']:.4f}, {stats['max']:.4f}]{ci_str}"
            )

    # Save summary
    out_path = output_file or (run_dir / "summary.json")
    summary = {
        "n_folds": len(folds),
        "n_subjects": results.get("n_subjects"),
        "target": target,
        "task_type": task_type,
        "aggregated_metrics": aggregated,
        "metrics_ci": metrics_ci,
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
    import sys
    from dataclasses import asdict

    from opencr.report.annexA import AnnexAConfig, generate_annex_a
    from opencr.repro.manifest import build_manifest, load_manifest, write_manifest

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

    config = AnnexAConfig()
    try:
        result = generate_annex_a(run_dir, output_dir, config=config)
    except Exception as e:
        console.print(f"[red]Error during report generation:[/red] {e}")
        raise typer.Exit(code=1) from e

    tables = result.get("tables", [])
    figures = result.get("figures", [])
    baseline_manifest = load_manifest(run_dir / "manifest.json")
    dataset_hash = None
    data_card_path = None
    seed = None
    if baseline_manifest:
        dataset_hash = baseline_manifest.get("dataset_hash")
        data_card_value = baseline_manifest.get("data_card_path")
        if data_card_value:
            data_card_path = Path(data_card_value)
        seed = baseline_manifest.get("seed")

    manifest = build_manifest(
        stage="report_annexA",
        seed=seed,
        command_args=sys.argv,
        data_card_path=data_card_path,
        dataset_hash=dataset_hash,
        config={"annexA": asdict(config), "source_run": str(run_dir)},
        artifacts={"tables": tables, "figures": figures, "output_dir": str(output_dir)},
    )
    write_manifest(output_dir / "manifest.json", manifest)

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
        help="Output path for metrics JSON or directory for tables",
    ),
) -> None:
    """
    Export evaluation metrics and summary tables.

    Generates summary.csv and summary.md from results.json and
    saves a metrics.json snapshot.

    \b
    Example:
      opencr report metrics runs/eval --output docs/metrics.json
    """
    import json

    from opencr.report.metrics import generate_metrics_summary

    setup_logging()
    logger.info(f"Exporting metrics: {results_dir}")

    _validate_dir_exists(results_dir, "Results directory")
    output_path = Path(output_path)
    output_dir = output_path if output_path.suffix == "" else output_path.parent
    _ensure_dir(output_dir)
    json_path = output_path if output_path.suffix else output_dir / "metrics.json"

    result = generate_metrics_summary(results_dir, output_dir)

    results_path = results_dir / "results.json"
    with open(results_path, encoding="utf-8") as f:
        results = json.load(f)

    metrics_payload = {
        "target": results.get("target", "unknown"),
        "task_type": results.get("task_type", "unknown"),
        "n_folds": len(results.get("folds", [])),
        "aggregated_metrics": results.get("aggregated_metrics", {}),
        "metrics_ci": results.get("metrics_ci", {}),
        "source_results": str(results_path),
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)

    message = f"""[yellow]Metrics Export[/yellow]

Results: [cyan]{results_dir}[/cyan]
JSON: [cyan]{json_path}[/cyan]
Tables: [cyan]{', '.join(result['tables'])}[/cyan]"""

    console.print(
        Panel(
            message,
            title="Metrics Export",
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
    if quantize:
        console.print(
            "[red]Error:[/red] Quantization (int8) is only supported for future "
            "TFLite/Keras models; baseline sklearn exports do not apply. "
            "Remove --quantize."
        )
        raise typer.Exit(code=1)

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
        help="Path to ONNX or sklearn model artifact",
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

    Measures host-side inference latency and artifact size.

    \b
    Example:
      opencr edge benchmark models/model.onnx --iterations 1000 --device cpu
    """
    import json

    from opencr.edge.benchmark import benchmark_edge_model, write_benchmark_tables

    setup_logging()
    logger.info(f"Running benchmark: {model_path}")

    _validate_file_exists(model_path, "Model file")

    if iterations <= 0:
        console.print(f"[red]Error:[/red] iterations must be positive, got {iterations}")
        raise typer.Exit(code=1)

    if warmup < 0:
        console.print(f"[red]Error:[/red] warmup must be non-negative, got {warmup}")
        raise typer.Exit(code=1)

    output_path = Path(output_path)
    output_dir = output_path if output_path.suffix == "" else output_path.parent
    _ensure_dir(output_dir)
    json_path = output_path if output_path.suffix else output_dir / "benchmark.json"

    benchmark = benchmark_edge_model(
        model_path,
        iterations=iterations,
        warmup=warmup,
        device=device,
    )

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(benchmark, f, indent=2)

    table_paths = write_benchmark_tables(output_dir, benchmark)

    status = benchmark.get("inference", {}).get("status", "unknown")
    summary = benchmark.get("inference", {}).get("summary") or {}

    message = f"""[yellow]Edge Benchmark[/yellow]

Model: [cyan]{model_path}[/cyan]
Device: {device}
Iterations: {iterations}
Warmup: {warmup}
Status: {status}
Mean: {summary.get('mean_ms', 'N/A')} ms
Output: [cyan]{json_path}[/cyan]
Summary: [cyan]{table_paths['csv']}[/cyan]"""

    console.print(
        Panel(
            message,
            title="Edge Benchmark",
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
