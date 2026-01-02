"""
Edge Export Module.

Exports trained models to edge-ready formats:
- sklearn models -> ONNX (optional) or pickle stub
- Planned: TFLite export for deep models
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opencr.logging import get_logger

logger = get_logger(__name__)

# Check for optional ONNX support
try:
    import onnx
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False
    logger.debug("skl2onnx not installed, ONNX export disabled")


@dataclass
class EdgeBudget:
    """Edge deployment budget estimation."""

    model_size_bytes: int = 0
    model_size_kb: float = 0.0
    model_size_mb: float = 0.0
    estimated_ram_kb: float = 0.0
    estimated_latency_ms: float = 0.0
    target_latency_ms: float = 100.0
    meets_budget: bool = False
    format: str = "unknown"
    quantization: str = "none"
    n_features: int = 0
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "model_size": {
                "bytes": self.model_size_bytes,
                "kb": round(self.model_size_kb, 2),
                "mb": round(self.model_size_mb, 4),
            },
            "estimated_ram_kb": round(self.estimated_ram_kb, 2),
            "latency": {
                "estimated_ms": round(self.estimated_latency_ms, 2),
                "target_ms": self.target_latency_ms,
                "meets_budget": self.meets_budget,
            },
            "format": self.format,
            "quantization": self.quantization,
            "n_features": self.n_features,
            "notes": self.notes,
            "plan_b": {
                "strategy": "playback",
                "description": "Pre-computed predictions stored on device",
                "fallback_available": True,
            },
        }


@dataclass
class EdgeConfig:
    """Configuration for edge export."""

    target_format: str = "auto"  # auto, onnx, pickle (tflite planned)
    quantize: bool = False
    target_latency_ms: float = 100.0
    target_ram_kb: float = 512.0


def export_edge_model(
    run_dir: Path,
    output_dir: Path,
    config: EdgeConfig | None = None,
) -> dict[str, Any]:
    """
    Export a trained model to edge-ready format.

    Args:
        run_dir: Path to baseline run directory.
        output_dir: Output directory for edge artifacts.
        config: Export configuration.

    Returns:
        Dictionary with export results and budget.
    """
    if config is None:
        config = EdgeConfig()
    if config.quantize:
        raise ValueError(
            "Quantization (int8) is only supported for future TFLite/Keras models; "
            "baseline sklearn exports do not apply. Remove --quantize."
        )

    run_dir = Path(run_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load results to get metadata
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"results.json not found in {run_dir}")

    with open(results_path) as f:
        results = json.load(f)

    n_features = results.get("n_features", 0)
    model_type = results.get("model_type", "unknown")

    # Find first fold model
    model_path = None
    for fold_dir in sorted(run_dir.glob("fold_*")):
        candidate = fold_dir / "model.joblib"
        if candidate.exists():
            model_path = candidate
            break

    if model_path is None:
        raise FileNotFoundError(f"No model.joblib found in {run_dir}")

    from opencr.models.baseline import BaselineModel

    model_wrapper = BaselineModel.load(model_path)
    sklearn_model = model_wrapper.model

    # Determine export format
    export_format = config.target_format
    if export_format == "auto":
        export_format = "onnx" if HAS_ONNX else "pickle"
    if export_format == "onnx" and not HAS_ONNX:
        logger.warning("skl2onnx not installed; falling back to pickle export")
        export_format = "pickle"

    # Export based on format
    if export_format == "onnx":
        edge_path, budget = _export_onnx(sklearn_model, output_dir, n_features, config)
    else:
        edge_path, budget = _export_pickle_stub(
            sklearn_model, model_path, output_dir, n_features, config
        )

    # Save budget
    budget_path = output_dir / "edge_budget.json"
    with open(budget_path, "w") as f:
        json.dump(budget.to_dict(), f, indent=2)

    # Save export manifest
    manifest = {
        "source_run": str(run_dir),
        "model_type": model_type,
        "export_format": budget.format,
        "quantization": "not_applicable",
        "edge_model_path": str(edge_path),
        "budget_path": str(budget_path),
        "config": {
            "target_format": config.target_format,
            "quantize": config.quantize,
            "target_latency_ms": config.target_latency_ms,
            "target_ram_kb": config.target_ram_kb,
        },
        "status": "success",
        "limitations": _get_limitations(budget.format),
    }

    manifest_path = output_dir / "export_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.info(f"Edge export complete: {output_dir}")

    return {
        "edge_model_path": str(edge_path),
        "budget_path": str(budget_path),
        "manifest_path": str(manifest_path),
        "budget": budget.to_dict(),
        "format": budget.format,
    }


def _export_onnx(
    model: Any,
    output_dir: Path,
    n_features: int,
    config: EdgeConfig,
) -> tuple[Path, EdgeBudget]:
    """Export sklearn model to ONNX format."""
    logger.info("Exporting to ONNX format")

    # Define input shape
    if n_features == 0:
        n_features = 45  # Default feature count

    initial_type = [("input", FloatTensorType([None, n_features]))]

    try:
        onnx_model = convert_sklearn(model, initial_types=initial_type)
    except Exception as e:
        logger.warning(f"ONNX conversion failed: {e}, falling back to pickle")
        return _export_pickle_stub(model, None, output_dir, n_features, config)

    # Save ONNX model
    onnx_path = output_dir / "model.onnx"
    onnx.save_model(onnx_model, str(onnx_path))

    # Calculate size
    model_size = os.path.getsize(onnx_path)

    # Estimate RAM (ONNX roughly 2x model size at runtime)
    estimated_ram = model_size * 2

    # Estimate latency (very rough: sklearn models are fast)
    estimated_latency = 1.0 + (n_features * 0.01)  # ~1-2ms for tree models

    budget = EdgeBudget(
        model_size_bytes=model_size,
        model_size_kb=model_size / 1024,
        model_size_mb=model_size / (1024 * 1024),
        estimated_ram_kb=estimated_ram / 1024,
        estimated_latency_ms=estimated_latency,
        target_latency_ms=config.target_latency_ms,
        meets_budget=estimated_latency <= config.target_latency_ms,
        format="onnx",
        quantization="float32",
        n_features=n_features,
        notes=[
            "ONNX export from sklearn model",
            "Can run on ONNX Runtime (CPU)",
            "Quantization is not applied by this tool",
        ],
    )

    return onnx_path, budget


def _export_pickle_stub(
    model: Any,
    _original_path: Path | None,
    output_dir: Path,
    n_features: int,
    config: EdgeConfig,
) -> tuple[Path, EdgeBudget]:
    """Export as pickle stub with documentation."""
    logger.info("Exporting as pickle stub (ONNX not available)")

    # Save model
    stub_path = output_dir / "model_stub.joblib"
    import joblib

    joblib.dump(model, stub_path)
    model_size = os.path.getsize(stub_path)

    # Estimate RAM (pickle models need full Python runtime)
    estimated_ram = model_size * 3 + 50 * 1024  # Model + Python overhead

    # Estimate latency
    estimated_latency = 2.0 + (n_features * 0.02)

    budget = EdgeBudget(
        model_size_bytes=model_size,
        model_size_kb=model_size / 1024,
        model_size_mb=model_size / (1024 * 1024),
        estimated_ram_kb=estimated_ram / 1024,
        estimated_latency_ms=estimated_latency,
        target_latency_ms=config.target_latency_ms,
        meets_budget=estimated_latency <= config.target_latency_ms,
        format="pickle_stub",
        quantization="none",
        n_features=n_features,
        notes=[
            "Pickle stub - requires Python runtime on edge",
            "For true edge deployment, install skl2onnx for ONNX export",
            "Alternative: Use Plan B (playback) strategy",
            "Future: Deep model with TFLite quantization",
        ],
    )

    # Create placeholder for future TFLite
    _create_tflite_placeholder(output_dir)

    return stub_path, budget


def _create_tflite_placeholder(output_dir: Path) -> None:
    """Create placeholder structure for future TFLite support."""
    import textwrap

    tflite_dir = output_dir / "tflite_future"
    tflite_dir.mkdir(exist_ok=True)

    readme = textwrap.dedent(
        """\
        # TFLite Export (Future - H4)

        This directory is prepared for future TFLite model export and quantization.

        ## Planned Features (H4 Deep Model)

        1. **Model Architecture**
           - 1D CNN or LSTM for temporal patterns
           - ~100K parameters target
           - Input: 30s windows @ 100Hz

        2. **Quantization (planned)**
           - Post-training quantization to int8
           - Representative dataset from training data
           - Target: <100KB model size

        3. **Deployment**
           - TensorFlow Lite Micro for MCU
           - Android/iOS via TFLite interpreter
           - ONNX Runtime Mobile as alternative

        ## Current Status

        - [ ] Deep model training
        - [ ] TFLite conversion
        - [ ] Int8 quantization
        - [ ] Edge benchmark

        ## Fallback (Plan B)

        If model deployment is infeasible:
        - Pre-compute predictions for known scenarios
        - Store lookup table on device
        - ~10KB storage requirement
        """
    )

    with open(tflite_dir / "README.md", "w") as f:
        f.write(readme)


def _get_limitations(format_type: str) -> list[str]:
    """Get limitations for the export format."""
    if format_type == "onnx":
        return [
            "Tree ensemble models may have large ONNX graphs",
            "No GPU acceleration for tree models",
            "Consider model compression for very large forests",
        ]
    elif format_type == "pickle_stub":
        return [
            "Requires Python runtime on edge device",
            "Not suitable for MCU deployment",
            "Use ONNX or TFLite for true edge deployment",
            "Plan B (playback) recommended for resource-constrained devices",
        ]
    else:
        return ["Unknown format - review manually"]


def estimate_edge_budget(
    model_path: Path,
    n_features: int = 45,
    target_latency_ms: float = 100.0,
) -> EdgeBudget:
    """
    Estimate edge budget for a model without exporting.

    Args:
        model_path: Path to model file.
        n_features: Number of input features.
        target_latency_ms: Target latency constraint.

    Returns:
        EdgeBudget estimation.
    """
    model_size = os.path.getsize(model_path)

    # Detect format
    suffix = model_path.suffix.lower()
    if suffix == ".onnx":
        format_type = "onnx"
        ram_multiplier = 2.0
        latency_base = 1.0
    elif suffix in (".joblib", ".pkl", ".pickle"):
        format_type = "pickle"
        ram_multiplier = 3.0
        latency_base = 2.0
    elif suffix == ".tflite":
        format_type = "tflite"
        ram_multiplier = 1.5
        latency_base = 0.5
    else:
        format_type = "unknown"
        ram_multiplier = 3.0
        latency_base = 5.0

    estimated_ram = model_size * ram_multiplier
    estimated_latency = latency_base + (n_features * 0.01)

    return EdgeBudget(
        model_size_bytes=model_size,
        model_size_kb=model_size / 1024,
        model_size_mb=model_size / (1024 * 1024),
        estimated_ram_kb=estimated_ram / 1024,
        estimated_latency_ms=estimated_latency,
        target_latency_ms=target_latency_ms,
        meets_budget=estimated_latency <= target_latency_ms,
        format=format_type,
        quantization="none",
        n_features=n_features,
        notes=[f"Estimated from {format_type} model file"],
    )
