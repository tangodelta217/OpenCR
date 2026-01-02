"""Host-side benchmark utilities for edge artifacts."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from opencr.logging import get_logger

logger = get_logger(__name__)


def _summarize_timings(timings_ms: list[float]) -> dict[str, float] | None:
    if not timings_ms:
        return None
    values = np.array(timings_ms, dtype=np.float64)
    return {
        "mean_ms": float(values.mean()),
        "std_ms": float(values.std()),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
        "p50_ms": float(np.percentile(values, 50)),
        "p95_ms": float(np.percentile(values, 95)),
    }


def _benchmark_callable(fn, iterations: int, warmup: int) -> list[float]:
    for _ in range(max(warmup, 0)):
        fn()
    timings = []
    for _ in range(max(iterations, 0)):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        timings.append((end - start) * 1000.0)
    return timings


def _infer_n_features_from_metadata(model_path: Path) -> int | None:
    metadata_path = model_path.parent / "metadata.json"
    if metadata_path.exists():
        with open(metadata_path, encoding="utf-8") as f:
            metadata = json.load(f)
        feature_names = metadata.get("feature_names")
        if isinstance(feature_names, list) and feature_names:
            return len(feature_names)

    schema_path = model_path.parent / "feature_schema.json"
    if schema_path.exists():
        with open(schema_path, encoding="utf-8") as f:
            schema = json.load(f)
        feature_names = schema.get("feature_names")
        if isinstance(feature_names, list) and feature_names:
            return len(feature_names)

    return None


def _normalize_shape(shape: list[Any]) -> list[int]:
    normalized = []
    for dim in shape:
        if isinstance(dim, (int, np.integer)) and dim > 0:
            normalized.append(int(dim))
        else:
            normalized.append(1)
    return normalized


def _benchmark_onnx(model_path: Path, iterations: int, warmup: int) -> dict[str, Any]:
    try:
        import onnxruntime as ort
    except ImportError:
        return {
            "status": "skipped",
            "reason": "onnxruntime not installed",
        }

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = _normalize_shape(list(input_meta.shape))
    input_data = np.random.randn(*input_shape).astype(np.float32)

    timings = _benchmark_callable(
        lambda: session.run(None, {input_name: input_data}),
        iterations,
        warmup,
    )

    return {
        "status": "ok",
        "runtime": "onnxruntime",
        "input_name": input_name,
        "input_shape": input_shape,
        "batch_size": input_shape[0] if input_shape else 1,
        "iterations": iterations,
        "warmup": warmup,
        "timings_ms": timings,
        "summary": _summarize_timings(timings),
    }


def _benchmark_sklearn(model_path: Path, iterations: int, warmup: int) -> dict[str, Any]:
    import joblib

    model = joblib.load(model_path)
    n_features = getattr(model, "n_features_in_", None)
    if not n_features:
        n_features = _infer_n_features_from_metadata(model_path)
    if not n_features:
        n_features = 1

    input_data = np.random.randn(1, int(n_features)).astype(np.float32)

    timings = _benchmark_callable(
        lambda: model.predict(input_data),
        iterations,
        warmup,
    )

    return {
        "status": "ok",
        "runtime": "sklearn",
        "input_shape": [1, int(n_features)],
        "batch_size": 1,
        "iterations": iterations,
        "warmup": warmup,
        "timings_ms": timings,
        "summary": _summarize_timings(timings),
    }


def benchmark_edge_model(
    model_path: Path,
    iterations: int = 100,
    warmup: int = 10,
    device: str = "cpu",
) -> dict[str, Any]:
    """Run a host-side benchmark on a model artifact."""
    model_path = Path(model_path)
    suffix = model_path.suffix.lower()
    size_bytes = model_path.stat().st_size

    benchmark: dict[str, Any] = {
        "model_path": str(model_path),
        "model_format": suffix.lstrip(".") if suffix else "unknown",
        "device": device,
        "artifact": {
            "size_bytes": size_bytes,
            "size_kb": round(size_bytes / 1024, 2),
        },
        "inference": {},
        "notes": [],
    }

    if device != "cpu":
        benchmark["notes"].append("Only CPU execution is supported for host benchmark.")

    if suffix == ".onnx":
        benchmark["inference"] = _benchmark_onnx(model_path, iterations, warmup)
    elif suffix in (".joblib", ".pkl", ".pickle"):
        benchmark["inference"] = _benchmark_sklearn(model_path, iterations, warmup)
    else:
        benchmark["inference"] = {
            "status": "skipped",
            "reason": f"Unsupported model format: {suffix or 'unknown'}",
        }

    if benchmark["inference"].get("status") != "ok":
        reason = benchmark["inference"].get("reason", "unknown reason")
        logger.warning(f"Benchmark skipped or partial: {reason}")

    return benchmark


def write_benchmark_tables(output_dir: Path, benchmark: dict[str, Any]) -> dict[str, Path]:
    """Write benchmark summary tables (CSV and Markdown)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary = benchmark.get("inference", {}).get("summary") or {}
    status = benchmark.get("inference", {}).get("status", "unknown")
    fmt = benchmark.get("model_format", "unknown")
    size_kb = benchmark.get("artifact", {}).get("size_kb", "N/A")
    iterations = benchmark.get("inference", {}).get("iterations", "N/A")
    batch_size = benchmark.get("inference", {}).get("batch_size", "N/A")

    csv_path = output_dir / "benchmark_summary.csv"
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(
            "model_format,size_kb,iterations,batch_size,status,mean_ms,p50_ms,p95_ms,min_ms,max_ms\n"
        )
        f.write(
            f"{fmt},{size_kb},{iterations},{batch_size},{status},"
            f"{summary.get('mean_ms', 'N/A')},"
            f"{summary.get('p50_ms', 'N/A')},"
            f"{summary.get('p95_ms', 'N/A')},"
            f"{summary.get('min_ms', 'N/A')},"
            f"{summary.get('max_ms', 'N/A')}\n"
        )

    md_path = output_dir / "benchmark_summary.md"
    lines = [
        "# Edge Benchmark Summary",
        "",
        "| Format | Size (KB) | Iterations | Batch | Status | Mean (ms) | P50 (ms) | P95 (ms) | Min (ms) | Max (ms) |",
        "|--------|-----------|------------|-------|--------|-----------|----------|----------|----------|----------|",
        f"| {fmt} | {size_kb} | {iterations} | {batch_size} | {status} | "
        f"{summary.get('mean_ms', 'N/A')} | {summary.get('p50_ms', 'N/A')} | "
        f"{summary.get('p95_ms', 'N/A')} | {summary.get('min_ms', 'N/A')} | "
        f"{summary.get('max_ms', 'N/A')} |",
    ]
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    return {"csv": csv_path, "md": md_path}
