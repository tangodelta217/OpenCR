# Edge rules (P0 export + budget + benchmark)

## P0 scope
- Baseline sklearn models:
  - ONNX export is OPTIONAL (depends on skl2onnx).
  - If dependency is missing, export must degrade gracefully (pickle/joblib) and record that ONNX is unavailable.

## Budget honesty
- meets_budget must not be optimistic:
  - Each dimension (latency/flash/ram) has pass/fail/unknown.
  - meets_budget:
    - true only if all known dimensions pass and none fail
    - false if any fails
    - unknown if any dimension is unknown

## Quantization
- For sklearn baseline, int8 quantization is "not applicable" and should error clearly if requested.
- Roadmap is fine, but do not claim int8 exists unless an actual .tflite is produced.

## Benchmark
- Host benchmark must measure:
  - artifact size (bytes)
  - inference time (ms) and/or feature extraction time
