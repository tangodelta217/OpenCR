# Edge Deployment Strategy (P0 -> H4)

## P0: Current Status (Baseline Models)

The current implementation supports standard machine learning models (Random Forest, Gradient Boosting) trained via `scikit-learn`.

### Deployment Artifacts
1. **ONNX Export**
   - **Supported**: RandomForest, GradientBoosting, SVM
   - **Toolchain**: `skl2onnx` + `onnxruntime`
   - **Performance**: CPU-optimized, no quantization (float32)
   - **Quantization**: Not applicable for baseline sklearn exports (`--quantize` errors)
   - **Limitations**: Large tree ensembles can produce large ONNX graphs.

2. **Pickle Stub (Fallback)**
   - **Usage**: Use when ONNX export fails or for rapid prototyping.
   - **Requirements**: Full Python runtime + `scikit-learn`.
   - **Constraint**: Not suitable for microcontrollers (MCUs).

3. **Budget Estimation**
   - Automatically generates `edge_budget.json`.
   - Estimates/measures: Model Size, Flash usage, Inference Latency.
   - RAM usage is not measured for baseline sklearn exports (reported as unknown).
   - Includes "Plan B" (Playback strategy) data structure.

### Budget Semantics (P0)
- **Flash**: measured as the total size of exported model artifact + `export_manifest.json`.
- **Latency**: estimated from feature count (heuristic).
- **RAM**: unknown for baseline sklearn exports (no runtime measurement).
- **meets_budget**:
  - `true` only if latency/flash/ram are all known and pass.
  - `false` if any dimension fails.
  - `null` if any dimension is unknown.

---

## H4: Future Roadmap (Deep Learning)

The goal for Horizon 4 (H4) is to deploy deep learning models (1D-CNN, LSTM) to ultra-low power devices.

### Target Architecture
- **Model**: Lightweight 1D-CNN using depthwise separable convolutions.
- **Input**: 30s PPG/BioZ windows @ 100Hz.
- **Framework**: PyTorch -> TFLite -> TFLite Micro.

### Optimization Pipeline (Planned)
1. **Quantization Aware Training (QAT)**
   - Train with fake quantization nodes to minimize accuracy loss.
2. **Post-Training Quantization (PTQ)**
   - Convert float32 weights/activations to int8.
   - Target size: < 100 KB.
3. **Microcontroller Deployment**
   - **Platform**: ESP32 / Arduino Nano 33 BLE Sense.
   - **Inference Engine**: TensorFlow Lite for Microcontrollers.

### Deployment Checklist (H4, not implemented yet)
- [ ] Train PyTorch model with `opencr train --model cnn_v1`
- [ ] Export to ONNX (future): `opencr edge export --format onnx`
- [ ] Convert to TFLite (future): `opencr edge export --format tflite --quantize`
- [ ] Verify accuracy on PC using TFLite interpreter
- [ ] Measure power consumption on reference board
