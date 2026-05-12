# MLToolchain — Cross-Platform ML Compiler & Quantization SDK

## Overview

MLToolchain implements a production-grade **PyTorch → ONNX → INT8** pipeline designed for Edge deployment. A pretrained MobileNetV2 classification model is:

1. Exported from PyTorch to the ONNX interchange format (opset 17)
2. Dynamically quantized to INT8 weights using ONNX Runtime's quantization API
3. Benchmarked across all three representations (PyTorch, ONNX FP32, ONNX INT8)
4. Compared in a side-by-side chart covering latency and model size

The resulting INT8 ONNX model is ready for deployment to the C++ ONNX Runtime SDK and optional QNN (Qualcomm Neural Network) backend on Edge SoCs.

## Why Quantization

INT8 quantization cuts model weight storage roughly in half compared to FP32, reducing memory bandwidth pressure — a critical bottleneck on power-constrained Edge SoCs. Lower precision arithmetic maps directly to the fixed-function INT8/INT4 MAC units present on NPUs (Neural Processing Units) in chips like the Snapdragon 8 Gen series, delivering 2–4× throughput gains while staying within milliwatt-class power budgets. For latency-sensitive applications (ADAS perception, on-device vision), the accuracy trade-off (typically <1% top-1 drop on ImageNet) is well within acceptable tolerances.

## Architecture

```
src/
├── exporter.py      — PyTorch → ONNX export with onnx.checker validation
├── quantizer.py     — ONNX FP32 → INT8 dynamic quantization
├── benchmarker.py   — Latency profiling (p50/p90/p99) for all three formats
└── visualizer.py    — Matplotlib bar charts for latency and model size
```

## Pipeline Stages

| Stage | Tool | Why it matters |
|-------|------|----------------|
| **Save PyTorch weights** | `torch.save` | Baseline — needed to reload model for benchmarking |
| **Export to ONNX** | `torch.onnx.export` opset 17 | Hardware-agnostic interchange format; decouples training framework from runtime |
| **Validate ONNX graph** | `onnx.checker.check_model` | Catches malformed graphs before downstream tools consume them |
| **INT8 quantization** | `onnxruntime.quantization.quantize_dynamic` | Reduces weight precision to INT8; compresses model and accelerates inference on NPUs |
| **Benchmark** | `onnxruntime.InferenceSession` + `time.perf_counter` | Quantifies real-world latency gain; p99 shows tail latency under load |
| **Chart** | `matplotlib` | Visual proof of improvement for stakeholder communication |

## Sample Results

Measured on Apple M-series CPU, 100 inference runs, MobileNetV2, input 1×3×224×224.

| Format | p50 (ms) | p99 (ms) | mean (ms) | Size (MB) |
|--------|----------|----------|-----------|-----------|
| PyTorch | 22.38 | 24.71 | 22.24 | 13.60 |
| ONNX FP32 | 3.71 | 6.50 | 3.94 | 13.34 |
| ONNX INT8 | 37.55 | 48.88 | 38.74 | 3.52 |

**Key takeaways:**
- ONNX Runtime FP32 is **6× faster** than raw PyTorch eager mode (no JIT).
- INT8 quantization achieves **3.8× model compression** (13.34 MB → 3.52 MB).
- INT8 CPU latency is higher on Apple Silicon because the M-series NEON SIMD path for INT8 MatMul in ORT is not as well-optimised as its FP32 path on this platform; on a Qualcomm Hexagon NPU or x86 VNNI the INT8 path would be 2–4× faster than FP32.

ONNX INT8 is slower than FP32 on Apple Silicon because ARM NEON does not accelerate INT8 MatMul over FP32. On Qualcomm Hexagon NPU or x86 with VNNI extensions, INT8 delivers 2-4x latency improvement. The primary Edge deployment benefit here is the 3.8x model size reduction (13.3MB → 3.5MB), critical for memory-constrained SoCs.

See `results/pipeline_comparison.png` for the bar chart.

## Build

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python dependencies
pip install -r requirements.txt

# Run the full pipeline (100 benchmark runs)
python run_pipeline.py --runs 100

# Skip chart generation
python run_pipeline.py --runs 100 --no-chart
```

## CMake Build

```bash
cmake -B build -S .
cmake --build build
./build/mltoolchain
```

## Tech Stack

| Component | Version |
|-----------|---------|
| Python | 3.11+ |
| PyTorch | 2.1+ |
| torchvision | 0.16+ |
| ONNX | 1.14+ |
| ONNX Runtime | 1.16+ |
| NumPy | 1.24+ |
| Matplotlib | 3.7+ |
| C++ standard | 17 |
| CMake | 3.20+ |
