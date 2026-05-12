import os
import time
import numpy as np
import torch
import torchvision.models as models
import onnxruntime as ort


def _percentile(data, p):
    return float(np.percentile(data, p))


def _run_pytorch(model, dummy_np, n_runs):
    dummy = torch.tensor(dummy_np)
    latencies = []
    with torch.no_grad():
        for _ in range(n_runs):
            t0 = time.perf_counter()
            model(dummy)
            latencies.append(time.perf_counter() - t0)
    return latencies


def _run_ort(session, dummy_np, n_runs):
    input_name = session.get_inputs()[0].name
    latencies = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        session.run(None, {input_name: dummy_np})
        latencies.append(time.perf_counter() - t0)
    return latencies


def _stats(latencies, model_path):
    arr = np.array(latencies) * 1000  # convert to ms
    return {
        "p50_ms": _percentile(arr, 50),
        "p90_ms": _percentile(arr, 90),
        "p99_ms": _percentile(arr, 99),
        "mean_ms": float(arr.mean()),
        "min_ms": float(arr.min()),
        "max_ms": float(arr.max()),
        "model_size_mb": round(os.path.getsize(model_path) / (1024 ** 2), 2),
    }


def benchmark_all(pytorch_path, onnx_path, quantized_path, n_runs=100):
    dummy_np = np.random.randn(1, 3, 224, 224).astype(np.float32)

    print(f"[1/3] Benchmarking PyTorch (n={n_runs})...")
    model = models.mobilenet_v2(weights=None)
    model.load_state_dict(torch.load(pytorch_path, map_location="cpu", weights_only=True))
    model.eval()
    pt_latencies = _run_pytorch(model, dummy_np, n_runs)

    print(f"[2/3] Benchmarking ONNX FP32 (n={n_runs})...")
    sess_fp32 = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    fp32_latencies = _run_ort(sess_fp32, dummy_np, n_runs)

    print(f"[3/3] Benchmarking ONNX INT8 (n={n_runs})...")
    sess_int8 = ort.InferenceSession(quantized_path, providers=["CPUExecutionProvider"])
    int8_latencies = _run_ort(sess_int8, dummy_np, n_runs)

    return {
        "pytorch": _stats(pt_latencies, pytorch_path),
        "onnx_fp32": _stats(fp32_latencies, onnx_path),
        "onnx_int8": _stats(int8_latencies, quantized_path),
    }
