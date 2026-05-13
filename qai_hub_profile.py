#!/usr/bin/env python3
"""
MobileNetV2 — Qualcomm AI Hub profiling on Snapdragon 8 Gen 3 (Hexagon NPU).

Jobs submitted:
  1. Compile FP32  → QNN runtime
  2. Quantize INT8 (100 calibration samples) + Compile INT8 → QNN runtime
  3. Profile FP32 compiled model
  4. Profile INT8 compiled model
"""

import getpass
import json
import os
import sys

import numpy as np
import torch
import torchvision.models as models


# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

def _get_token() -> str:
    token = os.environ.get("QAI_HUB_API_TOKEN", "").strip()
    if token:
        print("[auth] Using token from QAI_HUB_API_TOKEN env var.")
        return token
    token = getpass.getpass("[auth] Enter your Qualcomm AI Hub API token (hidden): ").strip()
    if not token:
        sys.exit("ERROR: API token is required — aborting.")
    return token


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def _prepare_mobilenetv2():
    print("[model] Loading pretrained MobileNetV2 (ImageNet weights)...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()
    dummy = torch.randn(1, 3, 224, 224)

    traced = torch.jit.trace(model, dummy)
    print("[model] Tracing complete.")

    # Export ONNX — required by the quantize job
    os.makedirs("models/onnx", exist_ok=True)
    onnx_path = "models/onnx/mobilenetv2_hub.onnx"
    print(f"[model] Exporting ONNX → {onnx_path} ...")
    torch.onnx.export(
        model, dummy, onnx_path,
        opset_version=17,
        dynamo=False,
        input_names=["input_1"],
        output_names=["output"],
    )
    print("[model] ONNX export complete.")
    return traced, onnx_path


# ---------------------------------------------------------------------------
# Result parsing
# ---------------------------------------------------------------------------

def _parse_profile(profile_data: dict, label: str) -> dict:
    """Extract latency, memory, and compute-unit breakdown from qai_hub profile dict.

    Profile schema (from download_profile()):
      execution_summary.all_inference_times  — list of per-run times in µs
      execution_summary.estimated_inference_peak_memory — bytes
      execution_detail  — list of {name, type, compute_unit, execution_time, execution_cycles}
    """
    metrics = {"label": label}

    summary = profile_data.get("execution_summary", {})

    # Inference times (µs). Skip index 0 (cold/JIT first run) to match AI Hub UI p50/p99.
    raw_times = summary.get("all_inference_times", [])
    if raw_times:
        warm = np.array(raw_times[1:] if len(raw_times) > 1 else raw_times)
        metrics["p50_us"] = float(np.percentile(warm, 50))
        metrics["p99_us"] = float(np.percentile(warm, 99))
        metrics["p50_ms"] = round(metrics["p50_us"] / 1000, 4)
        metrics["p99_ms"] = round(metrics["p99_us"] / 1000, 4)
    else:
        metrics["p50_us"] = metrics["p99_us"] = None
        metrics["p50_ms"] = metrics["p99_ms"] = None

    # Peak memory during inference (bytes → MB)
    peak_bytes = summary.get("estimated_inference_peak_memory")
    metrics["peak_memory_mb"] = round(peak_bytes / (1024 ** 2), 1) if peak_bytes else None

    # Operator breakdown — execution_detail contains per-op rows; skip I/O bookkeeping entries
    _IO_TYPES = {"Input", "input", "Output", "output", "QNN"}
    layers = [l for l in profile_data.get("execution_detail", []) if l.get("type") not in _IO_TYPES]

    npu_ops = cpu_ops = gpu_ops = 0
    npu_time_us = cpu_time_us = 0.0
    for layer in layers:
        unit = (layer.get("compute_unit") or "").upper()
        exec_time = float(layer.get("execution_time", 0) or 0)
        if any(k in unit for k in ("NPU", "HTP", "DSP", "HEXAGON")):
            npu_ops += 1
            npu_time_us += exec_time
        elif "CPU" in unit:
            cpu_ops += 1
            cpu_time_us += exec_time
        elif "GPU" in unit:
            gpu_ops += 1

    metrics["total_layers"] = len(layers)
    metrics["npu_ops"] = npu_ops
    metrics["cpu_ops"] = cpu_ops
    metrics["gpu_ops"] = gpu_ops
    total_time = npu_time_us + cpu_time_us
    # If execution_time is 0 for all NPU layers (cycle-only reporting), fall back to 100% NPU
    if total_time == 0 and npu_ops > 0:
        metrics["npu_time_pct"] = 100.0
    elif total_time > 0:
        metrics["npu_time_pct"] = round(npu_time_us / total_time * 100, 1)
    else:
        metrics["npu_time_pct"] = None

    return metrics


def _print_metrics(m: dict):
    print(f"\n  === {m['label']} ===")
    if m.get("p50_ms") is not None:
        print(f"  Inference latency  p50 : {m['p50_ms']:.3f} ms")
        print(f"  Inference latency  p99 : {m['p99_ms']:.3f} ms")
    else:
        print("  Inference latency       : (not available in profile dump)")
    if m.get("peak_memory_mb") is not None:
        print(f"  Peak memory             : {m['peak_memory_mb']:.3f} MB")
    else:
        print("  Peak memory             : (not reported)")
    if m["total_layers"]:
        print(f"  Layers total            : {m['total_layers']}")
        print(f"    NPU ops               : {m['npu_ops']}")
        print(f"    CPU ops               : {m['cpu_ops']}")
        print(f"    GPU ops               : {m['gpu_ops']}")
        if m.get("npu_time_pct") is not None:
            print(f"    NPU time share        : {m['npu_time_pct']} %")
    else:
        print("  Layer breakdown         : (visit the job URL for the full layer table)")


def _fmt_line(f, label, value):
    f.write(f"  {label:<26}: {value}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    try:
        import qai_hub as hub
    except ImportError:
        sys.exit("ERROR: qai_hub not installed. Run: pip install qai-hub")

    token = _get_token()
    client = hub.Client(config=hub.ClientConfig(token))

    device = hub.Device("Snapdragon 8 Elite QRD")
    traced_model, onnx_path = _prepare_mobilenetv2()

    input_specs = {"input_1": ((1, 3, 224, 224), "float32")}
    qnn_option = "--target_runtime qnn_context_binary"

    # ------------------------------------------------------------------
    # Job 1 — Compile FP32
    # ------------------------------------------------------------------
    print("\n[job1] Submitting FP32 compile job (QNN runtime)...")
    compile_fp32 = client.submit_compile_job(
        model=traced_model,
        device=device,
        input_specs=input_specs,
        options=qnn_option,
        name="mobilenetv2_fp32_compile",
    )
    print(f"[job1] FP32 compile URL  : {compile_fp32.url}")

    # ------------------------------------------------------------------
    # Job 2a — Quantize INT8 (100 calibration samples)
    # ------------------------------------------------------------------
    print("\n[job2a] Submitting INT8 quantize job (100 calibration samples)...")
    # calibration_data must be Mapping[str, list[np.ndarray]]
    cal_data = {
        "input_1": [
            np.random.randn(1, 3, 224, 224).astype(np.float32)
            for _ in range(100)
        ]
    }
    quantize_job = client.submit_quantize_job(
        model=onnx_path,
        calibration_data=cal_data,
        weights_dtype=hub.QuantizeDtype.INT8,
        activations_dtype=hub.QuantizeDtype.INT8,
        name="mobilenetv2_int8_quantize",
    )
    print(f"[job2a] Quantize URL     : {quantize_job.url}")

    print("[job2a] Waiting for quantize job to complete (uploads 100 samples first)...")
    qstatus = quantize_job.wait()
    if qstatus.failure:
        sys.exit(f"[job2a] Quantize job FAILED: {qstatus.message}")
    print("[job2a] Quantize job complete.")

    # ------------------------------------------------------------------
    # Job 2b — Compile INT8
    # ------------------------------------------------------------------
    print("\n[job2b] Submitting INT8 compile job (QNN runtime)...")
    compile_int8 = client.submit_compile_job(
        model=quantize_job.get_target_model(),
        device=device,
        input_specs=input_specs,
        options=qnn_option,
        name="mobilenetv2_int8_compile",
    )
    print(f"[job2b] INT8 compile URL : {compile_int8.url}")

    # ------------------------------------------------------------------
    # Wait for both compile jobs
    # ------------------------------------------------------------------
    print("\n[compile] Waiting for FP32 compile job...")
    fp32_cstatus = compile_fp32.wait()
    if fp32_cstatus.failure:
        sys.exit(f"[job1] FP32 compile FAILED: {fp32_cstatus.message}")
    print("[compile] FP32 compile done.")

    print("[compile] Waiting for INT8 compile job...")
    int8_cstatus = compile_int8.wait()
    if int8_cstatus.failure:
        sys.exit(f"[job2b] INT8 compile FAILED: {int8_cstatus.message}")
    print("[compile] INT8 compile done.")

    # ------------------------------------------------------------------
    # Job 3 — Profile FP32
    # ------------------------------------------------------------------
    print("\n[job3] Submitting FP32 profile job (real Snapdragon 8 Elite hardware)...")
    profile_fp32 = client.submit_profile_job(
        model=compile_fp32.get_target_model(),
        device=device,
        name="mobilenetv2_fp32_profile",
    )
    print(f"[job3] FP32 profile URL  : {profile_fp32.url}")

    # ------------------------------------------------------------------
    # Job 4 — Profile INT8
    # ------------------------------------------------------------------
    print("\n[job4] Submitting INT8 profile job (real Snapdragon 8 Elite hardware)...")
    profile_int8 = client.submit_profile_job(
        model=compile_int8.get_target_model(),
        device=device,
        name="mobilenetv2_int8_profile",
    )
    print(f"[job4] INT8 profile URL  : {profile_int8.url}")

    # ------------------------------------------------------------------
    # Wait for profile jobs (actual NPU execution — may take several minutes)
    # ------------------------------------------------------------------
    print("\n[profile] Waiting for FP32 profile job (real hardware queuing may take a few minutes)...")
    fp32_pstatus = profile_fp32.wait()
    if fp32_pstatus.failure:
        sys.exit(f"[job3] FP32 profile FAILED: {fp32_pstatus.message}")
    print("[profile] FP32 profile done.")

    print("[profile] Waiting for INT8 profile job...")
    int8_pstatus = profile_int8.wait()
    if int8_pstatus.failure:
        sys.exit(f"[job4] INT8 profile FAILED: {int8_pstatus.message}")
    print("[profile] INT8 profile done.")

    # ------------------------------------------------------------------
    # Extract and display results
    # ------------------------------------------------------------------
    fp32_data = profile_fp32.download_profile()
    int8_data = profile_int8.download_profile()

    fp32_metrics = _parse_profile(fp32_data, "MobileNetV2 FP32 (QNN / Snapdragon 8 Elite QRD)")
    int8_metrics = _parse_profile(int8_data, "MobileNetV2 INT8 (QNN / Snapdragon 8 Elite QRD)")

    print("\n" + "=" * 64)
    print("  PROFILING RESULTS — Snapdragon 8 Elite QRD (Hexagon NPU)")
    print("=" * 64)
    _print_metrics(fp32_metrics)
    _print_metrics(int8_metrics)
    print()

    if fp32_metrics.get("p50_ms") and int8_metrics.get("p50_ms"):
        speedup = fp32_metrics["p50_ms"] / int8_metrics["p50_ms"]
        print(f"  INT8 speedup vs FP32 (p50): {speedup:.2f}x")
    print("=" * 64)

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    os.makedirs("results", exist_ok=True)
    output = {
        "device": "Snapdragon 8 Elite QRD",
        "runtime": "QNN (Hexagon NPU)",
        "jobs": {
            "compile_fp32_url": compile_fp32.url,
            "quantize_int8_url": quantize_job.url,
            "compile_int8_url": compile_int8.url,
            "profile_fp32_url": profile_fp32.url,
            "profile_int8_url": profile_int8.url,
        },
        "fp32": fp32_metrics,
        "int8": int8_metrics,
        "raw_fp32_profile": fp32_data,
        "raw_int8_profile": int8_data,
    }

    txt_path = "results/qai_hub_results.txt"
    json_path = "results/qai_hub_results.json"

    with open(txt_path, "w") as f:
        f.write("=" * 64 + "\n")
        f.write("QUALCOMM AI HUB — MobileNetV2 Profiling Results\n")
        f.write("Device : Snapdragon 8 Elite QRD\n")
        f.write("Runtime: QNN (Hexagon NPU)\n")
        f.write("=" * 64 + "\n\n")

        f.write("JOB URLs\n")
        _fmt_line(f, "Compile FP32", compile_fp32.url)
        _fmt_line(f, "Quantize INT8", quantize_job.url)
        _fmt_line(f, "Compile INT8", compile_int8.url)
        _fmt_line(f, "Profile FP32", profile_fp32.url)
        _fmt_line(f, "Profile INT8", profile_int8.url)
        f.write("\n")

        for m in [fp32_metrics, int8_metrics]:
            f.write(f"{m['label']}\n")
            _fmt_line(f, "p50 latency", f"{m.get('p50_ms')} ms")
            _fmt_line(f, "p99 latency", f"{m.get('p99_ms')} ms")
            _fmt_line(f, "Peak memory", f"{m.get('peak_memory_mb')} MB")
            _fmt_line(f, "Total layers", m.get("total_layers"))
            _fmt_line(f, "NPU ops", m.get("npu_ops"))
            _fmt_line(f, "CPU ops", m.get("cpu_ops"))
            _fmt_line(f, "GPU ops", m.get("gpu_ops"))
            _fmt_line(f, "NPU time share", f"{m.get('npu_time_pct')} %")
            f.write("\n")

        if fp32_metrics.get("p50_ms") and int8_metrics.get("p50_ms"):
            speedup = fp32_metrics["p50_ms"] / int8_metrics["p50_ms"]
            f.write(f"INT8 speedup vs FP32 (p50): {speedup:.2f}x\n")

    with open(json_path, "w") as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved:\n  {txt_path}\n  {json_path}")


if __name__ == "__main__":
    main()
