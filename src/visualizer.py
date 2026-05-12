import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def generate_charts(results):
    os.makedirs("results", exist_ok=True)

    labels = ["PyTorch", "ONNX FP32", "ONNX INT8"]
    keys = ["pytorch", "onnx_fp32", "onnx_int8"]

    p50 = [results[k]["p50_ms"] for k in keys]
    p99 = [results[k]["p99_ms"] for k in keys]
    sizes = [results[k]["model_size_mb"] for k in keys]

    x = np.arange(len(labels))
    bar_width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(
        "MLToolchain — PyTorch → ONNX → INT8 Quantization Pipeline",
        fontsize=13,
        fontweight="bold",
    )

    bars1 = ax1.bar(x - bar_width / 2, p50, bar_width, label="p50", color="#4C72B0")
    bars2 = ax1.bar(x + bar_width / 2, p99, bar_width, label="p99", color="#DD8452")
    ax1.set_xlabel("Format")
    ax1.set_ylabel("Latency (ms)")
    ax1.set_title("Inference Latency Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels)
    ax1.legend()
    ax1.bar_label(bars1, fmt="%.1f", padding=2, fontsize=8)
    ax1.bar_label(bars2, fmt="%.1f", padding=2, fontsize=8)

    bars3 = ax2.bar(labels, sizes, color=["#4C72B0", "#DD8452", "#55A868"])
    ax2.set_xlabel("Format")
    ax2.set_ylabel("Size (MB)")
    ax2.set_title("Model Size Comparison")
    ax2.bar_label(bars3, fmt="%.1f MB", padding=2, fontsize=8)

    plt.tight_layout()
    out_path = "results/pipeline_comparison.png"
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Chart saved → {out_path}")
    return out_path
