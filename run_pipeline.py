import argparse
import json
import os
from datetime import datetime, timezone

from src.exporter import export_pytorch_to_onnx
from src.quantizer import quantize_model
from src.benchmarker import benchmark_all
from src.visualizer import generate_charts


def print_summary(results):
    fmt = "{:<15} {:>10} {:>10} {:>10} {:>12}"
    header = fmt.format("Format", "p50 (ms)", "p99 (ms)", "mean (ms)", "size (MB)")
    sep = "-" * len(header)
    print(f"\n{sep}")
    print(header)
    print(sep)
    for key, label in [("pytorch", "PyTorch"), ("onnx_fp32", "ONNX FP32"), ("onnx_int8", "ONNX INT8")]:
        r = results[key]
        print(fmt.format(
            label,
            f"{r['p50_ms']:.2f}",
            f"{r['p99_ms']:.2f}",
            f"{r['mean_ms']:.2f}",
            f"{r['model_size_mb']:.2f}",
        ))
    print(sep)


def main():
    parser = argparse.ArgumentParser(description="MLToolchain — end-to-end ML pipeline")
    parser.add_argument("--runs", type=int, default=100, help="Number of benchmark inference runs")
    parser.add_argument("--no-chart", action="store_true", help="Skip chart generation")
    args = parser.parse_args()

    print("\n=== MLToolchain Pipeline ===\n")

    print("--- Stage 1: Export ---")
    onnx_path = export_pytorch_to_onnx()

    print("\n--- Stage 2: Quantize ---")
    quantized_path = quantize_model(onnx_path)

    print("\n--- Stage 3: Benchmark ---")
    pytorch_path = "models/pytorch/mobilenetv2.pt"
    results = benchmark_all(pytorch_path, onnx_path, quantized_path, n_runs=args.runs)

    if not args.no_chart:
        print("\n--- Stage 4: Visualize ---")
        generate_charts(results)

    os.makedirs("results", exist_ok=True)
    results_path = "results/results.json"
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), "runs": args.runs, "results": results}
    with open(results_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nResults saved → {results_path}")

    print_summary(results)
    print()


if __name__ == "__main__":
    main()
