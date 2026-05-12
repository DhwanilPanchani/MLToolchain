import os
from onnxruntime.quantization import quantize_dynamic, QuantType


def quantize_model(onnx_path):
    os.makedirs("models/quantized", exist_ok=True)
    quantized_path = "models/quantized/mobilenetv2_int8.onnx"

    print("[1/2] Applying INT8 dynamic quantization...")
    quantize_dynamic(
        model_input=onnx_path,
        model_output=quantized_path,
        weight_type=QuantType.QInt8,
    )

    orig_size = os.path.getsize(onnx_path) / (1024 ** 2)
    quant_size = os.path.getsize(quantized_path) / (1024 ** 2)
    ratio = orig_size / quant_size

    print(f"[2/2] Quantization complete:")
    print(f"      Original ONNX : {orig_size:.2f} MB")
    print(f"      INT8 ONNX     : {quant_size:.2f} MB")
    print(f"      Compression   : {ratio:.2f}x")

    return quantized_path
