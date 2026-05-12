import os
import torch
import torchvision.models as models
import onnx
import numpy as np


def export_pytorch_to_onnx():
    os.makedirs("models/pytorch", exist_ok=True)
    os.makedirs("models/onnx", exist_ok=True)

    print("[1/4] Loading pretrained MobileNetV2...")
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    model.eval()

    pytorch_path = "models/pytorch/mobilenetv2.pt"
    torch.save(model.state_dict(), pytorch_path)
    pt_size = os.path.getsize(pytorch_path) / (1024 ** 2)
    print(f"[2/4] Saved PyTorch weights → {pytorch_path} ({pt_size:.2f} MB)")

    dummy_input = torch.randn(1, 3, 224, 224)
    onnx_path = "models/onnx/mobilenetv2.onnx"

    print("[3/4] Exporting to ONNX (opset 17, legacy scripted path)...")
    # Use the legacy TorchScript-based exporter (dynamo=False) so all weights are
    # embedded in the .onnx file and onnxruntime quantization can run shape inference.
    torch.onnx.export(
        model,
        dummy_input,
        onnx_path,
        opset_version=17,
        dynamo=False,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
    )

    print("[4/4] Validating ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    onnx_size = os.path.getsize(onnx_path) / (1024 ** 2)
    print(f"      ONNX model valid → {onnx_path} ({onnx_size:.2f} MB)")

    return onnx_path
