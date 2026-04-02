import os
from onnxruntime.quantization import quantize_dynamic

quantize_dynamic(
    "deployment/cnn.onnx",
    "deployment/cnn_int8.onnx"
)

print("ONNX quantization complete!")

fp32_size = os.path.getsize("deployment/cnn.onnx") / 1024
int8_size = os.path.getsize("deployment/cnn_int8.onnx") / 1024

print(f"FP32 model size: {fp32_size:.2f} KB")
print(f"INT8 model size: {int8_size:.2f} KB")