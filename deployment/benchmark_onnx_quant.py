import time
import numpy as np
import onnxruntime as ort

so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 1

# Load FP32 model
session_fp32 = ort.InferenceSession("deployment/cnn.onnx",so)

# Load INT8 model
session_int8 = ort.InferenceSession("deployment/cnn_int8.onnx",so)

# Get input name
input_name_fp32 = session_fp32.get_inputs()[0].name
input_name_int8 = session_int8.get_inputs()[0].name

# Dummy input
input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)

# Warmup
for _ in range(10):
    session_fp32.run(None,{input_name_fp32: input_data})
    session_int8.run(None,{input_name_int8: input_data})

# Benchmark FP32
start = time.time()
for _ in range(100):
    session_fp32.run(None,{input_name_fp32: input_data})
end = time.time()
fp32_latency = (end - start) / 100 * 1000 #ms

# Benchmark INT8
start = time.time()
for _ in range(100):
    session_int8.run(None,{input_name_int8: input_data})
end = time.time()
int8_latency = (end - start) / 100 * 1000 #ms

print(f"FP32 ONNX Latency: {fp32_latency:.4f} ms")
print(f"INT8 ONNX Latency: {int8_latency:.4f} ms")