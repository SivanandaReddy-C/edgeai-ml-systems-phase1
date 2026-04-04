import time
import numpy as np
import onnxruntime as ort

# Session options
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 1

# Load model (use FP32 ONNX for clarity)
session = ort.InferenceSession("deployment/cnn.onnx", so)
input_name = session.get_inputs()[0].name

# Batch sizes to test
batch_sizes = [1, 2, 4, 8, 16, 32]

for batch in batch_sizes:
    input_data = np.random.randn(batch, 1, 28, 28).astype(np.float32)

    # Warmup
    for _ in range(10):
        session.run(None, {input_name: input_data})

    # Timing
    start = time.perf_counter()
    runs = 100

    for _ in range(runs):
        session.run(None, {input_name: input_data})

    end = time.perf_counter()

    total_time = (end - start) / runs
    latency_per_batch = total_time * 1000
    latency_per_sample = (total_time / batch) * 1000

    print(f"\nBatch size: {batch}")
    print(f"Latency (batch): {latency_per_batch:.4f} ms")
    print(f"Latency (per sample): {latency_per_sample:.4f} ms")