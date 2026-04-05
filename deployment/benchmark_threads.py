import time
import numpy as np
import onnxruntime as ort

# Test different thread counts
thread_settings = [1, 2, 4, 8]

# Input
input_data = np.random.randn(8, 1, 28, 28).astype(np.float32)

for num_threads in thread_settings:
    # Session options
    so = ort.SessionOptions()
    so.intra_op_num_threads = num_threads
    so.inter_op_num_threads = 1
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    # Load model
    session = ort.InferenceSession("deployment/cnn.onnx", so)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(20):
        session.run(None, {input_name: input_data})

    # Benchmark
    runs = 200
    start = time.perf_counter()

    for _ in range(runs):
        session.run(None, {input_name: input_data})

    end = time.perf_counter()

    latency = (end - start) / runs * 1000 #ms

    print(f"Threads: {num_threads} | Latency: {latency:.4f} ms")