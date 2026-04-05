import time
import numpy as np
import onnxruntime as ort

input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)
print("Available providers:", ort.get_available_providers())
providers_list = [
    ["CPUExecutionProvider"],["AzureExecutionProvider"],
]

for providers in providers_list:
    print(f"\nProvider: {providers}")

    session = ort.InferenceSession(
        "deployment/cnn.onnx",
        providers=providers
    )

    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(20):
        session.run(None,{input_name: input_data})

    # Benchmark
    runs = 200
    start = time.perf_counter()

    for _ in range(runs):
        session.run(None, {input_name: input_data})

    end = time.perf_counter()

    latency = (end - start) / runs * 1000
    print(f"Latency: {latency:.4f} ms")