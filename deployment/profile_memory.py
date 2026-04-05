from memory_profiler import profile
import numpy as np
import onnxruntime as ort

@profile
def run_inference():
    so = ort.SessionOptions()
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL

    session = ort.InferenceSession("deployment/cnn.onnx",so)
    input_name = session.get_inputs()[0].name

    input_data = np.random.randn(1, 1, 28, 28).astype(np.float32)

    for _ in range(100):
        session.run(None,{input_name: input_data})

if __name__ == "__main__":
    run_inference()