import onnx

model = onnx.load("deployment/cnn.onnx")

onnx.checker.check_model(model)

print("ONNX model is valid")