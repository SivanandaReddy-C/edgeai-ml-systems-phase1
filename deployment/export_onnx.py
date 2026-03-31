import torch
from models.cnn import CNN

#Load model
model = CNN()
model.load_state_dict(torch.load("best_cnn.pth"))
model.eval()

# Dummy input (VERY IMPORTANT)
dummy_input = torch.randn(1, 1, 28, 28)

# Export to ONNX
torch.onnx.export(
    model,
    dummy_input,
    "deployment/cnn.onnx",
    input_names=["input"],
    output_names=["output"],
    opset_version=11
)

print("Model exported to cnn.onnx")