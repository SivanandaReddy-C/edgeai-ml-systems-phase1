import torch
from models.cnn import CNN
torch._dynamo.config.suppress_errors = True

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
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    dynamo=False  
)

print("Model exported to cnn.onnx")