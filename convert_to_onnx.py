import os
import torch
import torch.nn as nn

# Define the exact model architecture
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.pooling = nn.MaxPool2d(2, 2)
        self.relu = nn.ReLU()
        self.flatten = nn.Flatten()
        self.linear = nn.Linear((128 * 16 * 16), 128)
        self.output = nn.Linear(128, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)

        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)
        return x

# Load model weights
device = torch.device("cpu")
model = Net()
model.load_state_dict(torch.load("model.pth", map_location=device))
model.eval()

# Create dummy input matching the input shape (batch_size=1, channels=3, height=128, width=128)
dummy_input = torch.randn(1, 3, 128, 128, requires_grad=True)

# Export to ONNX
onnx_path = "model.onnx"
print(f"Exporting model to {onnx_path}...")
torch.onnx.export(
    model,
    dummy_input,
    onnx_path,
    export_params=True,
    opset_version=11,
    do_constant_folding=True,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}}
)
print("Model successfully exported to ONNX format!")
