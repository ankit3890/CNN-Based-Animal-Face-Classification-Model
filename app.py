import io
import os
import torch
import torch.nn as nn
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from torchvision import transforms

# 1. Define Model Architecture
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
        self.output = nn.Linear(128, 3)  # Three classes: Cat, Dog, Wild

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

# Initialize Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Trained Weights
model_path = os.path.join(os.path.dirname(__file__), "model.pth")
if not os.path.exists(model_path):
    raise RuntimeError(f"Trained model state dict not found at {model_path}!")

model = Net().to(device)
try:
    model.load_state_dict(torch.load(model_path, map_location=device))
except Exception as e:
    # In case the file is a full serialized model instead of a state_dict
    try:
        model = torch.load(model_path, map_location=device)
    except Exception as e2:
        raise RuntimeError(f"Failed to load model weights: {e}\n{e2}")

model.eval()

# 2. Define Image Transformations
# These exact transforms were used in the training pipeline
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])

# Label Encoder indices mapping: alphabetical sorted classes
# Index 0: cat, Index 1: dog, Index 2: wild
class_names = ["Cat", "Dog", "Wild Animal"]

# 3. Create FastAPI App
app = FastAPI(title="Animal Face Classifier API")

# Ensure static files directory exists
static_dir = os.path.join(os.path.dirname(__file__), "static")
os.makedirs(static_dir, exist_ok=True)

# Serve Root Index
@app.get("/")
def get_index():
    index_file = os.path.join(static_dir, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"message": "Server is running, but static/index.html is missing. Please build the frontend."}

# Define Prediction Endpoint
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Verify file is an image if content_type is provided
    if file.content_type and not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image format.")
    
    try:
        # Read image bytes
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        
        # Apply preprocess transformations
        input_tensor = transform(image).to(device)
        input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension (1, 3, 128, 128)
        
        # Inference
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.softmax(outputs, dim=1).squeeze().tolist()
            predicted_idx = torch.argmax(outputs, dim=1).item()
        
        # Format response
        res = {
            "prediction": class_names[predicted_idx],
            "probabilities": {
                class_names[i]: float(probabilities[i]) for i in range(len(class_names))
            }
        }
        return JSONResponse(content=res)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {str(e)}")

# Mount Static Files Directory
app.mount("/static", StaticFiles(directory=static_dir), name="static")
