import io
import os
import numpy as np
import onnxruntime as ort
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

# Initialize ONNX Runtime Session
model_path = os.path.join(os.path.dirname(__file__), "model.onnx")
if not os.path.exists(model_path):
    raise RuntimeError(f"Trained ONNX model not found at {model_path}!")

# Load the session (automatically detects model.onnx.data in the same folder)
session = ort.InferenceSession(model_path)
input_name = session.get_inputs()[0].name

# Label Encoder indices mapping: alphabetical sorted classes
# Index 0: Cat, Index 1: Dog, Index 2: Wild Animal
class_names = ["Cat", "Dog", "Wild Animal"]

# Helper function to preprocess images using pure numpy & PIL (no PyTorch/torchvision dependencies)
def preprocess_image(image: Image.Image) -> np.ndarray:
    # 1. Resize (Bilinear interpolation matches torchvision default)
    image = image.resize((128, 128), Image.Resampling.BILINEAR)
    
    # 2. ToTensor (convert to float32, scale to [0, 1], transpose HWC to CHW)
    img_arr = np.array(image).astype(np.float32) / 255.0
    img_arr = img_arr.transpose(2, 0, 1)
    
    # 3. Normalize (ImageNet stats)
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32).reshape(3, 1, 1)
    std = np.array([0.229, 0.224, 0.225], dtype=np.float32).reshape(3, 1, 1)
    img_arr = (img_arr - mean) / std
    
    # 4. Add batch dimension (1, 3, 128, 128)
    img_arr = np.expand_dims(img_arr, axis=0)
    return img_arr

# Helper function to compute softmax
def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

# Create FastAPI App
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
        
        # Preprocess
        input_data = preprocess_image(image)
        
        # Inference using ONNX Runtime
        outputs = session.run(None, {input_name: input_data})
        logits = outputs[0][0]  # Shape: (3,)
        
        # Calculate probabilities using Softmax
        probabilities = softmax(logits).tolist()
        predicted_idx = int(np.argmax(logits))
        
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
