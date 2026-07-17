# Animal Face Classification: Model and Web Application

This repository contains a PyTorch deep learning project and an accompanying FastAPI web application designed to classify animal faces into three distinct categories: Cat, Dog, and Wild Animal.

---

## Application Workflow Diagram

The sequence diagram below represents the exact flow of data from the web interface, through the FastAPI backend server, and into the PyTorch CNN model for inference:

```mermaid
sequenceDiagram
    autonumber
    actor User as Client (Web Browser)
    participant Server as FastAPI Backend (app.py)
    participant Model as PyTorch CNN Model (Net)

    User->>Server: POST /predict (Sends Image File)
    Note over Server: Read image bytes and convert to RGB
    Note over Server: Preprocessing Transforms: Resize(128x128) -> ToTensor() -> Normalize()
    Server->>Model: Forward Pass (Tensor of shape 1x3x128x128)
    Model->>Server: Return logits (Tensor of shape 1x3)
    Note over Server: Apply Softmax to logits to get probabilities
    Note over Server: Map highest probability index to class name
    Server->>User: JSON response (Predicted Class and Probabilities)
    Note over User: Update UI elements and animate probability bars
```

---

## Model Architecture Diagram

The flowchart below represents the structure of the custom Convolutional Neural Network (CNN) defined in the project:

```mermaid
graph TD
    Input[Input Tensor: 3x128x128] --> Conv1[Conv2D: 32 channels, kernel 3x3, pad 1]
    Conv1 --> Pool1[MaxPool2D: 2x2, stride 2]
    Pool1 --> ReLU1[ReLU Activation]
    
    ReLU1 --> Conv2[Conv2D: 64 channels, kernel 3x3, pad 1]
    Conv2 --> Pool2[MaxPool2D: 2x2, stride 2]
    Pool2 --> ReLU2[ReLU Activation]
    
    ReLU2 --> Conv3[Conv2D: 128 channels, kernel 3x3, pad 1]
    Conv3 --> Pool3[MaxPool2D: 2x2, stride 2]
    Pool3 --> ReLU3[ReLU Activation]
    
    ReLU3 --> Flat[Flatten: 32768 features]
    Flat --> FC1[Linear: 32768 to 128]
    FC1 --> ReLU4[ReLU Activation]
    ReLU4 --> Out[Output Linear: 128 to 3 Logits]
```

---

## Project Overview

The project uses the Animal Faces High Quality (AFHQ) dataset containing animal faces categorized into:
*   Cat
*   Dog
*   Wild (e.g., foxes, lions, tigers, leopards)

The training pipeline runs for 10 epochs using the Adam optimizer (learning rate 1e-4) and CrossEntropyLoss, achieving high performance on the test set.

*   **Test Dataset Accuracy**: 97.56%
*   **Test Loss**: 0.011
*   **Training Split**: 70% Train (11,291 images), 15% Validation (2,419 images), 15% Test (2,420 images)

---

## Web Application Design

The web application provides a professional interface configured to match a clean and professional dark color palette. It features:
*   **Dashboard metrics**: Displaying accuracy, loss, training epochs, and target classes.
*   **Minimalist palette**: Using solid dark slate backgrounds, solid gray borders, and no gradient color schemes.
*   **Inference Dashboard**: Featuring a drag-and-drop file upload zone, live image preview, and prediction class outputs with bar graphs. Emojis are excluded in favor of clean text labels.
*   **Model Card Section**: Outlining network parameters and layers.

---

## Installation and Execution

1.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
2.  **Start the web server**:
    ```bash
    python -m uvicorn app:app --reload --host 127.0.0.1 --port 8000
    ```
3.  **Access the Dashboard**: Open your web browser and navigate to `http://127.0.0.1:8000/`.
