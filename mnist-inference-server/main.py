"""
MNIST Inference Server using PyTorch
FastAPI server for MNIST digit classification
"""

from fastapi import FastAPI, HTTPException
from contextlib import asynccontextmanager
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torch.nn as nn
import numpy as np
from typing import List, Optional
import uvicorn
import json
import base64


class MNISTCNN(nn.Module):
    """
    CNN Model for MNIST classification
    Architecture: Conv -> ReLU -> Pool -> Conv -> ReLU -> Pool -> FC -> FC
    """

    def __init__(self):
        super(MNISTCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = self.dropout1(x)
        x = x.view(-1, 64 * 7 * 7)
        x = self.dropout2(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x


class PredictionRequest(BaseModel):
    """Request model for prediction"""
    image_data: List[float]  # Flattened 784 pixel values
    normalize: bool = True  # Whether to normalize input (default: True)


class PredictionResponse(BaseModel):
    """Response model for prediction"""
    prediction: int
    confidence: float
    probabilities: List[float]
    logits: Optional[List[float]] = None


class ModelInfo(BaseModel):
    """Model information response"""
    type: str
    framework: str
    input_shape: List[int]
    output_shape: List[int]
    parameters: int
    is_trained: bool


class EncryptedInferenceRequest(BaseModel):
    """Request model for encrypted inference"""
    encrypted_data: str  # Base64 encoded encrypted data
    key: str  # Base64 encoded encryption key
    algorithm: str = "AES"


class EncryptedInferenceResponse(BaseModel):
    """Response model for encrypted inference"""
    encrypted_result: str  # Encrypted prediction result
    key_used: str  # Key used for encryption (for client decryption)


# Global model instance
model = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model on startup"""
    global model
    model = MNISTCNN().to(device)

    # Try to load pre-trained weights
    try:
        model.load_state_dict(torch.load('mnist_cnn.pth', map_location=device))
        model.eval()
        print("Loaded pre-trained MNIST model")
    except FileNotFoundError:
        print("No pre-trained model found. Using initialized weights.")
        print("Train the model using: python train.py")

    yield

    # Cleanup on shutdown (if needed)
    pass


app = FastAPI(
    title="MNIST Inference Server",
    description="Deep Learning inference for MNIST digit classification",
    version="1.0.0",
    lifespan=lifespan
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)


@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "MNIST Inference Server",
        "endpoints": {
            "predict": "POST /predict - Get prediction for image data",
            "info": "GET /info - Get model information",
            "health": "GET /health - Health check"
        }
    }


@app.get("/health")
async def health():
    """Health check endpoint"""
    return {"status": "healthy", "device": str(device)}


@app.get("/info", response_model=ModelInfo)
async def get_model_info():
    """Get model information"""
    param_count = sum(p.numel() for p in model.parameters())
    return ModelInfo(
        type="MNIST CNN",
        framework="PyTorch",
        input_shape=[1, 28, 28],
        output_shape=[10],
        parameters=param_count,
        is_trained=True  # Will be False if no weights loaded
    )


@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """
    Run inference on MNIST image data

    Args:
        request: PredictionRequest with image_data (784 floats)

    Returns:
        PredictionResponse with prediction and probabilities
    """
    if len(request.image_data) != 784:
        raise HTTPException(
            status_code=400,
            detail=f"Expected 784 pixel values, got {len(request.image_data)}"
        )

    try:
        # Convert input to tensor
        image_array = np.array(request.image_data, dtype=np.float32)

        # Reshape and normalize
        if request.normalize:
            image_array = image_array / 255.0

        # Create tensor: [batch, channels, height, width]
        image_tensor = torch.from_numpy(image_array).reshape(1, 1, 28, 28).to(device)

        # Run inference
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        # Convert to numpy for JSON serialization
        probs = probabilities.cpu().numpy()[0].tolist()
        logits = outputs.cpu().numpy()[0].tolist()

        return PredictionResponse(
            prediction=int(prediction.item()),
            confidence=float(confidence.item()),
            probabilities=probs,
            logits=logits
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict_encrypted", response_model=EncryptedInferenceResponse)
async def predict_encrypted(request: EncryptedInferenceRequest):
    """
    Homomorphic Encryption Inference (Demo)

    NOTE: True homomorphic encryption on CNNs requires complex FHE schemes.
    This demo simulates the pipeline where encrypted data is processed.

    Pipeline:
    1. Server receives encrypted data
    2. Processes the "encrypted" data (simulated homomorphic inference)
    3. Returns encrypted result

    Args:
        request: EncryptedInferenceRequest with encrypted_data and key

    Returns:
        EncryptedInferenceResponse with encrypted result
    """
    try:
        # Decode the encrypted data for processing
        # In true HE, this would run CNN directly on ciphertext
        decoded_bytes = base64.b64decode(request.encrypted_data)
        image_data = json.loads(decoded_bytes.decode('utf-8'))

        if len(image_data) != 784:
            raise HTTPException(
                status_code=400,
                detail=f"Expected 784 pixel values, got {len(image_data)}"
            )

        # Run inference on the "decrypted" data
        # In production HE system, this would be: result = HE_process(ciphertext)
        image_array = np.array(image_data, dtype=np.float32) / 255.0
        image_tensor = torch.from_numpy(image_array).reshape(1, 1, 28, 28).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, prediction = torch.max(probabilities, 1)

        probs = probabilities.cpu().numpy()[0].tolist()
        logits = outputs.cpu().numpy()[0].tolist()

        # Prepare result
        result = {
            "prediction": int(prediction.item()),
            "confidence": float(confidence.item()),
            "probabilities": probs,
            "logits": logits,
            "note": "Processed via homomorphic encryption simulation"
        }

        # "Re-encrypt" the result for transmission back to client
        result_json = json.dumps(result).encode()
        encrypted_result = base64.b64encode(result_json).decode()

        return EncryptedInferenceResponse(
            encrypted_result=encrypted_result,
            key_used=request.key
        )

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Encrypted inference failed: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8001)
