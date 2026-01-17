# MNIST Inference Server

PyTorch-based FastAPI server for MNIST digit classification using a CNN model.

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Train the model (downloads MNIST data automatically)
python train.py --epochs 10

# Start the server
python main.py
```

Or using uvicorn directly:
```bash
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## API Endpoints

### POST /predict
Run inference on MNIST image data.

**Request:**
```json
{
  "image_data": [0.0, 0.1, ..., 1.0],  // 784 pixel values
  "normalize": true
}
```

**Response:**
```json
{
  "prediction": 7,
  "confidence": 0.9923,
  "probabilities": [0.0001, 0.0002, ..., 0.9923],
  "logits": [-5.2, -4.1, ..., 8.3]
}
```

### GET /info
Get model information.

### GET /health
Health check endpoint.

## Model Architecture

- Conv2d(1 -> 32, 3x3) -> ReLU -> MaxPool2d
- Conv2d(32 -> 64, 3x3) -> ReLU -> MaxPool2d
- Dropout(0.25)
- Flatten
- Dense(64*7*7 -> 128) -> ReLU -> Dropout(0.5)
- Dense(128 -> 10)

Expected accuracy: ~99% on test set after training.

## Training

The training script automatically downloads the MNIST dataset. Training typically takes 5-10 minutes on CPU, less on GPU.

```bash
python train.py --epochs 10 --batch-size 128 --lr 0.001
```

The trained model is saved as `mnist_cnn.pth`.
