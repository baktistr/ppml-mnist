# PPML MNIST - Privacy-Preserving ML with Homomorphic Encryption

A web application demonstrating privacy-preserving machine learning using simulated homomorphic encryption for digit classification.

## Architecture

- **Frontend**: React + Tailwind CSS
- **Backend**: FastAPI + PyTorch (MNIST CNN)

## Deployment on Vercel

### Option 1: Deploy as Separate Projects (Recommended)

#### Frontend Deployment

1. **Create a new Vercel project from the frontend folder:**
   ```bash
   cd frontend
   # Install dependencies if needed
   npm install
   ```

2. **Go to [vercel.com/new](https://vercel.com/new)**
   - Import repository: `baktistr/ppml-mnist`
   - Set **Root Directory** to `frontend`
   - **Framework Preset**: Create React App
   - Click **Deploy**

#### Backend Deployment

1. **Create another Vercel project for the backend:**
   - Go to [vercel.com/new](https://vercel.com/new)
   - Import repository: `baktistr/ppml-mnist`
   - Set **Root Directory** to `mnist-inference-server`
   - **Framework Preset**: Python (FastAPI will be auto-detected)
   - Click **Deploy**

2. **Note your backend URL** (e.g., `https://ppml-mnist-backend.vercel.app`)

3. **Update frontend API URL:**
   - Open [Vercel Dashboard](https://vercel.com/dashboard)
   - Go to your frontend project
   - Settings → Environment Variables
   - Add: `REACT_APP_API_URL` = `https://ppml-mnist-backend.vercel.app`
   - Redeploy the frontend

---

### Option 2: Monorepo Setup (Single Project)

1. **Create root `vercel.json`:**
   ```json
   {
     "buildCommand": "echo 'No build command for root'",
     "installCommand": "echo 'No install command for root'"
   }
   ```

2. **Import your project on Vercel:**
   - Set Root Directory: `/` (root)

3. **Vercel will auto-detect both projects** and deploy them separately.

---

## Environment Variables

### Frontend
- `REACT_APP_API_URL`: Backend API URL (default: `http://localhost:8001`)

### Backend
- `PYTHON_VERSION`: `3.9`

---

## Local Development

### Frontend
```bash
cd frontend
npm install
npm start
# Runs on http://localhost:3000
```

### Backend
```bash
cd mnist-inference-server
pip install -r requirements.txt
uvicorn main:app --host 0.0.0.0 --port 8001
# Runs on http://localhost:8001
```

---

## Model Training

The backend uses a pre-trained MNIST CNN model. To train:

```bash
cd mnist-inference-server
python train.py
```

This will create `mnist_cnn.pth` which is used for inference.

---

## Pipeline Stages

1. **Input**: Raw 28x28 pixel data from drawing canvas
2. **Encrypt**: Base64 encoding (simulates HE)
3. **Send**: Transmit encrypted data to server
4. **Server**: ML inference on "encrypted" data
5. **Recv**: Receive encrypted results
6. **Decrypt**: Decode results using secret key
7. **Result**: Display prediction with confidence
