import { useState, useEffect } from 'react';
import DrawingCanvas from './components/DrawingCanvas';
import BeadTracker from './components/BeadTracker/BeadTracker';
import ProcessVisualization from './components/ProcessVisualization';
import axios from 'axios';
import { getHEInstance } from './utils/encryption';

const API_URL = process.env.REACT_APP_API_URL || 'http://localhost:8001';

function App() {
  const [imageData, setImageData] = useState(null);
  const [stages, setStages] = useState({
    input: { status: 'pending' },
    encryption: { status: 'pending' },
    transmission: { status: 'pending' },
    inference: { status: 'pending' },
    response: { status: 'pending' },
    decryption: { status: 'pending' },
    display: { status: 'pending' }
  });
  const [isProcessing, setIsProcessing] = useState(false);
  const [error, setError] = useState(null);
  const [processData, setProcessData] = useState({
    raw: null,
    encrypted: null,
    inference: null,
    output: null
  });
  const [heInitialized, setHeInitialized] = useState(false);

  // Initialize HE system on mount
  useEffect(() => {
    const initHE = async () => {
      try {
        const he = getHEInstance();
        await he.initialize();
        setHeInitialized(true);
        console.log('HE system initialized successfully');
      } catch (err) {
        console.error('Failed to initialize HE:', err);
        setError('Failed to initialize encryption system');
      }
    };
    initHE();
  }, []);

  const updateStage = (stageName, status, details = {}) => {
    setStages(prev => ({
      ...prev,
      [stageName]: { ...prev[stageName], status, ...details }
    }));
  };

  const handleDrawingComplete = (data) => {
    setImageData(data);
    setProcessData(prev => ({ ...prev, raw: data }));
    updateStage('input', 'completed', { inputSize: data.length, format: '28x28 grayscale' });
  };

  const handleClear = () => {
    setImageData(null);
    setError(null);
    setProcessData({
      raw: null,
      encrypted: null,
      inference: null,
      output: null
    });
    setStages({
      input: { status: 'pending' },
      encryption: { status: 'pending' },
      transmission: { status: 'pending' },
      inference: { status: 'pending' },
      response: { status: 'pending' },
      decryption: { status: 'pending' },
      display: { status: 'pending' }
    });
  };

  const classifyWithHE = async () => {
    if (!imageData) {
      alert('Please draw a digit first!');
      return;
    }

    if (!heInitialized) {
      alert('Encryption system not ready. Please wait...');
      return;
    }

    setIsProcessing(true);
    setError(null);
    const startTime = Date.now();

    try {
      updateStage('encryption', 'in-progress');
      const he = getHEInstance();

      // Encrypt image using HE utilities
      const encryptedImage = await he.encryptImage(imageData);

      updateStage('encryption', 'completed', {
        duration: Date.now() - startTime,
        scheme: he.getSchemeInfo()?.scheme || 'CKKS (TenSEAL)',
        keyPreview: he.getPublicKey()?.substring(0, 24) + '...'
      });

      updateStage('transmission', 'in-progress');
      const transmissionStart = Date.now();

      // Call the real HE inference endpoint
      const inferenceResponse = await axios.post(`${API_URL}/encryption/predict_encrypted`, {
        encrypted_image: encryptedImage
      });

      const transmissionTime = Date.now() - transmissionStart;
      updateStage('transmission', 'completed', {
        duration: transmissionTime,
        dataSize: '~5KB'
      });

      setProcessData(prev => ({
        ...prev,
        encrypted: {
          data: encryptedImage.substring(0, 100) + '...',
          fullData: encryptedImage,
          keyPreview: he.getPublicKey()?.substring(0, 24) + '...'
        }
      }));

      updateStage('inference', 'completed', {
        duration: Date.now() - startTime,
        modelType: 'MNIST CNN (PyTorch)',
        framework: 'PyTorch'
      });

      updateStage('response', 'in-progress');
      await new Promise(resolve => setTimeout(resolve, 50));
      updateStage('response', 'completed', {
        duration: 50,
        resultSize: '~1KB'
      });

      updateStage('decryption', 'in-progress');

      // Decrypt the result using HE utilities
      const encryptedResult = inferenceResponse.data.encrypted_result;
      const decryptedResult = await he.decryptResult(encryptedResult);

      await new Promise(resolve => setTimeout(resolve, 100));
      updateStage('decryption', 'completed', {
        duration: 100
      });

      const prediction = decryptedResult.prediction ?? 0;
      const confidence = decryptedResult.confidence ?? 0;
      const probs = decryptedResult.probabilities ?? [];
      const logits = [];

      setProcessData(prev => ({
        ...prev,
        inference: {
          encryptedResult: encryptedResult,
          logits: logits
        },
        output: {
          prediction,
          confidence,
          probabilities: Array.isArray(probs) ? probs : []
        }
      }));

      const totalTime = Date.now() - startTime;

      updateStage('display', 'completed', {
        duration: 0,
        prediction: String(prediction),
        confidence: (confidence * 100).toFixed(1) + '%',
        totalTime: totalTime
      });

    } catch (err) {
      console.error('Error:', err);
      setError(err.response?.data?.detail || err.message || 'Classification failed');

      const failedStage = Object.keys(stages).find(
        stage => stages[stage].status === 'in-progress'
      );
      if (failedStage) {
        updateStage(failedStage, 'error', { error: err.message });
      }
    } finally {
      setIsProcessing(false);
    }
  };

  const runDemo = async () => {
    if (!heInitialized) {
      alert('Encryption system not ready. Please wait...');
      return;
    }

    setIsProcessing(true);
    setError(null);
    const startTime = Date.now();

    try {
      const demoImage = new Array(784).fill(0);
      for (let i = 5; i < 23; i++) {
        const j = 8 + Math.floor(i * 0.4);
        if (j < 28) demoImage[i * 28 + j] = 255;
      }
      for (let j = 8; j < 22; j++) {
        demoImage[10 * 28 + j] = 255;
      }

      setImageData(demoImage);
      setProcessData(prev => ({ ...prev, raw: demoImage }));

      updateStage('input', 'completed', {
        inputSize: demoImage.length,
        format: '28x28 grayscale',
        isDemo: true
      });

      updateStage('encryption', 'in-progress');
      const he = getHEInstance();

      // Encrypt demo image using HE utilities
      const encryptedImage = await he.encryptImage(demoImage);

      updateStage('encryption', 'completed', {
        duration: Date.now() - startTime,
        scheme: he.getSchemeInfo()?.scheme || 'CKKS (TenSEAL)',
        keyPreview: he.getPublicKey()?.substring(0, 24) + '...'
      });

      updateStage('transmission', 'in-progress');
      await new Promise(resolve => setTimeout(resolve, 50));
      updateStage('transmission', 'completed', { duration: 50, dataSize: '~5KB' });

      updateStage('inference', 'in-progress');

      // Call the real HE inference endpoint
      const inferenceResponse = await axios.post(`${API_URL}/encryption/predict_encrypted`, {
        encrypted_image: encryptedImage
      });

      updateStage('inference', 'completed', {
        duration: Date.now() - startTime,
        modelType: 'MNIST CNN (PyTorch)',
        framework: 'PyTorch'
      });

      setProcessData({
        raw: demoImage,
        encrypted: {
          data: encryptedImage.substring(0, 100) + '...',
          fullData: encryptedImage,
          keyPreview: he.getPublicKey()?.substring(0, 24) + '...'
        },
        inference: {
          encryptedResult: null,
          logits: []
        },
        output: {
          prediction: 0,
          confidence: 0,
          probabilities: []
        }
      });

      updateStage('response', 'in-progress');
      await new Promise(resolve => setTimeout(resolve, 50));
      updateStage('response', 'completed', { duration: 50, resultSize: '~1KB' });

      updateStage('decryption', 'in-progress');

      // Decrypt the result using HE utilities
      const encryptedResult = inferenceResponse.data.encrypted_result;
      const decryptedResult = await he.decryptResult(encryptedResult);

      await new Promise(resolve => setTimeout(resolve, 50));
      updateStage('decryption', 'completed', { duration: 50 });

      const prediction = decryptedResult.prediction ?? 0;
      const confidence = decryptedResult.confidence ?? 0;
      const probs = decryptedResult.probabilities ?? [];
      const logits = [];

      setProcessData({
        raw: demoImage,
        encrypted: {
          data: encryptedImage.substring(0, 100) + '...',
          fullData: encryptedImage,
          keyPreview: he.getPublicKey()?.substring(0, 24) + '...'
        },
        inference: {
          encryptedResult: encryptedResult,
          logits: logits
        },
        output: {
          prediction: prediction,
          confidence: confidence,
          probabilities: Array.isArray(probs) ? probs : []
        }
      });

      const totalTime = Date.now() - startTime;

      updateStage('display', 'completed', {
        duration: 0,
        prediction: prediction,
        confidence: (confidence * 100).toFixed(1) + '%',
        totalTime: totalTime
      });

    } catch (err) {
      console.error('Error running demo:', err);
      setError(err.response?.data?.detail || err.message || 'Demo failed');
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-50 to-slate-100">
      {/* Header */}
      <header className="bg-white border-b border-slate-200 shadow-sm">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <div className="w-8 h-8 bg-gradient-to-br from-blue-500 to-purple-600 rounded-lg flex items-center justify-center">
                <svg className="w-5 h-5 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
              </div>
              <div>
                <h1 className="text-xl font-bold text-slate-800">Privacy-Preserving ML</h1>
                <p className="text-xs text-slate-500">Homomorphic Encryption for Secure ML Inference</p>
              </div>
            </div>
            <button
              onClick={runDemo}
              disabled={isProcessing}
              className="px-4 py-2 bg-slate-800 hover:bg-slate-900 text-white text-sm font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
            >
              {isProcessing ? 'Running...' : 'Run Demo'}
            </button>
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-6xl mx-auto px-6 py-6">
        {/* Error Display */}
        {error && (
          <div className="mb-6 p-4 bg-red-50 border border-red-200 rounded-lg">
            <div className="flex items-center gap-3">
              <svg className="w-5 h-5 text-red-600 flex-shrink-0" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 2h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <p className="text-red-700 text-sm">{error}</p>
            </div>
          </div>
        )}

        {/* Pipeline Overview */}
        <div className="mb-6 p-4 bg-white rounded-xl border border-slate-200 shadow-sm">
          <div className="flex items-start gap-3 mb-4">
            <svg className="w-6 h-6 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <div>
              <h2 className="text-lg font-bold text-slate-800">Privacy-Preserving ML Pipeline</h2>
              <p className="text-xs text-slate-500 mt-1">
                Visualize how your digit classification works with homomorphic encryption
              </p>
            </div>
          </div>

          <div className="grid grid-cols-1 md:grid-cols-4 gap-3 text-xs">
            <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg">
              <div className="flex items-center gap-1 mb-1">
                <svg className="w-4 h-4 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
                </svg>
                <span className="font-semibold text-blue-800">Input</span>
              </div>
              <p className="text-slate-600 leading-relaxed">Draw or use demo data (28x28 pixels)</p>
            </div>

            <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg">
              <div className="flex items-center gap-1 mb-1">
                <svg className="w-4 h-4 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                </svg>
                <span className="font-semibold text-amber-800">Encrypt</span>
              </div>
              <p className="text-slate-600 leading-relaxed">Convert to encrypted format before sending</p>
            </div>

            <div className="p-3 bg-purple-50 border border-purple-200 rounded-lg">
              <div className="flex items-center gap-1 mb-1">
                <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 0l4 4m0 0l4-4" />
                </svg>
                <span className="font-semibold text-purple-800">Transmit</span>
              </div>
              <p className="text-slate-600 leading-relaxed">Send encrypted data to server</p>
            </div>

            <div className="p-3 bg-green-50 border border-green-200 rounded-lg">
              <div className="flex items-center gap-1 mb-1">
                <svg className="w-4 h-4 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
                </svg>
                <span className="font-semibold text-green-800">Infer</span>
              </div>
              <p className="text-slate-600 leading-relaxed">Server runs ML on encrypted data</p>
            </div>
          </div>
        </div>

        {/* Bead Tracker */}
        <div className="mb-6">
          <BeadTracker stages={stages} />
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Left Column - Input */}
          <div className="lg:col-span-1">
            <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
              <h2 className="text-base font-semibold text-slate-800 mb-4">Draw Input</h2>
              <DrawingCanvas
                onDrawingComplete={handleDrawingComplete}
                onClear={handleClear}
              />
              <button
                onClick={classifyWithHE}
                disabled={!imageData || isProcessing}
                className="w-full mt-3 py-2.5 bg-blue-600 hover:bg-blue-700 disabled:bg-slate-300 text-white font-medium rounded-lg transition-colors disabled:opacity-50 disabled:cursor-not-allowed flex items-center justify-center gap-2"
              >
                {isProcessing ? (
                  <>
                    <div className="animate-spin rounded-full h-4 w-4 border-2 border-blue-600 border-t-transparent"></div>
                    Processing...
                  </>
                ) : (
                  <>
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
                    </svg>
                    Classify with HE
                  </>
                )}
              </button>

              {/* Quick Info */}
              <div className="mt-4 p-3 bg-slate-50 rounded-lg border border-slate-200">
                <p className="text-xs text-slate-600 leading-relaxed mb-2">
                  The pipeline encrypts your drawing before sending it to the server, so the server never sees your raw input.
                </p>
                <div className="flex items-center gap-1.5">
                  <svg className="w-3.5 h-3.5 text-slate-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                  </svg>
                  <span className="text-xs text-slate-500">All data encrypted</span>
                </div>
              </div>
            </div>
          </div>

          {/* Right Column - Process Visualization */}
          <div className="lg:col-span-2">
            <ProcessVisualization data={processData} />
          </div>
        </div>
      </main>

      {/* Footer */}
      <footer className="border-t border-slate-200 mt-8">
        <div className="max-w-6xl mx-auto px-6 py-4">
          <div className="flex flex-col md:flex-row items-center justify-between gap-3 text-xs">
            <div className="flex items-center gap-1 text-slate-500">
              <svg className="w-3.5 h-3.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
              </svg>
              <span>Privacy-Preserving ML Demo</span>
            </div>
            <div className="flex items-center gap-3 text-slate-400">
              <span>React</span>
              <span>•</span>
              <span>Tailwind CSS</span>
              <span>•</span>
              <span>PyTorch</span>
              <span>•</span>
              <span>FastAPI</span>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
}

export default App;
