import { useRef, useEffect } from 'react';

const DrawingCanvas = ({ onDrawingComplete, onClear }) => {
  const canvasRef = useRef(null);
  const isDrawing = useRef(false);
  const lastX = useRef(0);
  const lastY = useRef(0);

  useEffect(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Set canvas size
    canvas.width = 280;
    canvas.height = 280;
    
    // Initialize white background
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    
    // Set drawing style
    ctx.strokeStyle = 'black';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    const startDrawing = (e) => {
      isDrawing.current = true;
      const rect = canvas.getBoundingClientRect();
      lastX.current = e.clientX - rect.left;
      lastY.current = e.clientY - rect.top;
    };

    const draw = (e) => {
      if (!isDrawing.current) return;
      
      const canvas = canvasRef.current;
      const ctx = canvas.getContext('2d');
      const rect = canvas.getBoundingClientRect();
      const x = e.clientX - rect.left;
      const y = e.clientY - rect.top;

      ctx.beginPath();
      ctx.moveTo(lastX.current, lastY.current);
      ctx.lineTo(x, y);
      ctx.stroke();

      lastX.current = x;
      lastY.current = y;
    };

    const stopDrawing = () => {
      if (isDrawing.current) {
        isDrawing.current = false;
        handleCanvasData();
      }
    };

    // Mouse events
    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);

    // Touch events
    canvas.addEventListener('touchstart', (e) => {
      e.preventDefault();
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      isDrawing.current = true;
      lastX.current = touch.clientX - rect.left;
      lastY.current = touch.clientY - rect.top;
    });

    canvas.addEventListener('touchmove', (e) => {
      e.preventDefault();
      if (!isDrawing.current) return;
      const touch = e.touches[0];
      const rect = canvas.getBoundingClientRect();
      const x = touch.clientX - rect.left;
      const y = touch.clientY - rect.top;

      const ctx = canvas.getContext('2d');
      ctx.beginPath();
      ctx.moveTo(lastX.current, lastY.current);
      ctx.lineTo(x, y);
      ctx.stroke();

      lastX.current = x;
      lastY.current = y;
    });

    canvas.addEventListener('touchend', stopDrawing);

    return () => {
      canvas.removeEventListener('mousedown', startDrawing);
      canvas.removeEventListener('mousemove', draw);
      canvas.removeEventListener('mouseup', stopDrawing);
      canvas.removeEventListener('mouseout', stopDrawing);
    };
  }, []);

  const handleCanvasData = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    
    // Get image data
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    
    // Convert to grayscale and resize to 28x28
    const resized = resizeAndGrayscale(imageData, 28, 28);
    
    if (onDrawingComplete) {
      onDrawingComplete(resized);
    }
  };

  const resizeAndGrayscale = (imageData, newWidth, newHeight) => {
    const { width, height, data } = imageData;
    const resized = new Array(newWidth * newHeight);

    // Simple downscaling and grayscale conversion
    for (let y = 0; y < newHeight; y++) {
      for (let x = 0; x < newWidth; x++) {
        const srcX = Math.floor((x / newWidth) * width);
        const srcY = Math.floor((y / newHeight) * height);
        const srcIndex = (srcY * width + srcX) * 4;

        // Convert to grayscale (MNIST format: black=0, white=255)
        // Canvas has white background, so we need to invert
        const r = data[srcIndex];
        const g = data[srcIndex + 1];
        const b = data[srcIndex + 2];
        const gray = 0.299 * r + 0.587 * g + 0.114 * b;

        // Invert: 255 (white) -> 0, 0 (black) -> 255
        resized[y * newWidth + x] = Math.round(255 - gray);
      }
    }

    return resized;
  };

  const clearCanvas = () => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext('2d');
    ctx.fillStyle = 'white';
    ctx.fillRect(0, 0, canvas.width, canvas.height);
    if (onClear) onClear();
  };

  return (
    <div className="flex flex-col items-center gap-3">
      <div className="relative">
        <canvas
          ref={canvasRef}
          className="border-2 border-slate-600 rounded-lg cursor-crosshair shadow-lg bg-white"
          style={{ touchAction: 'none' }}
        />
      </div>

      <button
        onClick={clearCanvas}
        className="px-4 py-2 bg-slate-700 hover:bg-slate-600 text-slate-200 text-sm font-medium rounded-lg transition-colors"
      >
        Clear Canvas
      </button>

      <p className="text-xs text-slate-400 text-center">
        Draw a digit (0-9) on the canvas
      </p>
    </div>
  );
};

export default DrawingCanvas;
