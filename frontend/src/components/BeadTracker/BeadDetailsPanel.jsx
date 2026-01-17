import { motion, AnimatePresence } from 'framer-motion';

const BeadDetailsPanel = ({ isOpen, onClose, stage, details }) => {
  if (!isOpen || !stage) return null;

  const getStageInfo = () => {
    switch (stage) {
      case 'input':
        return {
          title: 'Raw Input Stage',
          description: 'User provides digit image data for classification',
          technicalDetails: [
            `Input Format: ${details?.format || '28x28 grayscale image'}`,
            `Input Size: ${details?.inputSize || '784'} pixels`,
            `Data Range: 0-255 (MNIST format, inverted)`,
            `Purpose: Raw pixel data ready for encryption`
          ],
          icon: 'input'
        };
      case 'encryption':
        return {
          title: 'Encryption Stage',
          description: 'Data is converted to encrypted format using homomorphic encryption',
          technicalDetails: [
            `Encryption Scheme: ${details?.scheme || 'CKKS Homomorphic'}`,
            `Security Level: ${details?.securityLevel || '128-bit security'}`,
            `Encryption Method: Public Key Encryption`,
            `Processing Time: ${details?.duration || '0'}ms`
          ],
          icon: 'lock'
        };
      case 'transmission':
        return {
          title: 'Transmission Stage',
          description: 'Encrypted data is transmitted to the ML server',
          technicalDetails: [
            `Protocol: HTTPS (TLS 1.3)`,
            `Data Size: ${details?.dataSize || '~5KB encrypted data'}`,
            `Privacy: End-to-end encryption maintained`,
            `Network Time: ${details?.duration || '0'}ms`
          ],
          icon: 'cloud-upload'
        };
      case 'inference':
        return {
          title: 'Inference Stage',
          description: 'Server performs ML inference on encrypted data',
          technicalDetails: [
            `Model: ${details?.modelType || 'MNIST CNN (Convolutional Neural Network)'}`,
            `Framework: ${details?.framework || 'PyTorch'}`,
            `Processing: Homomorphic operations (simulated)`,
            `Inference Time: ${details?.duration || '0'}ms`
          ],
          icon: 'cpu'
        };
      case 'response':
        return {
          title: 'Response Stage',
          description: 'Encrypted prediction results are sent back to client',
          technicalDetails: [
            `Result Size: ${details?.resultSize || '~1KB'}`,
            `Content: Encrypted prediction with probabilities`,
            `Privacy: Still in encrypted format`,
            `Response Time: ${details?.duration || '0'}ms`
          ],
          icon: 'download'
        };
      case 'decryption':
        return {
          title: 'Decryption Stage',
          description: 'Client decrypts results using secret key',
          technicalDetails: [
            `Method: Private key decryption`,
            `Decryption Time: ${details?.duration || '0'}ms`,
            `Result: Plaintext classification result`,
            `Accuracy Loss: <1% (CKKS approximation)`
          ],
          icon: 'unlock'
        };
      case 'display':
        return {
          title: 'Display Stage',
          description: 'Final prediction is shown to user',
          technicalDetails: [
            `Predicted Class: ${details?.prediction || 'N/A'}`,
            `Confidence: ${details?.confidence || 'N/A'}`,
            `Total Pipeline: ${details?.totalTime || '0'}ms`,
            `Privacy: Fully preserved throughout pipeline`
          ],
          icon: 'chart'
        };
      default:
        return {
          title: 'Stage Information',
          description: 'Pipeline stage details',
          technicalDetails: ['Details not available'],
          icon: 'circle'
        };
    }
  };

  const info = getStageInfo();

  const getStageNumber = () => {
    switch (stage) {
      case 'input': return 1;
      case 'encryption': return 2;
      case 'transmission': return 3;
      case 'inference': return 4;
      case 'response': return 5;
      case 'decryption': return 6;
      case 'display': return 7;
      default: return 0;
    }
  };

  const getIconForStage = () => {
    const icons = {
      input: (
        <svg className="w-5 h-5 text-blue-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
        </svg>
      ),
      lock: (
        <svg className="w-5 h-5 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
        </svg>
      ),
      'cloud-upload': (
        <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 7h12m0 0l-4-4m4 4l-4 4m0 0l4 4m0 0l4 4m0 0l4 4m0 0l4 4m0 0l4-4l-4 4l-4 4m-4-4l4-4l4-4l4 4" />
        </svg>
      ),
      cpu: (
        <svg className="w-5 h-5 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9.75 17L9 20l-1 1h8l-1-1-.75-3M3 13h18M5 17h14a2 2 0 002-2V5a2 2 0 00-2-2H5a2 2 0 00-2 2v10a2 2 0 002 2z" />
        </svg>
      ),
      download: (
        <svg className="w-5 h-5 text-green-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16l-4-4m0 0l4-4m-4 4h18" />
        </svg>
      ),
      unlock: (
        <svg className="w-5 h-5 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M8 11V7a4 4 0 118 0m-4 8v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2z" />
        </svg>
      ),
      chart: (
        <svg className="w-5 h-5 text-slate-700" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      ),
      circle: (
        <svg className="w-5 h-5 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
        </svg>
      )
    };
    return icons[info.icon] || icons.circle;
  };

  return (
    <AnimatePresence>
      {isOpen && (
        <>
          {/* Backdrop */}
          <motion.div
            initial={{ opacity: 0 }}
            animate={{ opacity: 1 }}
            exit={{ opacity: 0 }}
            onClick={onClose}
            className="fixed inset-0 bg-black/30 backdrop-blur-sm z-50 flex items-center justify-center p-4"
          >
            {/* Panel */}
            <motion.div
              initial={{ scale: 0.95, opacity: 0, y: 20 }}
              animate={{ scale: 1, opacity: 1, y: 0 }}
              exit={{ scale: 0.95, opacity: 0, y: 20 }}
              transition={{ duration: 0.2 }}
              className="bg-white rounded-2xl shadow-2xl border border-slate-200 max-w-lg w-full max-h-[80vh] overflow-y-auto"
              onClick={(e) => e.stopPropagation()}
            >
              {/* Header */}
              <div className="bg-slate-100 p-4 rounded-t-xl border-b border-slate-200">
                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <span className="flex items-center justify-center w-6 h-6 bg-slate-700 rounded text-xs font-bold text-white">
                      {getStageNumber()}
                    </span>
                    <h2 className="text-base font-semibold text-slate-800 flex items-center gap-2">
                      {getIconForStage()}
                      {info.title}
                    </h2>
                  </div>
                  <button
                    onClick={onClose}
                    className="text-slate-500 hover:bg-slate-200 rounded p-1 transition-colors"
                  >
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                    </svg>
                  </button>
                </div>
                <p className="text-slate-600 text-xs mt-1">{info.description}</p>
              </div>

              {/* Content */}
              <div className="p-5">
                {/* Technical Details */}
                {info.technicalDetails.length > 0 && (
                  <div className="mb-5">
                    <h3 className="text-sm font-semibold text-slate-700 mb-2 flex items-center gap-2">
                      <svg className="w-4 h-4 text-blue-500" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                      </svg>
                      Technical Details
                    </h3>
                    <div className="bg-slate-50 rounded-lg p-3 space-y-2">
                      {info.technicalDetails.map((detail, idx) => (
                        <div key={idx} className="flex items-start gap-2">
                          <svg className="w-3.5 h-3.5 text-blue-500 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                          </svg>
                          <span className="text-slate-700 text-sm leading-relaxed">{detail}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}

                {/* Privacy Note */}
                <div className="bg-emerald-50 border border-emerald-200 rounded-lg p-3">
                  <div className="flex items-start gap-2">
                    <svg className="w-5 h-5 text-emerald-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
                    </svg>
                    <div>
                      <h4 className="font-semibold text-emerald-800 text-sm">Privacy Guarantee</h4>
                      <p className="text-xs text-emerald-700 mt-1 leading-relaxed">
                        {stage === 'inference'
                          ? 'Server processes data without seeing raw input (simulated HE)'
                          : 'Your data remains encrypted at this stage'}
                      </p>
                    </div>
                  </div>
                </div>

                {/* Status */}
                {details?.status && (
                  <div className="mt-4 flex items-center justify-between text-sm">
                    <span className="text-slate-500">Status:</span>
                    <span className={`font-semibold ${
                      details.status === 'completed' ? 'text-emerald-600' :
                      details.status === 'in-progress' ? 'text-amber-600' :
                      details.status === 'error' ? 'text-red-600' :
                      'text-slate-500'
                    }`}>
                      {details.status.charAt(0).toUpperCase() + details.status.slice(1)}
                    </span>
                  </div>
                )}
              </div>

              {/* Footer */}
              <div className="bg-slate-50 px-5 py-3 rounded-b-xl border-t border-slate-200">
                <button
                  onClick={onClose}
                  className="w-full bg-slate-700 hover:bg-slate-800 text-white py-2 px-4 rounded-lg font-medium text-sm transition-colors"
                >
                  Close Details
                </button>
              </div>
            </motion.div>
          </motion.div>
        </>
      )}
    </AnimatePresence>
  );
};

export default BeadDetailsPanel;
