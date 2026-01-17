import React, { useState } from 'react';

const ProcessVisualization = ({ data }) => {
  const [activeTab, setActiveTab] = useState('raw');

  const tabs = [
    { id: 'raw', label: 'Raw Input', icon: 'I' },
    { id: 'encrypted', label: 'Encrypted', icon: 'E' },
    { id: 'inference', label: 'Inference (Encrypted)', icon: 'N' },
    { id: 'output', label: 'Output (Decrypted)', icon: 'O' }
  ];

  const renderRawData = () => {
    if (!data.raw) return <p className="text-slate-500 text-sm">No raw data available</p>;

    const gridSize = 28;
    const pixels = data.raw;

    return (
      <div className="space-y-4">
        <div className="flex justify-center">
          <div
            className="grid gap-px bg-slate-300 border border-slate-400 rounded-lg overflow-hidden shadow-sm"
            style={{
              gridTemplateColumns: `repeat(${gridSize}, 1fr)`,
              width: '168px',
              height: '168px'
            }}
          >
            {pixels.map((pixel, i) => (
              <div
                key={i}
                className="w-full h-full"
                style={{
                  backgroundColor: `rgb(${pixel}, ${pixel}, ${pixel})`
                }}
                title={`[${Math.floor(i / 28)}, ${i % 28}]: ${pixel}`}
              />
            ))}
          </div>
        </div>
        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex justify-between p-2 bg-slate-100 rounded border border-slate-200">
            <span className="text-slate-600">Format</span>
            <span className="text-slate-800 font-mono">28x28</span>
          </div>
          <div className="flex justify-between p-2 bg-slate-100 rounded border border-slate-200">
            <span className="text-slate-600">Pixels</span>
            <span className="text-slate-800 font-mono">{pixels.length}</span>
          </div>
          <div className="flex justify-between p-2 bg-slate-100 rounded border border-slate-200">
            <span className="text-slate-600">Min</span>
            <span className="text-slate-800 font-mono">{Math.min(...pixels)}</span>
          </div>
          <div className="flex justify-between p-2 bg-slate-100 rounded border border-slate-200">
            <span className="text-slate-600">Max</span>
            <span className="text-slate-800 font-mono">{Math.max(...pixels)}</span>
          </div>
        </div>
        <div className="p-3 bg-blue-50 border border-blue-200 rounded-lg text-xs text-blue-700">
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-blue-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>This raw pixel data will be encrypted before sending to server</span>
          </div>
        </div>
      </div>
    );
  };

  const renderEncryptedData = () => {
    if (!data.encrypted) return <p className="text-slate-500 text-sm">No encrypted data available</p>;

    const encryptedStr = typeof data.encrypted === 'string'
      ? data.encrypted
      : data.encrypted.fullData || JSON.stringify(data.encrypted);

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2 pb-2">
          <svg className="w-4 h-4 text-amber-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
          </svg>
          <span className="text-sm font-semibold text-amber-700">Encrypted Input Data</span>
        </div>

        <div className="bg-slate-900 rounded-lg p-3 border border-slate-600 shadow-sm">
          <div className="text-xs font-mono text-emerald-400 break-all leading-relaxed">
            {encryptedStr.substring(0, 200)}
            {encryptedStr.length > 200 && '...'}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex justify-between p-2 bg-slate-100 rounded border border-slate-200">
            <span className="text-slate-600">Type</span>
            <span className="text-slate-800 font-mono text-xs">Base64</span>
          </div>
          <div className="flex justify-between p-2 bg-slate-100 rounded border border-slate-200">
            <span className="text-slate-600">Size</span>
            <span className="text-slate-800 font-mono">{encryptedStr.length} chars</span>
          </div>
          {data.encrypted.keyPreview && (
            <div className="flex justify-between p-2 bg-slate-100 rounded border border-slate-200 col-span-2">
              <span className="text-slate-600">Key Preview</span>
              <span className="text-slate-800 font-mono text-xs">{data.encrypted.keyPreview}</span>
            </div>
          )}
        </div>

        <div className="p-3 bg-amber-50 border border-amber-200 rounded-lg text-xs text-amber-700">
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-amber-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 15v2m-6 4h12a2 2 0 002-2v-6a2 2 0 00-2-2H6a2 2 0 00-2 2v6a2 2 0 002 2zm10-10V7a4 4 0 00-8 0v4h8z" />
            </svg>
            <span>Data is encrypted before transmission. Server never sees raw input.</span>
          </div>
        </div>
      </div>
    );
  };

  const renderInferenceData = () => {
    if (!data.inference) return <p className="text-slate-500 text-sm">No inference data available</p>;

    if (data.inference.encryptedResult) {
      return (
        <div className="space-y-4">
          <div className="flex items-center gap-2 pb-2">
            <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <span className="text-sm font-semibold text-purple-700">Encrypted Inference Result</span>
          </div>

          <div className="bg-slate-900 rounded-lg p-3 border border-slate-600 shadow-sm">
            <div className="text-xs font-mono text-cyan-400 break-all leading-relaxed">
              {data.inference.encryptedResult.substring(0, 200)}
              {data.inference.encryptedResult.length > 200 && '...'}
            </div>
          </div>

          <div className="grid grid-cols-2 gap-2 text-xs">
            <div className="flex justify-between p-2 bg-slate-100 rounded border border-slate-200">
              <span className="text-slate-600">Status</span>
              <span className="text-emerald-600 font-semibold">Processed</span>
            </div>
            <div className="flex justify-between p-2 bg-slate-100 rounded border border-slate-200">
              <span className="text-slate-600">Size</span>
              <span className="text-slate-800 font-mono">{data.inference.encryptedResult.length} chars</span>
            </div>
          </div>

          <div className="p-3 bg-purple-50 border border-purple-200 rounded-lg text-xs text-purple-700">
            <div className="flex items-start gap-2">
              <svg className="w-4 h-4 text-purple-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
              </svg>
              <span>Server processed encrypted data without seeing raw input</span>
            </div>
          </div>
        </div>
      );
    }

    const logits = data.inference.logits || [];
    if (logits.length === 0) {
      return (
        <div className="space-y-4">
          <div className="flex items-center gap-2 pb-2">
            <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
            </svg>
            <span className="text-sm font-semibold text-purple-700">Inference on Encrypted Data</span>
          </div>
          <p className="text-sm text-slate-500 italic">Encrypted result will appear here after classification</p>
        </div>
      );
    }

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2 pb-2">
          <svg className="w-4 h-4 text-purple-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <span className="text-sm font-semibold text-purple-700">Encrypted Inference Result</span>
        </div>

        <div>
          <p className="text-xs text-slate-600 mb-2 font-medium">Logits (Pre-Softmax)</p>
          <div className="space-y-1">
            {logits.map((logit, i) => (
              <div key={i} className="flex items-center gap-2">
                <span className="w-4 text-xs font-mono text-slate-500">{i}</span>
                <div className="flex-1 h-4 bg-slate-200 rounded overflow-hidden">
                  <div
                    className="h-full bg-purple-500 flex items-center justify-end pr-1 text-xs text-white font-mono"
                    style={{
                      width: `${Math.max(3, Math.min(100, (logit - Math.min(...logits)) / (Math.max(...logits) - Math.min(...logits)) * 100))}%`
                    }}
                  >
                    {logit.toFixed(1)}
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="grid grid-cols-2 gap-2 text-xs">
          <div className="flex justify-between p-2 bg-slate-100 rounded border border-slate-200">
            <span className="text-slate-600">Max Logit</span>
            <span className="text-slate-800 font-mono">{Math.max(...logits).toFixed(1)}</span>
          </div>
          <div className="flex justify-between p-2 bg-slate-100 rounded border border-slate-200">
            <span className="text-slate-600">Layers</span>
            <span className="text-slate-800 font-mono">CNN (2 conv)</span>
          </div>
        </div>
      </div>
    );
  };

  const renderOutputData = () => {
    if (!data.output) return <p className="text-slate-500 text-sm">No output data available</p>;

    const { prediction, confidence, probabilities } = data.output;

    return (
      <div className="space-y-4">
        <div className="flex items-center gap-2 pb-2">
          <svg className="w-4 h-4 text-emerald-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
          </svg>
          <span className="text-sm font-semibold text-emerald-700">Decrypted Result</span>
        </div>

        {/* Main Prediction Display */}
        <div className="text-center p-6 bg-slate-800 rounded-xl shadow-lg">
          <div className="text-7xl font-bold text-white mb-2">{prediction}</div>
          <div className="text-sm text-slate-300">
            {(confidence * 100).toFixed(1)}% confidence
          </div>
        </div>

        {/* Probability Distribution */}
        <div>
          <p className="text-xs text-slate-600 mb-2 font-medium">Probability Distribution</p>
          <div className="space-y-1">
            {Array.isArray(probabilities) && probabilities.length > 0 ? (
              probabilities
                .map((p, i) => ({ digit: i, probability: p }))
                .sort((a, b) => b.probability - a.probability)
                .map((p, index) => (
                  <div key={p.digit} className="flex items-center gap-2">
                    <span className="w-4 text-xs font-mono text-slate-500">{p.digit}</span>
                    <div className="flex-1 h-4 bg-slate-200 rounded overflow-hidden">
                      <div
                        className={`h-full flex items-center justify-end pr-2 text-xs text-white font-mono ${
                          index === 0 ? 'bg-emerald-600' : 'bg-slate-500'
                        }`}
                        style={{ width: `${Math.max(3, (p.probability || 0) * 100)}%` }}
                      >
                        {((p.probability || 0) * 100).toFixed(0)}%
                      </div>
                    </div>
                  </div>
                ))
            ) : (
              <p className="text-xs text-slate-500 italic">No probabilities available</p>
            )}
          </div>
        </div>

        <div className="p-3 bg-emerald-50 border border-emerald-200 rounded-lg text-xs text-emerald-700">
          <div className="flex items-start gap-2">
            <svg className="w-4 h-4 text-emerald-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" />
            </svg>
            <span>Final decrypted result. Data remained encrypted throughout the process.</span>
          </div>
        </div>
      </div>
    );
  };

  const renderContent = () => {
    switch (activeTab) {
      case 'raw':
        return renderRawData();
      case 'encrypted':
        return renderEncryptedData();
      case 'inference':
        return renderInferenceData();
      case 'output':
        return renderOutputData();
      default:
        return null;
    }
  };

  return (
    <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-5">
      <h3 className="text-lg font-semibold text-slate-800 mb-4">Process Pipeline</h3>

      {/* Tab Buttons */}
      <div className="flex gap-2 mb-4">
        {tabs.map(tab => (
          <button
            key={tab.id}
            onClick={() => setActiveTab(tab.id)}
            className={`
              flex-1 py-2 px-3 rounded-lg font-medium text-xs transition-all
              ${activeTab === tab.id
                ? 'bg-blue-600 text-white shadow-md'
                : 'bg-slate-100 text-slate-600 hover:bg-slate-200'
              }
              ${data[tab.id] ? '' : 'opacity-50 cursor-not-allowed'}
            `}
            disabled={!data[tab.id]}
          >
            <span className="hidden sm:inline">{tab.label}</span>
            <span className="sm:hidden">{tab.icon}</span>
          </button>
        ))}
      </div>

      {/* Tab Content */}
      <div className="min-h-[280px]">
        {renderContent()}
      </div>

      {/* Status Indicators */}
      <div className="flex gap-3 mt-4 pt-4 border-t border-slate-200">
        {tabs.map(tab => (
          <div
            key={tab.id}
            className={`flex items-center gap-1.5 text-xs ${
              data[tab.id] ? 'text-emerald-600' : 'text-slate-400'
            }`}
          >
            <div className={`w-1.5 h-1.5 rounded-full ${
              data[tab.id] ? 'bg-emerald-500' : 'bg-slate-300'
            }`} />
            {tab.label}
          </div>
        ))}
      </div>
    </div>
  );
};

export default ProcessVisualization;
