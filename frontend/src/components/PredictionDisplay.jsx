import React from 'react';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';

const PredictionDisplay = ({ prediction, confidence, probabilities }) => {
  if (!prediction && !probabilities) {
    return null;
  }

  const chartData = probabilities ? probabilities.map(p => ({
    digit: p.digit,
    probability: (p.probability * 100).toFixed(1)
  })) : [];

  const getConfidenceColor = (conf) => {
    if (conf >= 0.9) return 'text-emerald-600';
    if (conf >= 0.7) return 'text-amber-600';
    return 'text-red-600';
  };

  const getConfidenceBg = (conf) => {
    if (conf >= 0.9) return 'bg-emerald-50 border-emerald-200';
    if (conf >= 0.7) return 'bg-amber-50 border-amber-200';
    return 'bg-red-50 border-red-200';
  };

  return (
    <div className="bg-white rounded-xl shadow-sm border border-slate-200 p-6">
      <h3 className="text-2xl font-bold text-slate-800 mb-6 text-center">
        Classification Results
      </h3>

      {/* Main Prediction Display */}
      <div className="text-center mb-8">
        <div className={`inline-block px-12 py-8 rounded-2xl border-2 ${getConfidenceBg(confidence || 0)}`}>
          <div className="text-8xl font-bold text-slate-800 mb-2">
            {prediction}
          </div>
          <div className={`text-lg font-semibold ${getConfidenceColor(confidence || 0)}`}>
            {(confidence ? confidence * 100 : 0).toFixed(1)}% Confidence
          </div>
        </div>
      </div>

      {/* Confidence Bar Chart */}
      {chartData.length > 0 && (
        <div className="mb-6">
          <h4 className="text-lg font-semibold text-slate-700 mb-4">
            Probability Distribution
          </h4>
          <ResponsiveContainer width="100%" height={250}>
            <BarChart data={chartData}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e2e8f0" />
              <XAxis dataKey="digit" stroke="#64748b" />
              <YAxis
                label={{ value: 'Probability (%)', angle: -90, position: 'insideLeft', fill: '#64748b' }}
                domain={[0, 100]}
                stroke="#64748b"
              />
              <Tooltip
                formatter={(value) => `${value}%`}
                cursor={{ fill: 'rgba(59, 130, 246, 0.1)' }}
                contentStyle={{ backgroundColor: '#f8fafc', border: '1px solid #e2e8f0', borderRadius: '8px' }}
              />
              <Bar
                dataKey="probability"
                fill="#3B82F6"
                radius={[4, 4, 0, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* Top 3 Predictions */}
      {probabilities && probabilities.length > 0 && (
        <div>
          <h4 className="text-lg font-semibold text-slate-700 mb-4">
            Top 3 Predictions
          </h4>
          <div className="space-y-3">
            {probabilities.slice(0, 3).map((p, index) => (
              <div
                key={p.digit}
                className={`flex items-center justify-between p-4 rounded-lg ${
                  index === 0 ? 'bg-blue-50 border-2 border-blue-200' : 'bg-slate-50'
                }`}
              >
                <div className="flex items-center gap-3">
                  <span className="text-lg font-bold text-slate-600">
                    #{index + 1}
                  </span>
                  <span className="text-3xl font-bold text-slate-800">
                    {p.digit}
                  </span>
                </div>
                <div className="text-right">
                  <div className="text-2xl font-bold text-blue-600">
                    {(p.probability * 100).toFixed(1)}%
                  </div>
                  <div className="text-sm text-slate-600">probability</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Privacy Note */}
      <div className="mt-6 p-4 bg-emerald-50 rounded-lg border border-emerald-200">
        <div className="flex items-start gap-3">
          <svg className="w-6 h-6 text-emerald-600 flex-shrink-0 mt-0.5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12l2 2 4-4m5.618-4.016A11.955 11.955 0 0112 2.944a11.955 11.955 0 01-8.618 3.04A12.02 12.02 0 003 9c0 5.591 3.824 10.29 9 11.622 5.176-1.332 9-6.03 9-11.622 0-1.042-.133-2.052-.382-3.016z" />
          </svg>
          <div>
            <h4 className="font-semibold text-emerald-800">Privacy Preserved</h4>
            <p className="text-sm text-emerald-700 mt-1">
              Your drawing was encrypted using homomorphic encryption.
              The server performed ML inference without ever seeing your raw data.
            </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default PredictionDisplay;
