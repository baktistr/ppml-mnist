import { useState } from 'react';
import Bead from './Bead';
import BeadDetailsPanel from './BeadDetailsPanel';

const BeadTracker = ({ stages }) => {
  const [selectedStage, setSelectedStage] = useState(null);
  const [isPanelOpen, setIsPanelOpen] = useState(false);

  const handleBeadClick = (stage, details) => {
    setSelectedStage({ stage, details });
    setIsPanelOpen(true);
  };

  const handleClosePanel = () => {
    setIsPanelOpen(false);
    setSelectedStage(null);
  };

  const stageNames = [
    'input',
    'encryption',
    'transmission',
    'inference',
    'response',
    'decryption',
    'display'
  ];

  // Calculate overall progress
  const getProgress = () => {
    const completedCount = stageNames.filter(
      name => stages[name]?.status === 'completed'
    ).length;
    return (completedCount / stageNames.length) * 100;
  };

  return (
    <div className="w-full">
      {/* Bead Flow */}
      <div className="bg-white rounded-xl border border-slate-200 shadow-sm p-4">
        {/* Progress Bar */}
        <div className="mb-4">
          <div className="flex items-center justify-between text-xs mb-1">
            <span className="text-slate-600">Pipeline Progress</span>
            <span className="font-semibold text-slate-700">{Math.round(getProgress())}%</span>
          </div>
          <div className="h-2 bg-slate-200 rounded-full overflow-hidden">
            <div
              className="h-full bg-gradient-to-r from-blue-500 to-purple-600 transition-all duration-500 ease-out"
              style={{ width: `${getProgress()}%` }}
            />
          </div>
        </div>

        {/* Steps */}
        <div className="flex items-center justify-between gap-1">
          {stageNames.map((stageName, index) => (
            <Bead
              key={stageName}
              stage={stageName}
              status={stages[stageName]?.status || 'pending'}
              details={stages[stageName]}
              onClick={handleBeadClick}
              isSelected={selectedStage?.stage === stageName}
              index={index}
              total={stageNames.length}
            />
          ))}
        </div>

        {/* Legend */}
        <div className="mt-4 pt-3 border-t border-slate-100">
          <div className="flex flex-wrap items-center justify-center gap-4 text-xs">
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-slate-300" />
              <span className="text-slate-600">Pending</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-blue-500" />
              <span className="text-slate-600">In Progress</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-emerald-500" />
              <span className="text-slate-600">Completed</span>
            </div>
            <div className="flex items-center gap-1.5">
              <div className="w-2 h-2 rounded-full bg-red-500" />
              <span className="text-slate-600">Error</span>
            </div>
          </div>
        </div>

        {/* Total Time Display */}
        {stages.display?.totalTime && (
          <div className="mt-3 text-center">
            <div className="inline-flex items-center gap-2 bg-slate-100 px-3 py-1.5 rounded-full border border-slate-200">
              <span className="text-xs text-slate-500">Total Time:</span>
              <span className="text-sm font-bold text-slate-700">
                {stages.display.totalTime}ms
              </span>
            </div>
          </div>
        )}
      </div>

      {/* Details Panel */}
      <BeadDetailsPanel
        isOpen={isPanelOpen}
        onClose={handleClosePanel}
        stage={selectedStage?.stage}
        details={selectedStage?.details}
      />
    </div>
  );
};

export default BeadTracker;
