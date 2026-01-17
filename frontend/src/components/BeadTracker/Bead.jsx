const Bead = ({
  stage,
  status = 'pending',
  onClick,
  details,
  isSelected,
  index,
  total
}) => {
  const getStatusColor = () => {
    switch (status) {
      case 'completed': return 'bg-emerald-500 text-emerald-700';
      case 'in-progress': return 'bg-blue-500 text-blue-700';
      case 'error': return 'bg-red-500 text-red-700';
      default: return 'bg-slate-300 text-slate-600';
    }
  };

  const getCheckIcon = () => {
    return (
      <svg className="w-3 h-3" fill="none" stroke="currentColor" viewBox="0 0 24 24">
        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
      </svg>
    );
  };

  const getTitle = () => {
    switch (stage) {
      case 'input': return 'Input';
      case 'encryption': return 'Encrypt';
      case 'transmission': return 'Send';
      case 'inference': return 'Server';
      case 'response': return 'Recv';
      case 'decryption': return 'Decrypt';
      case 'display': return 'Result';
      default: return stage;
    }
  };

  const isLast = index === total - 1;

  return (
    <div
      className={`flex items-center ${isLast ? '' : 'flex-1'}`}
      onClick={() => onClick(stage, details)}
    >
      <div className="flex flex-col items-center gap-1 flex-1 min-w-0">
        {/* Step indicator */}
        <div className={`
          flex items-center justify-center w-6 h-6 rounded text-xs font-semibold transition-all
          ${getStatusColor()}
          ${isSelected ? 'ring-2 ring-slate-400 ring-offset-1' : ''}
        `}>
          {status === 'completed' ? getCheckIcon() : index + 1}
        </div>

        {/* Label */}
        <p className="text-[10px] font-medium text-slate-600 text-center leading-tight">
          {getTitle()}
        </p>

        {/* Duration */}
        {details?.duration && (
          <p className="text-[9px] text-slate-400">{details.duration}ms</p>
        )}
      </div>

      {/* Arrow */}
      {!isLast && (
        <div className="flex items-center justify-center px-1">
          <svg className="w-4 h-4 text-slate-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5l7 7-7 7" />
          </svg>
        </div>
      )}
    </div>
  );
};

export default Bead;
