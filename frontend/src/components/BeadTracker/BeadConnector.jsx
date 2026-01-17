import { motion } from 'framer-motion';

const BeadConnector = ({ fromStatus, toStatus, index }) => {
  const isActive = fromStatus !== 'pending' || toStatus === 'in-progress';

  const getConnectorColor = () => {
    if (fromStatus === 'completed' && toStatus !== 'pending') {
      return 'bg-emerald-500';
    } else if (fromStatus === 'in-progress' || toStatus === 'in-progress') {
      return 'bg-blue-500';
    } else if (fromStatus === 'error' || toStatus === 'error') {
      return 'bg-red-500';
    } else {
      return 'bg-slate-300';
    }
  };

  return (
    <div className="flex items-center justify-center h-0.5 w-6 md:w-10 relative">
      {/* Background line */}
      <div className="absolute w-full h-0.5 bg-slate-200 rounded-full" />

      {/* Active progress line */}
      <motion.div
        initial={{ width: '0%' }}
        animate={{
          width: isActive ? '100%' : '0%',
        }}
        transition={{
          duration: 0.5,
          delay: index * 0.1
        }}
        className={`
          absolute h-0.5 rounded-full
          ${getConnectorColor()}
          transition-colors duration-300
        `}
      >
        {/* Animated flow effect */}
        {isActive && (
          <motion.div
            className="absolute inset-0 bg-gradient-to-r from-transparent via-slate-400 to-transparent opacity-50"
            animate={{
              x: ['-100%', '100%'],
            }}
            transition={{
              duration: 1.5,
              repeat: Infinity,
              ease: 'linear',
            }}
          />
        )}
      </motion.div>
    </div>
  );
};

export default BeadConnector;
