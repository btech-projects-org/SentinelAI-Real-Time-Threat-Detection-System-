import React, { useEffect } from 'react';

export interface ToastMessage {
  id: string;
  type: 'success' | 'error' | 'warning' | 'info';
  title: string;
  message: string;
  duration?: number;
}

interface ToastProps extends ToastMessage {
  onClose: (id: string) => void;
}

const Toast: React.FC<ToastProps> = ({ id, type, title, message, duration = 5000, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(() => onClose(id), duration);
    return () => clearTimeout(timer);
  }, [id, duration, onClose]);

  const bgColor = {
    success: 'bg-green-900/90 border-green-500',
    error: 'bg-red-900/90 border-red-500',
    warning: 'bg-yellow-900/90 border-yellow-500',
    info: 'bg-blue-900/90 border-blue-500'
  }[type];

  const icon = {
    success: '✅',
    error: '❌',
    warning: '⚠️',
    info: 'ℹ️'
  }[type];

  const titleColor = {
    success: 'text-green-400',
    error: 'text-red-400',
    warning: 'text-yellow-400',
    info: 'text-blue-400'
  }[type];

  return (
    <div className="animate-in slide-in-from-top fade-in duration-300">
      <div className={`glass border-2 ${bgColor} p-4 rounded-lg shadow-2xl min-w-[350px] max-w-[500px]`}>
        <div className="flex items-start space-x-3">
          <span className="text-2xl mt-0.5">{icon}</span>
          <div className="flex-1">
            <h3 className={`font-black text-sm uppercase tracking-widest mb-1 ${titleColor}`}>
              {title}
            </h3>
            <p className="text-xs text-slate-300 leading-relaxed">
              {message}
            </p>
          </div>
          <button
            onClick={() => onClose(id)}
            className="text-slate-500 hover:text-white transition-colors text-lg font-bold"
          >
            ✕
          </button>
        </div>
      </div>
    </div>
  );
};

export default Toast;
