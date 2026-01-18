
import React from 'react';

interface Alert {
  id: string;
  type: string;
  camera: string;
  confidence: number;
  timestamp: string;
  dataset: string;
  status: string;
}

const AlertCard: React.FC<{ alert: Alert }> = ({ alert }) => {
  const isCritical = alert.status === 'CRITICAL' || alert.type === 'WEAPON';
  const displayType = alert.type === 'WEAPON' ? 'WEAPON DETECTED' : `${alert.type} DETECTED`;
  
  return (
    <div className={`glass rounded-2xl overflow-hidden transition-all duration-300 border-white/5 hover:border-white/20 group ${isCritical ? 'glow-red' : ''}`}>
      <div className={`${isCritical ? 'bg-red-600' : alert.type === 'CRIMINAL_FACE' ? 'bg-red-600' : 'bg-amber-600'} p-4 flex justify-between items-center shadow-lg`}>
        <div className="flex items-center space-x-2">
          <div className="w-2 h-2 rounded-full bg-white animate-pulse"></div>
          <span className="font-black text-white text-[10px] tracking-[0.2em] uppercase">{displayType}</span>
          <span className="text-white text-[10px] font-semibold">{(alert.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>

      <div className="p-4 space-y-3">
        <div className="grid grid-cols-2 gap-4">
          <div>
            <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Timeline</p>
            <p className="text-slate-300 text-[11px] font-medium">{new Date(alert.timestamp).toLocaleTimeString()}</p>
          </div>
          <div className="text-right">
            <p className="text-[10px] text-slate-500 uppercase font-black tracking-widest mb-1">Vector DB</p>
            <p className="text-slate-400 text-[9px] font-mono truncate max-w-[100px] ml-auto">CLUSTER0_SECURE</p>
          </div>
        </div>

        <button className="w-full bg-white/5 hover:bg-white/10 text-white py-3 rounded-xl text-[10px] font-black uppercase tracking-[0.1em] transition-all border border-white/5 group-hover:border-white/20">
          View Evidence Hash
        </button>
      </div>
    </div>
  );
};

export default AlertCard;
