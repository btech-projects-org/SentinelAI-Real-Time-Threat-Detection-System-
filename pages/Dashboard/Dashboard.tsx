
import React, { useState, useEffect } from 'react';
import VideoFeed from '../../components/VideoFeed';

const Dashboard: React.FC = () => {
  const [cameras] = useState([
    { id: 'cam_01', name: 'Alpha-7 Core Perimeter', type: 'HARDWARE_USB' }
  ]);
  
  const [logs, setLogs] = useState<string[]>([]);

  useEffect(() => {
    setLogs([
      `[${new Date().toLocaleTimeString()}] CORE_SYSTEM_ONLINE`,
      `[${new Date().toLocaleTimeString()}] NEURAL_ENGINE: FULL_AUTO_MODE`,
      `[${new Date().toLocaleTimeString()}] BIO_DATABASE: SYNC_COMPLETE`,
    ]);
  }, []);

  return (
    <div className="space-y-12 max-w-7xl mx-auto py-8">
      <div className="flex flex-col md:flex-row md:items-end justify-between gap-8 px-4 border-l-2 border-red-600 pl-8">
        <div>
          <h1 className="text-6xl font-black text-white tracking-tighter mb-2 italic uppercase">Auto_Monitor</h1>
          <div className="flex items-center space-x-3">
             <span className="text-[10px] font-black bg-red-600 text-black px-2 py-0.5 uppercase tracking-[0.2em]">Live_Stream</span>
             <p className="text-slate-500 text-xs font-mono uppercase tracking-widest">Cluster0 Neural Pipeline // Active Inference</p>
          </div>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-12 gap-10 px-4">
        <div className="lg:col-span-8 space-y-10">
          <VideoFeed camera={cameras[0]} detectionEnabled={true} />
        </div>
        
        <div className="lg:col-span-4 space-y-8">
          {/* Status Panel */}
          <div className="glass p-6 rounded-xl border-t-2 border-cyan-500/30 flex flex-col h-full max-h-[500px]">
            <div className="flex items-center justify-between mb-6 border-b border-white/10 pb-4">
               <h3 className="text-xs font-black text-cyan-500 uppercase tracking-[0.3em]">System_Log</h3>
               <span className="text-[9px] font-mono text-slate-500">AUTO_MODE</span>
            </div>
            
            <div className="flex-1 overflow-y-auto font-mono space-y-3 pr-2 scrollbar-thin">
              {logs.map((log, i) => (
                <div key={i} className="text-[10px] leading-relaxed">
                   <span className="text-cyan-800 mr-2">&gt;&gt;&gt;</span>
                   <span className="text-slate-400">{log}</span>
                </div>
              ))}
              <div className="text-[10px] text-cyan-500/30 animate-pulse italic mt-4">_LISTENING_FOR_MATCH_SIGNALS...</div>
            </div>
          </div>

          <div className="glass p-6 rounded-xl border border-white/5 bg-white/5">
             <div className="flex items-center space-x-4 mb-4">
                <div className="w-8 h-8 rounded bg-green-500/20 flex items-center justify-center">
                   <div className="w-2 h-2 bg-green-500 rounded-full animate-ping"></div>
                </div>
                <span className="text-[10px] font-black text-white uppercase tracking-widest">Bio_Engine_Status: Optimal</span>
             </div>
             <p className="text-[9px] text-slate-500 uppercase leading-relaxed font-bold">
                The neural engine is checking the video stream against the Vector Database every 500ms. All matches are automatically dispatched to Telegram.
             </p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Dashboard;
