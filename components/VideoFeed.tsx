
import React, { useRef, useEffect, useState } from 'react';
import { useCamera } from '../hooks/useCamera';
import { useInference, DetectionResult } from '../hooks/useInference';
import { useToast } from './ToastContainer';

interface Camera {
  id: string;
  name: string;
  type: string;
}

interface VideoFeedProps {
  camera: Camera;
  detectionEnabled: boolean;
}

const VideoFeed: React.FC<VideoFeedProps> = ({ camera, detectionEnabled }) => {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const [dispatchStatus, setDispatchStatus] = useState<string | null>(null);
  const [activeAlert, setActiveAlert] = useState<DetectionResult | null>(null);
  const { videoRef, isActive, error, startCamera } = useCamera();
  const { showToast } = useToast();

  // Debounce alerts to prevent spamming (2 second cooldown)
  const lastAlertTimeRef = useRef<number>(0);

  const handleAlert = React.useCallback((detection: DetectionResult) => {
    const now = Date.now();
    // Only show alert if 2 seconds have passed since last alert
    if (now - lastAlertTimeRef.current < 2000) {
      return;
    }

    setDispatchStatus(`COMM_LINK_ESTABLISHED: ${detection.id}`);
    setActiveAlert(detection);
    lastAlertTimeRef.current = now;
    
    // Show toast notification
    if (detection.type === 'CRIMINAL_FACE') {
      showToast({
        type: 'error',
        title: '🚨 CRITICAL ALERT',
        message: `Criminal detected: ${detection.name}\nConfidence: ${(detection.confidence * 100).toFixed(1)}%\nAlert sent to Telegram`,
        duration: 8000
      });
    } else if (detection.type === 'FACE') {
      showToast({
        type: 'info',
        title: '👤 Face Detected',
        message: `Face detected (${(detection.confidence * 100).toFixed(1)}%) - Alert sent to Telegram`,
        duration: 3000
      });
    } else if (detection.type === 'WEAPON') {
      showToast({
        type: 'warning',
        title: '⚠️ WEAPON ALERT',
        message: `Weapon detected: ${detection.label}\nConfidence: ${(detection.confidence * 100).toFixed(1)}%\nAlert sent to Telegram`,
        duration: 8000
      });
    }
    
    setTimeout(() => {
      setDispatchStatus(`PACKET_DISPATCHED_SUCCESS`);
      setTimeout(() => {
        setDispatchStatus(null);
        setTimeout(() => setActiveAlert(null), 4000);
      }, 2000);
    }, 1500);
  }, [showToast]);

  const { lastMatch, runInference, isScanningBiometrics } = useInference(handleAlert, detectionEnabled);

  useEffect(() => {
    let frameId: number | null = null;
    let frameCount = 0;
    let lastFrameTime = Date.now();
    let inferenceCounter = 0;
    
    if (!isActive) {
      return;
    }
    
    const canvas = canvasRef.current;
    const video = videoRef.current;
    
    if (!canvas || !video) {
      return;
    }
    
    const ctx = canvas.getContext('2d');
    if (!ctx) {
      return;
    }

    const render = () => {
      if (!canvas || !video || !isActive) {
        if (frameId !== null) {
          cancelAnimationFrame(frameId);
          frameId = null;
        }
        return;
      }
      
      try {
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        if (detectionEnabled) {
          frameCount++;
          inferenceCounter++;
          const now = Date.now();
          const fps = (frameCount / ((now - lastFrameTime) / 1000)).toFixed(1);
          
          // Send inference every 3rd frame to reduce backend load
          if (inferenceCounter % 3 === 0) {
            runInference(video);
          }
        }
        
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.04)';
        ctx.lineWidth = 1;
        for(let i=0; i<canvas.width; i+=100) { 
          ctx.beginPath(); 
          ctx.moveTo(i, 0); 
          ctx.lineTo(i, canvas.height); 
          ctx.stroke(); 
        }
        for(let j=0; j<canvas.height; j+=100) { 
          ctx.beginPath(); 
          ctx.moveTo(0, j); 
          ctx.lineTo(canvas.width, j); 
          ctx.stroke(); 
        }

        if (isScanningBiometrics) {
          ctx.fillStyle = '#06b6d4';
          ctx.font = 'bold 14px "JetBrains Mono"';
          ctx.fillText(`>>> SEARCHING_CLUSTER0_DATABASE...`, 40, 50);
          const scanY = (Date.now() % 1500) / 1500 * canvas.height;
          ctx.strokeStyle = '#06b6d4';
          ctx.beginPath(); 
          ctx.moveTo(0, scanY); 
          ctx.lineTo(canvas.width, scanY); 
          ctx.stroke();
        }

        if (lastMatch && detectionEnabled) {
          const [x, y, w, h] = lastMatch.box;
          const themeColor = lastMatch.type === 'CRIMINAL_FACE' ? '#06b6d4' : '#ff003c';
          ctx.strokeStyle = themeColor;
          ctx.lineWidth = 4;
          ctx.strokeRect(x * canvas.width, y * canvas.height, w * canvas.width, h * canvas.height);
          
          ctx.fillStyle = themeColor;
          ctx.fillRect(x * canvas.width, y * canvas.height - 30, 250, 30);
          ctx.fillStyle = '#000';
          ctx.font = 'bold 12px "JetBrains Mono"';
          // Handle both face (with name) and weapon (with type) detections
          const displayLabel = lastMatch.name || lastMatch.type || 'Unknown';
          const displayText = lastMatch.type === 'WEAPON' 
            ? `${lastMatch.type} [${lastMatch.metadata?.weapon_type || 'Unknown'}]` 
            : `${lastMatch.type}: ${displayLabel}`;
          ctx.fillText(displayText, x * canvas.width + 10, y * canvas.height - 10);
        }
      } catch (renderErr) {
        console.error('Render error:', renderErr);
      }
      
      frameId = requestAnimationFrame(render);
    };
    
    render();
    
    return () => {
      if (frameId !== null) {
        cancelAnimationFrame(frameId);
        frameId = null;
      }
    };
  }, [isActive, lastMatch, runInference, detectionEnabled, isScanningBiometrics]);

  return (
    <div className={`glass rounded-xl overflow-hidden transition-all duration-700 relative border-l-4 ${isActive ? 'glow-green border-l-green-500' : 'border-l-white/10'}`}>
      <div className="px-6 py-4 flex justify-between items-center bg-black/40 border-b border-white/5">
        <div className="flex items-center space-x-4">
          <div className={`w-3 h-3 rounded-full ${isActive ? 'bg-green-500 animate-pulse' : 'bg-slate-700'}`}></div>
          <div className="flex flex-col">
            <span className="text-[10px] text-slate-500 font-black uppercase tracking-widest leading-none mb-1">Source</span>
            <span className="text-sm font-bold text-white uppercase">{camera.name}</span>
          </div>
        </div>
        {isActive && (
          <div className="px-3 py-1 bg-red-600/10 rounded border border-red-600/30 flex items-center space-x-2">
            <span className="w-1.5 h-1.5 bg-red-600 rounded-full animate-pulse"></span>
            <span className="text-[10px] font-black text-red-500 uppercase tracking-widest">AUTO_SCAN_ACTIVE</span>
          </div>
        )}
      </div>
      
      <div className="relative aspect-video bg-black flex items-center justify-center overflow-hidden">
        {isActive && <div className="scanner-line"></div>}

        {activeAlert && (
          <div className={`absolute top-6 z-50 animate-in slide-in-from-right zoom-in ${activeAlert.type === 'CRIMINAL_FACE' || activeAlert.type === 'FACE' ? 'left-6' : 'right-6'}`}>
            <div className={`glass border-2 ${activeAlert.type === 'CRIMINAL_FACE' ? 'border-cyan-500 bg-cyan-950/90' : activeAlert.type === 'WEAPON' ? 'border-red-500 bg-red-950/90' : 'border-blue-500 bg-blue-950/90'} p-6 rounded shadow-2xl min-w-[320px]`}>
              <p className="text-[10px] font-black text-white/60 uppercase mb-2 tracking-widest">
                {activeAlert.type === 'WEAPON' ? 'Weapon_Detected' : activeAlert.type === 'CRIMINAL_FACE' ? 'Criminal_Match' : 'Identity_Verified'}
              </p>
              <h2 className="text-white font-black text-2xl tracking-tighter uppercase mb-4">
                {activeAlert.type === 'WEAPON' ? activeAlert.metadata?.weapon_type || 'Weapon' : activeAlert.name}
              </h2>
              <div className="bg-black/40 p-3 rounded border border-white/5 text-[10px] font-mono text-slate-300">
                {activeAlert.type === 'WEAPON' 
                  ? `Confidence: ${(activeAlert.confidence * 100).toFixed(1)}%\nTelegram alert dispatched` 
                  : activeAlert.notes || `Confidence: ${(activeAlert.confidence * 100).toFixed(1)}%`}
              </div>
            </div>
          </div>
        )}

        <video ref={videoRef} className="hidden" playsInline />
        <canvas ref={canvasRef} width={1280} height={720} className={`w-full h-auto ${isActive ? 'opacity-100' : 'opacity-0'} scan-grid`} />

        {!isActive && (
          <div className="absolute inset-0 flex flex-col items-center justify-center z-20 bg-black/50 backdrop-blur-sm">
            <button 
              onClick={startCamera} 
              className="px-12 py-5 bg-red-600 text-white font-black text-xs tracking-[0.3em] uppercase hover:bg-red-500 transition-all transform hover:scale-105 active:scale-95 shadow-[0_0_30px_rgba(220,38,38,0.4)] mb-6"
            >
              INITIALIZE_SURVEILLANCE
            </button>
            {error && (
              <div className="text-red-400 text-[10px] font-mono max-w-xs text-center">
                Error: {error}
              </div>
            )}
            {!error && (
              <div className="text-slate-400 text-[10px] font-mono max-w-xs text-center">
                Click button above to start camera<br/>Camera permissions will be requested
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default VideoFeed;
