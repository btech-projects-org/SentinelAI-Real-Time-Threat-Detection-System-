import { useState, useCallback, useEffect, useRef } from 'react';
import logger from '../utils/logger';

export interface DetectionResult {
  id: string;
  name: string;
  label?: string;
  type: 'CRIMINAL_FACE' | 'WEAPON' | 'VIOLENCE' | 'PERSON' | 'MOTION' | 'FACE';
  confidence: number;
  box: [number, number, number, number]; // [x, y, w, h]
  timestamp: string;
  notes?: string;
  metadata?: any;
}

export const useInference = (onAlertTriggered?: (detection: DetectionResult) => void, enabled: boolean = true) => {
  const [lastMatch, setLastMatch] = useState<DetectionResult | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [isScanningBiometrics, setIsScanningBiometrics] = useState(false);
  const [wsConnected, setWsConnected] = useState(false);
  const isMountedRef = useRef(true);
  const processingFrameRef = useRef(false);

  useEffect(() => {
    isMountedRef.current = true;
    
    return () => {
      isMountedRef.current = false;
    };
  }, []);

  useEffect(() => {
    if (!enabled) {
      setWsConnected(false);
      return;
    }
    
    // REST API is always "connected" if enabled
    setWsConnected(true);
    logger.log("✅ Connected to Real-Time Threat Engine (REST API)");
    
    return () => {
      setWsConnected(false);
    };
  }, [enabled]);

  const runInference = useCallback((videoElement: HTMLVideoElement) => {
    if (!enabled || !videoElement || !videoElement.videoWidth || !videoElement.videoHeight) {
      logger.debug(`runInference skipped: enabled=${enabled}, videoWidth=${videoElement?.videoWidth}, videoHeight=${videoElement?.videoHeight}`);
      return;
    }

    // Prevent concurrent frame processing
    if (processingFrameRef.current) {
      logger.debug("Frame processing already in progress, skipping");
      return;
    }

    try {
      const canvas = document.createElement('canvas');
      canvas.width = videoElement.videoWidth;
      canvas.height = videoElement.videoHeight;
      const ctx = canvas.getContext('2d');
      
      if (!ctx) {
        logger.error("Failed to get canvas context");
        return;
      }
      
      // Draw video frame to canvas
      ctx.drawImage(videoElement, 0, 0);
      logger.debug(`📹 Frame captured: ${canvas.width}x${canvas.height}`);
      
      // Convert to blob and send via REST API
      canvas.toBlob(
        async (blob) => {
          if (!blob || !isMountedRef.current) {
            logger.debug("Blob creation failed or component unmounted");
            processingFrameRef.current = false;
            return;
          }

          processingFrameRef.current = true;
          logger.debug(`📤 Sending frame blob (${blob.size} bytes) to API`);

          try {
            const formData = new FormData();
            formData.append('file', blob, 'frame.jpg');

            const apiUrl = import.meta.env.VITE_API_URL ? `${import.meta.env.VITE_API_URL}/v1/detect-frame` : 'http://localhost:8000/api/v1/detect-frame';
            logger.debug(`API URL: ${apiUrl}`);

            const response = await fetch(
              apiUrl,
              {
                method: 'POST',
                body: formData,
                signal: AbortSignal.timeout(30000), // 30s timeout for detection processing
              }
            );

            logger.log(`📨 Response received: ${response.status}`);

            if (!response.ok) {
              logger.error(`Detection API error: ${response.status} ${response.statusText}`);
              processingFrameRef.current = false;
              return;
            }

            const data = await response.json();
            logger.log(`✅ API returned ${data.detection_count} detections`);

            if (!isMountedRef.current) {
              processingFrameRef.current = false;
              return;
            }

            if (data.detections && Array.isArray(data.detections) && data.detections.length > 0) {
              logger.log(`🔍 Received ${data.detections.length} detections from API`);
              const rawDetection = data.detections[0];
              logger.log(`Detection: ${JSON.stringify({type: rawDetection.type, label: rawDetection.label, confidence: rawDetection.confidence, box: rawDetection.box})}`);
              
              // Normalize detection to ensure proper type classification
              let normalizedDetection = { ...rawDetection } as DetectionResult;
              
              // Ensure we have a type field
              if (!normalizedDetection.type && rawDetection.label) {
                normalizedDetection.type = rawDetection.label as any;
              }
              
              // Ensure we have a box (required for rendering)
              if (!normalizedDetection.box) {
                normalizedDetection.box = rawDetection.box || [0, 0, 0.5, 0.5];
              }
              
              // Ensure weapon detections have proper display name
              if (normalizedDetection.type === 'WEAPON') {
                if (!normalizedDetection.name) {
                  normalizedDetection.name = normalizedDetection.metadata?.weapon_type || rawDetection.type || 'Weapon';
                }
              }
              
              // Ensure criminal faces have proper name
              if (normalizedDetection.type === 'CRIMINAL_FACE' && !normalizedDetection.name) {
                normalizedDetection.name = normalizedDetection.metadata?.name || 'Unknown Criminal';
              }
              
              logger.log(`✅ Normalized detection: ${normalizedDetection.type} with name: ${normalizedDetection.name}`);
              setLastMatch(normalizedDetection);
              
              if (onAlertTriggered) {
                try {
                  onAlertTriggered(normalizedDetection);
                } catch (callbackErr) {
                  logger.error("Alert callback error:", callbackErr);
                }
              }
              
              setIsProcessing(true);

              // Keep detection visible for longer (5 seconds for weapons)
              const displayDuration = normalizedDetection.type === 'WEAPON' ? 5000 : 2000;
              setTimeout(() => {
                if (isMountedRef.current) {
                  setLastMatch(null);
                  setIsProcessing(false);
                }
              }, displayDuration);
            } else {
              logger.debug(`No detections in response. data.detections: ${data.detections}, length: ${data.detections?.length}`);
            }
          } catch (err) {
            logger.error("Detection request error:", err);
          } finally {
            processingFrameRef.current = false;
          }
        },
        'image/jpeg',
        0.8
      );
    } catch (err) {
      logger.error('Error capturing frame for inference:', err);
      processingFrameRef.current = false;
    }
  }, [enabled, onAlertTriggered]);

  return { lastMatch, runInference, isProcessing, isScanningBiometrics, wsConnected };
};
