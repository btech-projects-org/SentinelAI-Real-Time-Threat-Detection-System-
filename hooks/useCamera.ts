
import { useState, useRef, useCallback } from 'react';
import logger from '../utils/logger';

export const useCamera = () => {
  const [isActive, setIsActive] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  const startCamera = useCallback(async () => {
    try {
      setError(null);
      
      // 1. Validate browser support
      if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        throw new Error('UNSUPPORTED_BROWSER: Camera API not available');
      }
      
      // 2. Check permissions before requesting
      let permissionState: PermissionState | undefined;
      try {
        const permissionStatus = await navigator.permissions.query({ name: 'camera' as PermissionName });
        permissionState = permissionStatus.state;
        
        if (permissionState === 'denied') {
          throw new Error('PERMISSION_DENIED: Camera access blocked by user');
        }
      } catch (permErr) {
        // Permissions API not supported, continue anyway
        logger.warn('Permissions API not available:', permErr);
      }
      
      // 3. Request camera access with timeout
      const timeoutPromise = new Promise<never>((_, reject) => 
        setTimeout(() => reject(new Error('TIMEOUT: Camera initialization took too long')), 10000)
      );
      
      const streamPromise = navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 1280 }, 
          height: { ideal: 720 },
          facingMode: 'user'
        },
        audio: false 
      });
      
      const stream = await Promise.race([streamPromise, timeoutPromise]);
      
      // 4. Validate video element exists
      if (!videoRef.current) {
        stream.getTracks().forEach(track => track.stop());
        throw new Error('VIDEO_ELEMENT_NOT_READY');
      }
      
      // 5. Attach stream and setup handlers
      videoRef.current.srcObject = stream;
      
      videoRef.current.onloadedmetadata = () => {
        const video = videoRef.current;
        if (video) {
          video.play().then(() => {
            setIsActive(true);
          }).catch(playErr => {
            logger.error('Video play error:', playErr);
            setError('PLAYBACK_ERROR: Failed to start video');
            stream.getTracks().forEach(track => track.stop());
          });
        }
      };
      
      videoRef.current.onerror = (err) => {
        logger.error('Video element error:', err);
        setError('VIDEO_ERROR: Media playback failed');
        stream.getTracks().forEach(track => track.stop());
        setIsActive(false);
      };
      
    } catch (err) {
      let errorMessage = 'UNKNOWN_ERROR';
      
      if (err instanceof Error) {
        if (err.name === 'NotAllowedError') {
          errorMessage = 'PERMISSION_DENIED: User blocked camera access';
        } else if (err.name === 'NotFoundError') {
          errorMessage = 'DEVICE_NOT_FOUND: No camera detected';
        } else if (err.name === 'NotReadableError') {
          errorMessage = 'DEVICE_IN_USE: Camera already in use by another application';
        } else if (err.name === 'OverconstrainedError') {
          errorMessage = 'UNSUPPORTED_RESOLUTION: Requested resolution not supported';
        } else if (err.message.includes('TIMEOUT')) {
          errorMessage = err.message;
        } else {
          errorMessage = err.message;
        }
      }
      
      setError(errorMessage);
      logger.error('Camera initialization failed:', errorMessage, err);
    }
  }, []);

  const stopCamera = useCallback(() => {
    try {
      const video = videoRef.current;
      if (video && video.srcObject) {
        const stream = video.srcObject as MediaStream;
        const tracks = stream.getTracks();
        
        tracks.forEach(track => {
          track.stop();
          stream.removeTrack(track);
        });
        
        video.srcObject = null;
        video.onloadedmetadata = null;
        video.onerror = null;
        setIsActive(false);
        setError(null);
        
        logger.log('Camera stopped and cleaned up');
      }
    } catch (err) {
      logger.error('Error stopping camera:', err);
    }
  }, []);

  return { videoRef, isActive, error, startCamera, stopCamera };
};
