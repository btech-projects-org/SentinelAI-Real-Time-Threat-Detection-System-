import sys
import os
import subprocess
from pathlib import Path

# Robustness Patch: Add project root to sys.path to allow running executing this file directly
sys.path.append(str(Path(__file__).resolve().parent.parent))

# --- AUTO-VENV SWITCHER ---
# If running with global python, switch to venv automatically
def ensure_venv():
    # Check if running in a virtual environment
    is_venv = (hasattr(sys, 'real_prefix') or
               (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix))
    
    if not is_venv:
        print("🔄 Checking for virtual environment...")
        project_root = Path(__file__).resolve().parent.parent
        venv_path = None
        
        # Check potential venv locations
        for venv_name in [".venv", "venv"]:
            possible_path = project_root / venv_name / "Scripts" / "python.exe"
            if possible_path.exists():
                venv_path = possible_path
                break
        
        if venv_path:
            print(f"✅ Found venv at: {venv_path}")
            print(f"🔄 Switching to virtual environment...")
            
            # Re-execute this script with the venv python
            # Pass all original arguments
            args = [str(venv_path), __file__] + sys.argv[1:]
            
            # Flush stdout before switch
            sys.stdout.flush()
            
            try:
                subprocess.run(args)
                sys.exit(0) # Exit the global python process after child finishes
            except Exception as e:
                print(f"❌ Failed to switch to venv: {e}")
                # Fallthrough to try running anyway (might fail modules)
        else:
             print("⚠️  No virtual environment found. Running with global Python.")

ensure_venv()
# ---------------------------

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, File, UploadFile, Form, Depends, HTTPException, status, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.httpsredirect import HTTPSRedirectMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware
from backend.config.config import get_settings
from backend.database.mongodb import db
from backend.services.dl_threat_engine import dl_threat_engine
from backend.services.telegram_service import telegram_service
from backend.services.telegram_service import telegram_service
from backend.mcp_client.mcp_client import mcp_client
from backend.auth import security, deps
from backend.schemas.criminal import CriminalCreate
import logging
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from typing import List
import base64
from io import BytesIO
from datetime import datetime
from bson import ObjectId
import json
import asyncio
from fastapi.encoders import jsonable_encoder
from contextlib import asynccontextmanager
import time

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinel.main")
settings = get_settings()

# Custom JSON encoder for MongoDB ObjectId
class JSONEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, ObjectId):
            return str(o)
        if isinstance(o, datetime):
            return o.isoformat()
        return super().default(o)

# Lifespan event handler (replaces deprecated on_event)
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    await db.connect()
    dl_threat_engine.load_model()  # Load YOLOv8
    
    # Load criminal faces and generate embeddings
    await dl_threat_engine.load_criminals_from_db(db)
    
    logger.info("🧠 Deep Learning System Startup Complete")
    logger.info(f"   YOLOv8: {'✅ ENABLED' if dl_threat_engine.yolo_model else '❌ DISABLED'}")
    logger.info(f"   DeepFace: {'✅ ENABLED' if dl_threat_engine.deepface_available else '❌ DISABLED'}")
    logger.info(f"   Telegram: {'✅ ENABLED' if telegram_service.enabled else '❌ DISABLED'}")
    logger.info(f"   Criminal Profiles: {len(dl_threat_engine.criminal_embeddings)} loaded")
    
    yield
    
    # Shutdown
    await db.close()
    await mcp_client.close()
    logger.info("System Shutdown Complete")

class RateLimiterMiddleware(BaseHTTPMiddleware):
    """Simple in-memory rate limiter to reduce DoS risk."""

    def __init__(self, app: FastAPI, default_limit: int, default_window: int, path_limits: dict | None = None):
        super().__init__(app)
        self.default_limit = default_limit
        self.default_window = default_window
        self.path_limits = path_limits or {}
        self.requests: dict[str, list[float]] = {}
        self.lock = asyncio.Lock()

    def _get_limits(self, path: str) -> tuple[int, int]:
        for prefix, (limit, window) in self.path_limits.items():
            if path.startswith(prefix):
                return limit, window
        return self.default_limit, self.default_window

    async def dispatch(self, request, call_next):
        # Allow health and docs without limit
        if request.url.path in ("/health", "/docs", "/openapi.json", "/api/v1/token"):
            return await call_next(request)
        
        # Production Hardening: Removed localhost bypass
        # All clients must respect rate limits
        client_ip = request.client.host if request.client else "unknown"


        limit, window = self._get_limits(request.url.path)
        now = time.time()
        key = f"{client_ip}:{window}:{limit}"

        async with self.lock:
            timestamps = self.requests.get(key, [])
            cutoff = now - window
            timestamps = [t for t in timestamps if t >= cutoff]

            if len(timestamps) >= limit:
                retry_after = int(max(1, window - (now - timestamps[0])))
                return JSONResponse(
                    status_code=429,
                    content={
                        "detail": "Rate limit exceeded. Please slow down.",
                        "retry_after": retry_after,
                    },
                    headers={"Retry-After": str(retry_after)},
                )

            timestamps.append(now)
            self.requests[key] = timestamps

        return await call_next(request)


app = FastAPI(title=settings.APP_NAME, version=settings.VERSION, lifespan=lifespan)

# Thread pool for non-blocking inference
executor = ThreadPoolExecutor(max_workers=4)
DETECT_SEMAPHORE = asyncio.Semaphore(4)  # limit concurrent frame processing

def serialize_for_json(obj):
    """Convert MongoDB objects to JSON-serializable types"""
    if isinstance(obj, ObjectId):
        return str(obj)
    elif isinstance(obj, dict):
        return {k: serialize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [serialize_for_json(item) for item in obj]
    elif isinstance(obj, datetime):
        return obj.isoformat()
    return obj

# CORS (Strict Production Config)
origins = [origin.strip() for origin in settings.ALLOWED_ORIGINS.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Authorization", "Content-Type"],
)

# HTTPS Redirect (Enable in Production)
if not settings.DEBUG:
    app.add_middleware(HTTPSRedirectMiddleware)

# Rate limiting to mitigate DoS (defaults: 60 req/min per IP)
rate_limit_paths = {
    "/api/v1/detect-frame": (10, 60),        # 10 requests per 60s per IP
    "/api/v1/criminals/register": (5, 300),  # 5 requests per 5 minutes per IP
}

app.add_middleware(
    RateLimiterMiddleware,
    default_limit=60,
    default_window=60,
    path_limits=rate_limit_paths,
)

@app.get("/health")
async def health_check():
    return {
        "status": "ok",
        "version": settings.VERSION,
        "deep_learning": {
            "yolo_enabled": dl_threat_engine.yolo_model is not None,
            "deepface_enabled": dl_threat_engine.deepface_available,
            "yolo_model": settings.YOLO_MODEL_PATH if dl_threat_engine.yolo_model else None,
            "face_model": settings.DEEPFACE_MODEL if dl_threat_engine.deepface_available else None
        }
    }

@app.get("/api/v1/threat-engine-status")
async def threat_engine_status():
    return {
        "engine": "Deep Learning Threat Engine v2.0",
        "models": {
            "weapon_detection": {
                "enabled": dl_threat_engine.yolo_model is not None,
                "model": "YOLOv8" if dl_threat_engine.yolo_model else None,
                "confidence_threshold": settings.YOLO_CONFIDENCE_THRESHOLD
            },
            "face_recognition": {
                "enabled": dl_threat_engine.deepface_available,
                "model": settings.DEEPFACE_MODEL if dl_threat_engine.deepface_available else None,
                "distance_metric": settings.DEEPFACE_DISTANCE_METRIC,
                "match_threshold": settings.FACE_MATCH_THRESHOLD
            }
        },
        "criminal_database": {
            "profiles_loaded": len(dl_threat_engine.criminal_embeddings),
            "embeddings_generated": len(dl_threat_engine.criminal_embeddings)
        },
        "detection_mode": "DEEP_LEARNING"
    }

@app.post("/api/v1/token")
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Exchange username and password for JWT access token.
    For this single-admin system, username must be 'admin' and password matches env.
    """
    if form_data.username != "admin":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    # Verify password against settings.ADMIN_PASSWORD (simple matching as required by user for now)
    # Ideally use hash, but user prompt said 'simple system where admin password is stored in .env'
    # To be secure, we should verify form_data.password == settings.ADMIN_PASSWORD
    if form_data.password != settings.ADMIN_PASSWORD:
         raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    access_token = security.create_access_token(
        subject=form_data.username
    )
    return {"access_token": access_token, "token_type": "bearer"}

@app.post("/api/v1/detect-frame")
async def detect_frame(
    file: UploadFile = File(...),
    current_user: str = Depends(deps.get_current_user)
):
    """
    REST endpoint for real-time frame detection.
    Accepts JPEG/PNG frame and returns detections.
    """
    try:
        async with DETECT_SEMAPHORE:
            # Read frame data
            frame_data = await file.read()
            
            if not frame_data or len(frame_data) == 0:
                logger.warning(f"Empty frame data received")
                return {"error": "Empty frame data", "detections": []}
            # Decode frame
            nparr = np.frombuffer(frame_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                logger.warning(f"Failed to decode frame from {file.filename}")
                return {"error": "Frame decode failed", "detections": []}
            
            logger.debug(f"Frame decoded successfully: {frame.shape}")
            
            # Run detection
            try:
                detections = dl_threat_engine.detect(frame)
                logger.info(f"Detection complete: {len(detections)} total detections")
                for det in detections:
                    logger.info(f"  - {det.get('label', det.get('type'))}: {det.get('confidence'):.2%}")
            except Exception as detection_err:
                logger.error(f"Detection error: {detection_err}", exc_info=True)
                detections = []
            
            # Serialize detections
            serialized_detections = [serialize_for_json(det) for det in detections]
            
            # Save detections to database if any found
            if detections:
                for det in detections:
                    try:
                        await db.save_alert(det)
                    except Exception as db_err:
                        logger.error(f"Failed to save alert: {db_err}")
            
            return {
                "status": "success",
                "detections": serialized_detections,
                "detection_count": len(serialized_detections),
                "deep_learning_enabled": dl_threat_engine.yolo_model is not None
            }
    
    except Exception as e:
        logger.error(f"Frame detection error: {e}", exc_info=True)
        return {
            "status": "error",
            "error": str(e),
            "detections": []
        }

@app.post("/api/v1/criminals/register")
async def register_criminal(
    criminal_data: CriminalCreate = Depends(CriminalCreate.as_form),
    image: UploadFile = File(...),
    current_user: str = Depends(deps.get_current_user)
):
    """Register a criminal with biometric data"""
    try:
        # Pydantic has validated the input
        criminal_id = criminal_data.criminal_id
        name = criminal_data.name
        threat_level = criminal_data.threat_level
        description = criminal_data.description
        # Read image file
        image_data = await image.read()
        if not image_data:
            return {
                "status": "error",
                "message": "Image file is empty"
            }

        # Reject very large images (>4MB) to stay below MongoDB document limits
        if len(image_data) > 4 * 1024 * 1024:
            return {
                "status": "error",
                "message": "Image too large (max 4MB)"
            }

        image_base64 = base64.b64encode(image_data).decode()
        logger.info(f"Registering criminal {criminal_id} | image bytes: {len(image_data)} | base64 length: {len(image_base64)}")
        
        # Create criminal profile
        criminal_profile = {
            "criminal_id": criminal_id,
            "name": name,
            "threat_level": threat_level,
            "description": description,
            "image_base64": image_base64,
            "registered_at": datetime.utcnow().isoformat(),
            "status": "ACTIVE"
        }
        
        # Save to database
        saved_id = await db.save_criminal(criminal_profile)
        
        if saved_id:
            # Load the criminal face into threat engine immediately
            load_success = dl_threat_engine.load_single_criminal(criminal_profile)
            
            logger.info(f"✅ Criminal registered: {name} (ID: {criminal_id})")
            logger.info(f"   Face embedding generated: {'✅ Success' if load_success else '❌ Failed'}")
            
            return {
                "status": "success",
                "message": f"Criminal '{name}' registered successfully",
                "criminal_id": criminal_id,
                "database_id": saved_id,
                "embedding_generated": load_success,
                "total_criminals_loaded": len(dl_threat_engine.criminal_embeddings)
            }
        else:
            logger.error(f"Failed to save criminal: {name}")
            return {
                "status": "error",
                "message": "Failed to save criminal to database"
            }
    except Exception as e:
        logger.error(f"Criminal registration error: {str(e)}", exc_info=True)
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/v1/criminals/list")
async def list_criminals():
    """Get all registered criminals"""
    try:
        criminals = await db.get_all_criminals()
        # Don't send base64 images in list view for performance
        criminal_list = []
        for criminal in criminals:
            criminal_summary = {
                "criminal_id": criminal.get("criminal_id"),
                "name": criminal.get("name"),
                "threat_level": criminal.get("threat_level"),
                "status": criminal.get("status"),
                "registered_at": criminal.get("registered_at")
            }
            criminal_list.append(criminal_summary)
        
        logger.info(f"Listed {len(criminal_list)} criminals")
        return {
            "status": "success",
            "total": len(criminal_list),
            "criminals": criminal_list
        }
    except Exception as e:
        logger.error(f"Failed to list criminals: {str(e)}")
        return {
            "status": "error",
            "message": str(e),
            "criminals": []
        }

@app.get("/api/v1/criminals/{criminal_id}")
async def get_criminal(criminal_id: str):
    """Get a specific criminal profile"""
    try:
        criminal = await db.get_criminal_by_id(criminal_id)
        if criminal:
            return {
                "status": "success",
                "criminal": serialize_for_json(criminal)
            }
        else:
            return {
                "status": "not_found",
                "message": f"Criminal {criminal_id} not found"
            }
    except Exception as e:
        logger.error(f"Failed to get criminal: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/v1/criminals/reload")
async def reload_criminals(current_user: str = Depends(deps.get_current_user)):
    """Manually reload all criminal faces from database into memory"""
    try:
        await dl_threat_engine.load_criminals_from_db(db)
        return {
            "status": "success",
            "message": "Criminal face embeddings regenerated successfully",
            "total_loaded": len(dl_threat_engine.criminal_embeddings),
            "criminals": list(dl_threat_engine.criminals_data.keys())
        }
    except Exception as e:
        logger.error(f"Failed to reload criminals: {str(e)}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.get("/api/v1/telegram/status")
async def telegram_status():
    """Get Telegram bot configuration status"""
    return {
        "status": "success",
        "telegram_enabled": settings.TELEGRAM_ENABLED,
        "bot_token_configured": bool(settings.TELEGRAM_BOT_TOKEN),
        "chat_id_configured": bool(settings.TELEGRAM_CHAT_ID),
        "message": "Telegram alerts enabled" if settings.TELEGRAM_ENABLED else "Telegram alerts disabled"
    }

@app.post("/api/v1/telegram/test")
async def test_telegram_connection():
    """Test Telegram bot connection"""
    if not settings.TELEGRAM_ENABLED:
        return {
            "status": "error",
            "message": "Telegram not configured",
            "connected": False
        }
    
    try:
        is_connected = await telegram_service.test_connection()
        return {
            "status": "success" if is_connected else "error",
            "message": "Connected to Telegram API" if is_connected else "Failed to connect to Telegram API",
            "connected": is_connected
        }
    except Exception as e:
        logger.error(f"Telegram test failed: {e}")
        return {
            "status": "error",
            "message": str(e),
            "connected": False
        }

@app.post("/api/v1/telegram/send-alert")
async def send_telegram_alert(
    alert_type: str = Form(...),
    severity: str = Form(default="MEDIUM"),
    message: str = Form(...)
):
    """Send a manual alert to Telegram"""
    if not settings.TELEGRAM_ENABLED:
        return {
            "status": "error",
            "message": "Telegram not configured"
        }
    
    try:
        formatted_message = f"<b>🚨 {alert_type}</b>\n\n<b>Severity:</b> {severity}\n\n{message}"
        success = await telegram_service.send_message(formatted_message)
        
        # Save to database
        if success:
            telegram_record = {
                "message_id": f"msg_{datetime.utcnow().timestamp()}",
                "chat_id": settings.TELEGRAM_CHAT_ID,
                "text": message,
                "message_type": "manual_alert",
                "alert_type": alert_type,
                "severity": severity,
                "status": "sent",
                "timestamp": datetime.utcnow().isoformat()
            }
            await db.save_telegram_message(telegram_record)
        
        return {
            "status": "success" if success else "error",
            "message": "Alert sent to Telegram" if success else "Failed to send alert",
            "sent": success
        }
    except Exception as e:
        logger.error(f"Failed to send telegram alert: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/v1/telegram/test-criminal-alert")
async def test_criminal_alert(criminal_id: str = Form(...), criminal_name: str = Form(...)):
    """Test sending a criminal detection alert to Telegram"""
    if not settings.TELEGRAM_ENABLED:
        return {
            "status": "error",
            "message": "Telegram not configured"
        }
    
    try:
        # Get criminal info from database
        criminal = await db.get_criminal_by_id(criminal_id)
        
        if not criminal:
            return {
                "status": "error",
                "message": f"Criminal {criminal_id} not found"
            }
        
        threat_level = criminal.get("threat_level", "MEDIUM")
        
        # Send alert through threat engine
        await dl_threat_engine.send_criminal_alert_telegram(
            criminal_name=criminal_name,
            criminal_id=criminal_id,
            confidence=0.95,
            threat_level=threat_level
        )
        
        return {
            "status": "success",
            "message": f"Criminal alert sent for {criminal_name}",
            "criminal": {
                "id": criminal_id,
                "name": criminal_name,
                "threat_level": threat_level
            }
        }
    except Exception as e:
        logger.error(f"Failed to send criminal alert: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.post("/api/v1/telegram/test-weapon-alert")
async def test_weapon_alert(weapon_type: str = Form(...)):
    """Test sending a weapon detection alert to Telegram"""
    if not settings.TELEGRAM_ENABLED:
        return {
            "status": "error",
            "message": "Telegram not configured"
        }
    
    try:
        # Send weapon alert
        await dl_threat_engine.send_weapon_alert_telegram(
            weapon_type=weapon_type,
            confidence=0.92,
            location="Security Feed"
        )
        
        return {
            "status": "success",
            "message": f"Weapon alert sent for {weapon_type}",
            "weapon": {
                "type": weapon_type,
                "confidence": 0.92,
                "location": "Security Feed"
            }
        }
    except Exception as e:
        logger.error(f"Failed to send weapon alert: {e}")
        return {
            "status": "error",
            "message": str(e)
        }

@app.websocket("/ws/video-feed")
async def websocket_endpoint(websocket: WebSocket):
    connection_accepted = False
    
    try:
        await websocket.accept()
        connection_accepted = True
        logger.info("✅ WebSocket client connected")
        
        # Send initial ready message with error handling
        try:
            await websocket.send_json({
                "status": "ready",
                "message": "Deep Learning threat detection ready for video stream",
                "yolo_enabled": dl_threat_engine.yolo_model is not None,
                "deepface_enabled": dl_threat_engine.deepface_available
            })
        except Exception as send_err:
            logger.error(f"Failed to send ready message: {send_err}")
            raise  # Propagate to outer try-catch
            
    except Exception as accept_err:
        logger.error(f"❌ WebSocket connection failed: {accept_err}")
        if connection_accepted:
            try:
                await websocket.close(code=1011, reason=str(accept_err))
            except:
                pass
        return
        
    frame_count = 0
    consecutive_errors = 0
    MAX_CONSECUTIVE_ERRORS = 10
    
    try:
        while True:
            try:
                # Receive frame bytes with timeout
                data = await asyncio.wait_for(websocket.receive_bytes(), timeout=30.0)
                frame_count += 1
                
                # Validate data
                if not data or len(data) == 0:
                    logger.warning(f"Received empty frame data at frame {frame_count}")
                    await websocket.send_json({"error": "Empty frame data received"})
                    continue
                
                # Decode frame with validation
                nparr = np.frombuffer(data, np.uint8)
                if nparr.size == 0:
                    logger.error(f"Invalid frame buffer at frame {frame_count}")
                    await websocket.send_json({"error": "Invalid frame buffer"})
                    continue
                
                # Try to decode the image (supports JPEG, PNG, etc.)
                frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if frame is None:
                    logger.error(f"Frame decode failed at frame {frame_count}, data size: {len(data)} bytes")
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        raise Exception("Too many consecutive decode errors")
                    # Send error response
                    try:
                        await websocket.send_json({
                            "error": "Frame decode failed",
                            "frame_count": frame_count,
                            "data_size": len(data)
                        })
                    except:
                        pass
                    continue
                
                # Reset error counter on success
                consecutive_errors = 0
                
                # Run Inference
                try:
                    detections = dl_threat_engine.detect(frame)
                except Exception as detection_err:
                    logger.error(f"Threat engine detection error: {detection_err}", exc_info=True)
                    detections = []
                    consecutive_errors += 1
                    if consecutive_errors >= MAX_CONSECUTIVE_ERRORS:
                        raise Exception("Too many consecutive detection errors")
                    continue
                
                # Serialize detections
                serialized_detections = [serialize_for_json(det) for det in detections]
                
                # Build response
                response = {
                    "processed": True,
                    "frame_count": frame_count,
                    "deep_learning_enabled": dl_threat_engine.yolo_model is not None,
                    "detections": serialized_detections,
                    "detection_count": len(serialized_detections)
                }
                
                # Save detections to database
                if detections:
                    for det in detections:
                        try:
                            saved_id = await db.save_alert(det)
                            if saved_id:
                                logger.info(f"Detection saved: {det['label']} (confidence: {det['confidence']:.2f})")
                            else:
                                logger.warning(f"Failed to save detection: {det['label']}")
                        except Exception as db_err:
                            logger.error(f"Database save error: {db_err}")
                            # Continue processing even if DB save fails
                
                # Send response
                try:
                    await websocket.send_text(json.dumps(response, cls=JSONEncoder))
                except Exception as json_error:
                    logger.error(f"JSON serialization error: {json_error}")
                    await websocket.send_json({"error": "Serialization error", "frame_count": frame_count})
                    
            except asyncio.TimeoutError:
                logger.warning(f"WebSocket receive timeout after {frame_count} frames")
                await websocket.send_json({"status": "timeout", "message": "No data received for 30s"})
                break
            except Exception as frame_err:
                error_type = type(frame_err).__name__
                logger.error(f"Frame processing error at frame {frame_count}: [{error_type}] {frame_err}", exc_info=True)
                try:
                    await websocket.send_json({
                        "error": str(frame_err),
                        "error_type": error_type,
                        "frame_count": frame_count
                    })
                except:
                    # If we can't send error, connection is likely dead
                    logger.debug(f"Could not send error response, connection likely dead")
                    break
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected gracefully after {frame_count} frames")
    except Exception as e:
        logger.error(f"WebSocket critical error after {frame_count} frames: {e}")
        try:
            await websocket.close(code=1011, reason=str(e))
        except:
            pass

if __name__ == "__main__":
    import uvicorn
    logger.info(f"Starting {settings.APP_NAME} v{settings.VERSION}")
    logger.info(f"Server: http://{settings.HOST}:{settings.PORT}")
    logger.info(f"API Docs: http://{settings.HOST}:{settings.PORT}/docs")
    uvicorn.run(app, host=settings.HOST, port=settings.PORT, log_level="info")
