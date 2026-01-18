"""
Deep Learning Threat Detection Engine
Uses YOLOv8 for weapon detection and DeepFace for facial recognition
"""

import cv2
import numpy as np
import logging
from datetime import datetime
from typing import List, Optional, Dict
from pathlib import Path
import asyncio
import base64

# Deep Learning Imports
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    logging.warning("YOLOv8 not available. Install with: pip install ultralytics")

try:
    from deepface import DeepFace
    DEEPFACE_AVAILABLE = True
except ImportError:
    DEEPFACE_AVAILABLE = False
    logging.warning("DeepFace not available. Install with: pip install deepface")

from backend.config.config import get_settings

logger = logging.getLogger("sentinel.services.dl_threat_engine")

class Detection:
    def __init__(self, id: str, label: str, confidence: float, box: List[float], metadata: Optional[Dict] = None):
        self.id = id
        self.label = label
        self.confidence = confidence
        self.box = box  # [x, y, w, h] normalized
        self.timestamp = datetime.utcnow().isoformat()
        self.metadata = metadata or {}
    
    def to_dict(self):
        return {
            "id": self.id,
            "label": self.label,
            "confidence": self.confidence,
            "box": self.box,
            "timestamp": self.timestamp,
            "metadata": self.metadata
        }

class DeepLearningThreatEngine:
    """
    Production-grade threat detection using deep learning models:
    - YOLOv8 for weapon/threat object detection
    - DeepFace (Facenet512) for facial recognition
    """
    
    def __init__(self):
        self.settings = get_settings()
        
        # Model instances
        self.yolo_model = None
        self.yolo_available = YOLO_AVAILABLE
        self.deepface_available = DEEPFACE_AVAILABLE
        
        # Criminal face database (embeddings for fast comparison)
        self.criminal_embeddings = {}
        self.criminal_faces = {}  # Fallback: Raw images for histogram matching
        self.criminals_data = {}
        
        # Alert cooldown mechanism
        self.last_criminal_alert_time = {}
        self.last_weapon_alert_time = 0
        self.alert_cooldown = 30
        self.weapon_alert_cooldown = 20
        self.face_alert_cooldown = 30 # Cooldown for generic face alerts
        
        # Weapon classes that YOLO can detect
        self.weapon_classes = {
            'knife', 'scissors', 'gun', 'rifle', 'pistol', 'weapon',
            'sword', 'axe', 'baseball bat'
        }
        
        logger.info("🧠 Deep Learning Threat Engine initialized")
        logger.info(f"   YOLOv8: {'✅ Available' if YOLO_AVAILABLE else '❌ Not installed'}")
        logger.info(f"   DeepFace: {'✅ Available' if DEEPFACE_AVAILABLE else '❌ Not installed'}")

    def load_model(self):
        """Load YOLOv8 model for weapon detection."""
        if not self.yolo_available:
            logger.error("❌ YOLOv8 not available. Install with: pip install ultralytics")
            return
        
        try:
            model_path = Path(self.settings.YOLO_MODEL_PATH)
            
            # Auto-download YOLOv8n if model doesn't exist
            if not model_path.exists() and self.settings.AUTO_DOWNLOAD_MODELS:
                logger.info("📥 Downloading YOLOv8n model (first run only)...")
                model_path.parent.mkdir(parents=True, exist_ok=True)
                self.yolo_model = YOLO('yolov8n.pt')  # Auto-downloads
                logger.info("✅ YOLOv8n model downloaded successfully")
            else:
                logger.info(f"📂 Loading YOLOv8 model from: {model_path}")
                self.yolo_model = YOLO(str(model_path))
                logger.info("✅ YOLOv8 model loaded successfully")
            
            # Get model info
            logger.info(f"   Model: {self.yolo_model.model_name}")
            logger.info(f"   Classes: {len(self.yolo_model.names)} object types")
            logger.info(f"   Confidence threshold: {self.settings.YOLO_CONFIDENCE_THRESHOLD}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load YOLOv8 model: {e}")
            self.yolo_model = None

    async def load_criminals_from_db(self, db):
        """Load criminal faces and generate embeddings using DeepFace."""
        """Load criminal faces and generate embeddings using DeepFace."""
        # Note: We now proceed even if DeepFace is unavailable, to load raw images for fallback.
        if not self.deepface_available:
            logger.warning("DeepFace not available - Loading raw images for histogram matching fallback")
        
        try:
            criminals = await db.get_all_criminals()
            logger.info(f"🔍 Generating face embeddings for {len(criminals)} criminal profiles...")
            
            loaded_count = 0
            for criminal in criminals:
                criminal_id = criminal.get('criminal_id')
                image_base64 = criminal.get('image_base64')
                
                if criminal_id and image_base64:
                    try:
                        # Decode base64 image
                        img_data = base64.b64decode(image_base64)
                        nparr = np.frombuffer(img_data, np.uint8)
                        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                        
                        if img is not None:
                            if self.deepface_available:
                                # Generate face embedding using DeepFace
                                try:
                                    embedding = DeepFace.represent(
                                        img_path=img,
                                        model_name=self.settings.DEEPFACE_MODEL,
                                        detector_backend=self.settings.DEEPFACE_DETECTOR,
                                        enforce_detection=False
                                    )
                                    
                                    if embedding and len(embedding) > 0:
                                        # Store embedding (512-dimensional vector for Facenet512)
                                        self.criminal_embeddings[criminal_id] = embedding[0]['embedding']
                                        logger.info(f"   ✅ {criminal.get('name')} - embedding generated")
                                    else:
                                        logger.warning(f"   ⚠️  No face detected in {criminal_id} for embedding")
                                except Exception as e:
                                    logger.error(f"   ❌ DeepFace error for {criminal_id}: {e}")
                            
                            # MEMORY OPTIMIZATION: Only store raw faces if we are using the fallback system
                            if not self.deepface_available:
                                self.criminal_faces[criminal_id] = img
                                logger.info(f"   ✅ {criminal.get('name')} - raw face loaded (fallback)")
                            
                            self.criminals_data[criminal_id] = {
                                'name': criminal.get('name', 'Unknown'),
                                'threat_level': criminal.get('threat_level', 'MEDIUM'),
                                'description': criminal.get('description', ''),
                                'criminal_id': criminal_id
                            }
                            loaded_count += 1
                        else:
                            logger.warning(f"   ⚠️  Failed to decode image for {criminal_id}")
                    except Exception as e:
                        logger.error(f"   ❌ Error processing {criminal_id}: {e}")
            
            logger.info(f"✅ Loaded {loaded_count}/{len(criminals)} criminal profiles")
            logger.info(f"   Model: {self.settings.DEEPFACE_MODEL}")
            logger.info(f"   Distance metric: {self.settings.DEEPFACE_DISTANCE_METRIC}")
            
        except Exception as e:
            logger.error(f"❌ Failed to load criminals: {e}")

    def load_single_criminal(self, criminal_data: dict):
        """Load a single criminal face and generate embedding."""
        if not self.deepface_available:
            return False
        
        try:
            criminal_id = criminal_data.get('criminal_id')
            image_base64 = criminal_data.get('image_base64')
            
            if criminal_id and image_base64:
                # Decode image
                img_data = base64.b64decode(image_base64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Generate embedding
                    embedding = DeepFace.represent(
                        img_path=img,
                        model_name=self.settings.DEEPFACE_MODEL,
                        detector_backend=self.settings.DEEPFACE_DETECTOR,
                        enforce_detection=False
                    )
                    
                    if embedding and len(embedding) > 0:
                        self.criminal_embeddings[criminal_id] = embedding[0]['embedding']
                        self.criminals_data[criminal_id] = {
                            'name': criminal_data.get('name', 'Unknown'),
                            'threat_level': criminal_data.get('threat_level', 'MEDIUM'),
                            'description': criminal_data.get('description', ''),
                            'criminal_id': criminal_id
                        }
                        logger.info(f"✅ Embedding generated for: {criminal_data.get('name')}")
                        return True
            return False
        except Exception as e:
            logger.error(f"Error generating embedding: {e}")
            return False

    def detect_weapons_yolo(self, frame: np.ndarray):
        """Detect weapons AND people using YOLOv8."""
        if not self.yolo_model:
            return []
        
        try:
            # Run YOLOv8 inference
            results = self.yolo_model(
                frame,
                conf=self.settings.YOLO_CONFIDENCE_THRESHOLD,
                iou=self.settings.YOLO_IOU_THRESHOLD,
                verbose=False
            )
            
            weapons = []
            persons = []
            height, width = frame.shape[:2]
            
            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # Get class name
                    cls_id = int(box.cls[0])
                    class_name = self.yolo_model.names[cls_id].lower()
                    
                    # Get bounding box (normalized)
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    x, y, w, h = x1/width, y1/height, (x2-x1)/width, (y2-y1)/height
                    confidence = float(box.conf[0])
                    bbox = [float(x), float(y), float(w), float(h)]

                    # Check if it's a weapon
                    if any(weapon in class_name for weapon in self.weapon_classes):
                        weapons.append({
                            'type': class_name.title(),
                            'confidence': confidence,
                            'box': bbox,
                            'method': 'YOLOv8'
                        })
                        logger.warning(f"⚠️  WEAPON DETECTED: {class_name} (confidence: {confidence:.2%})")
                    
                    # Check if it's a person (for fighting detection)
                    elif class_name == 'person':
                        persons.append({
                            'type': 'Person',
                            'confidence': confidence,
                            'box': bbox,
                            'method': 'YOLOv8'
                        })
            
            return weapons, persons
            
        except Exception as e:
            logger.error(f"YOLOv8 detection error: {e}")
            return [], []

    def detect_fighting(self, persons: List[Dict]) -> List[Dict]:
        """
        Heuristic: Detect fighting based on high overlap (IoU) between two people.
        """
        if len(persons) < 2:
            return []
            
        fighting_events = []
        
        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1 = persons[i]
                p2 = persons[j]
                
                # Calculate Intersection over Union (IoU)
                box1 = p1['box'] # [x, y, w, h]
                box2 = p2['box']
                
                # Convert to x1, y1, x2, y2
                b1_x1, b1_y1, b1_x2, b1_y2 = box1[0], box1[1], box1[0]+box1[2], box1[1]+box1[3]
                b2_x1, b2_y1, b2_x2, b2_y2 = box2[0], box2[1], box2[0]+box2[2], box2[1]+box2[3]
                
                # Intersection
                x_left = max(b1_x1, b2_x1)
                y_top = max(b1_y1, b2_y1)
                x_right = min(b1_x2, b2_x2)
                y_bottom = min(b1_y2, b2_y2)
                
                if x_right < x_left or y_bottom < y_top:
                    intersection_area = 0.0
                else:
                    intersection_area = (x_right - x_left) * (y_bottom - y_top)
                
                # Union
                area1 = box1[2] * box1[3]
                area2 = box2[2] * box2[3]
                union_area = area1 + area2 - intersection_area
                
                iou = intersection_area / union_area if union_area > 0 else 0
                
                # Fighting Heuristic: Significant overlap (> 25%)
                if iou > 0.25:
                    fighting_events.append({
                        "id": f"violence_{int(datetime.now().timestamp()*1000)}",
                        "label": "VIOLENCE",
                        "type": "Fighting / Physical Conflict",
                        "confidence": 0.85, # Heuristic confidence
                        "box": box1, # Highlight one of the involved
                        "metadata": {
                            "detection_method": "Behavior Analysis (IoU Overlap)",
                            "severity": "CRITICAL",
                            "iou": iou
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    logger.warning(f"👊 FIGHTING DETECTED (IoU: {iou:.2f})")
        
        return fighting_events

    def match_face_deepface(self, face_roi: np.ndarray) -> tuple:
        """Match face against criminal database using DeepFace embeddings."""
        if not self.deepface_available or not self.criminal_embeddings:
            return None, 0.0
        
        try:
            # Generate embedding for detected face
            face_embedding = DeepFace.represent(
                img_path=face_roi,
                model_name=self.settings.DEEPFACE_MODEL,
                detector_backend=self.settings.DEEPFACE_DETECTOR,
                enforce_detection=False
            )
            
            if not face_embedding or len(face_embedding) == 0:
                return None, 0.0
            
            face_vec = np.array(face_embedding[0]['embedding'])
            
            # Compare with all criminal embeddings
            best_match = None
            best_distance = float('inf')
            
            for criminal_id, criminal_vec in self.criminal_embeddings.items():
                criminal_vec = np.array(criminal_vec)
                
                # Calculate distance based on metric
                if self.settings.DEEPFACE_DISTANCE_METRIC == 'cosine':
                    distance = 1 - np.dot(face_vec, criminal_vec) / (
                        np.linalg.norm(face_vec) * np.linalg.norm(criminal_vec)
                    )
                elif self.settings.DEEPFACE_DISTANCE_METRIC == 'euclidean':
                    distance = np.linalg.norm(face_vec - criminal_vec)
                else:  # euclidean_l2
                    distance = np.linalg.norm(face_vec - criminal_vec)
                
                if distance < best_distance:
                    best_distance = distance
                    best_match = criminal_id
            
            # Check if match is below threshold (lower distance = better match)
            if best_distance < self.settings.FACE_MATCH_THRESHOLD:
                confidence = 1 - best_distance  # Convert to similarity score
                logger.warning(f"🚨 CRIMINAL MATCH: {self.criminals_data[best_match]['name']} (distance: {best_distance:.4f})")
                return best_match, confidence
            
            return None, 0.0
            
        except Exception as e:
            logger.error(f"DeepFace matching error: {e}")
            return None, 0.0

    def match_face_histogram(self, face_roi: np.ndarray) -> tuple:
        """Fallback: Match face using OpenCV histogram comparison (Classic CV)."""
        if not self.criminal_faces:
             # Just in case, try to use embeddings if faces missing but embeddings present? No, histograms need raw pixels.
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        try:
            # Resize face for comparison
            face_resized = cv2.resize(face_roi, (100, 100))
            face_gray = cv2.cvtColor(face_resized, cv2.COLOR_BGR2GRAY) if len(face_resized.shape) == 3 else face_resized
            
            for criminal_id, criminal_img in self.criminal_faces.items():
                # Resize criminal image
                criminal_resized = cv2.resize(criminal_img, (100, 100))
                criminal_gray = cv2.cvtColor(criminal_resized, cv2.COLOR_BGR2GRAY) if len(criminal_resized.shape) == 3 else criminal_resized
                
                # Calculate similarity using histogram comparison
                hist1 = cv2.calcHist([face_gray], [0], None, [256], [0, 256])
                hist2 = cv2.calcHist([criminal_gray], [0], None, [256], [0, 256])
                
                # Normalize histograms
                cv2.normalize(hist1, hist1, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                cv2.normalize(hist2, hist2, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX)
                
                # Compare histograms
                score = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
                
                print(f"DEBUG: Comparing with {criminal_id}: score={score:.4f}")
                
                if score > best_score:
                    best_score = score
                    best_match = criminal_id
            
            # Threshold for positive match
            # Lowering to 0.40 for better recall with histogram matching
            if best_score > 0.40:
                # Log the raw score for debugging
                logger.info(f"Histogram match candidate: {best_match} score={best_score:.4f}")
                return best_match, best_score
                
        except Exception as e:
            logger.error(f"Histogram matching error: {e}")
        
        return None, 0.0

    def detect(self, frame: np.ndarray) -> List[Dict]:
        """
        Main detection pipeline using deep learning models.
        """
        detections = []
        
        if frame is None or frame.size == 0:
            return detections
        
        try:
            height, width = frame.shape[:2]
        except:
            return detections
        
        # 1. YOLO Weapon Detection
        # 1. YOLO Weapon & Fighting Detection
        try:
            weapons, persons = self.detect_weapons_yolo(frame)
            
            # Process Weapons
            for weapon in weapons:
                detection_dict = {
                    "id": f"weapon_{int(datetime.now().timestamp()*1000)}",
                    "label": "WEAPON",
                    "type": weapon['type'],
                    "confidence": float(weapon['confidence']),
                    "box": weapon['box'],
                    "metadata": {
                        "weapon_type": weapon['type'],
                        "detection_method": "YOLOv8 Deep Learning",
                        "severity": "CRITICAL" if weapon['confidence'] > 0.85 else "HIGH"
                    },
                    "timestamp": datetime.utcnow().isoformat()
                }
                detections.append(detection_dict)
                
                # Send Telegram alert
                try:
                    asyncio.create_task(
                        self.send_weapon_alert_telegram(
                            weapon_type=weapon['type'],
                            confidence=weapon['confidence'],
                            location="Real-time Feed"
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to queue weapon alert: {e}")

            # Process Fighting
            fights = self.detect_fighting(persons)
            for fight in fights:
                detections.append(fight)
                # Send Telegram alert for fighting
                try:
                    asyncio.create_task(
                        self.send_fighting_alert_telegram(
                            confidence=fight['confidence'],
                            location="Real-time Feed"
                        )
                    )
                except Exception as e:
                    logger.error(f"Failed to queue fighting alert: {e}")

        except Exception as e:
            logger.error(f"Weapon/Fighting detection error: {e}")
        
        # 2. Face Detection & Recognition using DeepFace
        if self.deepface_available:
            try:
                # Use DeepFace's built-in face detection
                faces = DeepFace.extract_faces(
                    img_path=frame,
                    detector_backend=self.settings.DEEPFACE_DETECTOR,
                    enforce_detection=False
                )
                
                for face_data in faces:
                    facial_area = face_data['facial_area']
                    x, y, w, h = facial_area['x'], facial_area['y'], facial_area['w'], facial_area['h']
                    
                    # Extract face ROI
                    face_roi = frame[y:y+h, x:x+w]
                    
                    # Normalize coordinates
                    box = [x / width, y / height, w / width, h / height]
                    
                    # Emit generic face detection
                    detections.append({
                        "id": f"face_{int(datetime.now().timestamp()*1000)}",
                        "label": "FACE",
                        "type": "FACE",
                        "confidence": float(face_data['confidence']) if face_data.get('confidence', 0) > 0 else 0.85,
                        "box": box,
                        "metadata": {
                            "detection_method": "DeepFace",
                            "severity": "INFO"
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # ALERT: Generic Face Detection (User requested alert for ALL faces)
                    try:
                         asyncio.create_task(
                            self.send_face_alert_telegram(
                                confidence=float(face_data['confidence']) if face_data.get('confidence', 0) > 0 else 0.85
                            )
                        )
                    except Exception as e:
                        logger.error(f"Failed to queue face alert: {e}")
                    
                    # Try criminal matching
                    matched_id, confidence = self.match_face_deepface(face_roi)
                    
                    if matched_id and matched_id in self.criminals_data:
                        criminal_info = self.criminals_data[matched_id]
                        
                        detection_dict = {
                            "id": f"criminal_match_{int(datetime.now().timestamp()*1000)}",
                            "label": "CRIMINAL_FACE",
                            "type": "CRIMINAL_FACE",
                            "name": criminal_info['name'],
                            "confidence": float(confidence),
                            "box": box,
                            "notes": criminal_info['description'],
                            "metadata": {
                                "name": criminal_info['name'],
                                "criminal_id": criminal_info['criminal_id'],
                                "threat_level": criminal_info['threat_level'],
                                "notes": criminal_info['description'],
                                "match_confidence": float(confidence),
                                "detection_method": f"DeepFace ({self.settings.DEEPFACE_MODEL})"
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        
                        detections.append(detection_dict)
                        
                        # Send Telegram alert
                        try:
                            asyncio.create_task(
                                self.send_criminal_alert_telegram(
                                    criminal_name=criminal_info['name'],
                                    criminal_id=matched_id,
                                    confidence=confidence,
                                    threat_level=criminal_info['threat_level']
                                )
                            )
                        except Exception as e:
                            logger.error(f"Failed to queue criminal alert: {e}")
                
            except Exception as e:
                logger.error(f"Face detection error: {e}")
        else:
            # Fallback: Use OpenCV Haar Cascade if DeepFace is unavailable
            try:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # Check if cascade is loaded (lazy load if needed)
                if not hasattr(self, 'face_cascade'):
                    cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                    self.face_cascade = cv2.CascadeClassifier(cascade_path)
                
                faces = self.face_cascade.detectMultiScale(
                    gray,
                    scaleFactor=1.1,
                    minNeighbors=4,  # Relaxed parameter
                    minSize=(30, 30)
                )
                
                logger.info(f"Fallback Detection: Found {len(faces)} face(s) using Haar Cascade")
                
                for (x, y, w, h) in faces:
                    # Normalize coordinates
                    box = [x / width, y / height, w / width, h / height]
                    
                    detections.append({
                        "id": f"face_{int(datetime.now().timestamp()*1000)}",
                        "label": "FACE",
                        "type": "FACE",
                        "confidence": 0.85, # Heuristic confidence
                        "box": box,
                        "metadata": {
                            "detection_method": "Haar Cascade (Fallback)",
                            "severity": "INFO"
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    
                    # Note: Cannot do criminal matching without DeepFace embeddings or separate logic
                    # To implement matching fallback, we'd need the Histogram logic from threat_engine.py
                    # For now, just detecting faces satisfies "face is not getting detected"
                    
                    # Try fallback matching (Histogram)
                    face_roi = frame[y:y+h, x:x+w]
                    matched_id, match_conf = self.match_face_histogram(face_roi)
                    
                    if matched_id and matched_id in self.criminals_data:
                        criminal_info = self.criminals_data[matched_id]
                        
                        detections.append({
                            "id": f"criminal_match_{int(datetime.now().timestamp()*1000)}",
                            "label": "CRIMINAL_FACE",
                            "type": "CRIMINAL_FACE",
                            "name": criminal_info['name'],
                            "confidence": float(match_conf),
                            "box": box,
                            "notes": criminal_info['description'],
                            "metadata": {
                                "name": criminal_info['name'],
                                "criminal_id": criminal_info['criminal_id'],
                                "threat_level": criminal_info['threat_level'],
                                "notes": criminal_info['description'],
                                "match_confidence": float(match_conf),
                                "detection_method": "Histogram Matching (Fallback)"
                            },
                            "timestamp": datetime.utcnow().isoformat()
                        })
                        
                        logger.warning(f"🚨 CRIMINAL MATCH (Histogram): {criminal_info['name']} (score: {match_conf:.4f})")

                        # Send Telegram alert
                        try:
                            asyncio.create_task(
                                self.send_criminal_alert_telegram(
                                    criminal_name=criminal_info['name'],
                                    criminal_id=matched_id,
                                    confidence=match_conf,
                                    threat_level=criminal_info['threat_level']
                                )
                            )
                        except Exception as e:
                            logger.error(f"Failed to queue criminal alert: {e}")
                    
            except Exception as e:
                logger.error(f"Fallback Haar detection error: {e}")
        
        return detections

    async def send_criminal_alert_telegram(self, criminal_name: str, criminal_id: str, confidence: float, threat_level: str):
        """Send Telegram alert for criminal detection."""
        try:
            from backend.services.telegram_service import telegram_service
            from backend.database.mongodb import db
            import time
            
            current_time = time.time()
            last_alert = self.last_criminal_alert_time.get(criminal_id, 0)
            
            if current_time - last_alert < self.alert_cooldown:
                return
            
            if telegram_service.enabled:
                severity_map = {
                    "CRITICAL": "CRITICAL",
                    "Violent Crimes": "HIGH",
                    "HIGH": "HIGH",
                    "MEDIUM": "MEDIUM",
                    "LOW": "LOW"
                }
                severity = severity_map.get(threat_level, "HIGH")
                
                success = await telegram_service.send_detection_alert(
                    detection_type="🚨 Criminal Face Match (DeepFace AI)",
                    severity=severity,
                    confidence=confidence,
                    location="Real-time Feed",
                    criminal_id=criminal_id,
                    criminal_name=criminal_name
                )
                
                if success:
                    self.last_criminal_alert_time[criminal_id] = current_time
                    logger.info(f"✅ Telegram alert sent for: {criminal_name}")
        except Exception as e:
            logger.error(f"Failed to send criminal alert: {e}")

    async def send_fighting_alert_telegram(self, confidence: float, location: str):
        """Send Telegram alert for fighting/violence detection."""
        try:
            from backend.services.telegram_service import telegram_service
            import time
            
            # Separate cooldown for fighting alerts (use weapon cooldown for now)
            current_time = time.time()
            if current_time - self.last_weapon_alert_time < self.weapon_alert_cooldown:
                return
            
            if telegram_service.enabled:
                success = await telegram_service.send_detection_alert(
                    detection_type="👊 Violent Activity Detected (Fighting)",
                    severity="CRITICAL",
                    confidence=confidence,
                    location=location
                )
                
                if success:
                    self.last_weapon_alert_time = current_time
                    logger.warning(f"⚠️  Telegram alert sent for fighting")
        except Exception as e:
            logger.error(f"Failed to send fighting alert: {e}")

    async def send_weapon_alert_telegram(self, weapon_type: str, confidence: float, location: str):
        """Send Telegram alert for weapon detection."""
        try:
            from backend.services.telegram_service import telegram_service
            import time
            
            current_time = time.time()
            if current_time - self.last_weapon_alert_time < self.weapon_alert_cooldown:
                return
            
            if telegram_service.enabled:
                severity = "CRITICAL" if confidence > 0.85 else "HIGH"
                
                success = await telegram_service.send_detection_alert(
                    detection_type=f"⚠️ {weapon_type} Detected (YOLOv8 AI)",
                    severity=severity,
                    confidence=confidence,
                    location=location
                )
                
                if success:
                    self.last_weapon_alert_time = current_time
                    logger.warning(f"⚠️  Telegram alert sent for weapon: {weapon_type}")
        except Exception as e:
            logger.error(f"Failed to send weapon alert: {e}")

    async def send_face_alert_telegram(self, confidence: float):
        """Send Telegram alert for generic face detection."""
        try:
            from backend.services.telegram_service import telegram_service
            import time
            
            current_time = time.time()
            # Use specific cooldown for generic faces
            if current_time - getattr(self, 'last_face_alert_time', 0) < self.face_alert_cooldown:
                return
            
            if telegram_service.enabled:
                success = await telegram_service.send_detection_alert(
                    detection_type="👤 Face Detected",
                    severity="INFO",
                    confidence=confidence,
                    location="Real-time Feed"
                )
                
                if success:
                    self.last_face_alert_time = current_time
                    logger.info(f"✅ Telegram alert sent for generic face")
        except Exception as e:
            logger.error(f"Failed to send face alert: {e}")

# Global instance
dl_threat_engine = DeepLearningThreatEngine()
