import cv2
import numpy as np
import logging
from datetime import datetime
from typing import List, Optional, Dict
import google.generativeai as genai
from backend.config.config import get_settings
import base64
import asyncio

logger = logging.getLogger("sentinel.services.threat_engine")

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

class ThreatEngine:
    def __init__(self, model_path: Optional[str] = None):
        self.model_path = model_path
        self.model = None
        self.fgbg = cv2.createBackgroundSubtractorMOG2(history=500, varThreshold=16, detectShadows=True)
        self.min_area = 500
        
        # Load Haar Cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)
        
        # Cache for criminal faces
        self.criminal_faces = {}
        self.criminals_data = {}
        
        # Debounce mechanism for alerts to prevent spam
        self.last_criminal_alert_time = {}  # Dict to store last alert time per criminal_id
        self.last_weapon_alert_time = 0  # Last time a weapon alert was sent
        self.alert_cooldown = 30  # Seconds between same alerts
        
        # Weapon tracking to avoid duplicate detections
        self.tracked_weapons = {}  # Dict to store tracked weapon positions
        self.weapon_alert_cooldown = 20  # Seconds between weapon alerts
        
        # Initialize Gemini API
        settings = get_settings()
        if settings.GEMINI_API_KEY:
            try:
                genai.configure(api_key=settings.GEMINI_API_KEY)
                self.gemini_available = True
                logger.info("✅ Gemini AI initialized for enhanced threat analysis")
            except Exception as e:
                logger.error(f"Failed to initialize Gemini: {e}")
                self.gemini_available = False
        else:
            self.gemini_available = False
            logger.warning("⚠️  Gemini API key not provided. Using heuristic mode only.")
        
        logger.info(f"ThreatEngine initialized with face detection enabled")

    def load_model(self):
        """Load face detection model and criminal database."""
        if self.model_path and "yolo" in self.model_path.lower():
            logger.info("Loading YOLO model (Simulated)...")
            self.model = "MOCK_YOLO_MODEL"
        else:
            logger.info("Using OpenCV Haar Cascade for face detection")
        
        logger.info("Face detection engine ready")
    
    async def load_criminals_from_db(self, db):
        """Load criminal faces from database for matching."""
        try:
            criminals = await db.get_all_criminals()
            logger.info(f"Loading {len(criminals)} criminal profiles for face matching...")
            
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
                            # Store the face image for matching
                            self.criminal_faces[criminal_id] = img
                            self.criminals_data[criminal_id] = {
                                'name': criminal.get('name', 'Unknown'),
                                'threat_level': criminal.get('threat_level', 'MEDIUM'),
                                'description': criminal.get('description', ''),
                                'criminal_id': criminal_id
                            }
                            logger.info(f"✅ Loaded face for: {criminal.get('name')}")
                        else:
                            logger.warning(f"Failed to decode image for {criminal_id}")
                    except Exception as e:
                        logger.error(f"Error loading criminal {criminal_id}: {e}")
            
            logger.info(f"✅ Loaded {len(self.criminal_faces)} criminal faces for recognition")
        except Exception as e:
            logger.error(f"Failed to load criminals from database: {e}")
    
    async def send_criminal_alert_telegram(self, criminal_name: str, criminal_id: str, confidence: float, threat_level: str):
        """Send Telegram alert when criminal is identified with cooldown to prevent spam."""
        try:
            from services.telegram_service import telegram_service
            from database.mongodb import db
            from datetime import datetime
            import time
            
            # Check cooldown - don't send alerts for same criminal too frequently
            current_time = time.time()
            last_alert = self.last_criminal_alert_time.get(criminal_id, 0)
            time_since_last = current_time - last_alert
            
            if time_since_last < self.alert_cooldown:
                logger.debug(f"⏱️  Cooldown active for {criminal_name}. Last alert was {time_since_last:.0f}s ago. Skipping alert.")
                return
            
            if telegram_service.enabled:
                # Determine severity based on threat level
                severity_map = {
                    "CRITICAL": "CRITICAL",
                    "Violent Crimes": "HIGH",
                    "HIGH": "HIGH",
                    "MEDIUM": "MEDIUM",
                    "LOW": "LOW"
                }
                
                severity = severity_map.get(threat_level, "HIGH")
                
                success = await telegram_service.send_detection_alert(
                    detection_type="Criminal Face Match",
                    severity=severity,
                    confidence=confidence,
                    location="Real-time Feed",
                    criminal_id=criminal_id,
                    criminal_name=criminal_name
                )
                
                if success:
                    # Update last alert time
                    self.last_criminal_alert_time[criminal_id] = current_time
                    logger.info(f"✅ Telegram alert sent for criminal: {criminal_name}")
                    
                    # Save alert to database
                    try:
                        telegram_record = {
                            "message_id": f"msg_{datetime.utcnow().timestamp()}",
                            "chat_id": telegram_service.chat_id,
                            "text": f"Criminal detected: {criminal_name} (Confidence: {confidence:.1%})",
                            "message_type": "criminal_detection",
                            "detection_type": "Criminal Face Match",
                            "severity": severity,
                            "criminal_id": criminal_id,
                            "criminal_name": criminal_name,
                            "confidence": float(confidence),
                            "status": "sent",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        await db.save_telegram_message(telegram_record)
                        logger.debug(f"Alert saved to database")
                    except Exception as db_err:
                        logger.error(f"Failed to save alert to database: {db_err}")
                else:
                    logger.warning(f"Failed to send telegram alert for {criminal_name}")
            else:
                logger.debug("Telegram not configured - skipping alert")
        except Exception as e:
            logger.error(f"Failed to send telegram alert: {e}")
    
    def load_single_criminal(self, criminal_data: dict):
        """Load a single criminal face into memory (sync version for immediate use)."""
        try:
            criminal_id = criminal_data.get('criminal_id')
            image_base64 = criminal_data.get('image_base64')
            
            if criminal_id and image_base64:
                # Decode base64 image
                img_data = base64.b64decode(image_base64)
                nparr = np.frombuffer(img_data, np.uint8)
                img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                
                if img is not None:
                    # Store the face image for matching
                    self.criminal_faces[criminal_id] = img
                    self.criminals_data[criminal_id] = {
                        'name': criminal_data.get('name', 'Unknown'),
                        'threat_level': criminal_data.get('threat_level', 'MEDIUM'),
                        'description': criminal_data.get('description', ''),
                        'criminal_id': criminal_id
                    }
                    logger.info(f"✅ Dynamically loaded face for: {criminal_data.get('name')} (ID: {criminal_id})")
                    return True
                else:
                    logger.warning(f"Failed to decode image for {criminal_id}")
                    return False
            return False
        except Exception as e:
            logger.error(f"Error loading single criminal face: {e}")
            return False

    def detect_weapons_by_edges(self, frame_np: np.ndarray) -> List[Dict]:
        """
        Alternative weapon detection using edge detection.
        Catches weapons that motion detection might miss.
        """
        weapons = []
        try:
            gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
            
            # Apply edge detection
            edges = cv2.Canny(gray, 50, 150)
            
            # Dilate to connect nearby edges
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            edges = cv2.dilate(edges, kernel, iterations=2)
            
            # Find contours from edges
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            height, width = frame_np.shape[:2]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Very lenient area for edge-based detection
                if area < 400:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Lenient size for edge detection
                if w < 8 or h < 12:
                    continue
                
                aspect_ratio = float(w) / h if h != 0 else 0
                
                # Detect thin or elongated objects
                is_thin_object = 0.02 < aspect_ratio <= 0.60
                is_medium_object = 0.5 < aspect_ratio <= 3.0
                
                if is_thin_object or is_medium_object:
                    # Edge-based detection is lower confidence
                    confidence = 0.60
                    
                    if is_thin_object:
                        weapon_type = "Knife"
                    elif aspect_ratio <= 1.5:
                        weapon_type = "Firearm"
                    else:
                        weapon_type = "Rifle/Long Object"
                    
                    # Apply geometric constraints
                    perimeter = cv2.arcLength(contour, True)
                    if perimeter > 0:
                        circularity = 4 * np.pi * area / (perimeter * perimeter)
                        # Weapons should NOT be circular
                        if circularity < 0.6:
                            confidence += 0.05
                    
                    if confidence >= 0.65:
                        box = [x / width, y / height, w / width, h / height]
                        weapons.append({
                            'type': weapon_type,
                            'confidence': min(confidence, 0.85),  # Edge detection caps at 0.85
                            'box': box,
                            'area': area,
                            'position': (x + w//2, y + h//2),
                            'method': 'edge_detection'
                        })
                        logger.debug(f"Weapon (edge) detected: {weapon_type} (conf: {confidence:.2%})")
            
            return weapons
        except Exception as e:
            logger.error(f"Error in edge-based weapon detection: {e}")
            return []
    
    def detect_weapons_by_darkness(self, frame_np: np.ndarray) -> List[Dict]:
        """
        Detect weapons by looking for dark thin objects (knives often appear darker).
        Ultra-aggressive detection method for real-world scenarios.
        """
        weapons = []
        try:
            gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
            height, width = gray.shape[:2]
            
            # Find dark pixels (threshold: < 120)
            dark_mask = gray < 120
            
            # Dilate dark regions
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            dark_mask = cv2.dilate(dark_mask.astype(np.uint8) * 255, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(dark_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Very small minimum area for this method
                if area < 300:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Very lenient size
                if w < 6 or h < 10:
                    continue
                
                # Skip if object is mostly the entire frame
                if w > width * 0.8 or h > height * 0.8:
                    continue
                
                aspect_ratio = float(w) / h if h != 0 else 0
                
                # Look for thin or medium elongated shapes
                if (0.01 < aspect_ratio <= 0.70) or (0.4 < aspect_ratio <= 3.5):
                    # This is a potential weapon
                    if 0.01 < aspect_ratio <= 0.70:
                        weapon_type = "Knife"
                        confidence = 0.65
                    elif aspect_ratio <= 1.8:
                        weapon_type = "Firearm"
                        confidence = 0.60
                    else:
                        weapon_type = "Rifle/Long Object"
                        confidence = 0.60
                    
                    if confidence >= 0.60:
                        box = [x / width, y / height, w / width, h / height]
                        weapons.append({
                            'type': weapon_type,
                            'confidence': min(confidence, 0.80),
                            'box': box,
                            'area': area,
                            'position': (x + w//2, y + h//2),
                            'method': 'darkness_detection'
                        })
                        logger.debug(f"Weapon (darkness) detected: {weapon_type} (conf: {confidence:.2%})")
            
            return weapons
        except Exception as e:
            logger.error(f"Error in darkness-based weapon detection: {e}")
            return []

    def detect_weapons(self, frame_np: np.ndarray) -> List[Dict]:
        """
        FAST weapon detection using ONLY edge detection.
        Optimized for real-time performance - completes in <100ms.
        """
        weapons = []
        
        try:
            # Use ONLY edge detection (fastest method)
            gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
            
            # Quick edge detection with optimized parameters
            edges = cv2.Canny(gray, 100, 200)  # Higher thresholds = faster
            
            # Minimal dilation for speed
            kernel = np.ones((3, 3), np.uint8)
            edges = cv2.dilate(edges, kernel, iterations=1)
            
            # Find contours
            contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            height, width = frame_np.shape[:2]
            
            # Process only largest contours for speed
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                # Quick area filter
                if area < 500 or area > width * height * 0.5:
                    continue
                
                x, y, w, h = cv2.boundingRect(contour)
                
                # Quick size filter
                if w < 10 or h < 15:
                    continue
                
                aspect_ratio = float(w) / h if h != 0 else 0
                
                # Weapon shape detection
                weapon_type = None
                confidence = 0.70
                
                # Knife: thin vertical or angled
                if 0.05 < aspect_ratio <= 0.6:
                    weapon_type = "Knife"
                    confidence = 0.75
                # Firearm: medium ratio
                elif 0.6 < aspect_ratio <= 2.5:
                    weapon_type = "Firearm"
                    confidence = 0.75
                # Rifle: long horizontal
                elif 2.5 < aspect_ratio <= 5.0:
                    weapon_type = "Rifle/Long Object"
                    confidence = 0.70
                
                if weapon_type:
                    box = [x / width, y / height, w / width, h / height]
                    weapons.append({
                        'type': weapon_type,
                        'confidence': confidence,
                        'box': box,
                        'area': area,
                        'position': (x + w//2, y + h//2)
                    })
                    
                    # Limit to 3 detections for speed
                    if len(weapons) >= 3:
                        break
            
            logger.debug(f"Fast weapon detection: {len(weapons)} found")
            return weapons
            
        except Exception as e:
            logger.error(f"Weapon detection error: {e}")
            return []
    
    def match_face(self, face_roi):
        """Match detected face against criminal database using template matching."""
        if not self.criminal_faces:
            logger.debug("No criminal faces loaded; skipping match")
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
                
                if score > best_score:
                    best_score = score
                    best_match = criminal_id
            
            # Threshold for positive match
            if best_score > 0.65:  # Slightly lower threshold to allow matches
                return best_match, best_score
                
        except Exception as e:
            logger.error(f"Face matching error: {e}")
        
        return None, 0.0

    def detect(self, frame_np: np.ndarray) -> List[Detection]:
        """
        Run face detection and criminal matching on video frame.
        """
        detections = []
        
        if frame_np is None:
            logger.error("Frame is None")
            return detections
        
        if not isinstance(frame_np, np.ndarray):
            logger.error(f"Frame is not numpy array, got {type(frame_np)}")
            return detections
            
        if frame_np.size == 0:
            logger.error("Frame is empty")
            return detections
            
        try:
            height, width = frame_np.shape[:2]
        except Exception as e:
            logger.error(f"Failed to get frame dimensions: {e}")
            return detections
        
        # Convert to grayscale for face detection
        try:
            gray = cv2.cvtColor(frame_np, cv2.COLOR_BGR2GRAY)
        except Exception as e:
            logger.error(f"Failed to convert frame to grayscale: {e}")
            return detections
        
        # Detect faces
        try:
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=4,
                minSize=(30, 30)
            )
        except Exception as e:
            logger.error(f"Face cascade detection failed: {e}")
            return detections
        
        logger.info(f"Detected {len(faces)} face(s) in frame")
        if len(faces) == 0:
            logger.warning("No faces detected in frame (Haar Cascade)")
        
        # Weapon Detection using simple heuristics
        try:
            weapon_detected = self.detect_weapons(frame_np)
            if weapon_detected:
                for weapon in weapon_detected:
                    weapon_type = weapon.get('type', 'Weapon')
                    confidence = weapon.get('confidence', 0.75)
                    
                    detection_dict = {
                        "id": f"weapon_{int(datetime.now().timestamp()*1000)}",
                        "label": "WEAPON",
                        "type": weapon_type,
                        "confidence": float(confidence),
                        "box": weapon.get('box', [0, 0, 1, 1]),
                        "metadata": {
                            "weapon_type": weapon_type,
                            "detection_method": "Heuristic Analysis",
                            "severity": "CRITICAL" if confidence > 0.85 else "HIGH"
                        },
                        "timestamp": datetime.utcnow().isoformat()
                    }
                    detections.append(detection_dict)
                    logger.warning(f"⚠️  WEAPON DETECTED: {weapon_type} (Confidence: {confidence:.2%})")
                    
                    # Send Telegram alert when weapon is detected
                    try:
                        asyncio.create_task(
                            self.send_weapon_alert_telegram(
                                weapon_type=weapon_type,
                                confidence=confidence,
                                location="Real-time Feed"
                            )
                        )
                    except Exception as e:
                        logger.error(f"Failed to queue weapon alert: {e}")
        except Exception as e:
            logger.debug(f"Weapon detection error: {e}")
        
        # Process face detection
        for (x, y, w, h) in faces:
            # Extract face ROI
            face_roi = frame_np[y:y+h, x:x+w]
            
            # Normalize coordinates
            box = [x / width, y / height, w / width, h / height]
            
            # Always emit a generic face detection for UI visibility
            detections.append({
                "id": f"face_{int(datetime.now().timestamp()*1000)}",
                "label": "FACE",
                "type": "FACE",
                "confidence": 1.0,
                "box": box,
                "metadata": {
                    "detection_method": "Haar Cascade",
                    "severity": "INFO"
                },
                "timestamp": datetime.utcnow().isoformat()
            })
            
            # Try to match against criminal database
            matched_id, confidence = self.match_face(face_roi)
            if not matched_id:
                logger.debug("Face detected but no criminal match (or below threshold)")
                continue
            
            if matched_id in self.criminals_data:
                # CRIMINAL MATCH FOUND!
                criminal_info = self.criminals_data[matched_id]
                
                detection = Detection(
                    id=f"criminal_match_{int(datetime.now().timestamp()*1000)}",
                    label="CRIMINAL_FACE",
                    confidence=float(confidence),
                    box=box,
                    metadata={
                        "name": criminal_info['name'],
                        "criminal_id": criminal_info['criminal_id'],
                        "threat_level": criminal_info['threat_level'],
                        "notes": criminal_info['description'],
                        "match_confidence": float(confidence),
                        "detection_method": "Face Recognition"
                    }
                )
                
                # Override to_dict to include criminal info
                det_dict = detection.to_dict()
                det_dict['name'] = criminal_info['name']
                det_dict['type'] = 'CRIMINAL_FACE'
                det_dict['notes'] = criminal_info['description']
                
                detections.append(det_dict)
                logger.warning(f"🚨 CRIMINAL DETECTED: {criminal_info['name']} (Confidence: {confidence:.2%})")
                
                # Send Telegram alert when criminal is identified
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
                    logger.error(f"Failed to queue telegram alert: {e}")
        
        return detections
    
    async def send_weapon_alert_telegram(self, weapon_type: str, confidence: float, location: str = "Real-time Feed"):
        """Send Telegram alert when weapon is detected with rate limiting."""
        try:
            from services.telegram_service import telegram_service
            from database.mongodb import db
            from datetime import datetime
            import time
            
            # Check cooldown - don't send weapon alerts too frequently
            current_time = time.time()
            time_since_last = current_time - self.last_weapon_alert_time
            
            if time_since_last < self.weapon_alert_cooldown:
                logger.debug(f"⏱️  Weapon alert cooldown active. Last alert: {time_since_last:.0f}s ago. Skipping.")
                return
            
            if telegram_service.enabled:
                # Determine severity based on weapon type and confidence
                severity = "CRITICAL" if confidence > 0.85 else "HIGH"
                
                success = await telegram_service.send_detection_alert(
                    detection_type=f"{weapon_type} Detected",
                    severity=severity,
                    confidence=confidence,
                    location=location
                )
                
                if success:
                    # Update last alert time only on successful send
                    self.last_weapon_alert_time = current_time
                    logger.warning(f"⚠️  Telegram alert sent for weapon: {weapon_type} (Confidence: {confidence:.1%})")
                    
                    # Save alert to database
                    try:
                        telegram_record = {
                            "message_id": f"msg_{datetime.utcnow().timestamp()}",
                            "chat_id": telegram_service.chat_id,
                            "text": f"⚠️ {weapon_type} detected (Confidence: {confidence:.1%}) at {location}",
                            "message_type": "weapon_detection",
                            "detection_type": f"{weapon_type} Detection",
                            "severity": severity,
                            "weapon_type": weapon_type,
                            "location": location,
                            "confidence": float(confidence),
                            "status": "sent",
                            "timestamp": datetime.utcnow().isoformat()
                        }
                        await db.save_telegram_message(telegram_record)
                        logger.debug(f"Weapon alert saved to database")
                    except Exception as db_err:
                        logger.error(f"Failed to save weapon alert to database: {db_err}")
                else:
                    logger.warning(f"Failed to send telegram alert for {weapon_type}")
            else:
                logger.debug("Telegram not configured - skipping weapon alert")
        except Exception as e:
            logger.error(f"Failed to send weapon telegram alert: {e}")

threat_engine = ThreatEngine()
