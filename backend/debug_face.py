import sys
import os
import asyncio
from pathlib import Path
import numpy as np
import cv2
import logging

# Setup paths
sys.path.append(str(Path(__file__).resolve().parent.parent))

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("debug_face")

async def test_face_detection_pipeline():
    print("🔍 Starting Face Detection Debug...")
    
    # 1. Test Imports
    try:
        from backend.config.config import get_settings
        from backend.database.mongodb import db
        from backend.services.dl_threat_engine import dl_threat_engine
        print("✅ Imports successful")
        
        settings = get_settings()
        print(f"   DeepFace Model: {settings.DEEPFACE_MODEL}")
        print(f"   Backend: {settings.DEEPFACE_DETECTOR}")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        return

    # 2. Test DeepFace Availability
    try:
        from deepface import DeepFace
        print(f"✅ DeepFace version: {DeepFace.__version__}")
    except ImportError as e:
        print(f"❌ DeepFace import failed: {e}")
        return
    except Exception as e:
        print(f"❌ DeepFace import crashed: {e}")
        return
    
    # 3. Connect to Database
    try:
        await db.connect()
        print("✅ Connected to MongoDB")
    except Exception as e:
        print(f"❌ Database connection failed: {e}")
        return

    # 4. Load Criminals
    print("\nLoad Criminals from DB:")
    try:
        criminals = await db.get_all_criminals()
        print(f"   Found {len(criminals)} criminals in DB")
        
        # 5. Try generating embedding for the first criminal
        if len(criminals) > 0:
            target = criminals[0]
            print(f"   Testing embedding for: {target.get('name')} (ID: {target.get('criminal_id')})")
            
            import base64
            img_data = base64.b64decode(target.get('image_base64'))
            nparr = np.frombuffer(img_data, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if img is not None:
                print(f"   Image decoded: {img.shape}")
                
                try:
                    embedding = DeepFace.represent(
                        img_path=img,
                        model_name=settings.DEEPFACE_MODEL,
                        detector_backend=settings.DEEPFACE_DETECTOR,
                        enforce_detection=False
                    )
                    
                    if embedding:
                        print("   ✅ DeepFace Embedding Generated Successfully!")
                        print(f"   Embedding vector length: {len(embedding[0]['embedding'])}")
                    else:
                        print("   ❌ Embedding generation returned empty")
                        
                except Exception as e:
                    print(f"   ❌ DeepFace Embedding Failed: {e}")
                    import traceback
                    traceback.print_exc()
            else:
                print("   ❌ Failed to decode base64 image")
    except Exception as e:
        print(f"❌ Error during debug checks: {e}")
        import traceback
        traceback.print_exc()
    
    await db.close()

if __name__ == "__main__":
    asyncio.run(test_face_detection_pipeline())
