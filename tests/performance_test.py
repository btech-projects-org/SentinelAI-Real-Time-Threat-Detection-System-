
import sys
import os
import time
import pytest
import numpy as np
import cv2
from fastapi.testclient import TestClient

# Add project root to sys.path
sys.path.append(os.getcwd())

from backend.main import app
from backend.config.config import get_settings

client = TestClient(app)
settings = get_settings()

def get_auth_token():
    response = client.post("/api/v1/token", data={"username": "admin", "password": settings.ADMIN_PASSWORD})
    if response.status_code != 200:
        raise Exception(f"Auth failed: {response.text}")
    return response.json()["access_token"]

def test_detection_latency():
    """Measure latency of detection endpoint."""
    token = get_auth_token()
    headers = {"Authorization": f"Bearer {token}"}
    
    # Generate dummy image
    img = np.zeros((480, 640, 3), dtype=np.uint8)
    _, img_encoded = cv2.imencode('.jpg', img)
    img_bytes = img_encoded.tobytes()
    
    timings = []
    
    print("\nStarting Performance Test (5 iterations)...")
    for i in range(5):
        start_time = time.time()
        files = {"file": ("test.jpg", img_bytes, "image/jpeg")}
        
        response = client.post("/api/v1/detect-frame", files=files, headers=headers)
        
        end_time = time.time()
        duration = end_time - start_time
        timings.append(duration)
        
        assert response.status_code == 200, f"Request failed: {response.text}"
        data = response.json()
        assert data["status"] == "success"
        
        print(f"  Iteration {i+1}: {duration:.4f}s")
    
    avg_latency = sum(timings) / len(timings)
    max_latency = max(timings)
    print(f"Average Latency: {avg_latency:.4f}s")
    print(f"Max Latency: {max_latency:.4f}s")
    
    # Assert performance criteria
    assert max_latency < 3.0, f"Max latency {max_latency:.4f}s exceeded 3s threshold"
    assert avg_latency < 1.0, f"Average latency {avg_latency:.4f}s exceeded 1s target"  # Stricter target for average

if __name__ == "__main__":
    try:
        test_detection_latency()
        print("✅ Performance Test Passed")
    except AssertionError as e:
        print(f"❌ Performance Test Failed: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
