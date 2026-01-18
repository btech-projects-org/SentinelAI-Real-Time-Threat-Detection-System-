
import sys
import os
import pytest
from fastapi.testclient import TestClient
from pathlib import Path

# Add project root to sys.path
sys.path.append(os.getcwd())

from backend.main import app
from backend.config.config import get_settings

client = TestClient(app)
settings = get_settings()

def test_public_endpoints():
    """Verify that public endpoints are accessible."""
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"

    response = client.get("/api/v1/telegram/status")
    assert response.status_code == 200

def test_protected_endpoints_no_auth():
    """Verify that protected endpoints reject unauthenticated access."""
    # Criminals Register
    response = client.post("/api/v1/criminals/register", data={"criminal_id": "test", "name": "test"})
    assert response.status_code == 401
    assert response.json()["detail"] == "Not authenticated"

    # Criminals Reload
    response = client.post("/api/v1/criminals/reload")
    assert response.status_code == 401
    
    # Detect Frame (Now protected)
    response = client.post("/api/v1/detect-frame")
    # This might be 401 or 422 depending on if it checks auth first or validation first.
    # FastAPI usually does dependencies (auth) first.
    assert response.status_code == 401

def test_login_flow():
    """Verify full login flow and authenticated access."""
    # 1. Login with wrong password
    response = client.post("/api/v1/token", data={"username": "admin", "password": "wrongpassword"})
    assert response.status_code == 401
    
    # 2. Login with correct password
    response = client.post("/api/v1/token", data={"username": "admin", "password": settings.ADMIN_PASSWORD})
    assert response.status_code == 200
    token_data = response.json()
    assert "access_token" in token_data
    assert token_data["token_type"] == "bearer"
    
    token = token_data["access_token"]
    headers = {"Authorization": f"Bearer {token}"}
    
    # 3. Access protected endpoint with token
    response = client.post("/api/v1/criminals/reload", headers=headers)
    assert response.status_code == 200
    assert response.json()["status"] == "success"

def test_cors_headers():
    """Verify CORS headers are set correctly."""
    origin = "http://localhost:5173"
    response = client.options("/health", headers={"Origin": origin, "Access-Control-Request-Method": "GET"})
    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == origin

if __name__ == "__main__":
    # specific run for debugging
    test_public_endpoints()
    test_protected_endpoints_no_auth()
    test_login_flow()
    test_cors_headers()
    print("✅ All Security Tests Passed")
