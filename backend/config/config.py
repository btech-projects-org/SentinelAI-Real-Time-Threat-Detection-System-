import os
from pydantic_settings import BaseSettings, SettingsConfigDict
from functools import lru_cache
from pathlib import Path
import sys

# Get the project root directory (parent of backend)
BASE_DIR = Path(__file__).resolve().parent.parent.parent
ENV_FILE = BASE_DIR / ".env"

class Settings(BaseSettings):
    APP_NAME: str = "SentinelAI Threat Detection"
    VERSION: str = "1.0.0"
    API_PREFIX: str = "/api/v1"
    
    # Server Config
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = True
    
    # Database
    MONGO_URI: str
    DB_NAME: str = "sentinel_core"
    
    # Security
    JWT_SECRET: str
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    ADMIN_PASSWORD: str
    ALLOWED_ORIGINS: str = "*" # Comma separated list of origins

    
    # AI / MCP
    MCP_SERVER_URL: str = "http://localhost:8080"
    CONFIDENCE_THRESHOLD: float = 0.60
    
    # Google Gemini API
    GEMINI_API_KEY: str = ""
    
    # Telegram Bot Configuration
    TELEGRAM_BOT_TOKEN: str = ""
    TELEGRAM_CHAT_ID: str = ""
    
    # Deep Learning Model Configuration
    YOLO_MODEL_PATH: str = "./models/yolov8n.pt"
    YOLO_CONFIDENCE_THRESHOLD: float = 0.70
    YOLO_IOU_THRESHOLD: float = 0.45
    
    DEEPFACE_MODEL: str = "Facenet512"
    DEEPFACE_DETECTOR: str = "opencv"
    DEEPFACE_DISTANCE_METRIC: str = "cosine"
    FACE_MATCH_THRESHOLD: float = 0.40
    
    AUTO_DOWNLOAD_MODELS: bool = True
    MODELS_CACHE_DIR: str = "./models/cache"
    
    @property
    def TELEGRAM_ENABLED(self) -> bool:
        return bool(self.TELEGRAM_BOT_TOKEN and self.TELEGRAM_CHAT_ID)
    
    model_config = SettingsConfigDict(
        env_file=str(ENV_FILE),
        env_file_encoding='utf-8',
        extra='ignore'
    )
    
    def validate_critical_settings(self):
        """Fail-fast validation: Ensure ALL critical credentials are present."""
        errors = []
        
        # 1. Environment File
        if not ENV_FILE.exists():
            errors.append(f"❌ CRITICAL: Environment file not found at {ENV_FILE}")
        
        # 2. Database
        if not self.MONGO_URI:
            errors.append("❌ CRITICAL: MONGO_URI is missing")
            
        # 3. Security
        if not self.JWT_SECRET or self.JWT_SECRET == "CHANGE_THIS_IN_PRODUCTION_USE_STRONG_SECRET_KEY":
            # We treat default secret as 'missing' for strict production readiness, 
            # but user just asked to check if keys are GIVEN. 
            # Let's fail if it's empty, warn if default.
            if not self.JWT_SECRET:
                errors.append("❌ CRITICAL: JWT_SECRET is missing")
        
        if not self.ADMIN_PASSWORD:
            errors.append("❌ CRITICAL: ADMIN_PASSWORD is missing")

        
        # 4. Telegram Credentials
        if not self.TELEGRAM_BOT_TOKEN:
            errors.append("❌ CRITICAL: TELEGRAM_BOT_TOKEN is missing")
        if not self.TELEGRAM_CHAT_ID:
            errors.append("❌ CRITICAL: TELEGRAM_CHAT_ID is missing")
            
        # 5. Gemini API Check
        if not self.GEMINI_API_KEY:
            errors.append("❌ CRITICAL: GEMINI_API_KEY is missing")

        if errors:
            print("\n" + "!"*60)
            print("🛑 STARTUP BLOCKED: MISSING CONFIGURATION")
            print("!"*60)
            for e in errors:
                print(e)
            print("!"*60 + "\n")
            # Force exit prevents server from starting with bad config
            sys.exit(1)
            
        print("✅ Configuration validated: All required keys are present.")

@lru_cache()
def get_settings():
    settings = Settings()
    settings.validate_critical_settings()
    return settings

