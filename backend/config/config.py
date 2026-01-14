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
        """Fail-fast validation for production-critical settings (strict in production, relaxed in DEBUG)."""
        errors = []
        warnings = []
        
        # Check environment file exists
        if not ENV_FILE.exists():
            errors.append(f"❌ CRITICAL: Environment file not found at {ENV_FILE}")
        
        # Validate database configuration
        if not getattr(self, 'MONGO_URI', None):
            errors.append("❌ CRITICAL: MONGO_URI is not configured in .env")
        
        # Validate JWT secret is not default
        if self.JWT_SECRET == "CHANGE_THIS_IN_PRODUCTION_USE_STRONG_SECRET_KEY":
            warnings.append("⚠️  WARNING: JWT_SECRET is using default value - change in production")
        
        # Validate JWT secret strength
        if len(self.JWT_SECRET) < 32:
            errors.append("❌ CRITICAL: JWT_SECRET must be at least 32 characters long")
        
        if errors or warnings:
            print("\n" + "="*60)
            print("SENTINELAI - CONFIGURATION VALIDATION")
            print("="*60)
            for w in warnings:
                print(w)
            for e in errors:
                print(e)
            print("="*60)
            if errors and not self.DEBUG:
                print("\nApplication cannot start. Fix the errors above and restart.")
                print("="*60 + "\n")
                sys.exit(1)
            else:
                print("\nStarting in DEBUG mode with relaxed validation.")
                print("="*60 + "\n")

@lru_cache()
def get_settings():
    settings = Settings()
    settings.validate_critical_settings()
    return settings

