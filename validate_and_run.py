
import os
import sys
import uvicorn
from dotenv import load_dotenv

# Add current directory to path so imports work
sys.path.append(os.getcwd())

def main():
    print("\n" + "="*60)
    print("🚀 SENTINELAI STARTUP SEQUENCE")
    print("="*60)
    
    # 1. Load Environment Variables
    env_path = os.path.join(os.getcwd(), ".env")
    if os.path.exists(env_path):
        print(f"📂 Loading environment from: {env_path}")
        load_dotenv(env_path)
    else:
        print(f"❌ ERROR: .env file not found at {env_path}")
        print("   Please create one based on .env.example")
        sys.exit(1)

    # 2. Validate Configuration
    print("🔐 Validating security credentials...")
    try:
        from backend.config.config import get_settings
        # This triggers the strict validation logic we added to config.py
        settings = get_settings()
        # Explicit call to be double sure, though get_settings() calls it via lru_cache
        settings.validate_critical_settings()
        
    except Exception as e:
        print(f"\n❌ Configuration Validation Failed: {e}")
        sys.exit(1)
        
    # 3. Start Server
    print("\n✅ All systems go. Starting backend server...")
    print("="*60 + "\n")
    
    uvicorn.run(
        "backend.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )

if __name__ == "__main__":
    main()
