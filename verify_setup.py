"""
SentinelAI Deep Learning Setup Verification
Checks if all deep learning dependencies are installed correctly
"""

import sys
import importlib

def check_import(module_name, package_name=None):
    """Check if a module can be imported."""
    try:
        importlib.import_module(module_name)
        print(f"✅ {package_name or module_name} - Installed")
        return True
    except ImportError:
        print(f"❌ {package_name or module_name} - NOT INSTALLED")
        return False

def check_torch():
    """Check PyTorch installation and CUDA availability."""
    try:
        import torch
        print(f"✅ PyTorch {torch.__version__} - Installed")
        
        if torch.cuda.is_available():
            print(f"   🚀 CUDA Available: {torch.cuda.get_device_name(0)}")
            print(f"   🚀 CUDA Version: {torch.version.cuda}")
        else:
            print(f"   ℹ️  CUDA: Not available (CPU only)")
        
        return True
    except ImportError:
        print(f"❌ PyTorch - NOT INSTALLED")
        return False

def check_yolo():
    """Check YOLOv8 installation."""
    try:
        from ultralytics import YOLO
        print(f"✅ YOLOv8 (Ultralytics) - Installed")
        
        # Try to load model (will download if not exists)
        try:
            model = YOLO('yolov8n.pt')
            print(f"   ℹ️  YOLOv8n model ready")
        except Exception as e:
            print(f"   ⚠️  YOLOv8n model will download on first run")
        
        return True
    except ImportError:
        print(f"❌ YOLOv8 (Ultralytics) - NOT INSTALLED")
        return False

def check_deepface():
    """Check DeepFace installation."""
    try:
        from deepface import DeepFace
        print(f"✅ DeepFace - Installed")
        
        # Check if models are available
        try:
            from deepface.basemodels import Facenet512
            print(f"   ℹ️  Facenet512 model available")
        except:
            print(f"   ⚠️  Facenet512 will download on first run")
        
        return True
    except ImportError:
        print(f"❌ DeepFace - NOT INSTALLED")
        return False

def main():
    print("="*60)
    print("SentinelAI Deep Learning Setup Verification")
    print("="*60)
    print()
    
    results = []
    
    # Core dependencies
    print("📦 Core Dependencies:")
    results.append(check_import('fastapi', 'FastAPI'))
    results.append(check_import('motor', 'Motor (MongoDB)'))
    results.append(check_import('cv2', 'OpenCV'))
    print()
    
    # Deep Learning
    print("🧠 Deep Learning Frameworks:")
    results.append(check_torch())
    results.append(check_import('torchvision', 'TorchVision'))
    results.append(check_yolo())
    results.append(check_deepface())
    results.append(check_import('tensorflow', 'TensorFlow'))
    print()
    
    # Optional
    print("🔧 Optional Dependencies:")
    check_import('google.generativeai', 'Google Gemini API')
    check_import('telegram', 'Python Telegram Bot')
    print()
    
    # Summary
    print("="*60)
    total = len(results)
    passed = sum(results)
    
    if passed == total:
        print(f"✅ ALL CHECKS PASSED ({passed}/{total})")
        print()
        print("🚀 You can now start the server with:")
        print("   cd backend")
        print("   python main.py")
        print()
        print("📖 See DEEP_LEARNING_SETUP.md for full documentation")
    else:
        failed = total - passed
        print(f"⚠️  SOME CHECKS FAILED ({failed}/{total} failed)")
        print()
        print("📦 Install missing dependencies:")
        print("   pip install -r requirements.txt")
        print()
        print("Or install individually:")
        if not check_import('torch', test_only=True):
            print("   pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu")
        if not check_import('ultralytics', test_only=True):
            print("   pip install ultralytics")
        if not check_import('deepface', test_only=True):
            print("   pip install deepface")
        if not check_import('tensorflow', test_only=True):
            print("   pip install tensorflow")
    
    print("="*60)

def check_import(module_name, package_name=None, test_only=False):
    """Check if module exists without printing (for test_only mode)."""
    if test_only:
        try:
            importlib.import_module(module_name)
            return True
        except ImportError:
            return False
    else:
        return check_import(module_name, package_name)

if __name__ == "__main__":
    main()
