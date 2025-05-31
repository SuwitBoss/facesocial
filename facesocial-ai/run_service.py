#!/usr/bin/env python3
"""
Face Detection Service Runner
Easy script to start the service for development
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

def check_requirements():
    """Check if required packages are installed"""
    required_packages = [
        "fastapi", "uvicorn", "opencv-python", "numpy", 
        "pydantic", "psutil", "pillow"
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package.replace("-", "_"))
        except ImportError:
            missing.append(package)
    
    if missing:
        print(f"❌ Missing packages: {', '.join(missing)}")
        print("Install with: pip install -r docker/requirements.txt")
        return False
    
    print("✅ All required packages are installed")
    return True


def check_model():
    """Check if YOLO model exists"""
    model_paths = [
        "facesocial-ai/models/face-detection/yolov10n-face.onnx",
        "./models/face-detection/yolov10n-face.onnx",
        "../models/face-detection/yolov10n-face.onnx"
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            print(f"✅ YOLO model found: {path}")
            return True
    
    print("⚠️  YOLO model not found. Service will work with MTCNN/MediaPipe only.")
    print("To download YOLO model, run: python scripts/download_models.py")
    return False


def run_tests():
    """Run test suite"""
    print("🧪 Running test suite...")
    
    # Change to service directory
    service_dir = Path(__file__).parent / "services" / "face-detection"
    if service_dir.exists():
        os.chdir(service_dir)
    
    try:
        result = subprocess.run([
            sys.executable, "scripts/test_service.py"
        ], check=False)
        return result.returncode == 0
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False


def start_service(host="0.0.0.0", port=8000, reload=True):
    """Start the FastAPI service"""
    print(f"🚀 Starting Face Detection Service on {host}:{port}")
    
    # Change to service directory
    service_dir = Path(__file__).parent / "services" / "face-detection"
    if service_dir.exists():
        os.chdir(service_dir)
    elif Path("app").exists():
        # Already in service directory
        pass
    else:
        print("❌ Cannot find service directory")
        return False
    
    cmd = [
        sys.executable, "-m", "uvicorn", 
        "app.main:app",
        "--host", host,
        "--port", str(port)
    ]
    
    if reload:
        cmd.append("--reload")
    
    try:
        subprocess.run(cmd, check=True)
        return True
    except KeyboardInterrupt:
        print("\n🛑 Service stopped by user")
        return True
    except Exception as e:
        print(f"❌ Failed to start service: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Face Detection Service Runner")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--no-reload", action="store_true", help="Disable auto-reload")
    parser.add_argument("--test", action="store_true", help="Run tests only")
    parser.add_argument("--skip-checks", action="store_true", help="Skip requirement checks")
    
    args = parser.parse_args()
    
    print("🤖 Face Detection Service")
    print("=" * 40)
    
    if not args.skip_checks:
        # Check requirements
        if not check_requirements():
            return 1
        
        # Check model
        check_model()
    
    if args.test:
        # Run tests only
        success = run_tests()
        return 0 if success else 1
    
    # Start service
    success = start_service(
        host=args.host,
        port=args.port,
        reload=not args.no_reload
    )
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())