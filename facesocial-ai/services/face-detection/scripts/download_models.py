#!/usr/bin/env python3
"""
Download and verify AI models for Face Detection Service
Downloads YOLOv10n-face model and verifies integrity
"""

import os
import sys
import requests
import hashlib
import logging
from pathlib import Path
from typing import Dict, Optional
import onnxruntime as ort


# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Model configurations
MODELS_CONFIG = {
    "yolov10n-face": {
        "filename": "yolov10n-face.onnx",
        "url": "https://github.com/jameslahm/yolov10/releases/download/v1.1/yolov10n.pt",
        "alt_url": "https://huggingface.co/jameslahm/yolov10n/resolve/main/yolov10n.pt",
        "onnx_url": "https://github.com/THU-MIG/yolov10/releases/download/v1.1/yolov10n.onnx",
        "expected_size_mb": 5.2,
        "sha256": None,  # Will be calculated after download
        "description": "YOLOv10n face detection model"
    }
}


class ModelDownloader:
    """Downloads and manages AI models"""
    
    def __init__(self, models_dir: str = "/app/models"):
        self.models_dir = Path(models_dir)
        self.face_detection_dir = self.models_dir / "face-detection"
        self.face_detection_dir.mkdir(parents=True, exist_ok=True)
    
    def download_file(self, url: str, destination: Path, chunk_size: int = 8192) -> bool:
        """Download file with progress tracking"""
        try:
            logger.info(f"Downloading from: {url}")
            logger.info(f"Destination: {destination}")
            
            response = requests.get(url, stream=True, timeout=30)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            downloaded = 0
            
            with open(destination, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        if total_size > 0:
                            progress = (downloaded / total_size) * 100
                            print(f"\rProgress: {progress:.1f}% ({downloaded}/{total_size} bytes)", end='')
            
            print()  # New line after progress
            logger.info(f"Download completed: {destination}")
            return True
            
        except Exception as e:
            logger.error(f"Download failed: {e}")
            if destination.exists():
                destination.unlink()  # Remove partial file
            return False
    
    def verify_onnx_model(self, model_path: Path) -> bool:
        """Verify ONNX model integrity"""
        try:
            logger.info(f"Verifying ONNX model: {model_path}")
            
            # Check file size
            file_size_mb = model_path.stat().st_size / (1024 * 1024)
            logger.info(f"File size: {file_size_mb:.2f} MB")
            
            if file_size_mb < 1.0:
                logger.error("Model file too small, likely corrupted")
                return False
            
            # Try to load with ONNX Runtime
            session = ort.InferenceSession(str(model_path), providers=['CPUExecutionProvider'])
            
            # Check input/output shapes
            inputs = session.get_inputs()
            outputs = session.get_outputs()
            
            logger.info(f"Model inputs: {len(inputs)}")
            for i, input_info in enumerate(inputs):
                logger.info(f"  Input {i}: {input_info.name} {input_info.shape} {input_info.type}")
            
            logger.info(f"Model outputs: {len(outputs)}")
            for i, output_info in enumerate(outputs):
                logger.info(f"  Output {i}: {output_info.name} {output_info.shape} {output_info.type}")
            
            # Basic validation for YOLOv10n-face
            if len(inputs) != 1:
                logger.error("Expected 1 input for YOLOv10n model")
                return False
            
            input_shape = inputs[0].shape
            if len(input_shape) != 4 or input_shape[1] != 3:
                logger.error(f"Expected input shape [N,3,H,W], got {input_shape}")
                return False
            
            logger.info("✅ Model verification successful")
            return True
            
        except Exception as e:
            logger.error(f"Model verification failed: {e}")
            return False
    
    def calculate_sha256(self, file_path: Path) -> str:
        """Calculate SHA256 hash of file"""
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()
    
    def download_yolo_model(self) -> bool:
        """Download YOLOv10n-face model"""
        model_config = MODELS_CONFIG["yolov10n-face"]
        model_path = self.face_detection_dir / model_config["filename"]
        
        # Check if model already exists and is valid
        if model_path.exists():
            logger.info(f"Model already exists: {model_path}")
            if self.verify_onnx_model(model_path):
                logger.info("✅ Existing model is valid")
                return True
            else:
                logger.warning("Existing model is invalid, re-downloading...")
                model_path.unlink()
        
        # Try to download from different sources
        urls_to_try = [
            model_config.get("onnx_url"),
            model_config.get("url"),
            model_config.get("alt_url")
        ]
        
        for url in urls_to_try:
            if not url:
                continue
                
            logger.info(f"Attempting download from: {url}")
            
            # Download to temporary file first
            temp_path = model_path.with_suffix('.tmp')
            
            if self.download_file(url, temp_path):
                # Verify downloaded model
                if self.verify_onnx_model(temp_path):
                    # Move to final location
                    temp_path.rename(model_path)
                    
                    # Calculate and log hash
                    file_hash = self.calculate_sha256(model_path)
                    logger.info(f"SHA256: {file_hash}")
                    
                    logger.info("✅ YOLOv10n-face model downloaded and verified successfully")
                    return True
                else:
                    logger.warning("Downloaded model failed verification")
                    temp_path.unlink()
            
        logger.error("❌ Failed to download YOLOv10n-face model from all sources")
        return False
    
    def check_existing_model(self) -> bool:
        """Check if YOLO model exists at expected location"""
        model_path = self.face_detection_dir / "yolov10n-face.onnx"
        
        if not model_path.exists():
            logger.info(f"Model not found at: {model_path}")
            return False
        
        logger.info(f"Found existing model: {model_path}")
        
        # Verify the existing model
        if self.verify_onnx_model(model_path):
            logger.info("✅ Existing model is valid and ready to use")
            return True
        else:
            logger.warning("❌ Existing model failed verification")
            return False
    
    def setup_all_models(self) -> bool:
        """Setup all required models"""
        logger.info("🚀 Setting up Face Detection models...")
        
        success = True
        
        # Check if model already exists and is valid
        if self.check_existing_model():
            logger.info("All models are ready!")
            return True
        
        # Download YOLOv10n-face model
        logger.info("📥 Downloading YOLOv10n-face model...")
        if not self.download_yolo_model():
            logger.error("Failed to download YOLOv10n-face model")
            success = False
        
        # Create model config file
        self.create_model_config()
        
        if success:
            logger.info("✅ All models setup completed successfully!")
        else:
            logger.error("❌ Model setup failed!")
        
        return success
    
    def create_model_config(self):
        """Create model configuration file"""
        config_content = """# Face Detection Models Configuration
# Generated automatically by download_models.py

models:
  yolo:
    path: "/app/models/face-detection/yolov10n-face.onnx"
    type: "face_detection"
    input_size: [640, 640]
    confidence_threshold: 0.25
    iou_threshold: 0.45
    provider: "CUDAExecutionProvider"
    fallback_provider: "CPUExecutionProvider"
  
  mtcnn:
    type: "face_detection"
    min_face_size: 20
    scale_factor: 0.709
    steps_threshold: [0.6, 0.7, 0.7]
    provider: "CPUExecutionProvider"
  
  mediapipe:
    type: "face_detection"
    model_selection: 0
    min_detection_confidence: 0.5
    provider: "CPUExecutionProvider"

# Performance settings
performance:
  gpu_memory_limit_mb: 4800
  max_batch_size: 4
  warmup_runs: 3
"""
        
        config_path = self.models_dir / "model_config.yaml"
        with open(config_path, 'w') as f:
            f.write(config_content)
        
        logger.info(f"Created model config: {config_path}")


def main():
    """Main function"""
    logger.info("🔧 Face Detection Model Setup")
    logger.info("=" * 50)
    
    # Parse command line arguments
    import argparse
    parser = argparse.ArgumentParser(description="Download face detection models")
    parser.add_argument("--models-dir", default="/app/models", 
                       help="Directory to store models")
    parser.add_argument("--force", action="store_true",
                       help="Force re-download even if models exist")
    parser.add_argument("--verify-only", action="store_true",
                       help="Only verify existing models, don't download")
    
    args = parser.parse_args()
    
    # Create downloader
    downloader = ModelDownloader(args.models_dir)
    
    if args.verify_only:
        # Only verify existing models
        success = downloader.check_existing_model()
    elif args.force:
        # Force re-download
        model_path = downloader.face_detection_dir / "yolov10n-face.onnx"
        if model_path.exists():
            model_path.unlink()
        success = downloader.setup_all_models()
    else:
        # Normal setup (download if needed)
        success = downloader.setup_all_models()
    
    if success:
        logger.info("🎉 Model setup completed successfully!")
        sys.exit(0)
    else:
        logger.error("💥 Model setup failed!")
        sys.exit(1)


if __name__ == "__main__":
    main()