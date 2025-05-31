# app/core/config.py - Fixed version
"""
Face Detection Service Configuration - FIXED
"""

import os
from typing import List, Optional, Dict, Any
from pydantic import BaseSettings, validator
from enum import Enum


class DetectionMode(str, Enum):
    """Detection modes for different use cases"""
    REALTIME = "realtime"    # MediaPipe - fastest
    BALANCED = "balanced"    # MTCNN - balanced speed/accuracy  
    ACCURATE = "accurate"    # YOLO - highest accuracy


class DetectorType(str, Enum):
    """Available detector types"""
    YOLO = "yolo"
    MTCNN = "mtcnn"
    MEDIAPIPE = "mediapipe"


class Settings(BaseSettings):
    """Main service configuration"""
    
    # Service Information
    SERVICE_NAME: str = "face-detection-service"
    VERSION: str = "1.0.0"
    DEBUG: bool = False
    ENVIRONMENT: str = "development"
    
    # API Configuration
    API_V1_STR: str = "/api/v1"
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    WORKERS: int = 1
    
    # Database Configuration
    POSTGRES_URL: str = "postgresql+asyncpg://facesocial_user:facesocial_2024@postgres:5432/facesocial"
    REDIS_URL: str = "redis://:redis_2024@redis:6379/0"
    
    # Model Paths - FIXED PATHS
    MODEL_BASE_PATH: str = "./models"  # ← Changed from /app/models
    
    @property 
    def YOLO_MODEL_PATH(self) -> str:
        """Dynamic YOLO model path"""
        # Check multiple possible locations
        possible_paths = [
            "./models/face-detection/yolov10n-face.onnx",           # Local development
            "/app/models/face-detection/yolov10n-face.onnx",       # Docker container
            "../models/face-detection/yolov10n-face.onnx",        # Relative path
            "../../models/face-detection/yolov10n-face.onnx"      # Alternative relative
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Fallback to default
        return "./models/face-detection/yolov10n-face.onnx"
    
    CONFIG_PATH: str = "./config"
    
    # GPU Configuration
    CUDA_VISIBLE_DEVICES: str = "0"
    VRAM_LIMIT_MB: int = 4800  # 6GB * 0.8
    MAX_BATCH_SIZE: int = 4
    MODEL_CACHE_SIZE: int = 2
    CUDA_MEMORY_FRACTION: float = 0.8
    
    # Detection Parameters - Default values
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7
    DEFAULT_IOU_THRESHOLD: float = 0.45
    DEFAULT_MIN_FACE_SIZE: int = 40
    DEFAULT_MAX_FACES: int = 100
    
    # YOLO Configuration
    YOLO_INPUT_SIZE: tuple = (640, 640)
    YOLO_CONFIDENCE_THRESHOLD: float = 0.25
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_WARMUP_RUNS: int = 3
    
    # MTCNN Configuration
    MTCNN_MIN_FACE_SIZE: int = 20
    MTCNN_SCALE_FACTOR: float = 0.709
    MTCNN_STEPS_THRESHOLD: List[float] = [0.6, 0.7, 0.7]
    
    # MediaPipe Configuration
    MEDIAPIPE_CONFIDENCE_THRESHOLD: float = 0.5
    MEDIAPIPE_MIN_DETECTION_CONFIDENCE: float = 0.5
    MEDIAPIPE_MIN_TRACKING_CONFIDENCE: float = 0.5
    
    # Performance Settings
    MAX_IMAGE_SIZE_MB: int = 10
    MAX_CONCURRENT_REQUESTS: int = 10
    REQUEST_TIMEOUT_SECONDS: int = 30
    
    # Quality Assessment
    MIN_FACE_QUALITY_SCORE: float = 0.3
    ENABLE_QUALITY_ASSESSMENT: bool = True
    QUALITY_METRICS: List[str] = ["sharpness", "brightness", "size"]
    
    # Caching
    CACHE_TTL_SECONDS: int = 3600  # 1 hour
    ENABLE_RESULT_CACHING: bool = True
    CACHE_MAX_SIZE: int = 1000
    
    # Logging
    LOG_LEVEL: str = "INFO"
    LOG_FORMAT: str = "json"
    LOG_FILE: Optional[str] = "./logs/face-detection.log"
    ENABLE_ACCESS_LOG: bool = True
    
    # Monitoring
    ENABLE_METRICS: bool = True
    METRICS_PORT: int = 9090
    HEALTH_CHECK_INTERVAL: int = 30
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 100
    RATE_LIMIT_PER_HOUR: int = 1000
    
    # Auto-detection Strategy
    AUTO_FALLBACK_ENABLED: bool = True
    FALLBACK_ORDER: List[DetectorType] = [
        DetectorType.YOLO,
        DetectorType.MTCNN, 
        DetectorType.MEDIAPIPE
    ]
    
    # Model Loading Strategy
    LAZY_LOADING: bool = True  # Load models only when needed
    MODEL_UNLOAD_TIMEOUT: int = 300  # Unload after 5 minutes of inactivity
    PRELOAD_MODELS: List[DetectorType] = [DetectorType.MEDIAPIPE]  # Always keep loaded
    
    # Batch Processing
    BATCH_QUEUE_SIZE: int = 100
    BATCH_TIMEOUT_SECONDS: int = 60
    MAX_BATCH_ITEMS: int = 10
    
    # Model validation
    def validate_model_paths(self) -> bool:
        """Validate that required models exist"""
        yolo_path = self.YOLO_MODEL_PATH
        
        if not os.path.exists(yolo_path):
            print(f"⚠️  YOLO model not found at: {yolo_path}")
            print(f"📍 Please ensure model exists or run: python scripts/download_models.py")
            return False
            
        print(f"✅ YOLO model found: {yolo_path}")
        return True
    
    class Config:
        env_file = ".env"
        case_sensitive = True


# Rest of the config classes remain the same...
class YOLOConfig:
    """YOLO-specific configuration"""
    
    def __init__(self, settings: Settings):
        self.model_path = settings.YOLO_MODEL_PATH
        self.input_size = settings.YOLO_INPUT_SIZE
        self.confidence_threshold = settings.YOLO_CONFIDENCE_THRESHOLD
        self.iou_threshold = settings.YOLO_IOU_THRESHOLD
        self.warmup_runs = settings.YOLO_WARMUP_RUNS
        
        # GPU settings
        self.gpu_memory_limit = settings.VRAM_LIMIT_MB * 1024 * 1024
        self.device_id = 0
        
        # ONNX Runtime settings
        self.ort_providers = [
            ('CUDAExecutionProvider', {
                'device_id': self.device_id,
                'arena_extend_strategy': 'kNextPowerOfTwo',
                'gpu_mem_limit': self.gpu_memory_limit,
                'cudnn_conv_algo_search': 'HEURISTIC',
                'enable_cuda_graph': False,
                'tunable_op_enable': True,
            }),
            'CPUExecutionProvider'
        ]


# Global settings instance
settings = Settings()

# Validate model paths on startup
if not settings.validate_model_paths():
    print("⚠️  Model validation failed - some features may not work")

# Detection configuration
detection_config = DetectionConfig(settings) if 'DetectionConfig' in globals() else None

# Export commonly used configs
__all__ = [
    "settings",
    "detection_config", 
    "DetectionMode",
    "DetectorType",
    "YOLOConfig"
]