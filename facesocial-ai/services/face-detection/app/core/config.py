# Service configuration
"""
Face Detection Service Configuration
Manages settings for all 3 detection models and GPU resources
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
    
    # Model Paths
    MODEL_BASE_PATH: str = "/app/models"
    YOLO_MODEL_PATH: str = "/app/models/yolov10n-face.onnx"
    CONFIG_PATH: str = "/app/config"
    
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
    LOG_FILE: Optional[str] = "/app/logs/face-detection.log"
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
    
    class Config:
        env_file = ".env"
        case_sensitive = True


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


class MTCNNConfig:
    """MTCNN-specific configuration"""
    
    def __init__(self, settings: Settings):
        self.min_face_size = settings.MTCNN_MIN_FACE_SIZE
        self.scale_factor = settings.MTCNN_SCALE_FACTOR
        self.steps_threshold = settings.MTCNN_STEPS_THRESHOLD
        self.device = 'cpu'  # MTCNN uses CPU only


class MediaPipeConfig:
    """MediaPipe-specific configuration"""
    
    def __init__(self, settings: Settings):
        self.confidence_threshold = settings.MEDIAPIPE_CONFIDENCE_THRESHOLD
        self.min_detection_confidence = settings.MEDIAPIPE_MIN_DETECTION_CONFIDENCE
        self.min_tracking_confidence = settings.MEDIAPIPE_MIN_TRACKING_CONFIDENCE
        self.max_num_faces = settings.DEFAULT_MAX_FACES


class DetectionConfig:
    """Combined detection configuration for all models"""
    
    def __init__(self, settings: Settings):
        self.settings = settings
        self.yolo = YOLOConfig(settings)
        self.mtcnn = MTCNNConfig(settings)
        self.mediapipe = MediaPipeConfig(settings)
        
    def get_config_for_mode(self, mode: DetectionMode) -> Dict[str, Any]:
        """Get configuration for specific detection mode"""
        mode_configs = {
            DetectionMode.REALTIME: {
                "detector": DetectorType.MEDIAPIPE,
                "confidence_threshold": self.mediapipe.confidence_threshold,
                "max_faces": min(10, self.settings.DEFAULT_MAX_FACES),
                "enable_quality_assessment": False,
                "enable_landmarks": False
            },
            DetectionMode.BALANCED: {
                "detector": DetectorType.MTCNN,
                "confidence_threshold": 0.7,
                "max_faces": self.settings.DEFAULT_MAX_FACES,
                "enable_quality_assessment": True,
                "enable_landmarks": True
            },
            DetectionMode.ACCURATE: {
                "detector": DetectorType.YOLO,
                "confidence_threshold": self.yolo.confidence_threshold,
                "max_faces": self.settings.DEFAULT_MAX_FACES,
                "enable_quality_assessment": True,
                "enable_landmarks": True
            }
        }
        return mode_configs.get(mode, mode_configs[DetectionMode.BALANCED])


# Global settings instance
settings = Settings()

# Detection configuration
detection_config = DetectionConfig(settings)

# Export commonly used configs
__all__ = [
    "settings",
    "detection_config", 
    "DetectionMode",
    "DetectorType",
    "YOLOConfig",
    "MTCNNConfig", 
    "MediaPipeConfig"
]