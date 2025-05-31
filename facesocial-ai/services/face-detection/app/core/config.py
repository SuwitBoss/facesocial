"""
Face Detection Service Configuration - SECURE VERSION
"""

import os
from typing import List, Optional, Dict, Any
from enum import Enum
from functools import lru_cache
import secrets

try:
    from pydantic_settings import BaseSettings
    from pydantic import field_validator, SecretStr
except ImportError:
    from pydantic import BaseSettings, validator as field_validator, SecretStr


class DetectionMode(str, Enum):
    """Detection modes for different use cases"""
    REALTIME = "realtime"    # MediaPipe - fastest
    BALANCED = "balanced"    # InsightFace - balanced speed/accuracy  
    ACCURATE = "accurate"    # YOLO - highest accuracy


class DetectorType(str, Enum):
    """Available detector types"""
    YOLO = "yolo"
    INSIGHTFACE = "insightface"
    MEDIAPIPE = "mediapipe"


class Settings(BaseSettings):
    """Main service configuration with secure defaults"""
    
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
    
    # Database Configuration - From Environment Variables
    POSTGRES_USER: str
    POSTGRES_PASSWORD: SecretStr
    POSTGRES_DB: str = "facesocial"
    POSTGRES_HOST: str = "postgres"
    POSTGRES_PORT: int = 5432
    
    @property
    def POSTGRES_URL(self) -> str:
        """Build PostgreSQL URL from components"""
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:"
            f"{self.POSTGRES_PASSWORD.get_secret_value()}@"
            f"{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )
    
    # Redis Configuration
    REDIS_PASSWORD: SecretStr
    REDIS_HOST: str = "redis"
    REDIS_PORT: int = 6379
    REDIS_DB: int = 0
    
    @property
    def REDIS_URL(self) -> str:
        """Build Redis URL from components"""
        return (
            f"redis://:{self.REDIS_PASSWORD.get_secret_value()}@"
            f"{self.REDIS_HOST}:{self.REDIS_PORT}/{self.REDIS_DB}"
        )
    
    # JWT Configuration
    JWT_SECRET_KEY: SecretStr = SecretStr(secrets.token_urlsafe(32))
    JWT_ALGORITHM: str = "HS256"
    JWT_ACCESS_TOKEN_EXPIRE_MINUTES: int = 30    
    # Model Paths
    MODEL_BASE_PATH: str = os.getenv("MODEL_PATH", "./models")
    
    @property 
    def YOLO_MODEL_PATH(self) -> str:
        """Get YOLO model path with fallback locations"""
        model_filename = "yolov10n-face.onnx"
        
        # Check environment variable first
        env_path = os.getenv("YOLO_MODEL_PATH")
        if env_path and os.path.exists(env_path):
            return env_path
        
        # Check standard locations
        possible_paths = [
            os.path.join(self.MODEL_BASE_PATH, "face-detection", model_filename),
            os.path.join("/app/models/face-detection", model_filename),
            os.path.join("./models/face-detection", model_filename),
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                return path
        
        # Return default path
        return os.path.join(self.MODEL_BASE_PATH, "face-detection", model_filename)
    
    # GPU Configuration
    CUDA_VISIBLE_DEVICES: str = "0"
    VRAM_LIMIT_MB: int = 4800
    MAX_BATCH_SIZE: int = 4
    MODEL_CACHE_SIZE: int = 2
    CUDA_MEMORY_FRACTION: float = 0.8
    FORCE_CPU: bool = False  # Allow GPU by default
    
    # Detection Parameters
    DEFAULT_CONFIDENCE_THRESHOLD: float = 0.7
    DEFAULT_IOU_THRESHOLD: float = 0.45
    DEFAULT_MIN_FACE_SIZE: int = 40
    DEFAULT_MAX_FACES: int = 100
    
    # YOLO Configuration
    YOLO_INPUT_SIZE: tuple = (640, 640)
    YOLO_CONFIDENCE_THRESHOLD: float = 0.25
    YOLO_IOU_THRESHOLD: float = 0.45
    YOLO_WARMUP_RUNS: int = 3
    
    # InsightFace Configuration
    INSIGHTFACE_MODEL_NAME: str = "buffalo_l"
    INSIGHTFACE_DET_SIZE: tuple = (640, 640)
    INSIGHTFACE_CTX_ID: int = 0  # GPU by default, -1 for CPU
    INSIGHTFACE_CONFIDENCE_THRESHOLD: float = 0.7
    
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
    CACHE_TTL_SECONDS: int = 3600
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
        DetectorType.INSIGHTFACE, 
        DetectorType.MEDIAPIPE
    ]
    
    # Model Loading Strategy
    LAZY_LOADING: bool = True
    MODEL_UNLOAD_TIMEOUT: int = 300
    PRELOAD_MODELS: List[DetectorType] = [DetectorType.MEDIAPIPE]
    
    # Batch Processing
    BATCH_QUEUE_SIZE: int = 100
    BATCH_TIMEOUT_SECONDS: int = 60
    MAX_BATCH_ITEMS: int = 10
    
    class Config:
        env_file = ".env"
        case_sensitive = True
        
        # Prevent secrets from being logged
        json_encoders = {
            SecretStr: lambda v: v.get_secret_value() if v else None
        }


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance"""
    return Settings()


# Global settings instance
settings = get_settings()