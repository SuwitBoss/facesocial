"""
Input validation utilities
"""

import re
from typing import Any, Dict, List, Optional
from pydantic import BaseModel, validator
import numpy as np


def validate_confidence_threshold(confidence: float) -> bool:
    """Validate confidence threshold"""
    return 0.0 <= confidence <= 1.0


def validate_face_count(count: int) -> bool:
    """Validate face count parameter"""
    return 1 <= count <= 100


def validate_face_size(size: int) -> bool:
    """Validate minimum face size"""
    return 10 <= size <= 500


def validate_image_array(image: np.ndarray) -> Dict[str, Any]:
    """Validate numpy image array"""
    result = {
        "valid": False,
        "error": None,
        "info": {}
    }
    
    try:
        if image is None:
            result["error"] = "Image is None"
            return result
        
        if not isinstance(image, np.ndarray):
            result["error"] = "Input is not a numpy array"
            return result
        
        if image.size == 0:
            result["error"] = "Image array is empty"
            return result
        
        # Check dimensions
        if len(image.shape) != 3:
            result["error"] = f"Expected 3D array, got {len(image.shape)}D"
            return result
        
        height, width, channels = image.shape
        
        if channels != 3:
            result["error"] = f"Expected 3 channels (RGB/BGR), got {channels}"
            return result
        
        # Check reasonable image size
        if height < 32 or width < 32:
            result["error"] = f"Image too small: {width}x{height}"
            return result
        
        if height > 4096 or width > 4096:
            result["error"] = f"Image too large: {width}x{height}"
            return result
        
        # Check data type
        if image.dtype not in [np.uint8, np.float32, np.float64]:
            result["error"] = f"Unsupported data type: {image.dtype}"
            return result
        
        result["valid"] = True
        result["info"] = {
            "width": width,
            "height": height,
            "channels": channels,
            "dtype": str(image.dtype),
            "size_mb": (image.nbytes / 1024 / 1024)
        }
        
        return result
        
    except Exception as e:
        result["error"] = f"Validation error: {str(e)}"
        return result


def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe storage"""
    # Remove or replace dangerous characters
    filename = re.sub(r'[^\w\-_\.]', '_', filename)
    
    # Limit length
    if len(filename) > 100:
        name, ext = filename.rsplit('.', 1) if '.' in filename else (filename, '')
        filename = name[:95] + f'.{ext}' if ext else name[:100]
    
    return filename


class DetectionOptions(BaseModel):
    """Validation model for detection options"""
    
    confidence_threshold: Optional[float] = 0.7
    max_faces: Optional[int] = 10
    min_face_size: Optional[int] = 40
    enable_quality_assessment: Optional[bool] = True
    enable_landmarks: Optional[bool] = False
    
    @validator('confidence_threshold')
    def validate_confidence(cls, v):
        if v is not None and not (0.0 <= v <= 1.0):
            raise ValueError('Confidence threshold must be between 0.0 and 1.0')
        return v
    
    @validator('max_faces')
    def validate_max_faces(cls, v):
        if v is not None and not (1 <= v <= 100):
            raise ValueError('Max faces must be between 1 and 100')
        return v
    
    @validator('min_face_size')
    def validate_min_size(cls, v):
        if v is not None and not (10 <= v <= 500):
            raise ValueError('Min face size must be between 10 and 500 pixels')
        return v
