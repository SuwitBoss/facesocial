"""
Pydantic response models for face detection service
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from enum import Enum


class QualityMetrics(BaseModel):
    """Face quality assessment metrics"""
    sharpness: float = Field(..., description="Face sharpness score (0-1)")
    brightness: float = Field(..., description="Face brightness score (0-1)")
    size: float = Field(..., description="Face size score (0-1)")
    overall_score: float = Field(..., description="Overall quality score (0-1)")


class BoundingBox(BaseModel):
    """Face bounding box coordinates"""
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")


class FaceDetection(BaseModel):
    """Single face detection result"""
    bbox: BoundingBox = Field(..., description="Face bounding box")
    confidence: float = Field(..., description="Detection confidence (0-1)")
    quality_metrics: Optional[QualityMetrics] = Field(None, description="Quality assessment")
    landmarks: Optional[List[Dict[str, float]]] = Field(None, description="Facial landmarks")
    face_id: Optional[str] = Field(None, description="Unique face identifier")


class DetectionStatus(str, Enum):
    """Detection status values"""
    SUCCESS = "success"
    PARTIAL = "partial"
    FAILED = "failed"
    ERROR = "error"


class DetectionResponse(BaseModel):
    """Response for single image face detection"""
    status: DetectionStatus = Field(..., description="Detection status")
    faces: List[FaceDetection] = Field(default_factory=list, description="Detected faces")
    face_count: int = Field(..., description="Number of faces detected")
    processing_time: float = Field(..., description="Processing time in seconds")
    detector_used: str = Field(..., description="Detector algorithm used")
    image_info: Dict[str, Any] = Field(default_factory=dict, description="Image metadata")
    message: Optional[str] = Field(None, description="Additional message or error details")


class BatchDetectionItem(BaseModel):
    """Single item in batch detection response"""
    image_id: str = Field(..., description="Image identifier")
    result: DetectionResponse = Field(..., description="Detection result for this image")


class BatchDetectionResponse(BaseModel):
    """Response for batch face detection"""
    status: DetectionStatus = Field(..., description="Overall batch status")
    results: List[BatchDetectionItem] = Field(default_factory=list, description="Individual results")
    total_images: int = Field(..., description="Total number of images processed")
    successful_detections: int = Field(..., description="Number of successful detections")
    failed_detections: int = Field(..., description="Number of failed detections")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    batch_id: Optional[str] = Field(None, description="Batch identifier")


class ErrorResponse(BaseModel):
    """Error response model"""
    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")
    timestamp: Optional[str] = Field(None, description="Error timestamp")
    request_id: Optional[str] = Field(None, description="Request identifier")


class HealthResponse(BaseModel):
    """Health check response"""
    status: str = Field(..., description="Service status")
    version: str = Field(..., description="Service version")
    timestamp: str = Field(..., description="Health check timestamp")
    gpu_available: bool = Field(..., description="GPU availability")
    models_loaded: List[str] = Field(default_factory=list, description="Currently loaded models")
    memory_usage: Optional[Dict[str, float]] = Field(None, description="Memory usage statistics")


class MetricsResponse(BaseModel):
    """Metrics response model"""
    total_requests: int = Field(..., description="Total number of requests")
    successful_requests: int = Field(..., description="Number of successful requests")
    failed_requests: int = Field(..., description="Number of failed requests")
    average_processing_time: float = Field(..., description="Average processing time")
    models_in_memory: List[str] = Field(default_factory=list, description="Models currently in memory")
    gpu_memory_usage: Optional[Dict[str, float]] = Field(None, description="GPU memory statistics")
    uptime_seconds: float = Field(..., description="Service uptime in seconds")


class ModelInfo(BaseModel):
    """Model information response"""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type (YOLO, InsightFace, MediaPipe)")
    loaded: bool = Field(..., description="Whether model is loaded")
    memory_usage_mb: Optional[float] = Field(None, description="Memory usage in MB")
    last_used: Optional[str] = Field(None, description="Last usage timestamp")


class ModelsResponse(BaseModel):
    """Models status response"""
    available_models: List[ModelInfo] = Field(default_factory=list, description="Available models")
    total_memory_usage_mb: float = Field(..., description="Total memory usage")
    gpu_memory_available_mb: float = Field(..., description="Available GPU memory")
    max_concurrent_models: int = Field(..., description="Maximum concurrent models")
