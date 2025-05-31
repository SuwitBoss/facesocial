# Pydantic request models
"""
Pydantic Response Models for Face Detection Service
"""

from typing import List, Optional, Dict, Any, Union
from pydantic import BaseModel, Field
from enum import Enum


class FaceBBox(BaseModel):
    """Face bounding box coordinates"""
    x: float = Field(..., description="X coordinate of top-left corner")
    y: float = Field(..., description="Y coordinate of top-left corner") 
    width: float = Field(..., description="Width of the bounding box")
    height: float = Field(..., description="Height of the bounding box")


class FaceLandmark(BaseModel):
    """Individual facial landmark point"""
    name: str = Field(..., description="Landmark name (e.g., 'left_eye', 'nose')")
    x: float = Field(..., description="X coordinate")
    y: float = Field(..., description="Y coordinate")


class FaceAttributes(BaseModel):
    """Face attributes from analysis"""
    estimated_age: Optional[float] = Field(None, description="Estimated age")
    estimated_gender: Optional[str] = Field(None, description="Estimated gender")
    gender_confidence: Optional[float] = Field(None, description="Gender confidence score")
    dominant_emotion: Optional[str] = Field(None, description="Dominant emotion")
    emotion_scores: Optional[Dict[str, float]] = Field(None, description="Emotion scores")


class DetectedFaceResponse(BaseModel):
    """Single detected face response"""
    bbox: FaceBBox
    confidence: float = Field(..., ge=0.0, le=1.0, description="Detection confidence score")
    quality_score: Optional[float] = Field(None, ge=0.0, le=1.0, description="Face quality score")
    landmarks: Optional[List[FaceLandmark]] = Field(None, description="Facial landmarks")
    attributes: Optional[FaceAttributes] = Field(None, description="Face attributes")


class ImageInfo(BaseModel):
    """Information about the processed image"""
    width: int = Field(..., description="Image width in pixels")
    height: int = Field(..., description="Image height in pixels")
    format: Optional[str] = Field(None, description="Image format")


class PerformanceMetrics(BaseModel):
    """Performance metrics for the detection"""
    inference_time: Optional[float] = Field(None, description="Model inference time in seconds")
    vram_usage: Optional[Dict[str, Any]] = Field(None, description="VRAM usage information")
    cpu_usage: Optional[float] = Field(None, description="CPU usage percentage")


class DetectionData(BaseModel):
    """Main detection result data"""
    faces: List[DetectedFaceResponse] = Field(default_factory=list, description="List of detected faces")
    total_faces: int = Field(..., description="Total number of faces detected")
    image_info: ImageInfo
    model_used: str = Field(..., description="Model used for detection")
    processing_time: float = Field(..., description="Total processing time in seconds")


class DetectionMetadata(BaseModel):
    """Metadata for the detection request"""
    request_id: str = Field(..., description="Unique request identifier")
    timestamp: float = Field(..., description="Unix timestamp")
    model_version: str = Field(..., description="Model version used")
    performance_metrics: Optional[PerformanceMetrics] = Field(None, description="Performance metrics")


class DetectionResponse(BaseModel):
    """Complete face detection response"""
    success: bool = Field(..., description="Whether the request was successful")
    data: DetectionData
    metadata: DetectionMetadata


class BatchDetectionResult(BaseModel):
    """Single image result in batch processing"""
    image_index: int = Field(..., description="Index of the image in the batch")
    filename: Optional[str] = Field(None, description="Original filename")
    success: bool = Field(..., description="Whether detection was successful for this image")
    faces: Optional[List[DetectedFaceResponse]] = Field(None, description="Detected faces")
    total_faces: Optional[int] = Field(None, description="Total faces detected")
    processing_time: Optional[float] = Field(None, description="Processing time for this image")
    model_used: Optional[str] = Field(None, description="Model used")
    error: Optional[str] = Field(None, description="Error message if failed")


class BatchSummary(BaseModel):
    """Summary of batch processing"""
    total_images: int = Field(..., description="Total number of images processed")
    successful_detections: int = Field(..., description="Number of successful detections")
    failed_detections: int = Field(..., description="Number of failed detections")
    total_faces_detected: int = Field(..., description="Total faces detected across all images")
    average_processing_time: float = Field(..., description="Average processing time per image")
    parallel_processing: bool = Field(..., description="Whether parallel processing was used")


class BatchDetectionData(BaseModel):
    """Batch detection result data"""
    results: List[BatchDetectionResult] = Field(..., description="Individual image results")
    summary: BatchSummary


class BatchDetectionResponse(BaseModel):
    """Complete batch detection response"""
    success: bool = Field(..., description="Whether the batch request was successful")
    data: BatchDetectionData
    metadata: DetectionMetadata


class ErrorDetail(BaseModel):
    """Error detail structure"""
    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Human readable error message")
    details: Optional[Dict[str, Any]] = Field(None, description="Additional error details")


class ErrorResponse(BaseModel):
    """Error response structure"""
    success: bool = Field(False, description="Always false for errors")
    error: ErrorDetail
    metadata: DetectionMetadata


class HealthStatus(BaseModel):
    """Health check status"""
    status: str = Field(..., description="Service health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    available_detectors: List[str] = Field(default_factory=list, description="Available detectors")
    response_time: str = Field(..., description="Response time")
    timestamp: float = Field(..., description="Timestamp")


class DetectorInfo(BaseModel):
    """Information about a specific detector"""
    available: bool = Field(..., description="Whether detector is available")
    status: str = Field(..., description="Detector status")
    total_detections: Optional[int] = Field(None, description="Total detections performed")
    average_processing_time: Optional[float] = Field(None, description="Average processing time")
    error_rate: Optional[float] = Field(None, description="Error rate")
    error: Optional[str] = Field(None, description="Error message if unavailable")


class SystemInfo(BaseModel):
    """System resource information"""
    cpu: Dict[str, Any] = Field(default_factory=dict, description="CPU information")
    memory: Dict[str, Any] = Field(default_factory=dict, description="Memory information")
    disk: Dict[str, Any] = Field(default_factory=dict, description="Disk information")


class GPUInfo(BaseModel):
    """GPU information"""
    total: float = Field(..., description="Total GPU memory")
    used: float = Field(..., description="Used GPU memory")
    free: float = Field(..., description="Free GPU memory")
    utilization: float = Field(..., description="GPU utilization percentage")


class VRAMStatus(BaseModel):
    """VRAM manager status"""
    loaded_models: int = Field(..., description="Number of loaded models")
    memory_usage_mb: float = Field(..., description="Memory usage in MB")
    memory_limit_mb: float = Field(..., description="Memory limit in MB")
    memory_utilization: float = Field(..., description="Memory utilization percentage")


class PerformanceInfo(BaseModel):
    """Performance information"""
    available_detectors: List[str] = Field(..., description="Available detectors")
    total_detectors: int = Field(..., description="Total number of detectors")
    high_error_rate: bool = Field(..., description="Whether there's high error rate")
    high_memory_usage: bool = Field(..., description="Whether there's high memory usage")


class CheckStatus(BaseModel):
    """Status checks"""
    detection_manager_initialized: bool = Field(..., description="Detection manager status")
    vram_manager_active: bool = Field(..., description="VRAM manager status")
    gpu_available: bool = Field(..., description="GPU availability")


class DetailedHealthResponse(BaseModel):
    """Detailed health check response"""
    status: str = Field(..., description="Overall health status")
    service: str = Field(..., description="Service name")
    version: str = Field(..., description="Service version")
    environment: str = Field(..., description="Environment")
    response_time: str = Field(..., description="Response time")
    timestamp: float = Field(..., description="Timestamp")
    detectors: Dict[str, DetectorInfo] = Field(..., description="Individual detector status")
    system: Dict[str, Any] = Field(..., description="System information")
    performance: PerformanceInfo
    checks: CheckStatus


class ModelInfo(BaseModel):
    """Model information"""
    name: str = Field(..., description="Model name")
    type: str = Field(..., description="Model type")
    use_case: str = Field(..., description="Primary use case")
    typical_speed: str = Field(..., description="Typical processing speed")
    memory_usage: str = Field(..., description="Memory usage")


class ModeInfo(BaseModel):
    """Detection mode information"""
    detector: str = Field(..., description="Detector used for this mode")
    description: str = Field(..., description="Mode description")


class ModelsResponse(BaseModel):
    """Available models response"""
    success: bool = Field(..., description="Whether request was successful")
    data: Dict[str, Any] = Field(..., description="Models and configuration data")


# Union types for responses
DetectionResponseType = Union[DetectionResponse, BatchDetectionResponse, ErrorResponse]
HealthResponseType = Union[HealthStatus, DetailedHealthResponse]