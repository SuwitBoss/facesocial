"""
Face Detection API Endpoints
Provides RESTful API for face detection with multiple models
"""

import asyncio
import time
import logging
from typing import Optional, List, Dict, Any
import io
import numpy as np
import cv2
from PIL import Image

from fastapi import APIRouter, HTTPException, UploadFile, File, Form, BackgroundTasks, status
from fastapi.responses import JSONResponse

from app.core.config import DetectionMode, DetectorType, settings
from app.services.detection.strategy import detection_manager
from app.models.responses import DetectionResponse, BatchDetectionResponse, ErrorResponse

router = APIRouter()
logger = logging.getLogger(__name__)


async def validate_image(image_file: UploadFile) -> np.ndarray:
    """Validate and convert uploaded image to numpy array"""
    
    # Check file size
    if image_file.size and image_file.size > settings.MAX_IMAGE_SIZE_MB * 1024 * 1024:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image size exceeds {settings.MAX_IMAGE_SIZE_MB}MB limit"
        )
    
    # Check content type
    allowed_types = ["image/jpeg", "image/jpg", "image/png", "image/webp"]
    if image_file.content_type not in allowed_types:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Unsupported image type. Allowed: {', '.join(allowed_types)}"
        )
    
    try:
        # Read image data
        image_data = await image_file.read()
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array (RGB)
        rgb_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        return bgr_array
        
    except Exception as e:
        logger.error(f"Failed to process image: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid image format or corrupted file"
        )


def create_detection_response(result, request_metadata: Dict[str, Any]) -> Dict[str, Any]:
    """Create standardized detection response"""
    
    # Convert DetectedFace objects to FaceDetection format
    faces = []
    for face in result.faces:
        face_detection = {
            "bbox": {
                "x": face.bbox.get("x", 0),
                "y": face.bbox.get("y", 0), 
                "width": face.bbox.get("width", 0),
                "height": face.bbox.get("height", 0)
            },
            "confidence": face.confidence,
            "landmarks": face.landmarks,
            "face_id": None
        }
        
        # Add quality metrics if available
        if face.quality_score is not None:
            face_detection["quality_metrics"] = {
                "sharpness": face.quality_score,
                "brightness": face.quality_score,
                "size": face.quality_score,
                "overall_score": face.quality_score
            }
        
        faces.append(face_detection)
    
    return {
        "status": "success",
        "faces": faces,
        "face_count": result.total_faces,
        "processing_time": result.processing_time,
        "detector_used": result.model_used,
        "image_info": result.image_info,
        "message": f"Successfully detected {result.total_faces} faces"
    }


@router.post("/", response_model=DetectionResponse)
async def detect_faces(
    image: UploadFile = File(..., description="Image file to detect faces in"),
    mode: Optional[DetectionMode] = Form(DetectionMode.BALANCED, description="Detection mode"),
    detector: Optional[DetectorType] = Form(None, description="Specific detector to use"),
    confidence_threshold: Optional[float] = Form(None, description="Confidence threshold (0.0-1.0)"),
    max_faces: Optional[int] = Form(None, description="Maximum number of faces to detect"),
    min_face_size: Optional[int] = Form(None, description="Minimum face size in pixels"),
    enable_quality_assessment: Optional[bool] = Form(True, description="Enable face quality assessment"),
    enable_landmarks: Optional[bool] = Form(False, description="Enable facial landmarks detection")
):
    """
    Detect faces in an uploaded image
    
    **Detection Modes:**
    - `realtime`: Fast detection using MediaPipe (best for live video)
    - `balanced`: Balanced speed/accuracy using InsightFace (general purpose)
    - `accurate`: High accuracy using YOLO (best for authentication)
    
    **Detectors:**
    - `yolo`: YOLOv10n-face (GPU, high accuracy)
    - `insightface`: InsightFace (GPU/CPU, balanced)
    - `mediapipe`: MediaPipe (CPU, real-time)
    
    If no detector is specified, it will be chosen automatically based on mode.
    """
    
    start_time = time.time()
    
    try:
        # Validate image
        image_array = await validate_image(image)
        
        # Prepare detection options
        options = {
            "enable_quality_assessment": enable_quality_assessment,
            "enable_landmarks": enable_landmarks
        }
        
        # Add optional parameters
        if confidence_threshold is not None:
            options["confidence_threshold"] = confidence_threshold
        if max_faces is not None:
            options["max_faces"] = max_faces
        if min_face_size is not None:
            options["min_face_size"] = min_face_size
        
        # Perform detection
        result = await detection_manager.detect_faces(
            image=image_array,
            mode=mode,
            detector=detector,
            options=options
        )
        
        # Create response
        request_metadata = {
            "request_id": f"detect_{int(time.time())}"
        }
        
        response = create_detection_response(result, request_metadata)
        
        logger.info(f"Detection completed: {result.total_faces} faces found "
                   f"using {result.model_used} in {result.processing_time:.3f}s")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Detection failed: {str(e)}"
        )


@router.post("/batch", response_model=BatchDetectionResponse)
async def batch_detect_faces(
    images: List[UploadFile] = File(..., description="Multiple image files"),
    mode: Optional[DetectionMode] = Form(DetectionMode.BALANCED, description="Detection mode"),
    detector: Optional[DetectorType] = Form(None, description="Specific detector to use"),
    confidence_threshold: Optional[float] = Form(None, description="Confidence threshold"),
    max_faces: Optional[int] = Form(None, description="Maximum faces per image"),
    parallel_processing: Optional[bool] = Form(True, description="Enable parallel processing")
):
    """
    Detect faces in multiple images (batch processing)
    
    Maximum 10 images per batch for performance optimization.
    Results are returned in the same order as input images.
    """
    
    # Validate batch size
    if len(images) > settings.MAX_BATCH_ITEMS:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Too many images. Maximum {settings.MAX_BATCH_ITEMS} images per batch."
        )
    
    start_time = time.time()
    
    try:
        # Prepare detection options
        options = {
            "enable_quality_assessment": True,
            "enable_landmarks": False
        }
        
        if confidence_threshold is not None:
            options["confidence_threshold"] = confidence_threshold
        if max_faces is not None:
            options["max_faces"] = max_faces
        
        # Process images
        results = []
        
        if parallel_processing and len(images) > 1:
            # Parallel processing
            tasks = []
            image_arrays = []
            
            # Validate all images first
            for i, image in enumerate(images):
                try:
                    image_array = await validate_image(image)
                    image_arrays.append((i, image_array, image.filename))
                except Exception as e:
                    # Add error result for failed validation
                    results.append({
                        "image_index": i,
                        "filename": image.filename,
                        "success": False,
                        "error": str(e)
                    })
            
            # Create detection tasks
            for i, image_array, filename in image_arrays:
                task = asyncio.create_task(
                    detection_manager.detect_faces(
                        image=image_array,
                        mode=mode,
                        detector=detector,
                        options=options
                    )
                )
                tasks.append((i, filename, task))
            
            # Wait for all tasks to complete
            for i, filename, task in tasks:
                try:
                    result = await task
                    results.append({
                        "image_index": i,
                        "filename": filename,
                        "success": True,
                        "faces": [
                            {
                                "bbox": face.bbox,
                                "confidence": face.confidence,
                                "quality_score": face.quality_score
                            }
                            for face in result.faces
                        ],
                        "total_faces": result.total_faces,
                        "processing_time": result.processing_time,
                        "model_used": result.model_used
                    })
                except Exception as e:
                    results.append({
                        "image_index": i,
                        "filename": filename,
                        "success": False,
                        "error": str(e)
                    })
        
        else:
            # Sequential processing
            for i, image in enumerate(images):
                try:
                    image_array = await validate_image(image)
                    
                    result = await detection_manager.detect_faces(
                        image=image_array,
                        mode=mode,
                        detector=detector,
                        options=options
                    )
                    
                    results.append({
                        "image_index": i,
                        "filename": image.filename,
                        "success": True,
                        "faces": [
                            {
                                "bbox": face.bbox,
                                "confidence": face.confidence,
                                "quality_score": face.quality_score
                            }
                            for face in result.faces
                        ],
                        "total_faces": result.total_faces,
                        "processing_time": result.processing_time,
                        "model_used": result.model_used
                    })
                    
                except Exception as e:
                    results.append({
                        "image_index": i,
                        "filename": image.filename,
                        "success": False,
                        "error": str(e)
                    })
          # Sort results by image index to maintain order
        results.sort(key=lambda x: x["image_index"])
        
        # Calculate statistics
        successful_results = [r for r in results if r["success"]]
        total_faces = sum(r.get("total_faces", 0) for r in successful_results)
        
        total_processing_time = time.time() - start_time
        
        # Convert results to BatchDetectionItem format
        batch_items = []
        for result in results:
            if result["success"]:
                # Convert the face data to proper format
                faces = []
                for face_data in result.get("faces", []):
                    face_detection = {
                        "bbox": {
                            "x": face_data["bbox"].get("x", 0) if isinstance(face_data["bbox"], dict) else 0,
                            "y": face_data["bbox"].get("y", 0) if isinstance(face_data["bbox"], dict) else 0,
                            "width": face_data["bbox"].get("width", 0) if isinstance(face_data["bbox"], dict) else 0,
                            "height": face_data["bbox"].get("height", 0) if isinstance(face_data["bbox"], dict) else 0
                        },
                        "confidence": face_data.get("confidence", 0),
                        "landmarks": face_data.get("landmarks"),
                        "face_id": None
                    }
                    
                    # Add quality metrics if available
                    if face_data.get("quality_score") is not None:
                        face_detection["quality_metrics"] = {
                            "sharpness": face_data["quality_score"],
                            "brightness": face_data["quality_score"],
                            "size": face_data["quality_score"],
                            "overall_score": face_data["quality_score"]
                        }
                    
                    faces.append(face_detection)
                
                detection_result = {
                    "status": "success",
                    "faces": faces,
                    "face_count": result.get("total_faces", 0),
                    "processing_time": result.get("processing_time", 0),
                    "detector_used": result.get("model_used", "unknown"),
                    "image_info": {"filename": result.get("filename", "")},
                    "message": f"Successfully detected {result.get('total_faces', 0)} faces"
                }
            else:
                detection_result = {
                    "status": "error",
                    "faces": [],
                    "face_count": 0,
                    "processing_time": 0,
                    "detector_used": "none",
                    "image_info": {"filename": result.get("filename", "")},
                    "message": result.get("error", "Detection failed")
                }
            
            batch_items.append({
                "image_id": result.get("filename", f"image_{result['image_index']}"),
                "result": detection_result
            })
        
        response = {
            "status": "success" if len(successful_results) > 0 else "failed",
            "results": batch_items,
            "total_images": len(images),
            "successful_detections": len(successful_results),
            "failed_detections": len(results) - len(successful_results),
            "total_processing_time": total_processing_time,
            "batch_id": f"batch_{int(time.time())}"
        }
        
        logger.info(f"Batch detection completed: {len(successful_results)}/{len(images)} "
                   f"images processed, {total_faces} faces found")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Batch detection failed: {e}", exc_info=True)
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Batch detection failed: {str(e)}"
        )


@router.get("/models")
async def get_available_models():
    """Get information about available detection models"""
    
    try:
        status = await detection_manager.get_status()
        
        return {
            "success": True,
            "data": {
                "available_detectors": status.get("available_detectors", []),
                "detector_info": {
                    "yolo": {
                        "name": "YOLOv10n-face",
                        "type": "GPU-accelerated",
                        "use_case": "High accuracy detection",
                        "typical_speed": "~300ms",
                        "memory_usage": "~800MB VRAM"
                    },                    "insightface": {
                        "name": "InsightFace",
                        "type": "GPU/CPU hybrid",
                        "use_case": "Balanced performance",
                        "typical_speed": "~200ms",
                        "memory_usage": "~400MB GPU/RAM"
                    },
                    "mediapipe": {
                        "name": "MediaPipe Face Detection",
                        "type": "CPU-optimized",
                        "use_case": "Real-time processing",
                        "typical_speed": "~50ms",
                        "memory_usage": "~100MB RAM"
                    }
                },
                "modes": {
                    "realtime": {
                        "detector": "mediapipe",
                        "description": "Fastest detection for live video"
                    },
                    "balanced": {                        "detector": "insightface",
                        "description": "Good balance of speed and accuracy"
                    },
                    "accurate": {
                        "detector": "yolo",
                        "description": "Highest accuracy for authentication"
                    }
                },
                "performance_stats": status.get("performance_stats", {}),
                "vram_status": status.get("vram_status", {})
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get model info: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve model information"
        )


@router.post("/benchmark")
async def benchmark_detectors(
    background_tasks: BackgroundTasks,
    iterations: Optional[int] = Form(5, description="Number of test iterations")
):
    """
    Benchmark all available detectors (development endpoint)
    
    This endpoint runs performance tests on all available detectors
    and returns timing and accuracy comparisons.
    """
    
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    
    if iterations > 20:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Maximum 20 iterations allowed"
        )
    
    try:
        # Start benchmark in background
        background_tasks.add_task(run_benchmark, iterations)
        
        return {
            "success": True,
            "message": f"Benchmark started with {iterations} iterations",
            "note": "Results will be logged to console. Check /performance for current stats."
        }
        
    except Exception as e:
        logger.error(f"Failed to start benchmark: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start benchmark"
        )


async def run_benchmark(iterations: int):
    """Run benchmark tests (background task)"""
    
    try:
        logger.info(f"Starting benchmark with {iterations} iterations...")
        
        # Create test image
        test_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        # Test each detector
        detectors = [DetectorType.MEDIAPIPE, DetectorType.INSIGHTFACE, DetectorType.YOLO]
        results = {}
        
        for detector in detectors:
            try:
                logger.info(f"Benchmarking {detector.value}...")
                
                times = []
                face_counts = []
                
                for i in range(iterations):
                    start_time = time.time()
                    
                    result = await detection_manager.detect_faces(
                        image=test_image,
                        detector=detector,
                        options={"enable_quality_assessment": False}
                    )
                    
                    processing_time = time.time() - start_time
                    times.append(processing_time)
                    face_counts.append(result.total_faces)
                
                results[detector.value] = {
                    "avg_time": sum(times) / len(times),
                    "min_time": min(times),
                    "max_time": max(times),
                    "avg_faces": sum(face_counts) / len(face_counts),
                    "iterations": iterations
                }
                
                logger.info(f"{detector.value} benchmark: "
                           f"avg={results[detector.value]['avg_time']:.3f}s")
                
            except Exception as e:
                logger.error(f"Benchmark failed for {detector.value}: {e}")
                results[detector.value] = {"error": str(e)}
        
        logger.info("Benchmark completed:")
        for detector, stats in results.items():
            if "error" not in stats:
                logger.info(f"  {detector}: {stats['avg_time']:.3f}s avg, "
                           f"{stats['avg_faces']:.1f} faces avg")
            else:
                logger.info(f"  {detector}: ERROR - {stats['error']}")
        
    except Exception as e:
        logger.error(f"Benchmark failed: {e}")