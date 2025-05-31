"""
Health Check Endpoints for Face Detection Service
Provides health monitoring and status information
"""

import time
import asyncio
import logging
from typing import Dict, Any

from fastapi import APIRouter, HTTPException, status
from fastapi.responses import JSONResponse

from app.core.config import settings, DetectorType
from app.services.detection.strategy import detection_manager
from app.services.gpu.vram_manager import vram_manager

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/health")
async def health_check():
    """
    Basic health check endpoint
    Returns service status and availability
    """
    
    try:
        start_time = time.time()
        
        # Check if detection manager is initialized
        if not detection_manager.initialized:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={
                    "status": "unhealthy",
                    "message": "Detection manager not initialized",
                    "timestamp": time.time()
                }
            )
        
        # Get basic status
        detection_status = await detection_manager.get_status()
        available_detectors = detection_status.get("available_detectors", [])
        
        # Determine overall health
        is_healthy = len(available_detectors) > 0
        
        response_time = time.time() - start_time
        
        return {
            "status": "healthy" if is_healthy else "degraded",
            "service": "Face Detection Service",
            "version": settings.VERSION,
            "available_detectors": available_detectors,
            "response_time": f"{response_time:.3f}s",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "unhealthy",
                "message": "Health check failed",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@router.get("/health/detailed")
async def detailed_health_check():
    """
    Detailed health check with comprehensive system information
    """
    
    try:
        start_time = time.time()
        
        # Get detection manager status
        detection_status = await detection_manager.get_status()
        
        # Get VRAM status
        vram_status = await vram_manager.get_status()
        
        # Get GPU information
        gpu_info = await vram_manager.get_gpu_memory_info()
        
        # Check individual detectors
        detector_health = {}
        for detector_type in DetectorType:
            try:
                strategy = detection_manager.strategies.get(detector_type)
                if strategy:
                    is_available = await strategy.is_available()
                    stats = strategy.get_stats()
                    
                    detector_health[detector_type.value] = {
                        "available": is_available,
                        "status": "healthy" if is_available else "unavailable",
                        "total_detections": stats.get("total_detections", 0),
                        "average_processing_time": stats.get("average_processing_time", 0),
                        "error_rate": stats.get("error_rate", 0)
                    }
                else:
                    detector_health[detector_type.value] = {
                        "available": False,
                        "status": "not_loaded"
                    }
            except Exception as e:
                detector_health[detector_type.value] = {
                    "available": False,
                    "status": "error",
                    "error": str(e)
                }
        
        # System resource information
        system_info = await _get_system_info()
        
        # Overall health assessment
        available_detectors = [d for d, info in detector_health.items() if info.get("available")]
        is_healthy = len(available_detectors) > 0
        
        # Performance indicators
        high_error_rate = any(
            info.get("error_rate", 0) > 0.1 
            for info in detector_health.values() 
            if info.get("available")
        )
        
        high_memory_usage = vram_status.get("memory_utilization", 0) > 90
        
        # Determine status
        if not is_healthy:
            overall_status = "critical"
        elif high_error_rate or high_memory_usage:
            overall_status = "degraded"
        else:
            overall_status = "healthy"
        
        response_time = time.time() - start_time
        
        response = {
            "status": overall_status,
            "service": "Face Detection Service",
            "version": settings.VERSION,
            "environment": settings.ENVIRONMENT,
            "response_time": f"{response_time:.3f}s",
            "timestamp": time.time(),
            "detectors": detector_health,
            "system": {
                "memory": system_info,
                "gpu": gpu_info,
                "vram_manager": vram_status
            },
            "performance": {
                "available_detectors": available_detectors,
                "total_detectors": len(DetectorType),
                "high_error_rate": high_error_rate,
                "high_memory_usage": high_memory_usage
            },
            "checks": {
                "detection_manager_initialized": detection_status.get("initialized", False),
                "vram_manager_active": len(vram_status.get("models", {})) >= 0,
                "gpu_available": gpu_info.get("total", 0) > 0
            }
        }
        
        return response
        
    except Exception as e:
        logger.error(f"Detailed health check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={
                "status": "critical",
                "message": "Detailed health check failed",
                "error": str(e),
                "timestamp": time.time()
            }
        )


@router.get("/health/ready")
async def readiness_check():
    """
    Kubernetes readiness probe endpoint
    Returns 200 if service is ready to handle requests
    """
    
    try:
        # Check if at least one detector is available
        detection_status = await detection_manager.get_status()
        available_detectors = detection_status.get("available_detectors", [])
        
        if len(available_detectors) == 0:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"ready": False, "reason": "No detectors available"}
            )
        
        # Quick VRAM check
        vram_status = await vram_manager.get_status()
        if vram_status.get("memory_utilization", 0) > 95:
            return JSONResponse(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                content={"ready": False, "reason": "VRAM usage too high"}
            )
        
        return {"ready": True, "detectors": available_detectors}
        
    except Exception as e:
        logger.error(f"Readiness check failed: {e}")
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"ready": False, "reason": str(e)}
        )


@router.get("/health/live")
async def liveness_check():
    """
    Kubernetes liveness probe endpoint
    Returns 200 if service process is alive
    """
    
    return {
        "alive": True,
        "service": "Face Detection Service",
        "timestamp": time.time()
    }


@router.post("/health/test")
async def health_test():
    """
    Run a simple detection test to verify functionality
    Only available in debug mode
    """
    
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        start_time = time.time()
        
        # Create a simple test image
        import numpy as np
        test_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Test each available detector
        results = {}
        detection_status = await detection_manager.get_status()
        
        for detector_name in detection_status.get("available_detectors", []):
            try:
                detector_type = DetectorType(detector_name)
                
                detection_start = time.time()
                result = await detection_manager.detect_faces(
                    image=test_image,
                    detector=detector_type,
                    options={"enable_quality_assessment": False}
                )
                detection_time = time.time() - detection_start
                
                results[detector_name] = {
                    "success": True,
                    "processing_time": f"{detection_time:.3f}s",
                    "faces_detected": result.total_faces,
                    "model_used": result.model_used
                }
                
            except Exception as e:
                results[detector_name] = {
                    "success": False,
                    "error": str(e)
                }
        
        total_time = time.time() - start_time
        
        # Determine overall test result
        successful_tests = sum(1 for r in results.values() if r.get("success"))
        test_passed = successful_tests > 0
        
        return {
            "test_passed": test_passed,
            "total_time": f"{total_time:.3f}s",
            "successful_detectors": successful_tests,
            "total_detectors": len(results),
            "results": results,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Health test failed: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Health test failed: {str(e)}"
        )


async def _get_system_info() -> Dict[str, Any]:
    """Get system resource information"""
    
    try:
        import psutil
        
        # CPU information
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_count = psutil.cpu_count()
        
        # Memory information
        memory = psutil.virtual_memory()
        memory_info = {
            "total_gb": round(memory.total / (1024**3), 2),
            "used_gb": round(memory.used / (1024**3), 2),
            "available_gb": round(memory.available / (1024**3), 2),
            "percent_used": memory.percent
        }
        
        # Disk information
        disk = psutil.disk_usage('/')
        disk_info = {
            "total_gb": round(disk.total / (1024**3), 2),
            "used_gb": round(disk.used / (1024**3), 2),
            "free_gb": round(disk.free / (1024**3), 2),
            "percent_used": round((disk.used / disk.total) * 100, 1)
        }
        
        return {
            "cpu": {
                "count": cpu_count,
                "usage_percent": cpu_percent
            },
            "memory": memory_info,
            "disk": disk_info
        }
        
    except Exception as e:
        logger.warning(f"Failed to get system info: {e}")
        return {
            "cpu": {"count": 0, "usage_percent": 0},
            "memory": {"total_gb": 0, "used_gb": 0, "percent_used": 0},
            "disk": {"total_gb": 0, "used_gb": 0, "percent_used": 0}
        }