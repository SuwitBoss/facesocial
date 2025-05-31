# app/api/v1/routes/metrics.py
"""
Metrics and Performance Monitoring Endpoints
"""

import time
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, status, Depends
from fastapi.responses import PlainTextResponse

from app.core.config import settings
from app.services.detection.strategy import detection_manager
from app.services.gpu.vram_manager import vram_manager
from app.services.utils.metrics import performance_tracker

router = APIRouter()
logger = logging.getLogger(__name__)


@router.get("/performance")
async def get_performance_metrics():
    """Get comprehensive performance metrics"""
    
    try:
        # Get detection manager stats
        detection_stats = await detection_manager.get_status()
        
        # Get VRAM stats
        vram_stats = await vram_manager.get_status()
        
        # Get service performance stats
        service_stats = performance_tracker.get_stats()
        
        # System resources
        import psutil
        system_stats = {
            "cpu_usage": psutil.cpu_percent(),
            "memory": {
                "used_percent": psutil.virtual_memory().percent,
                "used_gb": psutil.virtual_memory().used / (1024**3),
                "total_gb": psutil.virtual_memory().total / (1024**3)
            },
            "disk": {
                "used_percent": psutil.disk_usage('/').percent,
                "free_gb": psutil.disk_usage('/').free / (1024**3)
            }
        }
        
        return {
            "success": True,
            "data": {
                "service": service_stats,
                "detection": detection_stats,
                "vram": vram_stats,
                "system": system_stats,
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get performance metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve performance metrics"
        )


@router.get("/prometheus", response_class=PlainTextResponse)
async def get_prometheus_metrics():
    """Get metrics in Prometheus format"""
    
    try:
        # Get performance stats
        stats = performance_tracker.get_stats()
        detection_stats = await detection_manager.get_status()
        vram_stats = await vram_manager.get_status()
        
        # Generate Prometheus metrics
        metrics = []
        
        # Service metrics
        metrics.append(f"face_detection_requests_total {stats.get('total_requests', 0)}")
        metrics.append(f"face_detection_errors_total {stats.get('error_count', 0)}")
        metrics.append(f"face_detection_error_rate {stats.get('error_rate', 0)}")
        metrics.append(f"face_detection_requests_per_second {stats.get('requests_per_second', 0)}")
        
        if 'avg_detection_time' in stats:
            metrics.append(f"face_detection_avg_time_seconds {stats['avg_detection_time']}")
            metrics.append(f"face_detection_min_time_seconds {stats['min_detection_time']}")
            metrics.append(f"face_detection_max_time_seconds {stats['max_detection_time']}")
        
        # Memory metrics
        if 'current_memory_usage' in stats:
            metrics.append(f"face_detection_memory_usage_percent {stats['current_memory_usage']}")
        
        # VRAM metrics
        metrics.append(f"face_detection_vram_usage_mb {vram_stats.get('memory_usage_mb', 0)}")
        metrics.append(f"face_detection_vram_utilization_percent {vram_stats.get('memory_utilization', 0)}")
        metrics.append(f"face_detection_loaded_models {vram_stats.get('loaded_models', 0)}")
        
        # Detector availability
        available_detectors = detection_stats.get('available_detectors', [])
        for detector in ['yolo', 'insightface', 'mediapipe']:
            available = 1 if detector in available_detectors else 0
            metrics.append(f"face_detection_{detector}_available {available}")
        
        return "\n".join(metrics)
        
    except Exception as e:
        logger.error(f"Failed to generate Prometheus metrics: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate metrics"
        )


@router.get("/stats")
async def get_detector_stats():
    """Get detailed statistics for each detector"""
    
    try:
        detection_stats = await detection_manager.get_status()
        
        # Format stats for easy consumption
        formatted_stats = {}
        
        for detector_name, stats in detection_stats.get('performance_stats', {}).items():
            formatted_stats[detector_name] = {
                "available": stats.get('available', False),
                "total_detections": stats.get('total_detections', 0),
                "avg_processing_time": stats.get('average_processing_time', 0),
                "error_rate": stats.get('error_rate', 0),
                "status": "healthy" if stats.get('available') and stats.get('error_rate', 0) < 0.1 else "degraded"
            }
        
        return {
            "success": True,
            "data": {
                "detectors": formatted_stats,
                "overall": {
                    "available_detectors": len(detection_stats.get('available_detectors', [])),
                    "total_detectors": len(formatted_stats),
                    "healthy_detectors": len([s for s in formatted_stats.values() if s['status'] == 'healthy'])
                },
                "timestamp": time.time()
            }
        }
        
    except Exception as e:
        logger.error(f"Failed to get detector stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve detector statistics"
        )


@router.post("/reset-stats")
async def reset_performance_stats():
    """Reset performance statistics (admin only)"""
    
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        # Reset service stats
        performance_tracker.detection_times.clear()
        performance_tracker.memory_usage.clear()
        performance_tracker.error_count = 0
        performance_tracker.total_requests = 0
        performance_tracker.start_time = time.time()
        
        # Reset detector stats (if available)
        for strategy in detection_manager.strategies.values():
            strategy.total_detections = 0
            strategy.total_processing_time = 0.0
            strategy.error_count = 0
        
        return {
            "success": True,
            "message": "Performance statistics reset successfully",
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to reset stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to reset statistics"
        )


@router.get("/vram")
async def get_vram_status():
    """Get detailed VRAM status and model information"""
    
    try:
        vram_status = await vram_manager.get_status()
        
        return {
            "success": True,
            "data": vram_status,
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get VRAM status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve VRAM status"
        )


@router.get("/health-summary")
async def get_health_summary():
    """Get summarized health information for monitoring"""
    
    try:
        detection_stats = await detection_manager.get_status()
        vram_stats = await vram_manager.get_status()
        service_stats = performance_tracker.get_stats()
        
        # Determine overall health
        available_detectors = len(detection_stats.get('available_detectors', []))
        error_rate = service_stats.get('error_rate', 0)
        vram_utilization = vram_stats.get('memory_utilization', 0)
        
        if available_detectors == 0:
            health_status = "critical"
        elif error_rate > 0.1 or vram_utilization > 95:
            health_status = "degraded"
        else:
            health_status = "healthy"
        
        return {
            "status": health_status,
            "available_detectors": available_detectors,
            "error_rate": error_rate,
            "vram_utilization": vram_utilization,
            "uptime_seconds": service_stats.get('uptime_seconds', 0),
            "total_requests": service_stats.get('total_requests', 0),
            "timestamp": time.time()
        }
        
    except Exception as e:
        logger.error(f"Failed to get health summary: {e}")
        return {
            "status": "unknown",
            "error": str(e),
            "timestamp": time.time()
        }