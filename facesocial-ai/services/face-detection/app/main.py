"""
Face Detection Service - Main FastAPI Application
Supports YOLO, InsightFace, and MediaPipe detectors with automatic model selection
"""

import os
import asyncio
import logging
import time
from contextlib import asynccontextmanager
from typing import Dict, Any

# Force CPU-only execution for CUDA compatibility
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['ORT_LOGGING_LEVEL'] = '4'  # Only fatal errors for ONNX Runtime

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import JSONResponse
import uvicorn

from app.core.config import settings, DetectionMode, DetectorType
from app.services.detection.strategy import detection_manager
from app.services.gpu.vram_manager import vram_manager


# Configure logging
logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    
    # Startup
    logger.info("🚀 Starting Face Detection Service...")
    logger.info(f"Version: {settings.VERSION}")
    logger.info(f"Environment: {settings.ENVIRONMENT}")
    logger.info(f"VRAM Limit: {settings.VRAM_LIMIT_MB}MB")
    
    try:
        # Initialize detection manager
        await detection_manager.initialize()
        
        # Log available detectors
        status = await detection_manager.get_status()
        available = status.get("available_detectors", [])
        logger.info(f"Available detectors: {', '.join(available)}")
        
        # Log VRAM status
        vram_status = status.get("vram_status", {})
        logger.info(f"VRAM Manager: {vram_status.get('loaded_models', 0)} models loaded")
        
        logger.info("✅ Face Detection Service started successfully")
        
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        raise
    
    yield
    
    # Shutdown
    logger.info("🛑 Shutting down Face Detection Service...")
    try:
        await detection_manager.shutdown()
        logger.info("✅ Service shutdown complete")
    except Exception as e:
        logger.error(f"❌ Error during shutdown: {e}")


# Create FastAPI application
app = FastAPI(
    title="Face Detection Service",
    description="""
    Advanced Face Detection Service with multiple AI models:
    - YOLOv10n-face: High accuracy detection
    - InsightFace: Balanced performance 
    - MediaPipe: Real-time detection
    
    Automatic model selection based on use case with GPU memory management.
    """,
    version=settings.VERSION,
    lifespan=lifespan,
    docs_url="/docs" if settings.DEBUG else None,
    redoc_url="/redoc" if settings.DEBUG else None
)

# Add middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)


# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "success": False,
            "error": {
                "code": "INTERNAL_ERROR",
                "message": "An internal error occurred",
                "details": str(exc) if settings.DEBUG else None
            },
            "metadata": {
                "timestamp": time.time(),
                "request_id": getattr(request.state, 'request_id', None)
            }
        }
    )


# Request middleware for logging and timing
@app.middleware("http")
async def request_middleware(request: Request, call_next):
    """Request middleware for logging and performance tracking"""
    
    # Generate request ID
    import uuid
    request_id = str(uuid.uuid4())[:8]
    request.state.request_id = request_id
    
    # Log request
    start_time = time.time()
    logger.info(f"[{request_id}] {request.method} {request.url.path}")
    
    try:
        # Process request
        response = await call_next(request)
        
        # Log response
        processing_time = time.time() - start_time
        logger.info(f"[{request_id}] Response: {response.status_code} "
                   f"({processing_time:.3f}s)")
        
        # Add headers
        response.headers["X-Request-ID"] = request_id
        response.headers["X-Processing-Time"] = f"{processing_time:.3f}"
        
        return response
        
    except Exception as e:
        processing_time = time.time() - start_time
        logger.error(f"[{request_id}] Error: {e} ({processing_time:.3f}s)")
        raise


# Include API routes
from app.api.v1.routes.detection import router as detection_router
from app.api.v1.routes.health import router as health_router
from app.api.v1.routes.metrics import router as metrics_router

app.include_router(detection_router, prefix=f"{settings.API_V1_STR}/detect", tags=["Detection"])
app.include_router(health_router, prefix="", tags=["Health"])
app.include_router(metrics_router, prefix=f"{settings.API_V1_STR}/metrics", tags=["Metrics"])


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with service information"""
    
    try:
        status = await detection_manager.get_status()
        
        return {
            "service": "Face Detection Service",
            "version": settings.VERSION,
            "status": "healthy",
            "available_detectors": status.get("available_detectors", []),
            "modes": [mode.value for mode in DetectionMode],
            "endpoints": {
                "detection": f"{settings.API_V1_STR}/detect",
                "health": "/health",
                "metrics": f"{settings.API_V1_STR}/metrics",
                "docs": "/docs" if settings.DEBUG else None
            }
        }
        
    except Exception as e:
        logger.error(f"Error in root endpoint: {e}")
        raise HTTPException(status_code=500, detail="Service unavailable")


# Quick detection endpoint for testing
@app.post("/quick-detect")
async def quick_detect_endpoint(request: Request):
    """Quick detection endpoint for testing (development only)"""
    
    if not settings.DEBUG:
        raise HTTPException(status_code=404, detail="Not found")
    
    try:
        # Create a simple test response
        return {
            "message": "Quick detect endpoint - upload an image to /api/v1/detect",
            "available_modes": [mode.value for mode in DetectionMode],
            "available_detectors": [detector.value for detector in DetectorType],
            "example_curl": """
curl -X POST "http://localhost:8000/api/v1/detect" \\
     -F "image=@your_image.jpg" \\
     -F "mode=balanced"
            """.strip()
        }
        
    except Exception as e:
        logger.error(f"Error in quick detect: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Performance info endpoint
@app.get("/performance")
async def performance_info():
    """Get performance information"""
    
    try:
        status = await detection_manager.get_status()
        
        return {
            "vram_status": status.get("vram_status", {}),
            "detector_stats": status.get("performance_stats", {}),
            "system_info": {
                "vram_limit_mb": settings.VRAM_LIMIT_MB,
                "max_batch_size": settings.MAX_BATCH_SIZE,
                "model_cache_size": settings.MODEL_CACHE_SIZE
            }
        }
        
    except Exception as e:
        logger.error(f"Error getting performance info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    # Run with uvicorn for development
    uvicorn.run(
        "app.main:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=settings.DEBUG,
        workers=settings.WORKERS,
        log_level=settings.LOG_LEVEL.lower()
    )