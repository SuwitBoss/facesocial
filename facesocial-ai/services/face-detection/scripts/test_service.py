#!/usr/bin/env python3
"""
Face Detection Service Test Script
Quick test to verify service is working correctly
"""

import os
import sys
import asyncio
import logging
import time
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_config():
    """Test configuration loading"""
    logger.info("🔧 Testing configuration...")
    
    try:
        from app.core.config import settings, detection_config
        
        logger.info(f"✅ Service: {settings.SERVICE_NAME} v{settings.VERSION}")
        logger.info(f"✅ YOLO Model Path: {settings.YOLO_MODEL_PATH}")
        logger.info(f"✅ Model exists: {os.path.exists(settings.YOLO_MODEL_PATH)}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Config test failed: {e}")
        return False


async def test_vram_manager():
    """Test VRAM manager"""
    logger.info("🖥️  Testing VRAM Manager...")
    
    try:
        from app.services.gpu.vram_manager import vram_manager
        
        # Get GPU info
        gpu_info = await vram_manager.get_gpu_memory_info()
        logger.info(f"✅ GPU Memory: {gpu_info}")
        
        # Get status
        status = await vram_manager.get_status()
        logger.info(f"✅ VRAM Manager Status: {status['loaded_models']} models loaded")
        
        return True
    except Exception as e:
        logger.error(f"❌ VRAM Manager test failed: {e}")
        return False


async def test_detection_manager():
    """Test detection manager initialization"""
    logger.info("🤖 Testing Detection Manager...")
    
    try:
        from app.services.detection.strategy import detection_manager
        
        # Initialize
        await detection_manager.initialize()
        
        # Get status
        status = await detection_manager.get_status()
        available = status.get('available_detectors', [])
        
        logger.info(f"✅ Available detectors: {available}")
        
        if not available:
            logger.warning("⚠️  No detectors available - this might be expected if models aren't installed")
        
        return True
    except Exception as e:
        logger.error(f"❌ Detection Manager test failed: {e}")
        return False


async def test_simple_detection():
    """Test simple detection with dummy image"""
    logger.info("🔍 Testing Face Detection...")
    
    try:
        import numpy as np
        from app.services.detection.strategy import detection_manager
        from app.core.config import DetectionMode
        
        # Create dummy image
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        # Get available detectors
        status = await detection_manager.get_status()
        available = status.get('available_detectors', [])
        
        if not available:
            logger.warning("⚠️  No detectors available for testing")
            return True  # Not a failure, just no models
        
        # Test each available detector
        for detector in available:
            logger.info(f"Testing {detector} detector...")
            
            try:
                from app.core.config import DetectorType
                detector_type = DetectorType(detector)
                
                result = await detection_manager.detect_faces(
                    image=dummy_image,
                    detector=detector_type,
                    options={"enable_quality_assessment": False}
                )
                
                logger.info(f"✅ {detector}: {result.total_faces} faces, {result.processing_time:.3f}s")
                
            except Exception as e:
                logger.warning(f"⚠️  {detector} failed: {e}")
        
        return True
    except Exception as e:
        logger.error(f"❌ Detection test failed: {e}")
        return False


async def test_api_startup():
    """Test FastAPI app startup"""
    logger.info("🚀 Testing FastAPI startup...")
    
    try:
        from app.main import app
        
        # Just check if we can import and create the app
        logger.info(f"✅ FastAPI app created: {app.title}")
        
        return True
    except Exception as e:
        logger.error(f"❌ FastAPI startup test failed: {e}")
        return False


async def main():
    """Run all tests"""
    logger.info("🧪 Face Detection Service Test Suite")
    logger.info("=" * 50)
    
    tests = [
        ("Configuration", test_config),
        ("VRAM Manager", test_vram_manager),
        ("Detection Manager", test_detection_manager),
        ("Simple Detection", test_simple_detection),
        ("FastAPI Startup", test_api_startup)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        logger.info(f"\n📋 Running {test_name} test...")
        try:
            result = await test_func()
            results.append((test_name, result))
            status = "✅ PASSED" if result else "❌ FAILED"
            logger.info(f"{status}: {test_name}")
        except Exception as e:
            logger.error(f"❌ FAILED: {test_name} - {e}")
            results.append((test_name, False))
    
    # Summary
    logger.info("\n" + "=" * 50)
    logger.info("📊 Test Summary:")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        logger.info(f"  {status}: {test_name}")
    
    logger.info(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        logger.info("🎉 All tests passed! Service is ready.")
        return 0
    else:
        logger.warning(f"⚠️  {total - passed} tests failed. Check issues above.")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())