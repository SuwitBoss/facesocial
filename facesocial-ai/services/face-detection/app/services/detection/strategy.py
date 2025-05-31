"""
Face Detection Strategy Pattern
Routes detection requests to appropriate models based on use case
"""

import asyncio
import time
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
from enum import Enum

from app.core.config import DetectionMode, DetectorType, settings
from app.services.gpu.vram_manager import vram_manager


@dataclass
class DetectedFace:
    """Standardized face detection result"""
    bbox: Dict[str, float]  # {x, y, width, height}
    confidence: float
    landmarks: Optional[List[Dict[str, float]]] = None
    quality_score: Optional[float] = None
    attributes: Optional[Dict[str, Any]] = None


@dataclass
class DetectionResult:
    """Complete detection result with metadata"""
    faces: List[DetectedFace]
    total_faces: int
    processing_time: float
    model_used: str
    image_info: Dict[str, Any]
    performance_metrics: Dict[str, Any]


class DetectionStrategy(ABC):
    """Abstract base class for face detection strategies"""
    
    def __init__(self, name: str):
        self.name = name
        self.logger = logging.getLogger(f"{__name__}.{name}")
        self.total_detections = 0
        self.total_processing_time = 0.0
        self.error_count = 0
    
    @abstractmethod
    async def detect_faces(
        self, 
        image: Any, 
        options: Dict[str, Any]
    ) -> DetectionResult:
        """Detect faces in image"""
        pass
    
    @abstractmethod
    async def is_available(self) -> bool:
        """Check if detector is available and ready"""
        pass
    
    @abstractmethod
    async def warm_up(self) -> bool:
        """Initialize and warm up the detector"""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """Clean shutdown of detector"""
        pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        avg_time = (self.total_processing_time / self.total_detections 
                   if self.total_detections > 0 else 0)
        
        return {
            "name": self.name,
            "total_detections": self.total_detections,
            "average_processing_time": avg_time,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_detections, 1)
        }


class YOLODetectionStrategy(DetectionStrategy):
    """YOLO-based detection strategy for high accuracy"""
    
    def __init__(self):
        super().__init__("YOLO")
        self.model_name = "yolov10n-face"
        self.model_path = settings.YOLO_MODEL_PATH
        self.is_loaded = False
        
        # YOLO-specific settings
        from app.core.config import detection_config
        self.config = detection_config.yolo
    
    async def detect_faces(
        self, 
        image: Any, 
        options: Dict[str, Any]
    ) -> DetectionResult:
        """Detect faces using YOLO model"""
        start_time = time.time()
        
        try:
            # Load model if needed
            async with vram_manager.load_model(
                self.model_name, 
                self.model_path, 
                self.config.ort_providers
            ) as session:
                
                # Import detector class
                from app.services.detection.yolo_detector import EnhancedYOLODetector
                
                detector = EnhancedYOLODetector(session)
                faces = await detector.detect_faces(image, options)
                
                processing_time = time.time() - start_time
                
                # Update statistics
                self.total_detections += 1
                self.total_processing_time += processing_time
                vram_manager.total_inferences += 1
                
                return DetectionResult(
                    faces=faces,
                    total_faces=len(faces),
                    processing_time=processing_time,
                    model_used="YOLOv10n-face",
                    image_info={"width": image.shape[1], "height": image.shape[0]},
                    performance_metrics={
                        "inference_time": processing_time,
                        "vram_usage": await vram_manager.get_status()
                    }
                )
                
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"YOLO detection failed: {e}")
            raise
    
    async def is_available(self) -> bool:
        """Check if YOLO is available"""
        try:
            import os
            if not os.path.exists(self.model_path):
                self.logger.warning(f"YOLO model not found: {self.model_path}")
                return False
            
            # Check GPU availability
            gpu_info = await vram_manager.get_gpu_memory_info()
            if gpu_info["total"] == 0:
                self.logger.warning("No GPU available for YOLO")
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error checking YOLO availability: {e}")
            return False
    
    async def warm_up(self) -> bool:
        """Warm up YOLO model"""
        try:
            import numpy as np
            
            # Create dummy image for warmup
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            async with vram_manager.load_model(
                self.model_name, 
                self.model_path, 
                self.config.ort_providers
            ) as session:
                
                from app.services.detection.yolo_detector import EnhancedYOLODetector
                detector = EnhancedYOLODetector(session)
                
                # Run warmup inferences
                for _ in range(self.config.warmup_runs):
                    await detector.detect_faces(dummy_image, {})
                
                self.is_loaded = True
                self.logger.info("YOLO model warmed up successfully")
                return True
                
        except Exception as e:
            self.logger.error(f"YOLO warmup failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown YOLO detector"""
        if self.is_loaded:
            await vram_manager.unload_model(self.model_name)
            self.is_loaded = False


class MTCNNDetectionStrategy(DetectionStrategy):
    """MTCNN-based detection strategy for balanced performance"""
    
    def __init__(self):
        super().__init__("MTCNN")
        self.detector = None
        from app.core.config import detection_config
        self.config = detection_config.mtcnn
    
    async def detect_faces(
        self, 
        image: Any, 
        options: Dict[str, Any]
    ) -> DetectionResult:
        """Detect faces using MTCNN"""
        start_time = time.time()
        
        try:
            if self.detector is None:
                await self._load_detector()
            
            from app.services.detection.mtcnn_detector import MTCNNDetector
            
            faces = await MTCNNDetector.detect_faces_static(
                self.detector, image, options
            )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.total_detections += 1
            self.total_processing_time += processing_time
            
            return DetectionResult(
                faces=faces,
                total_faces=len(faces),
                processing_time=processing_time,
                model_used="MTCNN",
                image_info={"width": image.shape[1], "height": image.shape[0]},
                performance_metrics={
                    "inference_time": processing_time,
                    "cpu_usage": self._get_cpu_usage()
                }
            )
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"MTCNN detection failed: {e}")
            raise
    
    async def _load_detector(self):
        """Load MTCNN detector"""
        try:
            from mtcnn import MTCNN
            
            self.detector = MTCNN(
                min_face_size=self.config.min_face_size,
                scale_factor=self.config.scale_factor,
                steps_threshold=self.config.steps_threshold
            )
            
            self.logger.info("MTCNN detector loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load MTCNN: {e}")
            raise
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except:
            return 0.0
    
    async def is_available(self) -> bool:
        """Check if MTCNN is available"""
        try:
            import mtcnn
            return True
        except ImportError:
            self.logger.warning("MTCNN not installed")
            return False
    
    async def warm_up(self) -> bool:
        """Warm up MTCNN model"""
        try:
            await self._load_detector()
            
            # Test with dummy image
            import numpy as np
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            result = self.detector.detect_faces(dummy_image)
            
            self.logger.info("MTCNN model warmed up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"MTCNN warmup failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown MTCNN detector"""
        self.detector = None


class MediaPipeDetectionStrategy(DetectionStrategy):
    """MediaPipe-based detection strategy for real-time performance"""
    
    def __init__(self):
        super().__init__("MediaPipe")
        self.detector = None
        from app.core.config import detection_config
        self.config = detection_config.mediapipe
    
    async def detect_faces(
        self, 
        image: Any, 
        options: Dict[str, Any]
    ) -> DetectionResult:
        """Detect faces using MediaPipe"""
        start_time = time.time()
        
        try:
            if self.detector is None:
                await self._load_detector()
            
            from app.services.detection.mediapipe_detector import MediaPipeDetector
            
            faces = await MediaPipeDetector.detect_faces_static(
                self.detector, image, options
            )
            
            processing_time = time.time() - start_time
            
            # Update statistics
            self.total_detections += 1
            self.total_processing_time += processing_time
            
            return DetectionResult(
                faces=faces,
                total_faces=len(faces),
                processing_time=processing_time,
                model_used="MediaPipe",
                image_info={"width": image.shape[1], "height": image.shape[0]},
                performance_metrics={
                    "inference_time": processing_time,
                    "cpu_usage": self._get_cpu_usage()
                }
            )
            
        except Exception as e:
            self.error_count += 1
            self.logger.error(f"MediaPipe detection failed: {e}")
            raise
    
    async def _load_detector(self):
        """Load MediaPipe detector"""
        try:
            import mediapipe as mp
            
            self.mp_face_detection = mp.solutions.face_detection
            self.detector = self.mp_face_detection.FaceDetection(
                model_selection=0,  # 0 for close-range, 1 for full-range
                min_detection_confidence=self.config.min_detection_confidence
            )
            
            self.logger.info("MediaPipe detector loaded successfully")
            
        except Exception as e:
            self.logger.error(f"Failed to load MediaPipe: {e}")
            raise
    
    def _get_cpu_usage(self) -> float:
        """Get current CPU usage"""
        try:
            import psutil
            return psutil.cpu_percent()
        except:
            return 0.0
    
    async def is_available(self) -> bool:
        """Check if MediaPipe is available"""
        try:
            import mediapipe
            return True
        except ImportError:
            self.logger.warning("MediaPipe not installed")
            return False
    
    async def warm_up(self) -> bool:
        """Warm up MediaPipe model"""
        try:
            await self._load_detector()
            
            # Test with dummy image
            import numpy as np
            dummy_image = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
            
            results = self.detector.process(dummy_image)
            
            self.logger.info("MediaPipe model warmed up successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"MediaPipe warmup failed: {e}")
            return False
    
    async def shutdown(self):
        """Shutdown MediaPipe detector"""
        if self.detector:
            self.detector.close()
        self.detector = None


class FaceDetectionManager:
    """
    Main manager that routes detection requests to appropriate strategies
    """
    
    def __init__(self):
        self.strategies = {
            DetectorType.YOLO: YOLODetectionStrategy(),
            DetectorType.MTCNN: MTCNNDetectionStrategy(),
            DetectorType.MEDIAPIPE: MediaPipeDetectionStrategy()
        }
        
        self.logger = logging.getLogger(__name__)
        self.initialized = False
    
    async def initialize(self):
        """Initialize all available strategies"""
        self.logger.info("Initializing Face Detection Manager...")
        
        # Check availability and warm up strategies
        for detector_type, strategy in self.strategies.items():
            try:
                if await strategy.is_available():
                    if detector_type in settings.PRELOAD_MODELS or detector_type == DetectorType.MEDIAPIPE:
                        await strategy.warm_up()
                        self.logger.info(f"{detector_type.value} strategy ready")
                    else:
                        self.logger.info(f"{detector_type.value} strategy available (lazy loading)")
                else:
                    self.logger.warning(f"{detector_type.value} strategy not available")
            except Exception as e:
                self.logger.error(f"Failed to initialize {detector_type.value}: {e}")
        
        self.initialized = True
        self.logger.info("Face Detection Manager initialized")
    
    async def detect_faces(
        self,
        image: Any,
        mode: DetectionMode = DetectionMode.BALANCED,
        detector: Optional[DetectorType] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> DetectionResult:
        """
        Detect faces using specified mode or detector
        """
        if not self.initialized:
            await self.initialize()
        
        options = options or {}
        
        # Determine which detector to use
        if detector:
            target_detector = detector
        else:
            from app.core.config import detection_config
            mode_config = detection_config.get_config_for_mode(mode)
            target_detector = mode_config["detector"]
            options.update(mode_config)
        
        # Try primary detector
        try:
            strategy = self.strategies[target_detector]
            
            if not await strategy.is_available():
                raise RuntimeError(f"{target_detector.value} not available")
            
            return await strategy.detect_faces(image, options)
            
        except Exception as e:
            self.logger.warning(f"Primary detector {target_detector.value} failed: {e}")
            
            # Fallback strategy if enabled
            if settings.AUTO_FALLBACK_ENABLED and not detector:
                return await self._fallback_detection(image, target_detector, options)
            else:
                raise
    
    async def _fallback_detection(
        self,
        image: Any,
        failed_detector: DetectorType,
        options: Dict[str, Any]
    ) -> DetectionResult:
        """Try fallback detectors in order"""
        
        fallback_order = [d for d in settings.FALLBACK_ORDER if d != failed_detector]
        
        for fallback_detector in fallback_order:
            try:
                strategy = self.strategies[fallback_detector]
                
                if await strategy.is_available():
                    self.logger.info(f"Falling back to {fallback_detector.value}")
                    return await strategy.detect_faces(image, options)
                    
            except Exception as e:
                self.logger.warning(f"Fallback detector {fallback_detector.value} failed: {e}")
                continue
        
        raise RuntimeError("All detection strategies failed")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get status of all detection strategies"""
        status = {
            "initialized": self.initialized,
            "available_detectors": [],
            "performance_stats": {},
            "vram_status": await vram_manager.get_status()
        }
        
        for detector_type, strategy in self.strategies.items():
            is_available = await strategy.is_available()
            if is_available:
                status["available_detectors"].append(detector_type.value)
            
            status["performance_stats"][detector_type.value] = {
                "available": is_available,
                **strategy.get_stats()
            }
        
        return status
    
    async def shutdown(self):
        """Shutdown all strategies"""
        self.logger.info("Shutting down Face Detection Manager...")
        
        for strategy in self.strategies.values():
            try:
                await strategy.shutdown()
            except Exception as e:
                self.logger.error(f"Error shutting down strategy: {e}")
        
        await vram_manager.shutdown()
        self.logger.info("Face Detection Manager shutdown complete")


# Global detection manager instance
detection_manager = FaceDetectionManager()

__all__ = [
    "DetectedFace",
    "DetectionResult", 
    "DetectionStrategy",
    "FaceDetectionManager",
    "detection_manager"
]