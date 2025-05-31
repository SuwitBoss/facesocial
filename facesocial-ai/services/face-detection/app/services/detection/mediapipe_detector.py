# MediaPipe implementation
"""
MediaPipe Face Detection Implementation
Real-time optimized detector for live video streams
"""

import cv2
import numpy as np
import asyncio
import time
from typing import List, Dict, Any, Optional
import logging

from app.services.detection.strategy import DetectedFace
from app.core.config import settings


class MediaPipeDetector:
    """
    MediaPipe Face Detector optimized for real-time performance
    CPU-based detector with excellent speed for live applications
    """
    
    def __init__(self, face_detection=None, mp_face_detection=None):
        self.face_detection = face_detection
        self.mp_face_detection = mp_face_detection
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []
    
    @classmethod
    async def create_detector(cls, model_selection: int = 0, min_detection_confidence: float = 0.5):
        """
        Create MediaPipe detector instance
        
        Args:
            model_selection: 0 for short-range model (<2m), 1 for full-range model
            min_detection_confidence: Minimum confidence for detection
        """
        try:
            import mediapipe as mp
            
            mp_face_detection = mp.solutions.face_detection
            
            # Create face detection instance
            face_detection = mp_face_detection.FaceDetection(
                model_selection=model_selection,
                min_detection_confidence=min_detection_confidence
            )
            
            return cls(face_detection, mp_face_detection)
            
        except ImportError as e:
            raise RuntimeError("MediaPipe not installed. Install with: pip install mediapipe") from e
        except Exception as e:
            raise RuntimeError(f"Failed to create MediaPipe detector: {e}") from e
    
    @staticmethod
    async def detect_faces_static(detector, image: np.ndarray, options: Dict[str, Any]) -> List[DetectedFace]:
        """Static method for detection with existing detector"""
        mediapipe_detector = MediaPipeDetector(detector, None)
        return await mediapipe_detector.detect_faces(image, options)
    
    async def detect_faces(self, image: np.ndarray, options: Dict[str, Any]) -> List[DetectedFace]:
        """
        Detect faces using MediaPipe
        
        Args:
            image: Input image as numpy array (BGR format)
            options: Detection options
            
        Returns:
            List of detected faces optimized for real-time processing
        """
        
        if self.face_detection is None:
            raise RuntimeError("MediaPipe detector not initialized")
        
        # Parse options with real-time optimizations
        confidence_threshold = options.get('confidence_threshold', 0.5)
        max_faces = min(options.get('max_faces', 10), 10)  # Limit for real-time performance
        min_face_size = options.get('min_face_size', 20)
        enable_quality_assessment = options.get('enable_quality_assessment', False)  # Disabled by default for speed
        
        # 1. Preprocessing (minimal for speed)
        start_time = time.time()
        rgb_image = await self._preprocess_image(image)
        preprocessing_time = time.time() - start_time
        self.preprocessing_times.append(preprocessing_time)
        
        # 2. MediaPipe Detection
        start_time = time.time()
        results = await self._run_mediapipe_detection(rgb_image)
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # 3. Postprocessing
        start_time = time.time()
        faces = await self._postprocess_detections(
            results,
            image.shape,
            confidence_threshold,
            max_faces,
            min_face_size
        )
        postprocessing_time = time.time() - start_time
        self.postprocessing_times.append(postprocessing_time)
        
        # 4. Optional Quality Assessment (only if requested for real-time use)
        if enable_quality_assessment and faces:
            faces = await self._assess_face_quality(image, faces)
        
        return faces
    
    async def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Minimal preprocessing for MediaPipe (convert BGR to RGB)"""
        
        # MediaPipe expects RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Optional: resize for better performance if image is very large
        height, width = rgb_image.shape[:2]
        if max(height, width) > 1280:
            scale = 1280 / max(height, width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            rgb_image = cv2.resize(rgb_image, (new_width, new_height))
        
        return rgb_image
    
    async def _run_mediapipe_detection(self, rgb_image: np.ndarray):
        """Run MediaPipe detection in a thread to avoid blocking"""
        
        def detect_sync():
            """Synchronous detection function"""
            try:
                # MediaPipe detection - this is already optimized for speed
                results = self.face_detection.process(rgb_image)
                return results
            except Exception as e:
                self.logger.error(f"MediaPipe detection error: {e}")
                return None
        
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, detect_sync)
        
        return results
    
    async def _postprocess_detections(
        self,
        results,
        image_shape: tuple,
        confidence_threshold: float,
        max_faces: int,
        min_face_size: int
    ) -> List[DetectedFace]:
        """
        Postprocess MediaPipe detections
        
        MediaPipe output format:
        results.detections[i].location_data.relative_bounding_box
        - xmin, ymin, width, height (normalized 0-1)
        results.detections[i].score[0] - confidence score
        """
        
        if not results or not results.detections:
            return []
        
        faces = []
        image_height, image_width = image_shape[:2]
        
        for detection in results.detections:
            try:
                # Extract bounding box (normalized coordinates)
                bbox = detection.location_data.relative_bounding_box
                confidence = detection.score[0]
                
                # Filter by confidence
                if confidence < confidence_threshold:
                    continue
                
                # Convert normalized coordinates to pixel coordinates
                x = bbox.xmin * image_width
                y = bbox.ymin * image_height
                width = bbox.width * image_width
                height = bbox.height * image_height
                
                # Ensure coordinates are within image bounds
                x = max(0, min(x, image_width))
                y = max(0, min(y, image_height))
                width = min(width, image_width - x)
                height = min(height, image_height - y)
                
                # Filter by minimum face size
                if width < min_face_size or height < min_face_size:
                    continue
                
                # Extract key points if available
                landmarks = self._extract_mediapipe_landmarks(detection, image_width, image_height)
                
                # Create face detection
                face = DetectedFace(
                    bbox={
                        "x": float(x),
                        "y": float(y),
                        "width": float(width),
                        "height": float(height)
                    },
                    confidence=float(confidence),
                    landmarks=landmarks
                )
                
                faces.append(face)
                
            except Exception as e:
                self.logger.warning(f"Error processing MediaPipe detection: {e}")
                continue
        
        # Sort by confidence (highest first)
        faces.sort(key=lambda f: f.confidence, reverse=True)
        
        # Limit number of faces for real-time performance
        faces = faces[:max_faces]
        
        return faces
    
    def _extract_mediapipe_landmarks(self, detection, image_width: int, image_height: int) -> Optional[List[Dict[str, float]]]:
        """Extract key points from MediaPipe detection"""
        
        try:
            # MediaPipe face detection provides 6 key points
            if not hasattr(detection.location_data, 'relative_keypoints'):
                return None
            
            keypoints = detection.location_data.relative_keypoints
            if not keypoints:
                return None
            
            landmarks = []
            
            # MediaPipe key points indices:
            # 0: right eye, 1: left eye, 2: nose, 3: mouth, 4: right ear, 5: left ear
            keypoint_names = [
                "right_eye", "left_eye", "nose", "mouth", "right_ear", "left_ear"
            ]
            
            for i, keypoint in enumerate(keypoints):
                if i < len(keypoint_names):
                    landmarks.append({
                        "name": keypoint_names[i],
                        "x": float(keypoint.x * image_width),
                        "y": float(keypoint.y * image_height)
                    })
            
            return landmarks if landmarks else None
            
        except Exception as e:
            self.logger.debug(f"Failed to extract MediaPipe landmarks: {e}")
            return None
    
    async def _assess_face_quality(
        self, 
        image: np.ndarray, 
        faces: List[DetectedFace]
    ) -> List[DetectedFace]:
        """Lightweight quality assessment for real-time use"""
        
        for face in faces:
            bbox = face.bbox
            
            # Extract face region
            x1 = int(bbox["x"])
            y1 = int(bbox["y"])
            x2 = int(x1 + bbox["width"])
            y2 = int(y1 + bbox["height"])
            
            # Ensure valid crop coordinates
            x1 = max(0, x1)
            y1 = max(0, y1)
            x2 = min(image.shape[1], x2)
            y2 = min(image.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                face.quality_score = 0.0
                continue
            
            # Lightweight quality assessment
            quality_score = await self._calculate_lightweight_quality(bbox, face.confidence)
            face.quality_score = quality_score
        
        return faces
    
    async def _calculate_lightweight_quality(
        self, 
        bbox: Dict[str, float], 
        confidence: float
    ) -> float:
        """
        Lightweight quality calculation optimized for real-time use
        Focuses on size and confidence without expensive image analysis
        """
        
        try:
            # 1. Size adequacy score
            width, height = bbox["width"], bbox["height"]
            size_score = min((width * height) / (60 * 60), 1.0)  # Target: 60x60 minimum
            
            # 2. Aspect ratio score (faces should be roughly rectangular)
            aspect_ratio = width / height if height > 0 else 0
            if 0.7 <= aspect_ratio <= 1.3:  # Reasonable face aspect ratio
                aspect_score = 1.0
            else:
                aspect_score = max(0.0, 1.0 - abs(aspect_ratio - 1.0))
            
            # 3. Confidence factor
            confidence_factor = min(confidence / 0.8, 1.0)  # Normalize to 0.8 max
            
            # 4. Position score (faces near center are often better quality)
            # This is a simple heuristic for real-time use
            position_score = 1.0  # Simplified for now
            
            # Combined quality score with weights optimized for speed
            quality_score = (
                size_score * 0.4 +
                aspect_score * 0.2 +
                confidence_factor * 0.3 +
                position_score * 0.1
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.debug(f"Lightweight quality assessment failed: {e}")
            return confidence  # Fallback to confidence score
    
    def get_performance_stats(self) -> Dict[str, Any]:
        """Get performance statistics"""
        
        def safe_avg(times_list):
            return sum(times_list) / len(times_list) if times_list else 0.0
        
        return {
            "total_inferences": len(self.inference_times),
            "avg_inference_time": safe_avg(self.inference_times),
            "avg_preprocessing_time": safe_avg(self.preprocessing_times),
            "avg_postprocessing_time": safe_avg(self.postprocessing_times),
            "avg_total_time": (
                safe_avg(self.inference_times) + 
                safe_avg(self.preprocessing_times) + 
                safe_avg(self.postprocessing_times)
            ),
            "recent_inference_times": self.inference_times[-10:] if self.inference_times else [],
            "fps_estimate": 1.0 / safe_avg(self.inference_times) if safe_avg(self.inference_times) > 0 else 0
        }
    
    async def warmup(self, num_runs: int = 3):
        """Warm up the MediaPipe detector"""
        
        if self.face_detection is None:
            raise RuntimeError("MediaPipe detector not initialized")
        
        # Create dummy RGB image for warmup
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        for i in range(num_runs):
            try:
                await self.detect_faces(dummy_image, {"enable_quality_assessment": False})
            except Exception as e:
                self.logger.warning(f"MediaPipe warmup run {i+1} failed: {e}")
        
        # Clear warmup statistics
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []
        
        self.logger.info("MediaPipe detector warmed up successfully")
    
    def close(self):
        """Close MediaPipe detector and free resources"""
        if self.face_detection:
            self.face_detection.close()
            self.face_detection = None
    
    def __del__(self):
        """Destructor to ensure proper cleanup"""
        self.close()


# Real-time video processing utilities

class MediaPipeVideoProcessor:
    """
    Utility class for real-time video processing with MediaPipe
    Optimized for live video streams
    """
    
    def __init__(self, detector: MediaPipeDetector):
        self.detector = detector
        self.logger = logging.getLogger(__name__)
        
        # Frame tracking
        self.frame_count = 0
        self.fps_counter = 0
        self.last_fps_time = time.time()
        self.current_fps = 0.0
    
    async def process_frame(self, frame: np.ndarray, options: Dict[str, Any] = None) -> Dict[str, Any]:
        """
        Process a single video frame
        
        Returns:
            Dictionary with faces and performance info
        """
        
        if options is None:
            options = {"enable_quality_assessment": False}  # Disabled for real-time
        
        start_time = time.time()
        
        # Detect faces
        faces = await self.detector.detect_faces(frame, options)
        
        processing_time = time.time() - start_time
        
        # Update FPS counter
        self._update_fps_counter()
        
        # Prepare result
        result = {
            "faces": [
                {
                    "bbox": face.bbox,
                    "confidence": face.confidence,
                    "quality_score": face.quality_score
                }
                for face in faces
            ],
            "total_faces": len(faces),
            "processing_time": processing_time,
            "frame_number": self.frame_count,
            "fps": self.current_fps
        }
        
        self.frame_count += 1
        
        return result
    
    def _update_fps_counter(self):
        """Update FPS counter"""
        current_time = time.time()
        self.fps_counter += 1
        
        if current_time - self.last_fps_time >= 1.0:  # Update every second
            self.current_fps = self.fps_counter / (current_time - self.last_fps_time)
            self.fps_counter = 0
            self.last_fps_time = current_time
    
    def get_stats(self) -> Dict[str, Any]:
        """Get video processing statistics"""
        return {
            "frames_processed": self.frame_count,
            "current_fps": self.current_fps,
            "detector_stats": self.detector.get_performance_stats()
        }


# Utility functions for MediaPipe

def check_mediapipe_installation() -> bool:
    """Check if MediaPipe is properly installed"""
    try:
        import mediapipe as mp
        
        # Try to create a simple detector
        mp_face_detection = mp.solutions.face_detection
        detector = mp_face_detection.FaceDetection()
        detector.close()
        
        return True
        
    except ImportError:
        return False
    except Exception as e:
        print(f"MediaPipe installation check failed: {e}")
        return False


def install_mediapipe_requirements():
    """Install MediaPipe requirements if needed"""
    try:
        import subprocess
        import sys
        
        # Install MediaPipe
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "mediapipe>=0.10.0"
        ])
        
        print("MediaPipe installed successfully")
        return True
        
    except Exception as e:
        print(f"Failed to install MediaPipe: {e}")
        return False


def optimize_for_realtime(options: Dict[str, Any]) -> Dict[str, Any]:
    """Optimize detection options for real-time performance"""
    
    realtime_options = options.copy()
    
    # Real-time optimizations
    realtime_options.update({
        "enable_quality_assessment": False,  # Disable for speed
        "enable_landmarks": False,           # Disable for speed
        "max_faces": min(realtime_options.get("max_faces", 5), 5),  # Limit faces
        "confidence_threshold": max(realtime_options.get("confidence_threshold", 0.7), 0.7)  # Higher threshold
    })
    
    return realtime_options


__all__ = [
    "MediaPipeDetector", 
    "MediaPipeVideoProcessor",
    "check_mediapipe_installation", 
    "install_mediapipe_requirements",
    "optimize_for_realtime"
]