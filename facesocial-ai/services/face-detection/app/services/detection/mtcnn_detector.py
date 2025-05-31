"""
MTCNN Face Detection Implementation
Balanced performance detector with landmarks support
"""

import cv2
import numpy as np
import asyncio
import time
from typing import List, Dict, Any, Optional
import logging

from app.services.detection.strategy import DetectedFace
from app.core.config import settings


class MTCNNDetector:
    """
    MTCNN Face Detector with quality assessment
    CPU-based detector for balanced performance
    """
    
    def __init__(self, detector=None):
        self.detector = detector
        self.logger = logging.getLogger(__name__)
        
        # Performance tracking
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []
    
    @classmethod
    async def create_detector(cls):
        """Create MTCNN detector instance"""
        try:
            from mtcnn import MTCNN
            
            # Create MTCNN detector with optimized settings
            detector = MTCNN(
                min_face_size=20,      # Minimum face size in pixels
                scale_factor=0.709,    # Scale factor for image pyramid
                steps_threshold=[0.6, 0.7, 0.7],  # Thresholds for the 3 stages
                device='cpu'           # Force CPU usage
            )
            
            return cls(detector)
            
        except ImportError as e:
            raise RuntimeError("MTCNN not installed. Install with: pip install mtcnn") from e
        except Exception as e:
            raise RuntimeError(f"Failed to create MTCNN detector: {e}") from e
    
    @staticmethod
    async def detect_faces_static(detector, image: np.ndarray, options: Dict[str, Any]) -> List[DetectedFace]:
        """Static method for detection with existing detector"""
        mtcnn_detector = MTCNNDetector(detector)
        return await mtcnn_detector.detect_faces(image, options)
    
    async def detect_faces(self, image: np.ndarray, options: Dict[str, Any]) -> List[DetectedFace]:
        """
        Detect faces using MTCNN
        
        Args:
            image: Input image as numpy array (BGR format)
            options: Detection options
            
        Returns:
            List of detected faces with landmarks and quality scores
        """
        
        if self.detector is None:
            raise RuntimeError("MTCNN detector not initialized")
        
        # Parse options
        confidence_threshold = options.get('confidence_threshold', 0.7)
        max_faces = options.get('max_faces', 100)
        min_face_size = options.get('min_face_size', 20)
        enable_quality_assessment = options.get('enable_quality_assessment', True)
        enable_landmarks = options.get('enable_landmarks', True)
        
        # 1. Preprocessing
        start_time = time.time()
        rgb_image = await self._preprocess_image(image)
        preprocessing_time = time.time() - start_time
        self.preprocessing_times.append(preprocessing_time)
        
        # 2. MTCNN Detection
        start_time = time.time()
        detections = await self._run_mtcnn_detection(rgb_image)
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # 3. Postprocessing
        start_time = time.time()
        faces = await self._postprocess_detections(
            detections,
            image.shape,
            confidence_threshold,
            max_faces,
            min_face_size,
            enable_landmarks
        )
        postprocessing_time = time.time() - start_time
        self.postprocessing_times.append(postprocessing_time)
        
        # 4. Quality Assessment
        if enable_quality_assessment and faces:
            faces = await self._assess_face_quality(image, faces)
        
        return faces
    
    async def _preprocess_image(self, image: np.ndarray) -> np.ndarray:
        """Preprocess image for MTCNN (convert BGR to RGB)"""
        
        # MTCNN expects RGB format
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        return rgb_image
    
    async def _run_mtcnn_detection(self, rgb_image: np.ndarray) -> List[Dict[str, Any]]:
        """Run MTCNN detection in a thread to avoid blocking"""
        
        def detect_sync():
            """Synchronous detection function"""
            try:
                # MTCNN detection
                results = self.detector.detect_faces(rgb_image)
                return results if results else []
            except Exception as e:
                self.logger.error(f"MTCNN detection error: {e}")
                return []
        
        # Run in thread pool to avoid blocking the event loop
        loop = asyncio.get_event_loop()
        results = await loop.run_in_executor(None, detect_sync)
        
        return results
    
    async def _postprocess_detections(
        self,
        detections: List[Dict[str, Any]],
        image_shape: tuple,
        confidence_threshold: float,
        max_faces: int,
        min_face_size: int,
        enable_landmarks: bool
    ) -> List[DetectedFace]:
        """
        Postprocess MTCNN detections
        
        MTCNN output format:
        {
            'box': [x, y, width, height],
            'confidence': float,
            'keypoints': {
                'left_eye': (x, y),
                'right_eye': (x, y),
                'nose': (x, y),
                'mouth_left': (x, y),
                'mouth_right': (x, y)
            }
        }
        """
        
        if not detections:
            return []
        
        faces = []
        image_height, image_width = image_shape[:2]
        
        for detection in detections:
            try:
                # Extract detection data
                box = detection.get('box', [])
                confidence = detection.get('confidence', 0.0)
                keypoints = detection.get('keypoints', {})
                
                if len(box) != 4:
                    continue
                
                # Filter by confidence
                if confidence < confidence_threshold:
                    continue
                
                x, y, width, height = box
                
                # Ensure coordinates are within image bounds
                x = max(0, min(x, image_width))
                y = max(0, min(y, image_height))
                width = min(width, image_width - x)
                height = min(height, image_height - y)
                
                # Filter by minimum face size
                if width < min_face_size or height < min_face_size:
                    continue
                
                # Process landmarks if available and requested
                landmarks = None
                if enable_landmarks and keypoints:
                    landmarks = self._process_landmarks(keypoints)
                
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
                self.logger.warning(f"Error processing MTCNN detection: {e}")
                continue
        
        # Sort by confidence (highest first)
        faces.sort(key=lambda f: f.confidence, reverse=True)
        
        # Limit number of faces
        faces = faces[:max_faces]
        
        return faces
    
    def _process_landmarks(self, keypoints: Dict[str, tuple]) -> List[Dict[str, float]]:
        """Process MTCNN keypoints into standardized landmark format"""
        
        landmarks = []
        
        # MTCNN provides 5-point landmarks
        landmark_mapping = {
            'left_eye': 'left_eye',
            'right_eye': 'right_eye',
            'nose': 'nose',
            'mouth_left': 'left_mouth',
            'mouth_right': 'right_mouth'
        }
        
        for mtcnn_key, standard_key in landmark_mapping.items():
            if mtcnn_key in keypoints:
                x, y = keypoints[mtcnn_key]
                landmarks.append({
                    "name": standard_key,
                    "x": float(x),
                    "y": float(y)
                })
        
        return landmarks
    
    async def _assess_face_quality(
        self, 
        image: np.ndarray, 
        faces: List[DetectedFace]
    ) -> List[DetectedFace]:
        """Assess quality of detected faces"""
        
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
            
            face_crop = image[y1:y2, x1:x2]
            
            # Calculate quality metrics
            quality_score = await self._calculate_quality_score(face_crop, face)
            face.quality_score = quality_score
        
        return faces
    
    async def _calculate_quality_score(
        self, 
        face_crop: np.ndarray, 
        face: DetectedFace
    ) -> float:
        """
        Calculate comprehensive quality score for MTCNN detected face
        
        Takes into account:
        - Face size and resolution
        - Image sharpness
        - Landmark quality (if available)
        - Brightness and contrast
        """
        
        if face_crop.size == 0:
            return 0.0
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # 1. Size adequacy score
            height, width = face_crop.shape[:2]
            size_score = min((width * height) / (100 * 100), 1.0)  # Target: 100x100 minimum
            
            # 2. Sharpness score (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 500.0, 1.0)  # Normalize
            
            # 3. Brightness score
            mean_brightness = np.mean(gray) / 255.0
            if 0.2 <= mean_brightness <= 0.8:
                brightness_score = 1.0
            else:
                brightness_score = 1.0 - abs(mean_brightness - 0.5) * 1.5
            brightness_score = max(0.0, brightness_score)
            
            # 4. Contrast score
            contrast = np.std(gray) / 255.0
            contrast_score = min(contrast / 0.25, 1.0)
            
            # 5. Landmark quality score (if landmarks available)
            landmark_score = 1.0
            if face.landmarks and len(face.landmarks) >= 5:
                landmark_score = self._assess_landmark_quality(face.landmarks, face.bbox)
            
            # 6. MTCNN confidence as additional factor
            confidence_factor = min(face.confidence / 0.9, 1.0)  # Normalize to 0.9 max
            
            # Combined quality score with weights
            quality_score = (
                size_score * 0.2 +
                sharpness_score * 0.25 +
                brightness_score * 0.15 +
                contrast_score * 0.15 +
                landmark_score * 0.15 +
                confidence_factor * 0.1
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            self.logger.warning(f"Quality assessment failed: {e}")
            return 0.5
    
    def _assess_landmark_quality(self, landmarks: List[Dict[str, float]], bbox: Dict[str, float]) -> float:
        """Assess quality based on landmark positions"""
        
        if not landmarks or len(landmarks) < 5:
            return 0.5
        
        try:
            # Check if landmarks are within face bbox
            bbox_x, bbox_y = bbox["x"], bbox["y"]
            bbox_w, bbox_h = bbox["width"], bbox["height"]
            
            valid_landmarks = 0
            for landmark in landmarks:
                x, y = landmark["x"], landmark["y"]
                
                # Check if landmark is within bbox (with small tolerance)
                if (bbox_x - 5 <= x <= bbox_x + bbox_w + 5 and
                    bbox_y - 5 <= y <= bbox_y + bbox_h + 5):
                    valid_landmarks += 1
            
            # Calculate landmark validity score
            landmark_validity = valid_landmarks / len(landmarks)
            
            # Check landmark geometry (basic symmetry)
            symmetry_score = self._check_landmark_symmetry(landmarks)
            
            return (landmark_validity * 0.7 + symmetry_score * 0.3)
            
        except Exception as e:
            self.logger.warning(f"Landmark quality assessment failed: {e}")
            return 0.5
    
    def _check_landmark_symmetry(self, landmarks: List[Dict[str, float]]) -> float:
        """Check basic symmetry of facial landmarks"""
        
        try:
            # Find key landmarks
            left_eye = next((lm for lm in landmarks if lm["name"] == "left_eye"), None)
            right_eye = next((lm for lm in landmarks if lm["name"] == "right_eye"), None)
            nose = next((lm for lm in landmarks if lm["name"] == "nose"), None)
            
            if not all([left_eye, right_eye, nose]):
                return 0.5
            
            # Calculate eye distance and nose position relative to eyes
            eye_distance = abs(right_eye["x"] - left_eye["x"])
            eye_y_diff = abs(right_eye["y"] - left_eye["y"])
            
            # Check if eyes are roughly horizontal
            eye_level_score = 1.0 - min(eye_y_diff / max(eye_distance, 1), 1.0)
            
            # Check if nose is roughly centered between eyes
            eye_center_x = (left_eye["x"] + right_eye["x"]) / 2
            nose_center_diff = abs(nose["x"] - eye_center_x)
            nose_center_score = 1.0 - min(nose_center_diff / max(eye_distance/2, 1), 1.0)
            
            return (eye_level_score * 0.5 + nose_center_score * 0.5)
            
        except Exception as e:
            self.logger.warning(f"Symmetry check failed: {e}")
            return 0.5
    
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
            "recent_inference_times": self.inference_times[-10:] if self.inference_times else []
        }
    
    async def warmup(self, num_runs: int = 3):
        """Warm up the MTCNN detector"""
        
        if self.detector is None:
            raise RuntimeError("MTCNN detector not initialized")
        
        # Create dummy RGB image for warmup
        dummy_image = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        for i in range(num_runs):
            try:
                await self.detect_faces(dummy_image, {"enable_quality_assessment": False})
            except Exception as e:
                self.logger.warning(f"MTCNN warmup run {i+1} failed: {e}")
        
        # Clear warmup statistics
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []
        
        self.logger.info("MTCNN detector warmed up successfully")


# Utility functions for MTCNN

def check_mtcnn_installation() -> bool:
    """Check if MTCNN is properly installed"""
    try:
        import mtcnn
        from mtcnn import MTCNN
        
        # Try to create a simple detector
        detector = MTCNN()
        return True
        
    except ImportError:
        return False
    except Exception as e:
        print(f"MTCNN installation check failed: {e}")
        return False


def install_mtcnn_requirements():
    """Install MTCNN requirements if needed"""
    try:
        import subprocess
        import sys
        
        # Install MTCNN and its dependencies
        subprocess.check_call([
            sys.executable, "-m", "pip", "install", 
            "mtcnn", "tensorflow>=2.0"
        ])
        
        print("MTCNN installed successfully")
        return True
        
    except Exception as e:
        print(f"Failed to install MTCNN: {e}")
        return False


__all__ = ["MTCNNDetector", "check_mtcnn_installation", "install_mtcnn_requirements"]