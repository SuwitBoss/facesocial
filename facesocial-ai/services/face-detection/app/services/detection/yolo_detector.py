"""
YOLOv10n Face Detection Implementation
High-accuracy face detection with GPU optimization and quality assessment
"""

import cv2
import numpy as np
import asyncio
import time
from typing import List, Dict, Any, Tuple, Optional
import onnxruntime as ort

from app.services.detection.strategy import DetectedFace
from app.core.config import settings


class EnhancedYOLODetector:
    """
    Enhanced YOLO Face Detector with quality assessment and optimization
    """
    
    def __init__(self, session: ort.InferenceSession):
        self.session = session
        self.input_size = (640, 640)
        self.confidence_threshold = 0.25
        self.iou_threshold = 0.45
        
        # Get model input/output info
        self.input_name = self.session.get_inputs()[0].name
        self.output_names = [output.name for output in self.session.get_outputs()]
        
        # Performance tracking
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []
    
    async def detect_faces(
        self, 
        image: np.ndarray, 
        options: Dict[str, Any]
    ) -> List[DetectedFace]:
        """
        Detect faces in image with quality assessment
        
        Args:
            image: Input image as numpy array (BGR format)
            options: Detection options
            
        Returns:
            List of detected faces with quality scores
        """
        
        # Parse options
        confidence_threshold = options.get('confidence_threshold', self.confidence_threshold)
        max_faces = options.get('max_faces', 100)
        min_face_size = options.get('min_face_size', 40)
        enable_quality_assessment = options.get('enable_quality_assessment', True)
        
        # 1. Preprocessing
        start_time = time.time()
        input_tensor, scale_factor, pad_info = await self._preprocess_image(image)
        preprocessing_time = time.time() - start_time
        self.preprocessing_times.append(preprocessing_time)
        
        # 2. Inference
        start_time = time.time()
        outputs = await self._run_inference(input_tensor)
        inference_time = time.time() - start_time
        self.inference_times.append(inference_time)
        
        # 3. Postprocessing
        start_time = time.time()
        faces = await self._postprocess_detections(
            outputs, 
            scale_factor, 
            pad_info, 
            image.shape,
            confidence_threshold,
            max_faces,
            min_face_size
        )
        postprocessing_time = time.time() - start_time
        self.postprocessing_times.append(postprocessing_time)
        
        # 4. Quality Assessment
        if enable_quality_assessment and faces:
            faces = await self._assess_face_quality(image, faces)
        
        return faces
    
    async def _preprocess_image(self, image: np.ndarray) -> Tuple[np.ndarray, float, Dict[str, int]]:
        """
        Preprocess image for YOLO inference
        
        Returns:
            input_tensor: Preprocessed tensor for model
            scale_factor: Scaling factor applied
            pad_info: Padding information for coordinate conversion
        """
        
        original_height, original_width = image.shape[:2]
        target_width, target_height = self.input_size
        
        # Calculate scale factor to maintain aspect ratio
        scale = min(target_width / original_width, target_height / original_height)
        
        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image
        resized_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # Create padded image
        padded_image = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
        
        # Calculate padding offsets
        pad_x = (target_width - new_width) // 2
        pad_y = (target_height - new_height) // 2
        
        # Place resized image in padded image
        padded_image[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized_image
        
        # Convert to RGB and normalize
        rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        normalized_image = rgb_image.astype(np.float32) / 255.0
        
        # Convert to NCHW format (batch, channels, height, width)
        input_tensor = np.transpose(normalized_image, (2, 0, 1))
        input_tensor = np.expand_dims(input_tensor, axis=0)
        
        # Return preprocessing info
        pad_info = {
            "pad_x": pad_x,
            "pad_y": pad_y,
            "new_width": new_width,
            "new_height": new_height
        }
        
        return input_tensor, scale, pad_info
    
    async def _run_inference(self, input_tensor: np.ndarray) -> List[np.ndarray]:
        """Run ONNX inference"""
        
        try:
            # Run inference
            outputs = self.session.run(self.output_names, {self.input_name: input_tensor})
            return outputs
            
        except Exception as e:
            raise RuntimeError(f"YOLO inference failed: {e}")
    
    async def _postprocess_detections(
        self,
        outputs: List[np.ndarray],
        scale_factor: float,
        pad_info: Dict[str, int],
        original_shape: Tuple[int, int, int],
        confidence_threshold: float,
        max_faces: int,
        min_face_size: int
    ) -> List[DetectedFace]:
        """
        Postprocess YOLO outputs to extract face detections
        
        YOLOv10 output format: [batch, num_detections, 5]
        Each detection: [x1, y1, x2, y2, confidence]
        """
        
        if not outputs or len(outputs) == 0:
            return []
        
        # Get detections (assuming first output contains detections)
        detections = outputs[0]  # Shape: [1, N, 5]
        
        if len(detections.shape) != 3:
            return []
        
        detections = detections[0]  # Remove batch dimension: [N, 5]
        
        faces = []
        original_height, original_width = original_shape[:2]
        
        for detection in detections:
            if len(detection) < 5:
                continue
                
            x1, y1, x2, y2, confidence = detection[:5]
            
            # Filter by confidence threshold
            if confidence < confidence_threshold:
                continue
            
            # Convert coordinates back to original image space
            # Remove padding
            x1 = x1 - pad_info["pad_x"]
            y1 = y1 - pad_info["pad_y"]
            x2 = x2 - pad_info["pad_x"]
            y2 = y2 - pad_info["pad_y"]
            
            # Scale back to original size
            x1 = x1 / scale_factor
            y1 = y1 / scale_factor
            x2 = x2 / scale_factor
            y2 = y2 / scale_factor
            
            # Ensure coordinates are within image bounds
            x1 = max(0, min(x1, original_width))
            y1 = max(0, min(y1, original_height))
            x2 = max(0, min(x2, original_width))
            y2 = max(0, min(y2, original_height))
            
            # Calculate width and height
            width = x2 - x1
            height = y2 - y1
            
            # Filter by minimum face size
            if width < min_face_size or height < min_face_size:
                continue
            
            # Create face detection
            face = DetectedFace(
                bbox={
                    "x": float(x1),
                    "y": float(y1),
                    "width": float(width),
                    "height": float(height)
                },
                confidence=float(confidence)
            )
            
            faces.append(face)
        
        # Apply Non-Maximum Suppression
        faces = await self._apply_nms(faces, self.iou_threshold)
        
        # Limit number of faces
        faces = faces[:max_faces]
        
        return faces
    
    async def _apply_nms(self, faces: List[DetectedFace], iou_threshold: float) -> List[DetectedFace]:
        """Apply Non-Maximum Suppression to remove overlapping detections"""
        
        if len(faces) <= 1:
            return faces
        
        # Convert to format for OpenCV NMS
        boxes = []
        confidences = []
        
        for face in faces:
            bbox = face.bbox
            boxes.append([bbox["x"], bbox["y"], bbox["width"], bbox["height"]])
            confidences.append(face.confidence)
        
        boxes = np.array(boxes, dtype=np.float32)
        confidences = np.array(confidences, dtype=np.float32)
        
        # Apply NMS
        indices = cv2.dnn.NMSBoxes(
            boxes.tolist(),
            confidences.tolist(),
            score_threshold=self.confidence_threshold,
            nms_threshold=iou_threshold
        )
        
        # Extract kept faces
        if len(indices) > 0:
            indices = indices.flatten()
            return [faces[i] for i in indices]
        else:
            return []
    
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
            quality_score = await self._calculate_quality_score(face_crop)
            face.quality_score = quality_score
        
        return faces
    
    async def _calculate_quality_score(self, face_crop: np.ndarray) -> float:
        """
        Calculate comprehensive quality score for face crop
        
        Factors:
        - Sharpness (edge content)
        - Size adequacy
        - Brightness/contrast
        """
        
        if face_crop.size == 0:
            return 0.0
        
        try:
            # Convert to grayscale for analysis
            gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
            
            # 1. Sharpness score (Laplacian variance)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            sharpness_score = min(laplacian_var / 1000.0, 1.0)  # Normalize
            
            # 2. Size adequacy score
            height, width = face_crop.shape[:2]
            size_score = min((width * height) / (80 * 80), 1.0)  # Target: 80x80 minimum
            
            # 3. Brightness score
            mean_brightness = np.mean(gray) / 255.0
            # Optimal brightness is around 0.3-0.7
            if 0.3 <= mean_brightness <= 0.7:
                brightness_score = 1.0
            else:
                brightness_score = 1.0 - abs(mean_brightness - 0.5) * 2
            brightness_score = max(0.0, brightness_score)
            
            # 4. Contrast score
            contrast = np.std(gray) / 255.0
            contrast_score = min(contrast / 0.3, 1.0)  # Target: decent contrast
            
            # Combined quality score with weights
            quality_score = (
                sharpness_score * 0.4 +
                size_score * 0.3 +
                brightness_score * 0.2 +
                contrast_score * 0.1
            )
            
            return min(max(quality_score, 0.0), 1.0)
            
        except Exception as e:
            # If quality assessment fails, return neutral score
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
        """Warm up the model with dummy inputs"""
        
        # Create dummy input
        dummy_input = np.random.randint(0, 255, (640, 640, 3), dtype=np.uint8)
        
        for i in range(num_runs):
            try:
                await self.detect_faces(dummy_input, {})
            except Exception as e:
                print(f"Warmup run {i+1} failed: {e}")
        
        # Clear warmup statistics
        self.inference_times = []
        self.preprocessing_times = []
        self.postprocessing_times = []


# Utility functions for YOLO detector

def download_yolo_model(model_path: str, model_url: str) -> bool:
    """Download YOLO model if not exists"""
    
    import os
    import requests
    from pathlib import Path
    
    if os.path.exists(model_path):
        return True
    
    try:
        print(f"Downloading YOLO model to {model_path}...")
        
        # Create directory if not exists
        Path(model_path).parent.mkdir(parents=True, exist_ok=True)
        
        # Download model
        response = requests.get(model_url, stream=True)
        response.raise_for_status()
        
        with open(model_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        
        print(f"Model downloaded successfully to {model_path}")
        return True
        
    except Exception as e:
        print(f"Failed to download model: {e}")
        return False


def verify_yolo_model(model_path: str) -> bool:
    """Verify YOLO model file integrity"""
    
    import os
    
    try:
        if not os.path.exists(model_path):
            return False
        
        # Check file size (should be > 1MB for a real model)
        file_size = os.path.getsize(model_path)
        if file_size < 1024 * 1024:  # 1MB
            return False
        
        # Try to load with ONNX Runtime
        session = ort.InferenceSession(model_path, providers=['CPUExecutionProvider'])
        
        # Check input/output shapes
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        
        if len(inputs) != 1 or len(outputs) == 0:
            return False
        
        print(f"YOLO model verified: {model_path}")
        print(f"Input: {inputs[0].name} {inputs[0].shape}")
        print(f"Outputs: {[f'{o.name} {o.shape}' for o in outputs]}")
        
        return True
        
    except Exception as e:
        print(f"Model verification failed: {e}")
        return False


__all__ = ["EnhancedYOLODetector", "download_yolo_model", "verify_yolo_model"]