"""
InsightFace Face Detection Implementation - Optimized Version
High-quality face detection and recognition using InsightFace models with reduced warnings
"""

import asyncio
import logging
import time
import os
from typing import List, Dict, Any, Optional

import numpy as np
import cv2

from app.services.detection.strategy import DetectedFace


class InsightFaceDetector:
    """InsightFace-based face detector implementation with optimized ONNX Runtime settings"""
    
    def __init__(self, app=None):
        self.app = app
        self.logger = logging.getLogger(__name__)
    
    @staticmethod
    async def create_detector(
        model_name: str = "buffalo_l",
        det_size: tuple = (640, 640),
        ctx_id: int = 0
    ):
        """Create and initialize InsightFace detector with minimal warnings"""
        try:
            import insightface
            import onnxruntime as ort
            
            # Suppress ONNX Runtime warnings by setting environment variables
            os.environ['ORT_DISABLE_ALL_LOGS'] = '1'
            
            # Set up optimized session options to reduce warnings
            sess_options = ort.SessionOptions()
            sess_options.log_severity_level = 4  # Only show fatal errors
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_BASIC
            sess_options.enable_mem_pattern = False
            sess_options.enable_cpu_mem_arena = False
            
            # Optimized provider options
            providers = []
            if 'CUDAExecutionProvider' in ort.get_available_providers():
                cuda_options = {
                    'device_id': str(ctx_id),
                    'arena_extend_strategy': 'kSameAsRequested',
                    'cudnn_conv_algo_search': 'HEURISTIC',
                    'do_copy_in_default_stream': '1',
                    'cudnn_conv_use_max_workspace': '1',
                    'enable_cuda_graph': '0',
                    'tunable_op_enable': '0',
                    'tunable_op_tuning_enable': '0',
                    'use_tf32': '1',
                    'gpu_mem_limit': '4294967296',  # 4GB limit
                }
                providers.append(('CUDAExecutionProvider', cuda_options))
            
            providers.append(('CPUExecutionProvider', {
                'intra_op_num_threads': 4,
                'inter_op_num_threads': 2,
            }))
            
            # Create FaceAnalysis instance with optimized settings
            app = insightface.app.FaceAnalysis(
                providers=providers,
                session_options=sess_options
            )
            
            # Prepare the detector
            app.prepare(ctx_id=ctx_id, det_size=det_size)
            
            logging.getLogger(__name__).info(
                f"✅ InsightFace detector created: {model_name}, det_size: {det_size}"
            )
            
            return app
            
        except Exception as e:
            logging.getLogger(__name__).error(f"❌ Failed to create InsightFace detector: {e}")
            raise
    
    @staticmethod
    async def detect_faces_static(
        detector, 
        image: np.ndarray, 
        options: Dict[str, Any]
    ) -> List[DetectedFace]:
        """Detect faces using InsightFace detector"""
        logger = logging.getLogger(__name__)
        
        try:
            # Get options with defaults
            confidence_threshold = options.get("confidence_threshold", 0.7)
            max_faces = options.get("max_faces", 10)
            enable_quality_assessment = options.get("enable_quality_assessment", True)
            
            # Ensure image is in correct format (BGR for InsightFace)
            if len(image.shape) == 3 and image.shape[2] == 3:
                # Assume input is RGB, convert to BGR for InsightFace
                image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                image_bgr = image
            
            # Run face detection and analysis
            start_time = time.time()
            faces = detector.get(image_bgr)
            detection_time = time.time() - start_time
            
            logger.debug(f"InsightFace detected {len(faces)} faces in {detection_time:.3f}s")
            
            # Convert InsightFace results to our standard format
            detected_faces = []
            
            for i, face in enumerate(faces):
                if i >= max_faces:
                    break
                
                # Extract bounding box
                bbox = face.bbox.astype(int)
                x, y, x2, y2 = bbox
                width = x2 - x
                height = y2 - y
                
                # Filter by confidence
                confidence = float(face.det_score)
                if confidence < confidence_threshold:
                    continue
                
                # Extract landmarks (5 point landmarks)
                landmarks = []
                if hasattr(face, 'kps') and face.kps is not None:
                    for point in face.kps:
                        landmarks.append({
                            "x": float(point[0]),
                            "y": float(point[1])
                        })
                
                # Calculate quality score if enabled
                quality_score = None
                attributes = {}
                
                if enable_quality_assessment:
                    # Basic quality assessment based on face size and confidence
                    face_area = width * height
                    image_area = image.shape[0] * image.shape[1]
                    size_ratio = face_area / image_area
                    
                    # Quality score based on confidence, size, and clarity
                    quality_score = confidence * min(1.0, size_ratio * 20)
                    
                    # Add attributes if available
                    if hasattr(face, 'age') and face.age is not None:
                        attributes['age'] = int(face.age)
                    
                    if hasattr(face, 'gender') and face.gender is not None:
                        attributes['gender'] = 'male' if face.gender == 1 else 'female'
                    
                    # Face pose estimation based on landmarks
                    if landmarks and len(landmarks) >= 5:
                        attributes['pose'] = _estimate_face_pose(landmarks)
                
                detected_face = DetectedFace(
                    bbox={
                        "x": float(x),
                        "y": float(y), 
                        "width": float(width),
                        "height": float(height)
                    },
                    confidence=confidence,
                    landmarks=landmarks if landmarks else None,
                    quality_score=quality_score,
                    attributes=attributes if attributes else None
                )
                
                detected_faces.append(detected_face)
            
            logger.info(
                f"✅ InsightFace processed {len(detected_faces)} faces "
                f"(filtered from {len(faces)} total) in {detection_time:.3f}s"
            )
            
            return detected_faces
            
        except Exception as e:
            logger.error(f"❌ InsightFace detection failed: {e}")
            raise


def _estimate_face_pose(landmarks: List[Dict[str, float]]) -> str:
    """Estimate face pose from 5-point landmarks"""
    try:
        if len(landmarks) < 5:
            return "unknown"
        
        # Get key points (assuming 5-point landmark format)
        left_eye = landmarks[0]
        right_eye = landmarks[1]
        nose = landmarks[2]
        left_mouth = landmarks[3]
        right_mouth = landmarks[4]
        
        # Calculate angles
        eye_center_x = (left_eye["x"] + right_eye["x"]) / 2
        mouth_center_x = (left_mouth["x"] + right_mouth["x"]) / 2
        
        # Horizontal offset from center
        nose_offset = abs(nose["x"] - eye_center_x)
        mouth_offset = abs(mouth_center_x - eye_center_x)
        
        # Vertical alignment
        eye_y_diff = abs(left_eye["y"] - right_eye["y"])
        
        # Simple pose classification
        if nose_offset > 10 or mouth_offset > 15:
            if nose["x"] < eye_center_x:
                return "left_profile"
            else:
                return "right_profile"
        elif eye_y_diff > 8:
            return "tilted"
        else:
            return "frontal"
            
    except Exception:
        return "unknown"
