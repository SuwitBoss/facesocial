"""
Image processing utilities for Face Detection Service
"""

import cv2
import numpy as np
from typing import Tuple, Optional, List
from PIL import Image
import io
import base64


def resize_image(image: np.ndarray, target_size: Tuple[int, int], maintain_aspect: bool = True) -> np.ndarray:
    """Resize image to target size"""
    height, width = image.shape[:2]
    target_width, target_height = target_size
    
    if maintain_aspect:
        # Calculate scale factor
        scale = min(target_width / width, target_height / height)
        new_width = int(width * scale)
        new_height = int(height * scale)
        
        # Resize and pad if necessary
        resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        if new_width != target_width or new_height != target_height:
            # Create padded image
            padded = np.full((target_height, target_width, 3), 114, dtype=np.uint8)
            
            # Center the image
            y_offset = (target_height - new_height) // 2
            x_offset = (target_width - new_width) // 2
            padded[y_offset:y_offset + new_height, x_offset:x_offset + new_width] = resized
            
            return padded
        
        return resized
    else:
        return cv2.resize(image, target_size, interpolation=cv2.INTER_LINEAR)


def base64_to_image(base64_str: str) -> Optional[np.ndarray]:
    """Convert base64 string to numpy array"""
    try:
        # Remove data URL prefix if present
        if base64_str.startswith('data:image'):
            base64_str = base64_str.split(',')[1]
        
        # Decode base64
        image_data = base64.b64decode(base64_str)
        
        # Convert to PIL Image
        pil_image = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB
        if pil_image.mode != 'RGB':
            pil_image = pil_image.convert('RGB')
        
        # Convert to numpy array (RGB)
        rgb_array = np.array(pil_image)
        
        # Convert RGB to BGR for OpenCV
        bgr_array = cv2.cvtColor(rgb_array, cv2.COLOR_RGB2BGR)
        
        return bgr_array
        
    except Exception as e:
        print(f"Error converting base64 to image: {e}")
        return None


def image_to_base64(image: np.ndarray, format: str = 'JPEG') -> Optional[str]:
    """Convert numpy array to base64 string"""
    try:
        # Convert BGR to RGB
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        pil_image = Image.fromarray(rgb_image)
        
        # Save to bytes
        buffer = io.BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        
        # Encode to base64
        image_base64 = base64.b64encode(buffer.getvalue()).decode()
        
        return image_base64
        
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


def crop_face_region(image: np.ndarray, bbox: dict, padding: float = 0.2) -> np.ndarray:
    """Crop face region from image with optional padding"""
    x, y, width, height = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
    
    # Add padding
    pad_x = int(width * padding)
    pad_y = int(height * padding)
    
    # Calculate crop coordinates
    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))
    x2 = min(image.shape[1], int(x + width + pad_x))
    y2 = min(image.shape[0], int(y + height + pad_y))
    
    # Crop image
    cropped = image[y1:y2, x1:x2]
    
    return cropped


def validate_image_format(image: np.ndarray) -> bool:
    """Validate image format and properties"""
    if image is None or image.size == 0:
        return False
    
    # Check dimensions
    if len(image.shape) != 3 or image.shape[2] != 3:
        return False
    
    # Check image size
    height, width = image.shape[:2]
    if height < 32 or width < 32:
        return False
    
    if height > 4096 or width > 4096:
        return False
    
    return True
