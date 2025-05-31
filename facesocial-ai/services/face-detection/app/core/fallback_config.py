"""
Fallback configuration for development without GPU
"""

import os
from typing import Dict, Any

# Attempt to import GPUtil, but don't fail if it's not there
try:
    import GPUtil
except ImportError:
    GPUtil = None

# Assuming DetectorType is defined elsewhere, e.g., in app.core.config or a similar place
# For standalone use, you might need to define it or adjust the import
# from app.core.config import DetectorType # Example import

class DetectorType: # Placeholder if not imported
    MEDIAPIPE = "mediapipe"
    YOLO = "yolo"
    INSIGHTFACE = "insightface"


def get_development_config() -> Dict[str, Any]:
    """Get development configuration with CPU fallback"""
    
    has_gpu = False
    if GPUtil:
        try:
            gpus = GPUtil.getGPUs()
            has_gpu = len(gpus) > 0
        except Exception: # Broad exception if GPUtil fails for any reason
            pass
    
    return {
        "use_gpu": has_gpu,
        "fallback_to_cpu": True,
        "default_detector": DetectorType.MEDIAPIPE if not has_gpu else DetectorType.YOLO,
        "enable_caching": True,
        "log_level": "INFO" if has_gpu else "DEBUG"
    }

def apply_fallback_config(settings):
    """Apply fallback configuration to settings object (e.g., AppSettings)."""
    dev_config = get_development_config()
    
    if not dev_config["use_gpu"]:
        print("⚠️  No GPU detected, using CPU-optimized configuration")
        
        # These attributes need to exist on the settings object
        if hasattr(settings, 'VRAM_LIMIT_MB'):
            settings.VRAM_LIMIT_MB = 0
        if hasattr(settings, 'MAX_BATCH_SIZE'):
            settings.MAX_BATCH_SIZE = 1 # Smaller batch size for CPU
        if hasattr(settings, 'PRELOAD_MODELS'):
            # Ensure PRELOAD_MODELS is a list and can be appended to or set
            if not isinstance(settings.PRELOAD_MODELS, list):
                settings.PRELOAD_MODELS = []
            
            # Add MediaPipe if not already there, or set it as the only one
            # This logic might need adjustment based on how PRELOAD_MODELS is used
            if DetectorType.MEDIAPIPE not in settings.PRELOAD_MODELS:
                 settings.PRELOAD_MODELS.append(DetectorType.MEDIAPIPE)
            # If you want to *only* preload MediaPipe on CPU:
            # settings.PRELOAD_MODELS = [DetectorType.MEDIAPIPE]

        # Example: Adjusting detector if it's part of settings
        if hasattr(settings, 'DEFAULT_DETECTOR'):
            settings.DEFAULT_DETECTOR = dev_config["default_detector"]
        
        if hasattr(settings, 'LOG_LEVEL'):
            settings.LOG_LEVEL = dev_config["log_level"]

    # You might want to return the modified settings or just modify in place
    return settings

# Example usage (assuming you have an AppSettings class like in config.py)
if __name__ == '__main__':
    class MockAppSettings:
        VRAM_LIMIT_MB = 4096
        MAX_BATCH_SIZE = 8
        PRELOAD_MODELS = ["yolo", "insightface"] # Using strings for simplicity here
        DEFAULT_DETECTOR = "yolo"
        LOG_LEVEL = "INFO"

    mock_settings = MockAppSettings()
    print(f"Before fallback: GPU: {get_development_config()['use_gpu']}, Detector: {mock_settings.DEFAULT_DETECTOR}, Preload: {mock_settings.PRELOAD_MODELS}")
    
    updated_settings = apply_fallback_config(mock_settings)
    
    print(f"After fallback: Detector: {updated_settings.DEFAULT_DETECTOR}, Preload: {updated_settings.PRELOAD_MODELS}, VRAM: {updated_settings.VRAM_LIMIT_MB}")
    print(f"Development config: {get_development_config()}")
