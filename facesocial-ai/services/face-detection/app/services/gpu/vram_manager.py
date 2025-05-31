"""
VRAM Management System for Face Detection Service
Handles GPU memory allocation and model loading/unloading for 6GB VRAM limit
"""

import asyncio
import time
import logging
from typing import Dict, Optional, Any, List
from contextlib import asynccontextmanager
from dataclasses import dataclass
import psutil
import GPUtil
import onnxruntime as ort
from app.core.config import settings


@dataclass
class ModelInfo:
    """Information about a loaded model"""
    session: Optional[ort.InferenceSession]
    memory_usage_mb: float
    load_time: float
    last_used: float
    use_count: int


class VRAMManager:
    """
    Manages VRAM allocation for AI models with 6GB limit
    Features:
    - Dynamic model loading/unloading
    - Memory monitoring
    - Automatic cleanup
    - Performance optimization
    """
    
    def __init__(self, memory_limit_mb: int = 4800):
        self.memory_limit_mb = memory_limit_mb  # 80% of 6GB
        self.loaded_models: Dict[str, ModelInfo] = {}
        self.loading_locks: Dict[str, asyncio.Lock] = {}
        self.cleanup_task: Optional[asyncio.Task] = None
        self.logger = logging.getLogger(__name__)
          # Performance tracking
        self.total_inferences = 0
        self.memory_warnings = 0
        self.model_swaps = 0
        
        # Don't start cleanup task immediately - wait for event loop
    
    async def get_gpu_memory_info(self) -> Dict[str, float]:
        """Get current GPU memory usage with improved detection"""
        try:
            # Try pynvml first (more reliable)
            try:
                import pynvml
                pynvml.nvmlInit()
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                info = pynvml.nvmlDeviceGetMemoryInfo(handle)
                
                return {
                    "total": info.total / (1024**2),  # Convert to MB
                    "used": info.used / (1024**2),
                    "free": info.free / (1024**2),
                    "utilization": (info.used / info.total) * 100
                }
            except Exception as pynvml_error:
                self.logger.debug(f"pynvml failed, trying GPUtil: {pynvml_error}")
                # Fallback to GPUtil
                gpus = GPUtil.getGPUs()
                if gpus:
                    gpu = gpus[0]
                    return {
                        "total": gpu.memoryTotal,
                        "used": gpu.memoryUsed, 
                        "free": gpu.memoryFree,
                        "utilization": gpu.memoryUtil * 100
                    }
                else:
                    self.logger.debug("No GPUs found via GPUtil")
        except Exception as e:
            self.logger.warning(f"Failed to get GPU memory info: {e}")
        
        # Return defaults if no GPU detected
        self.logger.info("No GPU detected, using CPU fallback values")
        return {"total": 0, "used": 0, "free": 0, "utilization": 0}
    
    async def estimate_model_memory(self, model_path: str) -> float:
        """Estimate memory usage for a model (in MB)"""
        try:
            # Get file size as baseline
            import os
            file_size_mb = os.path.getsize(model_path) / (1024 * 1024)
            
            # Estimate runtime memory (usually 1.5-2x file size)
            estimated_memory = file_size_mb * 1.8
            
            # Add buffer for CUDA context and operations
            estimated_memory += 200  # 200MB buffer
            
            return estimated_memory
        except Exception as e:
            self.logger.warning(f"Failed to estimate memory for {model_path}: {e}")
            return 800  # Default estimate for YOLO
    
    async def can_load_model(self, model_path: str) -> bool:
        """Check if we have enough memory to load a model"""
        estimated_memory = await self.estimate_model_memory(model_path)
        current_usage = sum(model.memory_usage_mb for model in self.loaded_models.values())
        
        available_memory = self.memory_limit_mb - current_usage
        
        if estimated_memory > available_memory:
            self.logger.info(f"Need to free {estimated_memory - available_memory:.1f}MB for new model")
            return False
        
        return True
    
    async def free_memory_for_model(self, required_memory_mb: float) -> bool:
        """Free memory by unloading least recently used models"""
        current_usage = sum(model.memory_usage_mb for model in self.loaded_models.values())
        available_memory = self.memory_limit_mb - current_usage
        
        if available_memory >= required_memory_mb:
            return True
        
        memory_to_free = required_memory_mb - available_memory
        
        # Sort models by last used time (LRU)
        models_by_usage = sorted(
            self.loaded_models.items(),
            key=lambda x: x[1].last_used
        )
        
        freed_memory = 0
        for model_name, model_info in models_by_usage:
            if freed_memory >= memory_to_free:
                break
            
            self.logger.info(f"Unloading model {model_name} to free memory")
            await self.unload_model(model_name)
            freed_memory += model_info.memory_usage_mb
            self.model_swaps += 1
        
        return freed_memory >= memory_to_free
    
    @asynccontextmanager
    async def load_model(self, model_name: str, model_path: str, providers: List[Any]):
        """
        Context manager for loading and using a model
        Automatically handles memory management
        """
        session = None
        try:
            # Get or create lock for this model
            if model_name not in self.loading_locks:
                self.loading_locks[model_name] = asyncio.Lock()
            
            async with self.loading_locks[model_name]:
                # Check if model is already loaded
                if model_name in self.loaded_models:
                    session = self.loaded_models[model_name].session
                    self.loaded_models[model_name].last_used = time.time()
                    self.loaded_models[model_name].use_count += 1
                    yield session
                    return
                
                # Check memory availability
                estimated_memory = await self.estimate_model_memory(model_path)
                
                if not await self.can_load_model(model_path):
                    if not await self.free_memory_for_model(estimated_memory):
                        raise RuntimeError(f"Cannot free enough memory for model {model_name}")
                
                # Load the model
                start_time = time.time()
                session = await self._create_ort_session(model_path, providers)
                load_time = time.time() - start_time
                
                # Measure actual memory usage
                memory_after = await self.get_gpu_memory_info()
                actual_memory = estimated_memory  # Use estimate for now
                
                # Store model info
                self.loaded_models[model_name] = ModelInfo(
                    session=session,
                    memory_usage_mb=actual_memory,
                    load_time=load_time,
                    last_used=time.time(),
                    use_count=1
                )
                
                self.logger.info(f"Loaded model {model_name} in {load_time:.2f}s, "
                               f"using ~{actual_memory:.1f}MB")
                
                yield session
                
        except Exception as e:
            self.logger.error(f"Failed to load model {model_name}: {e}")
            raise
        finally:
            # Update last used time
            if model_name in self.loaded_models:
                self.loaded_models[model_name].last_used = time.time()
    
    async def _create_ort_session(self, model_path: str, providers: List[Any]) -> ort.InferenceSession:
        """Create ONNX Runtime session with optimized settings"""
        
        # Session options for memory optimization
        session_options = ort.SessionOptions()
        session_options.enable_mem_pattern = True
        session_options.enable_cpu_mem_arena = True
        session_options.enable_mem_reuse = True
        session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        session_options.inter_op_num_threads = 1
        session_options.intra_op_num_threads = 0
        
        # Create session
        session = ort.InferenceSession(
            model_path,
            sess_options=session_options,
            providers=providers
        )
        
        return session
    
    async def unload_model(self, model_name: str) -> bool:
        """Manually unload a model to free memory"""
        if model_name not in self.loaded_models:
            return False
        
        try:
            model_info = self.loaded_models[model_name]
            
            # Delete the session
            if model_info.session:
                del model_info.session
            
            # Remove from tracking            del self.loaded_models[model_name]
            
            self.logger.info(f"Unloaded model {model_name}, "
                           f"freed ~{model_info.memory_usage_mb:.1f}MB")
            
            # Force garbage collection
            import gc
            gc.collect()
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to unload model {model_name}: {e}")
            return False
    
    def _start_cleanup_task(self):
        """Start background task for automatic cleanup"""
        try:
            # Only create task if there's a running event loop
            loop = asyncio.get_running_loop()
            if self.cleanup_task is None or self.cleanup_task.done():
                self.cleanup_task = asyncio.create_task(self._cleanup_loop())
        except RuntimeError:
            # No event loop running, will start later when needed
            self.logger.debug("No event loop running, cleanup task will start when needed")
    
    async def _cleanup_loop(self):
        """Background loop for cleaning up unused models"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                await self._cleanup_unused_models()
                await self._check_memory_health()
            except asyncio.CancelledError:
                break
            except Exception as e:
                self.logger.error(f"Error in cleanup loop: {e}")
    
    async def _cleanup_unused_models(self):
        """Remove models that haven't been used recently"""
        current_time = time.time()
        inactive_threshold = 300  # 5 minutes
        
        models_to_remove = []
        
        for model_name, model_info in self.loaded_models.items():
            if current_time - model_info.last_used > inactive_threshold:
                models_to_remove.append(model_name)
        
        for model_name in models_to_remove:
            self.logger.info(f"Auto-unloading inactive model: {model_name}")
            await self.unload_model(model_name)
    
    async def _check_memory_health(self):
        """Check memory usage and warn if approaching limits"""
        current_usage = sum(model.memory_usage_mb for model in self.loaded_models.values())
        usage_percentage = (current_usage / self.memory_limit_mb) * 100
        
        if usage_percentage > 90:
            self.memory_warnings += 1
            self.logger.warning(f"High memory usage: {usage_percentage:.1f}% "
                              f"({current_usage:.1f}MB / {self.memory_limit_mb}MB)")
    
    async def get_status(self) -> Dict[str, Any]:
        """Get current VRAM manager status"""
        current_usage = sum(model.memory_usage_mb for model in self.loaded_models.values())
        gpu_info = await self.get_gpu_memory_info()
        
        return {
            "loaded_models": len(self.loaded_models),
            "memory_usage_mb": current_usage,
            "memory_limit_mb": self.memory_limit_mb,
            "memory_utilization": (current_usage / self.memory_limit_mb) * 100,
            "gpu_memory": gpu_info,
            "total_inferences": self.total_inferences,
            "memory_warnings": self.memory_warnings,
            "model_swaps": self.model_swaps,
            "models": {
                name: {
                    "memory_mb": info.memory_usage_mb,
                    "last_used": info.last_used,
                    "use_count": info.use_count,
                    "load_time": info.load_time
                }
                for name, info in self.loaded_models.items()
            }
        }
    
    async def shutdown(self):
        """Clean shutdown of VRAM manager"""
        if self.cleanup_task and not self.cleanup_task.done():
            self.cleanup_task.cancel()
        
        # Unload all models
        model_names = list(self.loaded_models.keys())
        for model_name in model_names:
            await self.unload_model(model_name)
        
        self.logger.info("VRAM Manager shutdown complete")


# Global VRAM manager instance
vram_manager = VRAMManager(memory_limit_mb=settings.VRAM_LIMIT_MB)

__all__ = ["VRAMManager", "vram_manager", "ModelInfo"]