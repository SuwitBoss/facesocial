"""
Performance metrics utilities
"""

import time
import psutil
from typing import Dict, Any, List
from collections import deque
import asyncio


class PerformanceTracker:
    """Track performance metrics for face detection"""
    
    def __init__(self, max_history: int = 1000):
        self.max_history = max_history
        self.detection_times = deque(maxlen=max_history)
        self.memory_usage = deque(maxlen=max_history)
        self.error_count = 0
        self.total_requests = 0
        self.start_time = time.time()
    
    def record_detection_time(self, duration: float):
        """Record detection processing time"""
        self.detection_times.append(duration)
        self.total_requests += 1
    
    def record_error(self):
        """Record an error occurrence"""
        self.error_count += 1
        self.total_requests += 1
    
    def record_memory_usage(self):
        """Record current memory usage"""
        try:
            memory_info = psutil.virtual_memory()
            self.memory_usage.append(memory_info.percent)
        except:
            pass
    
    def get_stats(self) -> Dict[str, Any]:
        """Get current performance statistics"""
        uptime = time.time() - self.start_time
        
        stats = {
            "uptime_seconds": uptime,
            "total_requests": self.total_requests,
            "error_count": self.error_count,
            "error_rate": self.error_count / max(self.total_requests, 1),
            "requests_per_second": self.total_requests / max(uptime, 1)
        }
        
        # Detection time statistics
        if self.detection_times:
            times = list(self.detection_times)
            stats.update({
                "avg_detection_time": sum(times) / len(times),
                "min_detection_time": min(times),
                "max_detection_time": max(times),
                "recent_detection_times": times[-10:]
            })
        
        # Memory statistics
        if self.memory_usage:
            memory_values = list(self.memory_usage)
            stats.update({
                "avg_memory_usage": sum(memory_values) / len(memory_values),
                "current_memory_usage": memory_values[-1] if memory_values else 0
            })
        
        return stats


# Global performance tracker
performance_tracker = PerformanceTracker()
