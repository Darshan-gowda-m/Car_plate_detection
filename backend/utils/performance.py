"""
Performance monitoring
"""
import time
import psutil
import threading
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional

class Timer:
    """Simple timer for performance measurement"""
    
    def __init__(self):
        self.start_time = None
        self.end_time = None
    
    def start(self):
        """Start timer"""
        self.start_time = time.time()
        self.end_time = None
    
    def stop(self) -> float:
        """Stop timer and return elapsed time"""
        if self.start_time is None:
            return 0.0
        
        self.end_time = time.time()
        return self.end_time - self.start_time
    
    def elapsed(self) -> float:
        """Get elapsed time without stopping"""
        if self.start_time is None:
            return 0.0
        
        if self.end_time is not None:
            return self.end_time - self.start_time
        
        return time.time() - self.start_time

class PerformanceMonitor:
    """Monitor system performance"""
    
    def __init__(self, max_samples: int = 1000):
        self.metrics = {
            'detection_time': deque(maxlen=max_samples),
            'ocr_time': deque(maxlen=max_samples),
            'total_time': deque(maxlen=max_samples),
            'detection_confidence': deque(maxlen=max_samples),
            'ocr_confidence': deque(maxlen=max_samples)
        }
        
        self.lock = threading.Lock()
    
    def record_detection(self, time_taken: float, confidence: float):
        """Record detection performance"""
        with self.lock:
            self.metrics['detection_time'].append(time_taken)
            self.metrics['detection_confidence'].append(confidence)
    
    def record_ocr(self, time_taken: float, confidence: float):
        """Record OCR performance"""
        with self.lock:
            self.metrics['ocr_time'].append(time_taken)
            self.metrics['ocr_confidence'].append(confidence)
    
    def record_total(self, time_taken: float):
        """Record total processing time"""
        with self.lock:
            self.metrics['total_time'].append(time_taken)
    
    def get_statistics(self) -> Dict:
        """Get performance statistics"""
        with self.lock:
            stats = {}
            
            for metric_name, values in self.metrics.items():
                if values:
                    stats[metric_name] = {
                        'count': len(values),
                        'avg': sum(values) / len(values),
                        'min': min(values),
                        'max': max(values),
                        'latest': values[-1] if values else 0
                    }
                else:
                    stats[metric_name] = {
                        'count': 0,
                        'avg': 0,
                        'min': 0,
                        'max': 0,
                        'latest': 0
                    }
            
            return stats
    
    def reset(self):
        """Reset all metrics"""
        with self.lock:
            for metric in self.metrics.values():
                metric.clear()

def get_system_metrics() -> Dict:
    """Get current system metrics"""
    try:
        cpu_percent = psutil.cpu_percent(interval=1)
        memory = psutil.virtual_memory()
        disk = psutil.disk_usage('/')
        
        return {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_mb': memory.available / (1024 * 1024),
            'memory_total_mb': memory.total / (1024 * 1024),
            'disk_percent': disk.percent,
            'disk_free_gb': disk.free / (1024 * 1024 * 1024),
            'disk_total_gb': disk.total / (1024 * 1024 * 1024),
            'timestamp': datetime.utcnow().isoformat()
        }
    except Exception as e:
        return {'error': str(e)}