"""
Caching system for improved performance
"""
import time
import hashlib
import json
import pickle
from typing import Any, Optional, Callable
from pathlib import Path
from loguru import logger

class CacheManager:
    """File-based caching system"""
    
    def __init__(self, cache_dir: str = '.cache', ttl: int = 3600):
        self.cache_dir = Path(cache_dir)
        self.ttl = ttl  # Time to live in seconds
        
        # Ensure cache directory exists
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Clean old cache entries on startup
        self.cleanup()
    
    def _get_cache_key(self, *args, **kwargs) -> str:
        """Generate cache key from function arguments"""
        # Create a string representation of arguments
        arg_str = str(args) + str(sorted(kwargs.items()))
        
        # Generate MD5 hash
        return hashlib.md5(arg_str.encode()).hexdigest()
    
    def _get_cache_path(self, key: str) -> Path:
        """Get file path for cache key"""
        return self.cache_dir / f"{key}.cache"
    
    def get(self, key: str) -> Optional[Any]:
        """Get value from cache"""
        cache_path = self._get_cache_path(key)
        
        if not cache_path.exists():
            return None
        
        try:
            # Check if cache is expired
            if self._is_expired(cache_path):
                cache_path.unlink()
                return None
            
            # Load cached data
            with open(cache_path, 'rb') as f:
                data = pickle.load(f)
            
            # Check TTL
            if time.time() > data.get('expires_at', 0):
                cache_path.unlink()
                return None
            
            logger.debug(f"Cache hit for key: {key}")
            return data.get('value')
            
        except Exception as e:
            logger.warning(f"Cache read error for {key}: {e}")
            # Remove corrupted cache file
            try:
                cache_path.unlink()
            except:
                pass
            return None
    
    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """Set value in cache"""
        try:
            cache_path = self._get_cache_path(key)
            
            # Prepare cache data
            cache_data = {
                'key': key,
                'value': value,
                'created_at': time.time(),
                'expires_at': time.time() + (ttl or self.ttl),
                'version': '1.0'
            }
            
            # Save to file
            with open(cache_path, 'wb') as f:
                pickle.dump(cache_data, f)
            
            logger.debug(f"Cache set for key: {key}")
            return True
            
        except Exception as e:
            logger.error(f"Cache write error for {key}: {e}")
            return False
    
    def delete(self, key: str) -> bool:
        """Delete value from cache"""
        try:
            cache_path = self._get_cache_path(key)
            if cache_path.exists():
                cache_path.unlink()
            return True
        except:
            return False
    
    def clear(self) -> bool:
        """Clear all cache"""
        try:
            for cache_file in self.cache_dir.glob('*.cache'):
                cache_file.unlink()
            return True
        except Exception as e:
            logger.error(f"Cache clear error: {e}")
            return False
    
    def cleanup(self) -> int:
        """Clean up expired cache entries"""
        deleted = 0
        
        for cache_file in self.cache_dir.glob('*.cache'):
            try:
                if self._is_expired(cache_file):
                    cache_file.unlink()
                    deleted += 1
            except:
                pass
        
        if deleted:
            logger.info(f"Cleaned up {deleted} expired cache entries")
        
        return deleted
    
    def _is_expired(self, cache_path: Path) -> bool:
        """Check if cache file is expired"""
        try:
            # Check file modification time
            file_age = time.time() - cache_path.stat().st_mtime
            return file_age > self.ttl
        except:
            return True
    
    def get_stats(self) -> dict:
        """Get cache statistics"""
        cache_files = list(self.cache_dir.glob('*.cache'))
        
        total_size = sum(f.stat().st_size for f in cache_files)
        
        # Analyze age distribution
        now = time.time()
        age_distribution = {
            '<1h': 0,
            '1-24h': 0,
            '>24h': 0
        }
        
        for cache_file in cache_files:
            try:
                age = now - cache_file.stat().st_mtime
                
                if age < 3600:
                    age_distribution['<1h'] += 1
                elif age < 86400:
                    age_distribution['1-24h'] += 1
                else:
                    age_distribution['>24h'] += 1
            except:
                pass
        
        return {
            'total_entries': len(cache_files),
            'total_size_bytes': total_size,
            'total_size_mb': total_size / (1024 * 1024),
            'age_distribution': age_distribution,
            'cache_dir': str(self.cache_dir)
        }


def cached(ttl: int = 3600, cache_dir: str = '.cache'):
    """
    Decorator for caching function results
    
    Args:
        ttl: Time to live in seconds
        cache_dir: Cache directory
    
    Returns:
        Decorated function
    """
    cache_manager = CacheManager(cache_dir, ttl)
    
    def decorator(func: Callable):
        def wrapper(*args, **kwargs):
            # Generate cache key
            cache_key = cache_manager._get_cache_key(
                func.__name__, 
                *args, 
                **kwargs
            )
            
            # Try to get from cache
            cached_result = cache_manager.get(cache_key)
            
            if cached_result is not None:
                return cached_result
            
            # Execute function
            result = func(*args, **kwargs)
            
            # Store in cache
            cache_manager.set(cache_key, result, ttl)
            
            return result
        
        return wrapper
    
    return decorator


class ImageCache:
    """Specialized cache for images and detection results"""
    
    def __init__(self, cache_dir: str = '.image_cache'):
        self.cache_manager = CacheManager(cache_dir, ttl=86400)  # 24 hours
        
    def get_detection(self, image_path: str, options: dict) -> Optional[dict]:
        """Get cached detection results"""
        cache_key = self._get_image_key(image_path, options, 'detection')
        return self.cache_manager.get(cache_key)
    
    def set_detection(self, image_path: str, options: dict, results: dict):
        """Cache detection results"""
        cache_key = self._get_image_key(image_path, options, 'detection')
        self.cache_manager.set(cache_key, results)
    
    def get_ocr(self, image_path: str, options: dict, engine: str) -> Optional[dict]:
        """Get cached OCR results"""
        cache_key = self._get_image_key(image_path, options, f'ocr_{engine}')
        return self.cache_manager.get(cache_key)
    
    def set_ocr(self, image_path: str, options: dict, engine: str, results: dict):
        """Cache OCR results"""
        cache_key = self._get_image_key(image_path, options, f'ocr_{engine}')
        self.cache_manager.set(cache_key, results)
    
    def _get_image_key(self, image_path: str, options: dict, suffix: str) -> str:
        """Generate cache key for image"""
        import os
        
        # Get file stats for versioning
        try:
            stat = os.stat(image_path)
            file_info = f"{stat.st_size}_{stat.st_mtime}"
        except:
            file_info = "unknown"
        
        # Combine all information
        key_data = {
            'path': image_path,
            'file_info': file_info,
            'options': options,
            'suffix': suffix
        }
        
        return hashlib.md5(json.dumps(key_data, sort_keys=True).encode()).hexdigest()