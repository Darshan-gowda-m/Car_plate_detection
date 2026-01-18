"""
Input Validation and Sanitization
"""
import re
import os
from pathlib import Path
from typing import Tuple, Dict, Any, Optional
import cv2
import numpy as np

class Validator:
    """Input validation utilities"""
    
    @staticmethod
    def validate_image_file(filepath: str) -> Tuple[bool, Optional[str]]:
        """Validate image file"""
        try:
            path = Path(filepath)
            
            # Check file exists
            if not path.exists():
                return False, "File does not exist"
            
            # Check file size (max 50MB)
            max_size = 50 * 1024 * 1024  # 50MB
            if path.stat().st_size > max_size:
                return False, f"File too large (max {max_size/1024/1024}MB)"
            
            # Check extension
            allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
            if path.suffix.lower() not in allowed_extensions:
                return False, f"Invalid file extension. Allowed: {allowed_extensions}"
            
            # Try to read with OpenCV
            img = cv2.imread(filepath)
            if img is None:
                return False, "Cannot read image file"
            
            # Check image dimensions
            h, w = img.shape[:2]
            if w < 50 or h < 50:
                return False, "Image too small (minimum 50x50 pixels)"
            if w > 10000 or h > 10000:
                return False, "Image too large (maximum 10000x10000 pixels)"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_image_array(image: np.ndarray) -> Tuple[bool, Optional[str]]:
        """Validate numpy image array"""
        try:
            if image is None:
                return False, "Image is None"
            
            if not isinstance(image, np.ndarray):
                return False, "Image is not a numpy array"
            
            if image.size == 0:
                return False, "Image is empty"
            
            # Check dimensions
            if len(image.shape) not in [2, 3]:
                return False, f"Invalid image shape: {image.shape}"
            
            # Check data type
            if image.dtype not in [np.uint8, np.uint16, np.float32, np.float64]:
                return False, f"Invalid image dtype: {image.dtype}"
            
            # Check values range
            if image.dtype == np.uint8:
                if np.any(image > 255) or np.any(image < 0):
                    return False, "Image values out of range [0, 255]"
            
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    @staticmethod
    def validate_plate_text(text: str) -> Tuple[bool, Optional[str]]:
        """Validate license plate text"""
        if not text or not isinstance(text, str):
            return False, "Invalid text"
        
        # Clean and normalize
        cleaned = text.strip().upper()
        
        # Check length
        if len(cleaned) < 2 or len(cleaned) > 20:
            return False, f"Text length {len(cleaned)} out of range [2, 20]"
        
        # Check for invalid characters
        # Allow alphanumeric, spaces, and common separators
        pattern = r'^[A-Z0-9\s\-\.]+$'
        if not re.match(pattern, cleaned):
            return False, "Contains invalid characters"
        
        return True, cleaned
    
    @staticmethod
    def validate_bounding_box(bbox: Any, image_shape: Tuple[int, int]) -> Tuple[bool, Optional[str]]:
        """Validate bounding box coordinates"""
        try:
            # Check type
            if not isinstance(bbox, (list, tuple)):
                return False, "Bounding box must be list or tuple"
            
            # Check length
            if len(bbox) != 4:
                return False, f"Bounding box must have 4 values, got {len(bbox)}"
            
            # Check values
            x1, y1, x2, y2 = bbox
            
            for val in [x1, y1, x2, y2]:
                if not isinstance(val, (int, float)):
                    return False, "Bounding box values must be numbers"
            
            # Check order
            if x1 >= x2:
                return False, f"x1 ({x1}) must be less than x2 ({x2})"
            if y1 >= y2:
                return False, f"y1 ({y1}) must be less than y2 ({y2})"
            
            # Check within image bounds
            h, w = image_shape[:2]
            if x1 < 0 or y1 < 0 or x2 > w or y2 > h:
                return False, f"Bounding box [{x1},{y1},{x2},{y2}] outside image [{w}x{h}]"
            
            # Check size
            bbox_w = x2 - x1
            bbox_h = y2 - y1
            min_size = 10  # pixels
            if bbox_w < min_size or bbox_h < min_size:
                return False, f"Bounding box too small: {bbox_w}x{bbox_h} (min {min_size}x{min_size})"
            
            max_size = min(w, h) * 0.8
            if bbox_w > max_size or bbox_h > max_size:
                return False, f"Bounding box too large: {bbox_w}x{bbox_h} (max {max_size})"
            
            return True, None
            
        except Exception as e:
            return False, f"Bounding box validation error: {str(e)}"
    
    @staticmethod
    def validate_confidence(confidence: Any) -> Tuple[bool, Optional[float]]:
        """Validate confidence score"""
        try:
            if not isinstance(confidence, (int, float)):
                return False, "Confidence must be a number"
            
            conf = float(confidence)
            
            if conf < 0 or conf > 1:
                return False, "Confidence must be between 0 and 1"
            
            return True, conf
            
        except Exception as e:
            return False, f"Confidence validation error: {str(e)}"