"""
Image utility functions
"""
import cv2
import numpy as np
from typing import Tuple, Optional, List
import base64
import io
from PIL import Image

def resize_image(image: np.ndarray, width: int = None, height: int = None) -> np.ndarray:
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    
    if width is None and height is None:
        return image
    
    if width is None:
        ratio = height / float(h)
        width = int(w * ratio)
    elif height is None:
        ratio = width / float(w)
        height = int(h * ratio)
    else:
        ratio = min(width / w, height / h)
        width = int(w * ratio)
        height = int(h * ratio)
    
    return cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)

def crop_image(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """Crop image using bounding box"""
    x1, y1, x2, y2 = bbox
    return image[y1:y2, x1:x2]

def add_margin_to_bbox(bbox: Tuple[int, int, int, int], 
                      margin: int,
                      image_shape: Tuple[int, int]) -> Tuple[int, int, int, int]:
    """Add margin to bounding box"""
    x1, y1, x2, y2 = bbox
    h, w = image_shape[:2]
    
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)
    
    return (x1, y1, x2, y2)

def draw_bounding_box(image: np.ndarray, 
                     bbox: Tuple[int, int, int, int],
                     label: str = None,
                     color: Tuple[int, int, int] = (0, 255, 0),
                     thickness: int = 2) -> np.ndarray:
    """Draw bounding box on image"""
    x1, y1, x2, y2 = bbox
    
    # Draw rectangle
    image = cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
    
    # Draw label if provided
    if label:
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        text_thickness = 1
        
        (text_width, text_height), baseline = cv2.getTextSize(
            label, font, font_scale, text_thickness
        )
        
        # Draw label background
        cv2.rectangle(
            image,
            (x1, y1 - text_height - 10),
            (x1 + text_width, y1),
            color,
            -1
        )
        
        # Draw text
        cv2.putText(
            image,
            label,
            (x1, y1 - 5),
            font,
            font_scale,
            (255, 255, 255),
            text_thickness,
            cv2.LINE_AA
        )
    
    return image

def numpy_to_base64(image: np.ndarray, format: str = 'JPEG') -> str:
    """Convert numpy array to base64 string"""
    if len(image.shape) == 3 and image.shape[2] == 3:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_rgb = image
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image_rgb)
    
    # Convert to bytes
    buffer = io.BytesIO()
    pil_image.save(buffer, format=format, quality=95)
    
    # Convert to base64
    img_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
    
    return f"data:image/{format.lower()};base64,{img_str}"

def base64_to_numpy(base64_str: str) -> Optional[np.ndarray]:
    """Convert base64 string to numpy array"""
    try:
        # Remove data URL prefix if present
        if 'base64,' in base64_str:
            base64_str = base64_str.split('base64,')[1]
        
        # Decode base64
        img_bytes = base64.b64decode(base64_str)
        
        # Convert bytes to numpy array
        nparr = np.frombuffer(img_bytes, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        return img
    except Exception as e:
        print(f"Error converting base64 to numpy: {e}")
        return None