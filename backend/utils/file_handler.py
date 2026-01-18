"""
File handling utilities
"""
import os
import shutil
import uuid
from pathlib import Path
from typing import Optional, Tuple
import cv2
import numpy as np
from PIL import Image
import io

def save_uploaded_file(file, upload_dir: str, allowed_extensions=None) -> Tuple[Optional[str], Optional[str]]:
    """Save uploaded file with validation"""
    if allowed_extensions is None:
        allowed_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif'}
    
    filename = file.filename
    if not filename:
        return None, "No filename provided"
    
    # Check extension
    ext = Path(filename).suffix.lower()
    if ext not in allowed_extensions:
        return None, f"File type {ext} not allowed. Allowed: {allowed_extensions}"
    
    # Generate unique filename
    unique_filename = f"{uuid.uuid4()}{ext}"
    save_path = Path(upload_dir) / unique_filename
    
    try:
        # Ensure directory exists
        Path(upload_dir).mkdir(parents=True, exist_ok=True)
        
        # Save file
        file.save(str(save_path))
        
        # Verify file was saved and is valid image
        if not validate_image(str(save_path)):
            # Clean up invalid file
            save_path.unlink(missing_ok=True)
            return None, "Invalid image file"
        
        return str(save_path), None
    
    except Exception as e:
        return None, f"Failed to save file: {str(e)}"

def validate_image(filepath: str) -> bool:
    """Validate that file is a valid image"""
    try:
        # Try with PIL
        with Image.open(filepath) as img:
            img.verify()
        
        # Try with OpenCV
        img = cv2.imread(filepath)
        return img is not None and img.size > 0
    
    except:
        return False

def cleanup_old_files(directory: str, max_age_days: int = 7):
    """Clean up files older than specified days"""
    import time
    current_time = time.time()
    cutoff_time = current_time - (max_age_days * 24 * 60 * 60)
    
    directory = Path(directory)
    if not directory.exists():
        return
    
    for filepath in directory.iterdir():
        if filepath.is_file():
            file_time = filepath.stat().st_mtime
            if file_time < cutoff_time:
                try:
                    filepath.unlink()
                except:
                    pass

def ensure_directory(path: str) -> bool:
    """Ensure directory exists, create if not"""
    try:
        Path(path).mkdir(parents=True, exist_ok=True)
        return True
    except:
        return False

def get_file_info(filepath: str) -> dict:
    """Get information about a file"""
    path = Path(filepath)
    
    if not path.exists():
        return None
    
    stats = path.stat()
    
    info = {
        'filename': path.name,
        'path': str(path),
        'size': stats.st_size,
        'created': stats.st_ctime,
        'modified': stats.st_mtime,
        'is_file': path.is_file(),
        'is_dir': path.is_dir()
    }
    
    # If it's an image, get dimensions
    if path.is_file() and path.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}:
        try:
            img = cv2.imread(str(path))
            if img is not None:
                info['dimensions'] = {
                    'width': img.shape[1],
                    'height': img.shape[0],
                    'channels': img.shape[2] if len(img.shape) > 2 else 1
                }
        except:
            pass
    
    return info

def convert_image_format(input_path: str, output_path: str, format: str = 'JPEG', quality: int = 95):
    """Convert image to different format"""
    try:
        with Image.open(input_path) as img:
            # Convert to RGB if needed
            if img.mode in ('RGBA', 'LA', 'P'):
                background = Image.new('RGB', img.size, (255, 255, 255))
                if img.mode == 'P':
                    img = img.convert('RGBA')
                background.paste(img, mask=img.split()[-1] if img.mode == 'RGBA' else None)
                img = background
            
            img.save(output_path, format=format, quality=quality, optimize=True)
            return True, None
    except Exception as e:
        return False, str(e)

def resize_image(input_path: str, output_path: str, max_size: Tuple[int, int] = (1920, 1080)):
    """Resize image while maintaining aspect ratio"""
    try:
        with Image.open(input_path) as img:
            img.thumbnail(max_size, Image.Resampling.LANCZOS)
            img.save(output_path, quality=95, optimize=True)
            return True, None
    except Exception as e:
        return False, str(e)

def image_to_base64(image_path: str) -> Optional[str]:
    """Convert image to base64 string"""
    try:
        with open(image_path, 'rb') as f:
            import base64
            return base64.b64encode(f.read()).decode('utf-8')
    except:
        return None

def base64_to_image(base64_str: str, output_path: str) -> bool:
    """Convert base64 string to image file"""
    try:
        import base64
        image_data = base64.b64decode(base64_str)
        
        with open(output_path, 'wb') as f:
            f.write(image_data)
        
        return True
    except:
        return False