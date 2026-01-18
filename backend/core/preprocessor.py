"""
Advanced Image Preprocessing for OCR Optimization
"""
import cv2
import numpy as np
from typing import Dict, Optional, List
import logging

logger = logging.getLogger(__name__)

class AdvancedImagePreprocessor:
    """Advanced image preprocessing with multiple techniques"""
    
    def __init__(self):
        self.methods = {
            'original': self._original,
            'grayscale': self._grayscale,
            'enhanced': self._enhance_contrast,
            'denoised': self._denoise,
            'threshold': self._adaptive_threshold,
            'morphological': self._morphological_ops,
            'clahe': self._clahe_enhancement,
            'edge_enhanced': self._edge_enhancement,
            'sharpen': self._sharpen,
            'histogram_equalized': self._histogram_equalization,
            'perspective_corrected': self._perspective_correction
        }
        
        logger.info(f"✅ Preprocessor initialized with {len(self.methods)} methods")
    
    def preprocess(self, image: np.ndarray, method: str = 'enhanced', 
                  **kwargs) -> np.ndarray:
        """
        Apply specific preprocessing method
        
        Args:
            image: Input image
            method: Preprocessing method name
            **kwargs: Method-specific parameters
            
        Returns:
            Preprocessed image
        """
        if method not in self.methods:
            logger.warning(f"Unknown method: {method}, using 'enhanced'")
            method = 'enhanced'
        
        try:
            return self.methods[method](image.copy(), **kwargs)
        except Exception as e:
            logger.error(f"Preprocessing error ({method}): {e}")
            return image
    
    def get_preprocessed_images(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Get multiple preprocessed versions
        
        Returns:
            Dictionary of method_name -> preprocessed_image
        """
        results = {}
        
        for method_name, method_func in self.methods.items():
            try:
                processed = method_func(image.copy())
                if processed is not None and processed.size > 0:
                    results[method_name] = processed
                    logger.debug(f"Preprocessed with {method_name}: {processed.shape}")
            except Exception as e:
                logger.warning(f"Failed {method_name}: {e}")
        
        return results
    
    def _original(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Return original image"""
        return image
    
    def _grayscale(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Convert to grayscale"""
        if len(image.shape) == 3:
            return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return image
    
    def _enhance_contrast(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance contrast using CLAHE"""
        if len(image.shape) == 3:
            # Convert to LAB color space
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l_channel, a, b = cv2.split(lab)
            
            # Apply CLAHE to L-channel
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            cl = clahe.apply(l_channel)
            
            # Merge channels
            limg = cv2.merge((cl, a, b))
            
            # Convert back to BGR
            enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
            return enhanced
        else:
            # Grayscale image
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
            return clahe.apply(image)
    
    def _denoise(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply noise reduction"""
        if len(image.shape) == 3:
            return cv2.fastNlMeansDenoisingColored(
                image, None, 10, 10, 7, 21
            )
        else:
            return cv2.fastNlMeansDenoising(
                image, None, 10, 7, 21
            )
    
    def _adaptive_threshold(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply adaptive thresholding"""
        gray = self._grayscale(image)
        
        # Adaptive Gaussian threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY, 11, 2
        )
        
        return thresh
    
    def _morphological_ops(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply morphological operations"""
        gray = self._grayscale(image)
        
        # Otsu's threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        
        # Morphological operations to remove noise
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        
        # Closing to remove small black points
        closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Opening to remove small white points
        opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel)
        
        return opened
    
    def _clahe_enhancement(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply CLAHE with parameters"""
        clip_limit = kwargs.get('clip_limit', 2.0)
        tile_size = kwargs.get('tile_size', 8)
        
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(tile_size, tile_size))
        return clahe.apply(gray)
    
    def _edge_enhancement(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Enhance edges for text detection"""
        gray = self._grayscale(image)
        
        # Apply bilateral filter to preserve edges
        filtered = cv2.bilateralFilter(gray, 9, 75, 75)
        
        # Edge detection using Canny
        edges = cv2.Canny(filtered, 50, 150)
        
        # Dilate edges slightly
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=1)
        
        # Combine with original
        result = cv2.addWeighted(gray, 0.7, dilated, 0.3, 0)
        
        return result
    
    def _sharpen(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Sharpen image"""
        kernel = np.array([[-1, -1, -1],
                          [-1,  9, -1],
                          [-1, -1, -1]])
        
        return cv2.filter2D(image, -1, kernel)
    
    def _histogram_equalization(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply histogram equalization"""
        if len(image.shape) == 3:
            # Convert to YCrCb
            ycrcb = cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb)
            channels = cv2.split(ycrcb)
            
            # Equalize Y channel
            cv2.equalizeHist(channels[0], channels[0])
            
            # Merge channels
            cv2.merge(channels, ycrcb)
            
            # Convert back to BGR
            return cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2BGR)
        else:
            return cv2.equalizeHist(image)
    
    def _perspective_correction(self, image: np.ndarray, **kwargs) -> np.ndarray:
        """Apply perspective correction"""
        gray = self._grayscale(image)
        
        # Edge detection
        edges = cv2.Canny(gray, 50, 150)
        
        # Find contours
        contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if not contours:
            return image
        
        # Find largest contour
        largest_contour = max(contours, key=cv2.contourArea)
        
        # Approximate contour
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        
        if len(approx) == 4:
            # Found a quadrilateral
            points = approx.reshape(4, 2)
            
            # Order points: top-left, top-right, bottom-right, bottom-left
            rect = np.zeros((4, 2), dtype="float32")
            
            s = points.sum(axis=1)
            rect[0] = points[np.argmin(s)]  # top-left
            rect[2] = points[np.argmax(s)]  # bottom-right
            
            diff = np.diff(points, axis=1)
            rect[1] = points[np.argmin(diff)]  # top-right
            rect[3] = points[np.argmax(diff)]  # bottom-left
            
            # Define destination points
            width = max(
                np.linalg.norm(rect[0] - rect[1]),
                np.linalg.norm(rect[2] - rect[3])
            )
            height = max(
                np.linalg.norm(rect[0] - rect[3]),
                np.linalg.norm(rect[1] - rect[2])
            )
            
            dst = np.array([
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ], dtype="float32")
            
            # Compute perspective transform
            M = cv2.getPerspectiveTransform(rect, dst)
            
            # Apply warp
            warped = cv2.warpPerspective(image, M, (int(width), int(height)))
            
            return warped
        
        return image
    
    def deskew(self, image: np.ndarray) -> np.ndarray:
        """Deskew image"""
        gray = self._grayscale(image)
        
        # Threshold
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find non-zero pixels
        coords = np.column_stack(np.where(thresh > 0))
        
        if len(coords) < 10:
            return image
        
        # Get angle
        angle = cv2.minAreaRect(coords)[-1]
        
        if angle < -45:
            angle = 90 + angle
        
        # Rotate if needed
        if abs(angle) > 0.5:
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            
            if len(image.shape) == 3:
                rotated = cv2.warpAffine(image, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
            else:
                rotated = cv2.warpAffine(image, M, (w, h),
                                        flags=cv2.INTER_CUBIC,
                                        borderMode=cv2.BORDER_REPLICATE)
            return rotated
        
        return image
    
    def resize_for_ocr(self, image: np.ndarray, 
                      min_dimension: int = 300) -> np.ndarray:
        """Resize image for optimal OCR"""
        h, w = image.shape[:2]
        
        # Calculate scale to reach minimum dimension
        scale = max(min_dimension / w, min_dimension / h)
        
        if scale > 1:
            new_w = int(w * scale)
            new_h = int(h * scale)
            
            if len(image.shape) == 3:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            else:
                resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
            
            return resized
        
        return image

# Alias for backward compatibility
ImagePreprocessor = AdvancedImagePreprocessor