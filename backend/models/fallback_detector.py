"""
Computer vision-based fallback plate detector
"""
import cv2
import numpy as np
from typing import List, Dict, Optional, Tuple
from loguru import logger

class FallbackDetector:
    """Computer vision-based plate detector for when YOLO fails"""
    
    def __init__(self):
        self.min_plate_aspect = 1.5
        self.max_plate_aspect = 6.0
        self.min_plate_area = 500  # pixels
        self.max_plate_area = 50000  # pixels
        
    def detect(self, image: np.ndarray, multi_scale: bool = True) -> List[Dict]:
        """
        Detect plates using computer vision techniques
        
        Args:
            image: Input image
            multi_scale: Use multi-scale detection
            
        Returns:
            List of detected plates
        """
        if multi_scale:
            return self._detect_multi_scale(image)
        else:
            return self._detect_single_scale(image)
    
    def _detect_single_scale(self, image: np.ndarray) -> List[Dict]:
        """Detect plates at single scale"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply bilateral filter
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Edge detection
            edges = cv2.Canny(filtered, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(
                edges.copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Sort contours by area
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            detected_plates = []
            
            for contour in contours:
                # Approximate contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
                
                # Check if it's a rectangle
                if len(approx) == 4:
                    # Get bounding rectangle
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / float(h)
                    area = w * h
                    
                    # Check plate criteria
                    if (self.min_plate_aspect <= aspect_ratio <= self.max_plate_aspect and
                        self.min_plate_area <= area <= self.max_plate_area):
                        
                        # Calculate confidence based on rectangularity
                        contour_area = cv2.contourArea(contour)
                        rectangularity = contour_area / area if area > 0 else 0
                        
                        confidence = min(1.0, rectangularity * 0.8 + 0.2)
                        
                        detected_plates.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': confidence,
                            'class': -1,
                            'method': 'cv_contour'
                        })
            
            return detected_plates
            
        except Exception as e:
            logger.error(f"Fallback detection error: {e}")
            return []
    
    def _detect_multi_scale(self, image: np.ndarray) -> List[Dict]:
        """Detect plates at multiple scales"""
        detected_plates = []
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        h, w = image.shape[:2]
        
        for scale in scales:
            if scale == 1.0:
                scaled_img = image
            else:
                new_w = int(w * scale)
                new_h = int(h * scale)
                scaled_img = cv2.resize(image, (new_w, new_h))
            
            # Detect at this scale
            plates = self._detect_single_scale(scaled_img)
            
            # Scale bounding boxes back
            for plate in plates:
                if scale != 1.0:
                    plate['bbox'] = [
                        int(plate['bbox'][0] / scale),
                        int(plate['bbox'][1] / scale),
                        int(plate['bbox'][2] / scale),
                        int(plate['bbox'][3] / scale)
                    ]
                detected_plates.append(plate)
        
        # Apply non-maximum suppression
        return self._non_max_suppression(detected_plates)
    
    def _non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply non-maximum suppression to remove duplicates"""
        if not detections:
            return []
        
        # Extract bounding boxes
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Initialize list of picked indices
        pick = []
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Compute area
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by confidence
        idxs = np.argsort(scores)[::-1]
        
        while len(idxs) > 0:
            # Grab last index
            last = len(idxs) - 1
            i = idxs[last]
            pick.append(i)
            
            # Find intersection
            xx1 = np.maximum(x1[i], x1[idxs[:last]])
            yy1 = np.maximum(y1[i], y1[idxs[:last]])
            xx2 = np.minimum(x2[i], x2[idxs[:last]])
            yy2 = np.minimum(y2[i], y2[idxs[:last]])
            
            # Compute width and height
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            # Compute IoU
            intersection = w * h
            union = area[i] + area[idxs[:last]] - intersection
            iou = intersection / union
            
            # Delete indexes with IoU > threshold
            idxs = np.delete(idxs, np.concatenate(([last],
                                                  np.where(iou > iou_threshold)[0])))
        
        # Return filtered detections
        return [detections[i] for i in pick]
    
    def detect_with_morphology(self, image: np.ndarray) -> List[Dict]:
        """Detect plates using morphological operations"""
        try:
            # Convert to grayscale
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            else:
                gray = image.copy()
            
            # Apply morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 5))
            tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)
            blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)
            
            # Combine
            combined = cv2.add(gray, tophat)
            combined = cv2.subtract(combined, blackhat)
            
            # Threshold
            _, thresh = cv2.threshold(combined, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
            
            # Find contours
            contours, _ = cv2.findContours(
                thresh.copy(),
                cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            detected_plates = []
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_plate_area <= area <= self.max_plate_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / float(h)
                    
                    if self.min_plate_aspect <= aspect_ratio <= self.max_plate_aspect:
                        # Calculate solidity
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = area / hull_area if hull_area > 0 else 0
                        
                        if solidity > 0.7:  # Plates are typically solid
                            confidence = min(1.0, solidity * 0.6 + (area / self.max_plate_area) * 0.4)
                            
                            detected_plates.append({
                                'bbox': [x, y, x + w, y + h],
                                'confidence': confidence,
                                'class': -1,
                                'method': 'cv_morphology'
                            })
            
            return detected_plates
            
        except Exception as e:
            logger.error(f"Morphology detection error: {e}")
            return []