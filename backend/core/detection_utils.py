"""
Detection utility functions
"""
import numpy as np
import cv2
from typing import List, Dict, Tuple

def filter_and_merge_detections(detections: List[Dict], image_shape: Tuple[int, int], 
                               iou_threshold: float = 0.3) -> List[Dict]:
    """
    Filter and merge detections with aggressive NMS
    
    Args:
        detections: List of detection dictionaries
        image_shape: (height, width) of image
        iou_threshold: IoU threshold for NMS
    
    Returns:
        Filtered detections
    """
    if not detections:
        return []
    
    # Extract boxes and scores
    boxes = []
    scores = []
    filtered_detections = []
    
    for det in detections:
        boxes.append(det['bbox'])
        scores.append(det['confidence'])
        filtered_detections.append(det)
    
    boxes = np.array(boxes)
    scores = np.array(scores)
    
    # Sort by confidence (descending)
    indices = np.argsort(scores)[::-1]
    boxes = boxes[indices]
    scores = scores[indices]
    
    keep = []
    
    while len(indices) > 0:
        # Pick the box with highest confidence
        current_idx = indices[0]
        keep.append(current_idx)
        
        if len(indices) == 1:
            break
        
        # Get IoU of current box with all other boxes
        current_box = boxes[0]
        other_boxes = boxes[1:]
        
        # Calculate IoU
        x1 = np.maximum(current_box[0], other_boxes[:, 0])
        y1 = np.maximum(current_box[1], other_boxes[:, 1])
        x2 = np.minimum(current_box[2], other_boxes[:, 2])
        y2 = np.minimum(current_box[3], other_boxes[:, 3])
        
        intersection = np.maximum(0, x2 - x1) * np.maximum(0, y2 - y1)
        
        area_current = (current_box[2] - current_box[0]) * (current_box[3] - current_box[1])
        area_others = (other_boxes[:, 2] - other_boxes[:, 0]) * (other_boxes[:, 3] - other_boxes[:, 1])
        union = area_current + area_others - intersection
        
        iou = intersection / union
        
        # Keep boxes with IoU less than threshold
        keep_indices = np.where(iou <= iou_threshold)[0]
        indices = indices[keep_indices + 1]
        boxes = boxes[keep_indices + 1]
        scores = scores[keep_indices + 1]
    
    # Get filtered detections
    result = [filtered_detections[i] for i in keep]
    
    # Additional filtering based on plate characteristics
    final_result = []
    img_height, img_width = image_shape[:2]
    
    for det in result:
        bbox = det['bbox']
        x1, y1, x2, y2 = bbox
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
            continue
        
        # Check size
        width = x2 - x1
        height = y2 - y1
        
        if width < 30 or height < 10:
            continue
        
        # Check aspect ratio
        aspect_ratio = width / height
        if not (2.0 <= aspect_ratio <= 5.0):
            continue
        
        # Check area
        area = width * height
        if not (800 <= area <= 50000):
            continue
        
        final_result.append(det)
    
    return final_result

def cluster_detections(detections: List[Dict], distance_threshold: int = 30) -> List[Dict]:
    """
    Cluster nearby detections to remove duplicates
    
    Args:
        detections: List of detection dictionaries
        distance_threshold: Distance threshold for clustering (pixels)
    
    Returns:
        Clustered detections
    """
    if len(detections) <= 1:
        return detections
    
    clustered = []
    used = [False] * len(detections)
    
    for i in range(len(detections)):
        if used[i]:
            continue
        
        # Start new cluster
        cluster = [i]
        used[i] = True
        current_bbox = detections[i]['bbox']
        
        # Calculate center
        x1, y1, x2, y2 = current_bbox
        cx1 = (x1 + x2) / 2
        cy1 = (y1 + y2) / 2
        
        # Find nearby detections
        for j in range(i + 1, len(detections)):
            if used[j]:
                continue
            
            other_bbox = detections[j]['bbox']
            ox1, oy1, ox2, oy2 = other_bbox
            cx2 = (ox1 + ox2) / 2
            cy2 = (oy1 + oy2) / 2
            
            # Calculate Euclidean distance
            distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
            
            if distance < distance_threshold:
                cluster.append(j)
                used[j] = True
        
        # Choose best detection from cluster (highest confidence)
        if cluster:
            best_idx = max(cluster, key=lambda idx: detections[idx]['confidence'])
            clustered.append(detections[best_idx])
    
    return clustered

def calculate_iou_matrix(boxes: np.ndarray) -> np.ndarray:
    """
    Calculate IoU matrix for all boxes
    
    Args:
        boxes: Nx4 array of bounding boxes
    
    Returns:
        NxN IoU matrix
    """
    n = len(boxes)
    iou_matrix = np.zeros((n, n))
    
    for i in range(n):
        for j in range(n):
            if i != j:
                iou_matrix[i, j] = calculate_iou(boxes[i], boxes[j])
    
    return iou_matrix

def calculate_iou(box1: List[int], box2: List[int]) -> float:
    """
    Calculate Intersection over Union between two boxes
    
    Args:
        box1: [x1, y1, x2, y2]
        box2: [x1, y1, x2, y2]
    
    Returns:
        IoU value
    """
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    union = area1 + area2 - intersection
    
    return intersection / union if union > 0 else 0

def validate_plate_bbox(bbox: List[int], image_shape: Tuple[int, int]) -> bool:
    """
    Validate if bounding box is likely a license plate
    
    Args:
        bbox: [x1, y1, x2, y2]
        image_shape: (height, width)
    
    Returns:
        True if valid plate, False otherwise
    """
    x1, y1, x2, y2 = bbox
    img_height, img_width = image_shape[:2]
    
    # Check bounds
    if x1 < 0 or y1 < 0 or x2 > img_width or y2 > img_height:
        return False
    
    # Check size
    width = x2 - x1
    height = y2 - y1
    
    if width < 30 or height < 10:
        return False
    
    # Check aspect ratio (plates are wider than tall)
    aspect_ratio = width / height
    if not (2.0 <= aspect_ratio <= 5.0):
        return False
    
    # Check area
    area = width * height
    if not (800 <= area <= 50000):
        return False
    
    return True

def estimate_plate_type(bbox: List[int]) -> str:
    """
    Estimate plate type based on aspect ratio
    
    Args:
        bbox: [x1, y1, x2, y2]
    
    Returns:
        Plate type string
    """
    x1, y1, x2, y2 = bbox
    width = x2 - x1
    height = y2 - y1
    aspect_ratio = width / height if height > 0 else 0
    
    if aspect_ratio > 4.0:
        return 'standard'  # Car plate
    elif aspect_ratio > 2.5:
        return 'large'     # Truck/bus plate
    else:
        return 'square'    # Motorcycle plate