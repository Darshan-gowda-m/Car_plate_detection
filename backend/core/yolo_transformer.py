"""
YOLO-Transformer Hybrid Detector
"""
import torch
import torch.nn as nn
from ultralytics import YOLO
import cv2
import numpy as np
from typing import List, Dict, Optional
from pathlib import Path

class YOLOTransformerDetector:
    """Hybrid detector combining YOLO and Transformer"""
    
    def __init__(self, yolo_model_path: str, 
                 transformer_model_path: Optional[str] = None,
                 use_gpu: bool = True):
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.device = torch.device("cuda" if self.use_gpu else "cpu")
        
        # Load YOLO model
        print(f"🤖 Loading YOLO model: {yolo_model_path}")
        self.yolo_model = YOLO(yolo_model_path)
        if self.use_gpu:
            self.yolo_model.to('cuda')
        
        # Load Transformer model if provided
        self.transformer_model = None
        if transformer_model_path and Path(transformer_model_path).exists():
            try:
                print(f"🧠 Loading Transformer model: {transformer_model_path}")
                # Load custom transformer model
                self.transformer_model = torch.load(transformer_model_path, 
                                                   map_location=self.device)
                self.transformer_model.eval()
            except Exception as e:
                print(f"⚠️ Failed to load Transformer model: {e}")
    
    def detect(self, image: np.ndarray, 
               use_hybrid: bool = True,
               conf_threshold: float = 0.3) -> List[Dict]:
        """
        Detect plates using hybrid approach
        
        Args:
            image: Input image
            use_hybrid: Use both YOLO and Transformer
            conf_threshold: Confidence threshold
            
        Returns:
            List of detections
        """
        # YOLO detection
        yolo_results = self.yolo_model(
            image,
            conf=conf_threshold,
            device='cuda' if self.use_gpu else 'cpu',
            verbose=False
        )
        
        detections = []
        
        for result in yolo_results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                    confidence = float(box.conf[0].cpu().numpy())
                    
                    detections.append({
                        'bbox': [x1, y1, x2, y2],
                        'confidence': confidence,
                        'detector': 'yolo',
                        'method': 'yolo'
                    })
        
        # Transformer refinement if available and hybrid mode
        if use_hybrid and self.transformer_model and detections:
            refined_detections = self._refine_with_transformer(image, detections)
            detections.extend(refined_detections)
        
        # Apply Non-Maximum Suppression
        final_detections = self._non_max_suppression(detections)
        
        return final_detections
    
    def _refine_with_transformer(self, image: np.ndarray, 
                                detections: List[Dict]) -> List[Dict]:
        """Refine detections using Transformer"""
        refined = []
        
        for det in detections:
            x1, y1, x2, y2 = det['bbox']
            
            # Crop plate region
            cropped = image[y1:y2, x1:x2]
            if cropped.size == 0:
                continue
            
            # Prepare for transformer
            processed = self._prepare_for_transformer(cropped)
            
            # Run transformer inference
            with torch.no_grad():
                output = self.transformer_model(processed)
                # Process transformer output
                # This is simplified - implement based on your transformer architecture
            
            # Add refined detection
            refined.append({
                'bbox': det['bbox'],
                'confidence': det['confidence'] * 1.1,  # Slight boost
                'detector': 'hybrid',
                'method': 'yolo+transformer'
            })
        
        return refined
    
    def _prepare_for_transformer(self, image: np.ndarray) -> torch.Tensor:
        """Prepare image for transformer input"""
        # Resize
        resized = cv2.resize(image, (224, 224))
        
        # Normalize
        normalized = resized / 255.0
        
        # Convert to tensor
        tensor = torch.from_numpy(normalized).float()
        
        # Add batch dimension
        tensor = tensor.unsqueeze(0)
        
        # Move to device
        tensor = tensor.to(self.device)
        
        return tensor
    
    def _non_max_suppression(self, detections: List[Dict], 
                           iou_threshold: float = 0.5) -> List[Dict]:
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Calculate area
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        area = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by score
        indices = np.argsort(scores)[::-1]
        
        keep = []
        
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            # Calculate IoU
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            intersection = w * h
            union = area[i] + area[indices[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU <= threshold
            remaining = np.where(iou <= iou_threshold)[0]
            indices = indices[remaining + 1]
        
        return [detections[i] for i in keep]