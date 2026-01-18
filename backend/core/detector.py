"""
Advanced Plate Detector with Multiple Detection Strategies
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Union
import time
import torch

# Try to import YOLO
try:
    from ultralytics import YOLO
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    print("⚠️ YOLO not available. Install: pip install ultralytics")

class AdvancedPlateDetector:
    """Advanced plate detector with multiple detection strategies"""
    
    def __init__(self, model_path: str, use_gpu: bool = True, conf_threshold: float = 0.3):
        """
        Initialize advanced plate detector
        
        Args:
            model_path: Path to YOLO model
            use_gpu: Whether to use GPU acceleration
            conf_threshold: Confidence threshold for detections
        """
        self.model_path = Path(model_path)
        self.use_gpu = use_gpu and torch.cuda.is_available()
        self.conf_threshold = conf_threshold
        
        # Initialize detection methods
        self.detection_methods = []
        
        # 1. YOLO Detection
        if YOLO_AVAILABLE:
            self.yolo_model = self._load_yolo_model()
            if self.yolo_model:
                self.detection_methods.append('yolo')
                print(f"✅ YOLO detection loaded (GPU: {'✅' if self.use_gpu else '❌'})")
        
        # 2. Haar Cascade
        self.haar_cascade = self._load_haar_cascade()
        if self.haar_cascade:
            self.detection_methods.append('haar')
            print("✅ Haar Cascade detection loaded")
        
        # 3. Edge-based detection
        self.detection_methods.append('edge')
        print("✅ Edge-based detection loaded")
        
        # 4. Color-based detection
        self.detection_methods.append('color')
        print("✅ Color-based detection loaded")
        
        if not self.detection_methods:
            raise RuntimeError("No detection methods available")
        
        print(f"📊 Detection methods: {', '.join(self.detection_methods)}")
        
        # Plate characteristics for validation
        self.min_plate_aspect_ratio = 2.0  # Increased from 1.5
        self.max_plate_aspect_ratio = 5.0   # Decreased from 6.0
        self.min_plate_area = 800           # Increased from 500
        self.max_plate_area = 50000         # Same
        
        # Performance tracking
        self.metrics = {
            'total_detections': 0,
            'successful_detections': 0,
            'avg_confidence': 0.0,
            'method_stats': {method: {'count': 0, 'avg_time': 0.0} for method in self.detection_methods}
        }
    
    def _load_yolo_model(self) -> Optional[YOLO]:
        """Load YOLO model"""
        try:
            if not self.model_path.exists():
                print(f"⚠️ Model not found at {self.model_path}")
                print("   Downloading default YOLOv8 model...")
                model = YOLO('yolov8n.pt')
            else:
                model = YOLO(str(self.model_path))
            
            # Warm up model
            device = 'cuda' if self.use_gpu else 'cpu'
            dummy_input = torch.randn(1, 3, 640, 640).to(device) if self.use_gpu else torch.randn(1, 3, 640, 640)
            model(dummy_input, verbose=False)
            
            return model
            
        except Exception as e:
            print(f"❌ Failed to load YOLO model: {e}")
            return None
    
    def _load_haar_cascade(self):
        """Load Haar Cascade for plate detection"""
        try:
            # Try to load pre-trained Haar cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_russian_plate_number.xml'
            if Path(cascade_path).exists():
                return cv2.CascadeClassifier(cascade_path)
            
            # Try alternative paths
            alt_paths = [
                '/usr/share/opencv4/haarcascades/haarcascade_russian_plate_number.xml',
                'models/haarcascade_russian_plate_number.xml'
            ]
            
            for path in alt_paths:
                if Path(path).exists():
                    return cv2.CascadeClassifier(path)
            
            print("⚠️ Haar cascade not found, skipping")
            return None
            
        except Exception as e:
            print(f"❌ Failed to load Haar cascade: {e}")
            return None
    
    def detect(self, image_input: Union[str, np.ndarray], 
               methods: List[str] = None,
               multi_scale: bool = True,
               conf_threshold: float = None) -> List[Dict]:
        """
        Detect plates using multiple strategies
        
        Args:
            image_input: Image path or numpy array
            methods: List of methods to use (None = all available)
            multi_scale: Use multi-scale detection
            conf_threshold: Confidence threshold
            
        Returns:
            List of detected plates with metadata
        """
        start_time = time.time()
        
        if conf_threshold is None:
            conf_threshold = self.conf_threshold
        
        # Load image
        image = self._load_image(image_input)
        if image is None:
            print("❌ Failed to load image")
            return []
        
        original_height, original_width = image.shape[:2]
        
        # Determine which methods to use
        if methods is None:
            methods_to_use = self.detection_methods
        else:
            methods_to_use = [m for m in methods if m in self.detection_methods]
        
        if not methods_to_use:
            print("⚠️ No valid detection methods specified")
            return []
        
        all_detections = []
        
        # Apply each detection method
        for method in methods_to_use:
            try:
                method_start = time.time()
                
                if method == 'yolo' and hasattr(self, 'yolo_model') and self.yolo_model:
                    detections = self._detect_with_yolo(image)
                elif method == 'haar' and self.haar_cascade:
                    detections = self._detect_with_haar(image)
                elif method == 'edge':
                    detections = self._detect_with_edges(image)
                elif method == 'color':
                    detections = self._detect_with_color(image)
                else:
                    continue
                
                method_time = time.time() - method_start
                
                # Add method metadata
                for det in detections:
                    det['method'] = method
                    det['method_time'] = method_time
                
                all_detections.extend(detections)
                
                # Update metrics
                if detections:
                    self.metrics['method_stats'][method]['count'] += len(detections)
                    self.metrics['method_stats'][method]['avg_time'] = (
                        self.metrics['method_stats'][method]['avg_time'] * 0.9 + 
                        method_time * 0.1
                    )
                
            except Exception as e:
                print(f"❌ {method} detection failed: {e}")
        
        # Multi-scale detection
        if multi_scale and len(all_detections) == 0:
            multi_detections = self._detect_multi_scale(image, methods_to_use)
            all_detections.extend(multi_detections)
        
        # Filter by confidence
        filtered_detections = [d for d in all_detections if d['confidence'] >= conf_threshold]
        
        # Validate plate characteristics
        validated_detections = []
        for det in filtered_detections:
            if self._validate_plate(det, original_width, original_height):
                validated_detections.append(det)
        
        # Apply Aggressive Non-Maximum Suppression with lower IoU threshold
        final_detections = self._non_max_suppression(validated_detections, iou_threshold=0.4)
        
        # Additional clustering to remove nearby duplicates
        final_detections = self._cluster_detections(final_detections, distance_threshold=30)
        
        # Sort by confidence
        final_detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Calculate plate characteristics
        for i, det in enumerate(final_detections):
            det['plate_id'] = i
            det['aspect_ratio'] = (det['bbox'][2] - det['bbox'][0]) / (det['bbox'][3] - det['bbox'][1])
            det['area'] = (det['bbox'][2] - det['bbox'][0]) * (det['bbox'][3] - det['bbox'][1])
            
            # Estimate plate type
            det['type'] = self._estimate_plate_type(det)
        
        # Update metrics
        total_time = time.time() - start_time
        self.metrics['total_detections'] += len(final_detections)
        if final_detections:
            self.metrics['successful_detections'] += 1
            self.metrics['avg_confidence'] = np.mean([d['confidence'] for d in final_detections])
        
        print(f"📊 Detection completed: {len(final_detections)} plates in {total_time:.3f}s")
        
        return final_detections
    
    def _detect_with_yolo(self, image: np.ndarray) -> List[Dict]:
        """Detect plates using YOLO"""
        detections = []
        
        try:
            results = self.yolo_model(
                image,
                conf=self.conf_threshold,
                device='cuda' if self.use_gpu else 'cpu',
                verbose=False,
                iou=0.4  # Lower IoU for NMS
            )
            
            for result in results:
                boxes = result.boxes
                if boxes is not None:
                    for box in boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                        confidence = float(box.conf[0].cpu().numpy())
                        
                        # Only add if it meets plate criteria
                        width = x2 - x1
                        height = y2 - y1
                        aspect_ratio = width / height if height > 0 else 0
                        
                        if self.min_plate_aspect_ratio <= aspect_ratio <= self.max_plate_aspect_ratio:
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': int(box.cls[0].cpu().numpy()) if box.cls is not None else 0
                            })
            
        except Exception as e:
            print(f"YOLO detection error: {e}")
        
        return detections
    
    def _detect_with_haar(self, image: np.ndarray) -> List[Dict]:
        """Detect plates using Haar Cascade"""
        detections = []
        
        try:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Detect plates
            plates = self.haar_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(30, 30)
            )
            
            for (x, y, w, h) in plates:
                # Calculate confidence based on aspect ratio and size
                aspect_ratio = w / h
                area = w * h
                
                if (self.min_plate_aspect_ratio <= aspect_ratio <= self.max_plate_aspect_ratio and
                    self.min_plate_area <= area <= self.max_plate_area):
                    
                    # Estimate confidence
                    confidence = min(1.0, 0.5 + (area / self.max_plate_area) * 0.5)
                    
                    detections.append({
                        'bbox': [x, y, x + w, y + h],
                        'confidence': confidence,
                        'class_id': -1  # Haar doesn't provide class
                    })
            
        except Exception as e:
            print(f"Haar detection error: {e}")
        
        return detections
    
    def _detect_with_edges(self, image: np.ndarray) -> List[Dict]:
        """Detect plates using edge detection"""
        detections = []
        
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
            # Apply bilateral filter to reduce noise
            filtered = cv2.bilateralFilter(gray, 11, 17, 17)
            
            # Edge detection
            edges = cv2.Canny(filtered, 30, 200)
            
            # Find contours
            contours, _ = cv2.findContours(
                edges.copy(),
                cv2.RETR_TREE,
                cv2.CHAIN_APPROX_SIMPLE
            )
            
            # Sort contours by area and take top 10 (reduced from 20)
            contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]
            
            for contour in contours:
                # Approximate contour
                perimeter = cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, 0.018 * perimeter, True)
                
                # Look for rectangular contours
                if len(approx) == 4:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = w / h
                    area = w * h
                    
                    if (self.min_plate_aspect_ratio <= aspect_ratio <= self.max_plate_aspect_ratio and
                        self.min_plate_area <= area <= self.max_plate_area):
                        
                        # Calculate contour solidity
                        hull = cv2.convexHull(contour)
                        hull_area = cv2.contourArea(hull)
                        solidity = cv2.contourArea(contour) / hull_area if hull_area > 0 else 0
                        
                        confidence = min(1.0, solidity * 0.7 + (area / self.max_plate_area) * 0.3)
                        
                        detections.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': confidence,
                            'class_id': -2  # Edge-based detection
                        })
            
        except Exception as e:
            print(f"Edge detection error: {e}")
        
        return detections
    
    def _detect_with_color(self, image: np.ndarray) -> List[Dict]:
        """Detect plates using color segmentation"""
        detections = []
        
        try:
            # Convert to HSV
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Define color ranges for plates (white/yellow common)
            # White plates
            lower_white = np.array([0, 0, 200])
            upper_white = np.array([180, 30, 255])
            mask_white = cv2.inRange(hsv, lower_white, upper_white)
            
            # Yellow plates
            lower_yellow = np.array([20, 100, 100])
            upper_yellow = np.array([30, 255, 255])
            mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)
            
            # Combine masks
            mask = cv2.bitwise_or(mask_white, mask_yellow)
            
            # Apply morphological operations
            kernel = np.ones((5, 5), np.uint8)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            
            # Find contours
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            for contour in contours:
                area = cv2.contourArea(contour)
                
                if self.min_plate_area <= area <= self.max_plate_area:
                    x, y, w, h = cv2.boundingRect(contour)
                    aspect_ratio = w / h
                    
                    if self.min_plate_aspect_ratio <= aspect_ratio <= self.max_plate_aspect_ratio:
                        # Calculate color coverage confidence
                        roi_mask = mask[y:y+h, x:x+w]
                        color_coverage = np.sum(roi_mask > 0) / (w * h)
                        
                        confidence = min(1.0, color_coverage * 0.6 + (area / self.max_plate_area) * 0.4)
                        
                        detections.append({
                            'bbox': [x, y, x + w, y + h],
                            'confidence': confidence,
                            'class_id': -3  # Color-based detection
                        })
            
        except Exception as e:
            print(f"Color detection error: {e}")
        
        return detections
    
    def _detect_multi_scale(self, image: np.ndarray, methods: List[str]) -> List[Dict]:
        """Detect plates at multiple scales"""
        all_detections = []
        scales = [0.5, 0.75, 1.0, 1.25, 1.5]
        
        original_height, original_width = image.shape[:2]
        
        for scale in scales:
            if scale == 1.0:
                scaled_image = image
            else:
                new_width = int(original_width * scale)
                new_height = int(original_height * scale)
                scaled_image = cv2.resize(image, (new_width, new_height))
            
            # Apply detection methods
            for method in methods:
                try:
                    if method == 'yolo' and hasattr(self, 'yolo_model') and self.yolo_model:
                        detections = self._detect_with_yolo(scaled_image)
                    elif method == 'haar' and self.haar_cascade:
                        detections = self._detect_with_haar(scaled_image)
                    elif method == 'edge':
                        detections = self._detect_with_edges(scaled_image)
                    elif method == 'color':
                        detections = self._detect_with_color(scaled_image)
                    else:
                        continue
                    
                    # Scale bounding boxes back to original size
                    for det in detections:
                        if scale != 1.0:
                            det['bbox'] = [
                                int(det['bbox'][0] / scale),
                                int(det['bbox'][1] / scale),
                                int(det['bbox'][2] / scale),
                                int(det['bbox'][3] / scale)
                            ]
                    
                    all_detections.extend(detections)
                    
                except Exception as e:
                    print(f"Multi-scale {method} detection failed: {e}")
        
        return all_detections
    
    def _validate_plate(self, detection: Dict, image_width: int, image_height: int) -> bool:
        """Validate if detection is a valid plate"""
        x1, y1, x2, y2 = detection['bbox']
        
        # Check bounds
        if x1 < 0 or y1 < 0 or x2 > image_width or y2 > image_height:
            return False
        
        # Check size
        width = x2 - x1
        height = y2 - y1
        
        if width < 30 or height < 10:  # Increased minimum width
            return False
        
        # Check aspect ratio
        aspect_ratio = width / height
        if not (self.min_plate_aspect_ratio <= aspect_ratio <= self.max_plate_aspect_ratio):
            return False
        
        # Check area
        area = width * height
        if not (self.min_plate_area <= area <= self.max_plate_area):
            return False
        
        return True
    
    def _estimate_plate_type(self, detection: Dict) -> str:
        """Estimate plate type based on characteristics"""
        x1, y1, x2, y2 = detection['bbox']
        width = x2 - x1
        height = y2 - y1
        aspect_ratio = width / height
        
        if aspect_ratio > 4:
            return 'standard'  # Standard rectangular plate
        elif aspect_ratio > 2.5:
            return 'large'     # Truck/bus plate
        else:
            return 'square'    # Motorcycle/square plate
    
    def _non_max_suppression(self, detections: List[Dict], iou_threshold: float = 0.5) -> List[Dict]:
        """Apply Non-Maximum Suppression"""
        if len(detections) == 0:
            return []
        
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])
        
        # Get coordinates
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Calculate area
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort by score
        indices = np.argsort(scores)[::-1]
        
        keep = []
        
        while len(indices) > 0:
            i = indices[0]
            keep.append(i)
            
            # Calculate IoU with remaining boxes
            xx1 = np.maximum(x1[i], x1[indices[1:]])
            yy1 = np.maximum(y1[i], y1[indices[1:]])
            xx2 = np.minimum(x2[i], x2[indices[1:]])
            yy2 = np.minimum(y2[i], y2[indices[1:]])
            
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            intersection = w * h
            union = areas[i] + areas[indices[1:]] - intersection
            iou = intersection / union
            
            # Keep boxes with IoU less than threshold
            remaining_indices = np.where(iou <= iou_threshold)[0]
            indices = indices[remaining_indices + 1]
        
        return [detections[i] for i in keep]
    
    def _cluster_detections(self, detections: List[Dict], distance_threshold: int = 30) -> List[Dict]:
        """Cluster nearby detections to remove duplicates"""
        if len(detections) <= 1:
            return detections
        
        clustered = []
        used = [False] * len(detections)
        
        for i in range(len(detections)):
            if used[i]:
                continue
                
            # Start new cluster with current detection
            cluster = [i]
            used[i] = True
            current_box = detections[i]['bbox']
            
            # Calculate center of current box
            x1, y1, x2, y2 = current_box
            cx1 = (x1 + x2) / 2
            cy1 = (y1 + y2) / 2
            
            # Find nearby detections
            for j in range(i + 1, len(detections)):
                if used[j]:
                    continue
                    
                other_box = detections[j]['bbox']
                ox1, oy1, ox2, oy2 = other_box
                cx2 = (ox1 + ox2) / 2
                cy2 = (oy1 + oy2) / 2
                
                # Calculate Euclidean distance between centers
                distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                
                if distance < distance_threshold:
                    cluster.append(j)
                    used[j] = True
            
            # Choose the best detection from cluster (highest confidence)
            if cluster:
                best_idx = max(cluster, key=lambda idx: detections[idx]['confidence'])
                clustered.append(detections[best_idx])
        
        return clustered
    
    def _load_image(self, image_input: Union[str, np.ndarray]) -> Optional[np.ndarray]:
        """Load image from path or numpy array"""
        try:
            if isinstance(image_input, str):
                if not Path(image_input).exists():
                    print(f"❌ Image not found: {image_input}")
                    return None
                
                image = cv2.imread(image_input)
                if image is None:
                    print(f"❌ Failed to read image: {image_input}")
                    return None
                
                # Convert BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
            elif isinstance(image_input, np.ndarray):
                image = image_input.copy()
                
                # Ensure proper format
                if len(image.shape) == 2:
                    image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
                elif image.shape[2] == 4:
                    image = cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
                elif image.shape[2] == 3:
                    # Assume RGB
                    pass
                else:
                    print(f"❌ Unsupported image shape: {image.shape}")
                    return None
            else:
                print(f"❌ Unsupported image type: {type(image_input)}")
                return None
            
            return image
            
        except Exception as e:
            print(f"❌ Error loading image: {e}")
            return None
    
    def crop_plate(self, image_input: Union[str, np.ndarray], bbox: List[int], 
                   margin: int = 5) -> Optional[np.ndarray]:
        """Crop plate region from image"""
        try:
            image = self._load_image(image_input)
            if image is None:
                return None
            
            # Convert back to BGR for OpenCV operations
            if len(image.shape) == 3 and image.shape[2] == 3:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            h, w = image.shape[:2]
            x1, y1, x2, y2 = bbox
            
            # Apply margin
            x1 = max(0, x1 - margin)
            y1 = max(0, y1 - margin)
            x2 = min(w, x2 + margin)
            y2 = min(h, y2 + margin)
            
            # Crop
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                print(f"⚠️ Empty crop: bbox={bbox}")
                return None
            
            return cropped
            
        except Exception as e:
            print(f"❌ Error cropping plate: {e}")
            return None
    
    def visualize_detection(self, image_input: Union[str, np.ndarray], 
                           detections: List[Dict],
                           output_path: Optional[str] = None) -> Optional[np.ndarray]:
        """Visualize detections on image"""
        try:
            image = self._load_image(image_input)
            if image is None:
                return None
            
            # Convert to BGR for drawing
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            
            # Define colors for different methods
            colors = {
                'yolo': (0, 255, 0),    # Green
                'haar': (255, 0, 0),    # Blue
                'edge': (0, 255, 255),  # Yellow
                'color': (255, 0, 255)  # Magenta
            }
            
            for det in detections:
                x1, y1, x2, y2 = det['bbox']
                confidence = det['confidence']
                method = det.get('method', 'unknown')
                plate_id = det.get('plate_id', 0)
                plate_type = det.get('type', 'unknown')
                
                # Get color for method
                color = colors.get(method, (255, 255, 255))
                
                # Draw bounding box
                cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
                
                # Draw label
                label = f"{method.upper()} #{plate_id}: {confidence:.2f} ({plate_type})"
                cv2.putText(image, label, (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            
            if output_path:
                cv2.imwrite(output_path, image)
                print(f"📸 Visualization saved to: {output_path}")
            
            return image
            
        except Exception as e:
            print(f"❌ Error visualizing detection: {e}")
            return None
    
    def get_metrics(self) -> Dict:
        """Get detection metrics"""
        return self.metrics.copy()

# Alias for backward compatibility
PlateDetector = AdvancedPlateDetector