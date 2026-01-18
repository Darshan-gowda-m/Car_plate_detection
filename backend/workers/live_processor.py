"""
Live Camera Processing Implementation
"""
import cv2
import threading
import queue
import time
import numpy as np
from datetime import datetime
from typing import Optional, Callable
import logging

logger = logging.getLogger(__name__)

class LiveCameraProcessor:
    """Process live camera feed"""
    
    def __init__(self, detector, ocr_manager):
        self.detector = detector
        self.ocr_manager = ocr_manager
        
        # Camera state
        self.camera = None
        self.is_running = False
        self.processing_thread = None
        
        # Frame processing
        self.frame_queue = queue.Queue(maxsize=2)
        self.processing_enabled = True
        self.frame_skip = 3
        self.frame_count = 0
        
        # Callbacks
        self.on_frame = None
        self.on_detection = None
        self.on_error = None
        
        # Statistics
        self.stats = {
            'frames_processed': 0,
            'plates_detected': 0,
            'avg_processing_time': 0,
            'fps': 0
        }
        
        # Last results
        self.last_results = None
    
    def start(self, camera_id=0, resolution=(1280, 720), fps=30):
        """Start camera processing"""
        try:
            self.camera = cv2.VideoCapture(camera_id)
            
            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {camera_id}")
            
            # Set camera properties
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, resolution[0])
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, resolution[1])
            self.camera.set(cv2.CAP_PROP_FPS, fps)
            
            # Get actual properties
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)
            
            logger.info(f"Camera started: {actual_width}x{actual_height} @ {actual_fps}fps")
            
            # Start processing thread
            self.is_running = True
            self.processing_thread = threading.Thread(
                target=self._camera_loop,
                daemon=True
            )
            self.processing_thread.start()
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            if self.on_error:
                self.on_error(str(e))
            return False
    
    def stop(self):
        """Stop camera processing"""
        self.is_running = False
        
        if self.processing_thread:
            self.processing_thread.join(timeout=2)
        
        if self.camera:
            self.camera.release()
            self.camera = None
        
        logger.info("Camera stopped")
    
    def _camera_loop(self):
        """Main camera loop"""
        frame_times = []
        
        while self.is_running and self.camera:
            try:
                ret, frame = self.camera.read()
                
                if not ret:
                    logger.warning("Failed to read frame")
                    time.sleep(0.1)
                    continue
                
                current_time = time.time()
                frame_times.append(current_time)
                
                # Keep only recent frame times
                frame_times = [t for t in frame_times if current_time - t < 1]
                self.stats['fps'] = len(frame_times)
                
                # Skip frames for performance
                self.frame_count += 1
                if self.frame_count % self.frame_skip != 0:
                    continue
                
                # Add to processing queue
                if self.frame_queue.empty() and self.processing_enabled:
                    self.frame_queue.put(frame)
                
                # Send frame to callback
                if self.on_frame:
                    self.on_frame(frame.copy())
                
                time.sleep(0.01)
                
            except Exception as e:
                logger.error(f"Camera loop error: {e}")
                if self.on_error:
                    self.on_error(str(e))
                time.sleep(0.1)
    
    def process_next_frame(self, timeout=1.0):
        """Process next available frame"""
        try:
            frame = self.frame_queue.get(timeout=timeout)
            return self._process_frame(frame)
        except queue.Empty:
            return None
    
    def _process_frame(self, frame):
        """Process a single frame"""
        start_time = time.time()
        
        try:
            # Detect plates
            detections = self.detector.detect(
                frame,
                use_yolo=True,
                use_fallback=False,  # Disable for performance
                multi_scale=False,
                conf_threshold=0.4
            )
            
            detection_time = time.time() - start_time
            
            # Update statistics
            self.stats['frames_processed'] += 1
            self.stats['avg_processing_time'] = (
                self.stats['avg_processing_time'] * 0.9 + 
                detection_time * 0.1
            )
            
            results = {
                'detections': detections,
                'processing_time': detection_time,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Process plates if found
            if detections:
                self.stats['plates_detected'] += 1
                
                # Process first plate only (for performance)
                first_detection = detections[0]
                cropped = self.detector.crop_plate(frame, first_detection['bbox'])
                
                if cropped is not None:
                    # Quick OCR
                    ocr_result = self.ocr_manager.extract_with_consensus(
                        cropped,
                        min_confidence=0.5,
                        require_agreement=1
                    )
                    
                    results.update({
                        'best_detection': first_detection,
                        'ocr_result': ocr_result
                    })
            
            self.last_results = results
            
            # Notify callback
            if self.on_detection:
                self.on_detection(results)
            
            return results
            
        except Exception as e:
            logger.error(f"Frame processing error: {e}")
            if self.on_error:
                self.on_error(str(e))
            return None
    
    def draw_results(self, frame, results):
        """Draw detection results on frame"""
        frame_copy = frame.copy()
        
        if not results or 'detections' not in results:
            return frame_copy
        
        # Draw detections
        for i, detection in enumerate(results['detections']):
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(frame_copy, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw label
            label = f"Plate {i+1}: {confidence:.2f}"
            cv2.putText(
                frame_copy,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2
            )
        
        # Draw OCR result if available
        if 'ocr_result' in results and results['ocr_result'].get('success'):
            text = results['ocr_result']['ocr_text']
            confidence = results['ocr_result']['confidence']
            
            # Draw at top of frame
            ocr_label = f"OCR: {text} ({confidence:.2f})"
            cv2.putText(
                frame_copy,
                ocr_label,
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 0),
                2
            )
        
        # Draw statistics
        stats_text = f"FPS: {self.stats['fps']} | Frames: {self.stats['frames_processed']}"
        cv2.putText(
            frame_copy,
            stats_text,
            (10, frame_copy.shape[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1
        )
        
        return frame_copy
    
    def get_stats(self):
        """Get processing statistics"""
        return self.stats.copy()