"""
Advanced Processing Worker with Hybrid Detection
"""
import threading
import queue
import time
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import logging
import cv2
import numpy as np

logger = logging.getLogger(__name__)

class AdvancedProcessingWorker:
    """Advanced worker with hybrid detection capabilities"""
    
    def __init__(self, yolo_detector, transformer_detector, 
                 ocr_manager, deep_ocr, socketio=None, max_workers: int = 4):
        self.yolo_detector = yolo_detector
        self.transformer_detector = transformer_detector
        self.ocr_manager = ocr_manager
        self.deep_ocr = deep_ocr
        self.socketio = socketio
        
        # Job tracking
        self.jobs = {}
        self.job_queue = queue.Queue()
        self.running = False
        
        # Worker threads
        self.workers = []
        self.max_workers = max_workers
        
        # Performance tracking
        self.metrics = {
            'total_jobs': 0,
            'completed_jobs': 0,
            'failed_jobs': 0,
            'avg_processing_time': 0,
            'detection_method_stats': {}
        }
        
        # Start worker threads
        self.start_workers()
    
    def start_workers(self):
        """Start worker threads"""
        self.running = True
        
        for i in range(self.max_workers):
            worker = threading.Thread(
                target=self._worker_loop,
                args=(i,),
                daemon=True
            )
            worker.start()
            self.workers.append(worker)
        
        logger.info(f"Started {self.max_workers} advanced worker threads")
    
    def _worker_loop(self, worker_id):
        """Worker thread main loop"""
        logger.debug(f"Advanced Worker {worker_id} started")
        
        while self.running:
            try:
                job = self.job_queue.get(timeout=1)
                
                # Poison pill
                if job is None:
                    break
                
                # Process job
                self._process_advanced_job(job, worker_id)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Advanced Worker {worker_id} error: {e}")
    
    def _process_advanced_job(self, job, worker_id):
        """Process an advanced job with hybrid detection"""
        job_id = job['id']
        
        try:
            # Update job status
            self.jobs[job_id]['status'] = 'processing'
            self.jobs[job_id]['started_at'] = datetime.utcnow()
            
            # Notify via WebSocket
            self._emit_progress(job_id, 0, f"Worker {worker_id} starting hybrid processing")
            
            # Process files with advanced pipeline
            results = self._advanced_processing_pipeline(job, worker_id)
            
            # Update job with results
            self.jobs[job_id]['results'] = results
            self.jobs[job_id]['status'] = 'completed'
            self.jobs[job_id]['completed_at'] = datetime.utcnow()
            
            # Calculate summary
            self._calculate_job_summary(job_id)
            
            # Final progress update
            self._emit_progress(job_id, 100, "Advanced processing completed")
            
            logger.info(f"Advanced Job {job_id} completed")
            
        except Exception as e:
            logger.error(f"Advanced Job {job_id} failed: {e}")
            self.jobs[job_id]['status'] = 'failed'
            self.jobs[job_id]['error'] = str(e)
            self.jobs[job_id]['completed_at'] = datetime.utcnow()
            
            self._emit_progress(job_id, 0, f"Job failed: {str(e)}", 'error')
    
    def _advanced_processing_pipeline(self, job, worker_id):
        """Advanced processing pipeline with hybrid detection"""
        from pathlib import Path
        import time
        
        filepath = job['filepath']
        options = job['options']
        
        results = {
            'filepath': filepath,
            'filename': Path(filepath).name,
            'processing_stages': {},
            'detection_results': {},
            'ocr_results': {},
            'quality_metrics': {}
        }
        
        # Stage 1: Image Loading and Preprocessing
        stage_start = time.time()
        image = cv2.imread(filepath)
        if image is None:
            raise ValueError(f"Failed to load image: {filepath}")
        
        # Enhanced preprocessing
        from backend.core.preprocessor import AdvancedImagePreprocessor
        preprocessor = AdvancedImagePreprocessor()
        preprocessed = preprocessor.get_preprocessed_images(image)
        
        results['processing_stages']['preprocessing'] = {
            'time': time.time() - stage_start,
            'preprocessed_versions': list(preprocessed.keys())
        }
        
        # Stage 2: Hybrid Detection with better filtering
        stage_start = time.time()
        detection_results = self._hybrid_detection(image, preprocessed, options)
        results['detection_results'] = detection_results
        results['processing_stages']['detection'] = {
            'time': time.time() - stage_start,
            'methods_used': detection_results.get('methods_used', []),
            'plate_count': len(detection_results.get('plates', []))
        }
        
        # Stage 3: Advanced OCR
        stage_start = time.time()
        ocr_results = self._advanced_ocr_processing(
            image, 
            detection_results.get('plates', []), 
            options
        )
        results['ocr_results'] = ocr_results
        results['processing_stages']['ocr'] = {
            'time': time.time() - stage_start,
            'engines_used': ocr_results.get('engines_used', []),
            'text_found': ocr_results.get('text_found', False)
        }
        
        # Stage 4: Quality Assessment
        stage_start = time.time()
        quality_metrics = self._assess_quality(
            image, 
            detection_results, 
            ocr_results
        )
        results['quality_metrics'] = quality_metrics
        results['processing_stages']['quality_assessment'] = {
            'time': time.time() - stage_start
        }
        
        # Calculate overall metrics
        results['total_processing_time'] = sum(
            stage['time'] for stage in results['processing_stages'].values()
        )
        results['success'] = quality_metrics.get('overall_quality', 0) > 0.5
        results['timestamp'] = datetime.utcnow().isoformat()
        
        return results
    
    def _hybrid_detection(self, image, preprocessed_images, options):
        """Perform hybrid detection using multiple methods with better filtering"""
        results = {
            'plates': [],
            'methods_used': [],
            'confidence_scores': []
        }
        
        all_detections = []
        
        # YOLO Detection - Only use YOLO for primary detection
        if self.yolo_detector and options.get('use_yolo', True):
            try:
                yolo_detections = self.yolo_detector.detect(
                    image,
                    methods=['yolo'],  # Only YOLO, no fallback methods
                    multi_scale=options.get('multi_scale', False),  # Disable multi-scale by default
                    conf_threshold=options.get('confidence_threshold', 0.4)  # Higher threshold
                )
                
                for det in yolo_detections:
                    det['detector'] = 'yolo'
                
                all_detections.extend(yolo_detections)
                results['methods_used'].append('yolo')
                
                logger.info(f"YOLO detected {len(yolo_detections)} plates")
                
            except Exception as e:
                logger.warning(f"YOLO detection failed: {e}")
        
        # Filter detections aggressively to remove duplicates
        if all_detections:
            # Apply NMS with lower IoU threshold
            filtered = self._apply_nms(all_detections, iou_threshold=0.3)
            
            # Additional clustering to remove nearby duplicates
            filtered = self._cluster_detections(filtered, distance_threshold=25)
            
            # Filter by plate characteristics
            filtered = self._filter_by_plate_characteristics(filtered, image.shape)
            
            results['plates'] = filtered
            results['confidence_scores'] = [d['confidence'] for d in filtered]
            
            logger.info(f"After filtering: {len(filtered)} plates")
        
        return results
    
    def _apply_nms(self, detections, iou_threshold=0.3):
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        filtered = []
        while detections:
            # Take highest confidence detection
            current = detections.pop(0)
            filtered.append(current)
            
            # Remove overlapping detections
            detections = [
                d for d in detections 
                if self._calculate_iou(current['bbox'], d['bbox']) < iou_threshold
            ]
        
        return filtered
    
    def _cluster_detections(self, detections, distance_threshold=25):
        """Cluster nearby detections to remove duplicates"""
        if len(detections) <= 1:
            return detections
        
        clustered = []
        used = [False] * len(detections)
        
        for i in range(len(detections)):
            if used[i]:
                continue
                
            cluster = [i]
            used[i] = True
            current_bbox = detections[i]['bbox']
            
            # Calculate center of current detection
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
                
                # Calculate distance between centers
                distance = np.sqrt((cx2 - cx1)**2 + (cy2 - cy1)**2)
                
                if distance < distance_threshold:
                    cluster.append(j)
                    used[j] = True
            
            # Choose best detection from cluster
            if cluster:
                best_idx = max(cluster, key=lambda idx: detections[idx]['confidence'])
                clustered.append(detections[best_idx])
        
        return clustered
    
    def _filter_by_plate_characteristics(self, detections, image_shape):
        """Filter detections by plate characteristics"""
        if not detections:
            return []
        
        filtered = []
        img_height, img_width = image_shape[:2]
        
        for det in detections:
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
            
            # Check aspect ratio (plates are typically wider than tall)
            aspect_ratio = width / height
            if not (2.0 <= aspect_ratio <= 5.0):
                continue
            
            # Check area
            area = width * height
            if not (800 <= area <= 50000):
                continue
            
            filtered.append(det)
        
        return filtered
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _advanced_ocr_processing(self, image, plates, options):
        """Perform advanced OCR processing"""
        results = {
            'plates': [],
            'engines_used': [],
            'text_found': False,
            'consensus_text': ''
        }
        
        for i, plate in enumerate(plates):
            # Crop plate
            x1, y1, x2, y2 = plate['bbox']
            cropped = image[y1:y2, x1:x2]
            
            if cropped.size == 0:
                continue
            
            plate_ocr = {
                'plate_id': i,
                'bbox': plate['bbox'],
                'detection_confidence': plate['confidence'],
                'ocr_results': {}
            }
            
            # Try multiple OCR engines
            ocr_engines = []
            
            # Standard OCR engines
            if self.ocr_manager:
                # Use only reliable engines
                standard_engines = ['easyocr', 'tesseract']
                if options.get('use_google_vision', False):
                    standard_engines.append('google')
                
                for engine in standard_engines:
                    try:
                        result = self.ocr_manager.extract_text(
                            cropped,
                            engines=[engine],
                            preprocess=True,
                            validate=True  # Enable validation
                        )
                        
                        if result and engine in result and result[engine].get('text'):
                            plate_ocr['ocr_results'][engine] = result[engine]
                            ocr_engines.append(engine)
                    except Exception as e:
                        logger.warning(f"{engine} OCR failed: {e}")
            
            # Deep OCR (if available)
            if self.deep_ocr and options.get('use_deep_ocr', True):
                try:
                    result = self.deep_ocr.extract_text(cropped)
                    if result and result.get('text'):
                        plate_ocr['ocr_results']['deep'] = result
                        ocr_engines.append('deep')
                except Exception as e:
                    logger.warning(f"Deep OCR failed: {e}")
            
            # Find best result with consensus
            if plate_ocr['ocr_results']:
                best_result = self._find_best_ocr_result(plate_ocr['ocr_results'])
                plate_ocr['best_ocr'] = best_result
                results['text_found'] = results['text_found'] or bool(best_result.get('text'))
            
            results['plates'].append(plate_ocr)
        
        results['engines_used'] = list(set(ocr_engines))
        
        # Find consensus text if multiple plates
        if results['plates']:
            all_texts = []
            for plate in results['plates']:
                if plate.get('best_ocr') and plate['best_ocr'].get('text'):
                    all_texts.append(plate['best_ocr']['text'])
            
            if all_texts:
                # Simple consensus: most common text
                from collections import Counter
                text_counts = Counter(all_texts)
                if text_counts:
                    results['consensus_text'] = text_counts.most_common(1)[0][0]
        
        return results
    
    def _find_best_ocr_result(self, ocr_results):
        """Find best OCR result from multiple engines"""
        best_conf = 0
        best_result = None
        
        for engine, result in ocr_results.items():
            if result and result.get('confidence', 0) > best_conf:
                best_conf = result['confidence']
                best_result = {
                    'engine': engine,
                    'text': result.get('text', ''),
                    'confidence': best_conf
                }
        
        return best_result or {'engine': 'none', 'text': '', 'confidence': 0}
    
    def _assess_quality(self, image, detection_results, ocr_results):
        """Assess overall processing quality"""
        quality = {
            'detection_quality': 0,
            'ocr_quality': 0,
            'image_quality': 0,
            'overall_quality': 0
        }
        
        # Detection quality
        if detection_results.get('plates'):
            confidences = detection_results.get('confidence_scores', [])
            quality['detection_quality'] = sum(confidences) / len(confidences) if confidences else 0
        
        # OCR quality
        if ocr_results.get('plates'):
            ocr_confidences = []
            for plate in ocr_results['plates']:
                if plate.get('best_ocr'):
                    ocr_confidences.append(plate['best_ocr'].get('confidence', 0))
            
            quality['ocr_quality'] = sum(ocr_confidences) / len(ocr_confidences) if ocr_confidences else 0
        
        # Image quality (simplified)
        if image is not None:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
            quality['image_quality'] = min(1.0, laplacian_var / 500)
        
        # Overall quality (weighted average)
        quality['overall_quality'] = (
            quality['detection_quality'] * 0.4 +
            quality['ocr_quality'] * 0.4 +
            quality['image_quality'] * 0.2
        )
        
        return quality
    
    def _calculate_job_summary(self, job_id):
        """Calculate job summary statistics"""
        job = self.jobs[job_id]
        results = job.get('results', {})
        
        summary = {
            'processing_time': results.get('total_processing_time', 0),
            'success': results.get('success', False),
            'plates_detected': len(results.get('detection_results', {}).get('plates', [])),
            'text_found': results.get('ocr_results', {}).get('text_found', False),
            'quality_score': results.get('quality_metrics', {}).get('overall_quality', 0),
            'detection_methods': results.get('detection_results', {}).get('methods_used', []),
            'ocr_engines': results.get('ocr_results', {}).get('engines_used', [])
        }
        
        job['summary'] = summary
    
    def _emit_progress(self, job_id, progress, message, level='info'):
        """Emit progress update via WebSocket"""
        if self.socketio:
            self.socketio.emit('job_progress', {
                'job_id': job_id,
                'progress': progress,
                'message': message,
                'level': level,
                'timestamp': datetime.utcnow().isoformat()
            })
    
    def submit_job(self, filepath, options=None):
        """Submit a job for advanced processing"""
        if options is None:
            options = {}
        
        job_id = str(uuid.uuid4())[:8]
        
        job = {
            'id': job_id,
            'filepath': filepath,
            'options': options,
            'status': 'pending',
            'created_at': datetime.utcnow(),
            'results': None,
            'summary': None
        }
        
        self.jobs[job_id] = job
        self.job_queue.put(job)
        
        logger.info(f"Advanced job {job_id} submitted: {filepath}")
        
        return job_id
    
    def get_job_status(self, job_id):
        """Get status of a job"""
        return self.jobs.get(job_id)
    
    def get_metrics(self):
        """Get worker metrics"""
        return self.metrics