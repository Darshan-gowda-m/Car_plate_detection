"""
Worker for batch processing
"""
import threading
import queue
import time
import json
import uuid
from datetime import datetime
from typing import List, Dict, Optional
import logging

logger = logging.getLogger(__name__)

class ProcessingWorker:
    """Worker for processing images in background"""
    
    def __init__(self, detector, ocr_manager, socketio=None):
        self.detector = detector
        self.ocr_manager = ocr_manager
        self.socketio = socketio
        
        # Job tracking
        self.jobs = {}
        self.job_queue = queue.Queue()
        self.running = False
        
        # Worker threads
        self.workers = []
        self.max_workers = 4
        
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
        
        logger.info(f"Started {self.max_workers} worker threads")
    
    def stop_workers(self):
        """Stop all worker threads"""
        self.running = False
        
        # Add poison pills to queue
        for _ in range(len(self.workers)):
            self.job_queue.put(None)
        
        # Wait for workers to finish
        for worker in self.workers:
            worker.join(timeout=5)
        
        self.workers.clear()
        logger.info("Worker threads stopped")
    
    def _worker_loop(self, worker_id):
        """Worker thread main loop"""
        logger.debug(f"Worker {worker_id} started")
        
        while self.running:
            try:
                job = self.job_queue.get(timeout=1)
                
                # Poison pill
                if job is None:
                    break
                
                # Process job
                self._process_job(job, worker_id)
                
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Worker {worker_id} error: {e}")
    
    def _process_job(self, job, worker_id):
        """Process a batch job"""
        job_id = job['id']
        
        try:
            # Update job status
            self.jobs[job_id]['status'] = 'processing'
            self.jobs[job_id]['started_at'] = datetime.utcnow()
            
            # Notify via WebSocket
            self._emit_progress(job_id, 0, f"Worker {worker_id} starting")
            
            # Process files
            results = []
            total_files = len(job['files'])
            
            for idx, filepath in enumerate(job['files']):
                try:
                    # Calculate progress
                    progress = (idx / total_files) * 100
                    self._emit_progress(
                        job_id,
                        progress,
                        f"Processing file {idx + 1}/{total_files}"
                    )
                    
                    # Process single file
                    result = self._process_single_file(filepath, job['options'])
                    results.append(result)
                    
                    # Update job
                    self.jobs[job_id]['processed_files'] += 1
                    
                except Exception as e:
                    logger.error(f"Failed to process {filepath}: {e}")
                    self.jobs[job_id]['failed_files'] += 1
                    results.append({
                        'filepath': filepath,
                        'error': str(e),
                        'success': False
                    })
            
            # Mark job as completed
            self.jobs[job_id]['status'] = 'completed'
            self.jobs[job_id]['completed_at'] = datetime.utcnow()
            self.jobs[job_id]['results'] = results
            
            # Calculate summary
            successful = sum(1 for r in results if r.get('success', False))
            total_time = sum(r.get('processing_time', 0) for r in results)
            
            self.jobs[job_id]['summary'] = {
                'total_files': total_files,
                'successful': successful,
                'failed': total_files - successful,
                'total_processing_time': total_time,
                'avg_processing_time': total_time / total_files if total_files > 0 else 0
            }
            
            # Final progress update
            self._emit_progress(job_id, 100, "Batch processing completed")
            
            logger.info(f"Job {job_id} completed: {successful}/{total_files} successful")
            
        except Exception as e:
            logger.error(f"Job {job_id} failed: {e}")
            self.jobs[job_id]['status'] = 'failed'
            self.jobs[job_id]['error'] = str(e)
            self.jobs[job_id]['completed_at'] = datetime.utcnow()
            
            self._emit_progress(job_id, 0, f"Job failed: {str(e)}", 'error')
    
    def _process_single_file(self, image_path, options):
        """Process single image file"""
        from pathlib import Path
        import cv2
        import time
        
        start_time = time.time()
        
        try:
            # Skip if detector not available
            if self.detector is None:
                return {
                    'success': False,
                    'filepath': image_path,
                    'error': 'Detector not available',
                    'processing_time': time.time() - start_time
                }
            
            # Detect plates
            detections = self.detector.detect(
                image_path,
                use_yolo=True,
                use_fallback=True,
                multi_scale=True
            )
            
            detection_time = time.time() - start_time
            
            if not detections:
                return {
                    'success': False,
                    'filepath': image_path,
                    'error': 'No plates detected',
                    'detection_time': detection_time,
                    'processing_time': detection_time
                }
            
            # Get best detection
            best_det = detections[0]
            
            # Crop plate
            cropped = self.detector.crop_plate(image_path, best_det['bbox'])
            
            if cropped is None:
                return {
                    'success': False,
                    'filepath': image_path,
                    'error': 'Failed to crop plate',
                    'detections': detections,
                    'detection_time': detection_time,
                    'processing_time': detection_time
                }
            
            # Run OCR if available
            ocr_results = {}
            best_ocr = None
            
            if self.ocr_manager:
                ocr_start = time.time()
                ocr_results = self.ocr_manager.extract_text(
                    cropped,
                    engines=['easyocr', 'tesseract'],
                    preprocess=True
                )
                ocr_time = time.time() - ocr_start
                
                # Find best OCR result
                best_confidence = 0
                for engine, result in ocr_results.items():
                    if result.get('confidence', 0) > best_confidence and result.get('text'):
                        best_confidence = result['confidence']
                        best_ocr = {
                            'engine': engine,
                            'text': result['text'],
                            'confidence': result['confidence']
                        }
            
            total_time = time.time() - start_time
            
            # Save cropped plate if requested
            cropped_path = None
            if options.get('save_output', False):
                from pathlib import Path
                output_dir = 'outputs'
                Path(output_dir).mkdir(exist_ok=True)
                
                cropped_filename = f"cropped_{Path(image_path).stem}.jpg"
                cropped_path = str(Path(output_dir) / cropped_filename)
                cv2.imwrite(cropped_path, cropped)
            
            return {
                'success': True,
                'filepath': image_path,
                'filename': Path(image_path).name,
                'detections': detections,
                'best_detection': best_det,
                'ocr_results': ocr_results,
                'best_ocr': best_ocr,
                'processing_time': total_time,
                'detection_time': detection_time,
                'ocr_time': ocr_time if 'ocr_time' in locals() else 0,
                'cropped_path': cropped_path,
                'timestamp': datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'success': False,
                'filepath': image_path,
                'error': str(e),
                'processing_time': time.time() - start_time
            }
    
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
    
    def process_batch_async(self, job_id, files, options=None):
        """Start batch processing asynchronously"""
        if options is None:
            options = {}
        
        # Create job
        job = {
            'id': job_id,
            'files': files,
            'options': options,
            'status': 'pending',
            'created_at': datetime.utcnow(),
            'total_files': len(files),
            'processed_files': 0,
            'failed_files': 0,
            'results': None,
            'summary': None
        }
        
        # Store job
        self.jobs[job_id] = job
        
        # Add to queue
        self.job_queue.put(job)
        
        logger.info(f"Batch job {job_id} queued with {len(files)} files")
        
        return job_id
    
    def get_job_status(self, job_id):
        """Get status of a job"""
        return self.jobs.get(job_id)
    
    def get_active_jobs(self):
        """Get list of active jobs"""
        return [
            job for job in self.jobs.values()
            if job['status'] in ['pending', 'processing']
        ]