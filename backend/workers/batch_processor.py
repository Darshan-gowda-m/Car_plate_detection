"""
Batch processing with progress tracking and parallel execution
"""
import os
import time
import json
import uuid
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Optional, Callable, Any
from pathlib import Path
from loguru import logger

from ..core.detector import PlateDetector
from ..core.ocr_engines import OCREngineManager
from ..utils.performance import PerformanceMonitor

class BatchProcessor:
    """Batch processor for multiple images with progress tracking"""
    
    def __init__(self, detector: PlateDetector, ocr_manager: OCREngineManager, 
                 max_workers: int = 4):
        self.detector = detector
        self.ocr_manager = ocr_manager
        self.max_workers = max_workers
        self.performance_monitor = PerformanceMonitor()
        
    def process_batch(self, image_paths: List[str], 
                     options: Dict[str, Any] = None,
                     progress_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """
        Process batch of images
        
        Args:
            image_paths: List of image file paths
            options: Processing options
            progress_callback: Callback for progress updates
            
        Returns:
            Dictionary with batch results
        """
        start_time = time.time()
        
        if options is None:
            options = {}
        
        # Validate inputs
        valid_paths = []
        invalid_files = []
        
        for path in image_paths:
            if os.path.exists(path):
                valid_paths.append(path)
            else:
                invalid_files.append({'path': path, 'error': 'File not found'})
        
        if not valid_paths:
            return {
                'success': False,
                'error': 'No valid image files found',
                'invalid_files': invalid_files,
                'total_files': 0,
                'processed_files': 0
            }
        
        # Create batch ID
        batch_id = str(uuid.uuid4())[:8]
        
        logger.info(f"Starting batch {batch_id} with {len(valid_paths)} files")
        
        # Process files
        results = []
        failed_files = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            future_to_path = {
                executor.submit(
                    self._process_single,
                    path,
                    options,
                    idx,
                    len(valid_paths)
                ): (idx, path)
                for idx, path in enumerate(valid_paths)
            }
            
            # Process completed tasks
            for future in as_completed(future_to_path):
                idx, path = future_to_path[future]
                
                try:
                    result = future.result(timeout=options.get('timeout', 60))
                    
                    if result.get('success'):
                        results.append(result)
                    else:
                        failed_files.append({
                            'path': path,
                            'error': result.get('error', 'Unknown error')
                        })
                    
                    # Update progress
                    progress = (len(results) + len(failed_files)) / len(valid_paths) * 100
                    
                    if progress_callback:
                        progress_callback({
                            'batch_id': batch_id,
                            'progress': progress,
                            'processed': len(results) + len(failed_files),
                            'total': len(valid_paths),
                            'current_file': Path(path).name
                        })
                    
                except Exception as e:
                    logger.error(f"Failed to process {path}: {e}")
                    failed_files.append({
                        'path': path,
                        'error': str(e)
                    })
        
        # Calculate statistics
        total_time = time.time() - start_time
        
        statistics = self._calculate_statistics(results, total_time)
        
        # Prepare response
        response = {
            'batch_id': batch_id,
            'success': True,
            'total_files': len(valid_paths),
            'processed_files': len(results),
            'failed_files': len(failed_files),
            'results': results,
            'failed_files_details': failed_files,
            'statistics': statistics,
            'processing_time': total_time,
            'timestamp': datetime.utcnow().isoformat(),
            'options': options
        }
        
        logger.info(f"Batch {batch_id} completed: {len(results)}/{len(valid_paths)} successful")
        
        return response
    
    def _process_single(self, image_path: str, options: Dict, 
                       file_index: int, total_files: int) -> Dict[str, Any]:
        """Process single image file"""
        try:
            file_start_time = time.time()
            
            # Detect plates
            detection_options = options.get('detection', {})
            detections = self.detector.detect(
                image_path,
                use_yolo=detection_options.get('use_yolo', True),
                use_fallback=detection_options.get('use_fallback', True),
                multi_scale=detection_options.get('multi_scale', True)
            )
            
            detection_time = time.time() - file_start_time
            
            if not detections:
                return {
                    'success': False,
                    'filepath': image_path,
                    'error': 'No plates detected',
                    'processing_time': detection_time,
                    'file_index': file_index
                }
            
            # Get best detection
            best_detection = detections[0]
            
            # Crop plate
            cropped = self.detector.crop_plate(
                image_path,
                best_detection['bbox'],
                margin=detection_options.get('margin', 10)
            )
            
            if cropped is None:
                return {
                    'success': False,
                    'filepath': image_path,
                    'error': 'Failed to crop plate',
                    'detections': detections,
                    'processing_time': detection_time,
                    'file_index': file_index
                }
            
            # Run OCR
            ocr_options = options.get('ocr', {})
            engines_to_use = []
            
            if ocr_options.get('engines', {}).get('easyocr', True):
                engines_to_use.append('easyocr')
            if ocr_options.get('engines', {}).get('tesseract', True):
                engines_to_use.append('tesseract')
            if ocr_options.get('engines', {}).get('google_vision', True):
                engines_to_use.append('google')
            
            ocr_results = self.ocr_manager.extract_text(
                cropped,
                engines=engines_to_use if engines_to_use else None,
                preprocess=ocr_options.get('preprocess', True)
            )
            
            ocr_time = time.time() - file_start_time - detection_time
            
            # Find best OCR result
            best_ocr = None
            best_confidence = 0
            
            for engine, result in ocr_results.items():
                if result['confidence'] > best_confidence and result['text']:
                    best_confidence = result['confidence']
                    best_ocr = {
                        'engine': engine,
                        'text': result['text'],
                        'confidence': result['confidence'],
                        'processing_time': result['processing_time']
                    }
            
            total_time = time.time() - file_start_time
            
            # Record performance metrics
            self.performance_monitor.record_detection(detection_time, best_detection['confidence'])
            
            if best_ocr:
                self.performance_monitor.record_ocr(
                    best_ocr['engine'],
                    ocr_time,
                    best_ocr['confidence']
                )
            
            # Prepare result
            result = {
                'success': True,
                'filepath': image_path,
                'filename': Path(image_path).name,
                'detections': detections,
                'best_detection': best_detection,
                'ocr_results': ocr_results,
                'best_ocr': best_ocr,
                'processing_time': total_time,
                'detection_time': detection_time,
                'ocr_time': ocr_time,
                'file_index': file_index,
                'timestamp': datetime.utcnow().isoformat()
            }
            
            # Save output files if requested
            output_options = options.get('output', {})
            if output_options.get('save_files', True):
                output_dir = output_options.get('output_dir', 'outputs')
                os.makedirs(output_dir, exist_ok=True)
                
                file_id = f"{file_index:04d}_{Path(image_path).stem}"
                
                # Save cropped plate
                if output_options.get('save_cropped', True):
                    cropped_path = os.path.join(output_dir, f"{file_id}_cropped.jpg")
                    cv2.imwrite(cropped_path, cropped)
                    result['cropped_path'] = cropped_path
                
                # Save detection visualization
                if output_options.get('save_visualization', True):
                    viz_path = os.path.join(output_dir, f"{file_id}_detection.jpg")
                    self.detector.visualize_detection(image_path, detections, viz_path)
                    result['visualization_path'] = viz_path
            
            return result
            
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return {
                'success': False,
                'filepath': image_path,
                'error': str(e),
                'file_index': file_index
            }
    
    def _calculate_statistics(self, results: List[Dict], total_time: float) -> Dict[str, Any]:
        """Calculate batch statistics"""
        if not results:
            return {
                'total_files': 0,
                'successful_files': 0,
                'avg_processing_time': 0,
                'avg_detection_time': 0,
                'avg_ocr_time': 0,
                'avg_confidence': 0
            }
        
        successful_results = [r for r in results if r.get('success')]
        
        if not successful_results:
            return {
                'total_files': len(results),
                'successful_files': 0,
                'avg_processing_time': 0,
                'avg_detection_time': 0,
                'avg_ocr_time': 0,
                'avg_confidence': 0
            }
        
        # Calculate averages
        total_processing_time = sum(r.get('processing_time', 0) for r in successful_results)
        total_detection_time = sum(r.get('detection_time', 0) for r in successful_results)
        total_ocr_time = sum(r.get('ocr_time', 0) for r in successful_results)
        
        # Calculate confidence
        confidences = []
        for r in successful_results:
            if r.get('best_ocr') and r['best_ocr'].get('confidence'):
                confidences.append(r['best_ocr']['confidence'])
        
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0
        
        # Engine performance
        engine_stats = {}
        for r in successful_results:
            if r.get('ocr_results'):
                for engine, ocr_result in r['ocr_results'].items():
                    if engine not in engine_stats:
                        engine_stats[engine] = {
                            'count': 0,
                            'total_time': 0,
                            'total_confidence': 0
                        }
                    
                    engine_stats[engine]['count'] += 1
                    engine_stats[engine]['total_time'] += ocr_result.get('processing_time', 0)
                    engine_stats[engine]['total_confidence'] += ocr_result.get('confidence', 0)
        
        # Format engine statistics
        formatted_engine_stats = {}
        for engine, stats in engine_stats.items():
            if stats['count'] > 0:
                formatted_engine_stats[engine] = {
                    'count': stats['count'],
                    'avg_time': stats['total_time'] / stats['count'],
                    'avg_confidence': stats['total_confidence'] / stats['count']
                }
        
        return {
            'total_files': len(results),
            'successful_files': len(successful_results),
            'failed_files': len(results) - len(successful_results),
            'success_rate': (len(successful_results) / len(results)) * 100,
            'total_processing_time': total_time,
            'avg_processing_time': total_processing_time / len(successful_results),
            'avg_detection_time': total_detection_time / len(successful_results),
            'avg_ocr_time': total_ocr_time / len(successful_results),
            'avg_confidence': avg_confidence,
            'engine_statistics': formatted_engine_stats,
            'performance_metrics': self.performance_monitor.get_statistics()
        }
    
    def export_results(self, results: Dict[str, Any], format: str = 'json') -> Optional[str]:
        """Export batch results to file"""
        try:
            output_dir = 'exports'
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            batch_id = results.get('batch_id', 'batch')
            
            if format == 'json':
                filename = os.path.join(output_dir, f"{batch_id}_{timestamp}.json")
                with open(filename, 'w') as f:
                    json.dump(results, f, indent=2, default=str)
                return filename
            
            elif format == 'csv':
                import csv
                
                filename = os.path.join(output_dir, f"{batch_id}_{timestamp}.csv")
                
                with open(filename, 'w', newline='', encoding='utf-8') as f:
                    writer = csv.writer(f)
                    
                    # Write header
                    writer.writerow([
                        'filename', 'success', 'detection_method', 
                        'detection_confidence', 'ocr_engine', 'ocr_text',
                        'ocr_confidence', 'processing_time', 'timestamp'
                    ])
                    
                    # Write data
                    for result in results.get('results', []):
                        if result.get('success'):
                            writer.writerow([
                                result.get('filename', ''),
                                'YES',
                                result.get('best_detection', {}).get('method', ''),
                                result.get('best_detection', {}).get('confidence', 0),
                                result.get('best_ocr', {}).get('engine', ''),
                                result.get('best_ocr', {}).get('text', ''),
                                result.get('best_ocr', {}).get('confidence', 0),
                                result.get('processing_time', 0),
                                result.get('timestamp', '')
                            ])
                        else:
                            writer.writerow([
                                Path(result.get('filepath', '')).name,
                                'NO',
                                '',
                                0,
                                '',
                                '',
                                0,
                                result.get('processing_time', 0),
                                result.get('timestamp', '')
                            ])
                
                return filename
            
            elif format == 'excel':
                import pandas as pd
                
                filename = os.path.join(output_dir, f"{batch_id}_{timestamp}.xlsx")
                
                # Prepare data
                data = []
                for result in results.get('results', []):
                    if result.get('success'):
                        data.append({
                            'Filename': result.get('filename', ''),
                            'Success': 'YES',
                            'Detection Method': result.get('best_detection', {}).get('method', ''),
                            'Detection Confidence': result.get('best_detection', {}).get('confidence', 0),
                            'OCR Engine': result.get('best_ocr', {}).get('engine', ''),
                            'OCR Text': result.get('best_ocr', {}).get('text', ''),
                            'OCR Confidence': result.get('best_ocr', {}).get('confidence', 0),
                            'Processing Time (s)': result.get('processing_time', 0),
                            'Timestamp': result.get('timestamp', '')
                        })
                    else:
                        data.append({
                            'Filename': Path(result.get('filepath', '')).name,
                            'Success': 'NO',
                            'Detection Method': '',
                            'Detection Confidence': 0,
                            'OCR Engine': '',
                            'OCR Text': '',
                            'OCR Confidence': 0,
                            'Processing Time (s)': result.get('processing_time', 0),
                            'Timestamp': result.get('timestamp', '')
                        })
                
                # Create DataFrame and save
                df = pd.DataFrame(data)
                df.to_excel(filename, index=False)
                
                return filename
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to export results: {e}")
            return None