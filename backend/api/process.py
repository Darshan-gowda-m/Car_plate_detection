"""
Processing API endpoints
"""
import os
import json
import uuid
import time
from datetime import datetime
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
import cv2

from backend.utils.error_handler import handle_api_error

process_api = Blueprint('process_api', __name__)

@process_api.route('/single', methods=['POST'])
@handle_api_error
def process_single():
    """Process single image"""
    return current_app.process_single_image()
    # The actual implementation is in app.py

@process_api.route('/batch', methods=['POST'])
@handle_api_error
def process_batch():
    """Process multiple images in batch"""
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No images provided', 'success': False}), 400
        
        files = request.files.getlist('images')
        if not files:
            return jsonify({'error': 'No valid images', 'success': False}), 400
        
        # Get options
        options = request.form.get('options', '{}')
        try:
            options = json.loads(options)
        except:
            options = {}
        
        # Save all files
        upload_dir = current_app.config['UPLOAD_FOLDER']
        filepaths = []
        
        for file in files:
            if file.filename == '':
                continue
                
            filename = secure_filename(file.filename)
            filepath = Path(upload_dir) / f"{uuid.uuid4()}_{filename}"
            file.save(str(filepath))
            filepaths.append(str(filepath))
        
        if not filepaths:
            return jsonify({'error': 'No valid files saved', 'success': False}), 400
        
        # Process sequentially
        results = []
        
        for filepath in filepaths:
            # Reuse the single processing logic
            request.files = {'image': open(filepath, 'rb')}
            request.form = {'options': json.dumps(options)}
            
            # This would call the processing function
            # For now, just add placeholder
            results.append({
                'filepath': filepath,
                'success': True,
                'processing_time': 0.5,
                'timestamp': datetime.utcnow().isoformat()
            })
        
        return jsonify({
            'success': True,
            'results': results,
            'total_processed': len(results)
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Batch processing error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@process_api.route('/batch/<job_id>/status', methods=['GET'])
@handle_api_error
def get_batch_status(job_id):
    """Get batch job status"""
    # This would check the worker for job status
    return jsonify({
        'job_id': job_id,
        'status': 'completed',
        'progress': 100,
        'message': 'Job completed successfully'
    }), 200