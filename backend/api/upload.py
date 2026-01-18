"""
Upload API endpoints
"""
import os
import uuid
import psutil
from pathlib import Path
from flask import Blueprint, request, jsonify, current_app
from werkzeug.utils import secure_filename
import cv2

from backend.utils.error_handler import handle_api_error

upload_api = Blueprint('upload_api', __name__)

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png', 'bmp', 'tiff', 'tif'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@upload_api.route('/single', methods=['POST'])
@handle_api_error
def upload_single():
    """Upload single file"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file part', 'success': False}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No selected file', 'success': False}), 400
        
        if not allowed_file(file.filename):
            return jsonify({
                'error': 'File type not allowed',
                'allowed_types': list(ALLOWED_EXTENSIONS),
                'success': False
            }), 400
        
        # Generate unique filename
        original_filename = secure_filename(file.filename)
        filename = f"{uuid.uuid4()}_{original_filename}"
        
        # Save file
        upload_folder = current_app.config['UPLOAD_FOLDER']
        filepath = Path(upload_folder) / filename
        file.save(str(filepath))
        
        # Get file info
        img = cv2.imread(str(filepath))
        if img is None:
            # Clean up invalid file
            filepath.unlink()
            return jsonify({'error': 'Invalid image file', 'success': False}), 400
        
        height, width = img.shape[:2]
        file_size = os.path.getsize(filepath)
        
        return jsonify({
            'success': True,
            'filename': original_filename,
            'saved_as': filename,
            'filepath': str(filepath),
            'dimensions': {
                'width': width,
                'height': height
            },
            'size': file_size,
            'size_human': f"{file_size/(1024*1024):.2f} MB"
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Upload error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500

@upload_api.route('/multiple', methods=['POST'])
@handle_api_error
def upload_multiple():
    """Upload multiple files"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files part', 'success': False}), 400
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No selected files', 'success': False}), 400
        
        upload_folder = current_app.config['UPLOAD_FOLDER']
        results = []
        errors = []
        
        for file in files:
            if file.filename == '':
                errors.append(f"Empty filename")
                continue
            
            if not allowed_file(file.filename):
                errors.append(f"Invalid file type: {file.filename}")
                continue
            
            try:
                original_filename = secure_filename(file.filename)
                filename = f"{uuid.uuid4()}_{original_filename}"
                filepath = Path(upload_folder) / filename
                file.save(str(filepath))
                
                results.append({
                    'original_name': original_filename,
                    'saved_as': filename,
                    'filepath': str(filepath)
                })
            except Exception as e:
                errors.append(f"Failed to save {file.filename}: {str(e)}")
        
        return jsonify({
            'success': True,
            'uploaded': len(results),
            'failed': len(errors),
            'results': results,
            'errors': errors
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Multiple upload error: {str(e)}")
        return jsonify({'error': str(e), 'success': False}), 500
    
@upload_api.route('/status', methods=['GET'])
@handle_api_error
def get_upload_status():
    """Get upload directory status"""
    try:
        from pathlib import Path
        import os
        
        upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
        output_dir = Path(current_app.config['OUTPUT_FOLDER'])
        temp_dir = Path(current_app.config['TEMP_FOLDER'])
        
        def get_dir_info(path):
            if not path.exists():
                return {
                    'exists': False,
                    'file_count': 0,
                    'total_size_mb': 0
                }
            
            files = list(path.iterdir())
            total_size = sum(f.stat().st_size for f in files if f.is_file())
            
            return {
                'exists': True,
                'path': str(path),
                'file_count': len([f for f in files if f.is_file()]),
                'total_size_mb': total_size / (1024 * 1024)
            }
        
        return jsonify({
            'success': True,
            'directories': {
                'uploads': get_dir_info(upload_dir),
                'outputs': get_dir_info(output_dir),
                'temp': get_dir_info(temp_dir)
            },
            'system': {
                'disk_usage_percent': psutil.disk_usage('/').percent if hasattr(psutil, 'disk_usage') else 0,
                'disk_free_gb': psutil.disk_usage('/').free / (1024**3) if hasattr(psutil, 'disk_usage') else 0
            },
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Upload status error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500