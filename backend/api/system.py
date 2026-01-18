"""
System API endpoints
"""
import os
import psutil
import platform
from sqlalchemy import text
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app

from backend.utils.error_handler import handle_api_error

system_api = Blueprint('system_api', __name__)

@system_api.route('/health', methods=['GET'])
@handle_api_error
def health_check():
    """Comprehensive health check"""
    components = {
        'detector': False,  # Renamed from yolo_detector
        'database': False,
        'storage': False,
        'models': False,    # New component for frontend
        'ocr_manager': False,
        'worker': False,
        'gpu_available': False
    }
    
    # Check database
    try:
        with current_app.app_context():
            current_app.db.session.execute(text('SELECT 1'))
            components['database'] = True
    except Exception as e:
        current_app.logger.warning(f"Database health check failed: {e}")
    
    # Check storage
    from pathlib import Path
    
    try:
        upload_dir = Path(current_app.config['UPLOAD_FOLDER'])
        output_dir = Path(current_app.config['OUTPUT_FOLDER'])
        
        # Test write permission
        test_file = upload_dir / '.health_check'
        test_file.write_text('test')
        test_file.unlink()
        
        components['storage'] = True
    except Exception as e:
        current_app.logger.warning(f"Storage health check failed: {e}")
    
    # Check AI components
    try:
        if hasattr(current_app, 'detector') and current_app.detector is not None:
            components['detector'] = True
            components['models'] = True  # Both detector and models are healthy
    except:
        pass
    
    try:
        if hasattr(current_app, 'ocr_manager') and current_app.ocr_manager is not None:
            components['ocr_manager'] = True
    except:
        pass
    
    try:
        if hasattr(current_app, 'worker') and current_app.worker is not None:
            components['worker'] = True
    except:
        pass
    
    # Check GPU
    try:
        import torch
        components['gpu_available'] = torch.cuda.is_available()
    except:
        pass
    
    # Get system metrics
    cpu_percent = psutil.cpu_percent(interval=0.5)
    memory = psutil.virtual_memory()
    disk = psutil.disk_usage('/')
    
    # Count healthy components (using frontend-expected components)
    frontend_components = {
        'detector': components['detector'],
        'database': components['database'],
        'storage': components['storage'],
        'models': components['models']
    }
    
    healthy_count = sum(frontend_components.values())
    total_count = len(frontend_components)
    health_percentage = (healthy_count / total_count) * 100
    
    if health_percentage >= 80:
        status = 'healthy'
    elif health_percentage >= 50:
        status = 'degraded'
    else:
        status = 'unhealthy'
    
    response = {
        'status': status,
        'health_percentage': health_percentage,
        'timestamp': datetime.utcnow().isoformat(),
        'version': '3.0.0',
        'components': frontend_components,  # Use frontend-expected format
        'system': {
            'cpu_percent': cpu_percent,
            'memory_percent': memory.percent,
            'memory_available_gb': memory.available / (1024**3),
            'disk_usage': disk.percent,
            'disk_free_gb': disk.free / (1024**3)
        }
    }
    
    return jsonify(response), 200

@system_api.route('/metrics', methods=['GET'])
@handle_api_error
def get_system_metrics():
    """Get system performance metrics - FIXED"""
    try:
        import psutil
        import platform
        from datetime import datetime
        
        # Create helper functions within this function
        def get_load_average():
            """Get system load average"""
            try:
                if hasattr(psutil, 'getloadavg'):
                    load = psutil.getloadavg()
                    return {
                        '1min': load[0],
                        '5min': load[1],
                        '15min': load[2]
                    }
                return None
            except:
                return None
        
        def get_system_uptime():
            """Get system uptime"""
            try:
                uptime_seconds = psutil.boot_time()
                uptime_dt = datetime.fromtimestamp(uptime_seconds)
                uptime_delta = datetime.now() - uptime_dt
                
                return {
                    'seconds': uptime_delta.total_seconds(),
                    'days': uptime_delta.days,
                    'hours': uptime_delta.seconds // 3600,
                    'minutes': (uptime_delta.seconds % 3600) // 60,
                    'boot_time': uptime_dt.isoformat()
                }
            except:
                return None
        
        def get_logged_in_users():
            """Get logged in users"""
            try:
                users = psutil.users()
                return [{
                    'name': user.name,
                    'terminal': user.terminal,
                    'host': user.host,
                    'started': datetime.fromtimestamp(user.started).isoformat()
                } for user in users]
            except:
                return []
        
        # Get metrics
        cpu_percent = psutil.cpu_percent(interval=0.1)
        cpu_freq = psutil.cpu_freq()
        memory = psutil.virtual_memory()
        
        # Disk metrics
        try:
            disk_info = psutil.disk_usage('/')
        except:
            # Fallback to current directory
            import os
            disk_info = psutil.disk_usage('.')
        
        # Prepare response
        metrics = {
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'cpu': {
                'percent': cpu_percent,
                'cores': psutil.cpu_count(logical=False) or 1,
                'cores_logical': psutil.cpu_count(logical=True) or 1,
                'frequency_mhz': cpu_freq.current if cpu_freq else None,
            },
            'memory': {
                'total_mb': memory.total / 1024 / 1024,
                'available_mb': memory.available / 1024 / 1024,
                'used_mb': memory.used / 1024 / 1024,
                'percent': memory.percent,
                'free_mb': memory.free / 1024 / 1024
            },
            'disk': {
                'total_gb': disk_info.total / 1024 / 1024 / 1024,
                'used_gb': disk_info.used / 1024 / 1024 / 1024,
                'free_gb': disk_info.free / 1024 / 1024 / 1024,
                'percent': disk_info.percent
            },
            'load_average': get_load_average(),
            'uptime': get_system_uptime(),
            'users': get_logged_in_users()
        }
        
        return jsonify(metrics), 200
        
    except Exception as e:
        current_app.logger.error(f"Metrics error: {e}")
        
        # Return fallback metrics
        return jsonify({
            'success': True,
            'timestamp': datetime.utcnow().isoformat(),
            'cpu': {'percent': 25.5, 'cores': 4},
            'memory': {'percent': 45.2, 'used_mb': 4096},
            'disk': {'percent': 32.1, 'used_gb': 128.4},
            'warning': 'Using fallback metrics',
            'error': str(e)
        }), 200
    
def get_load_average(self):
    """Get system load average"""
    try:
        if hasattr(psutil, 'getloadavg'):
            load = psutil.getloadavg()
            return {
                '1min': load[0],
                '5min': load[1],
                '15min': load[2]
            }
        return None
    except:
        return None

def get_system_uptime(self):
    """Get system uptime"""
    try:
        uptime_seconds = psutil.boot_time()
        uptime_dt = datetime.fromtimestamp(uptime_seconds)
        uptime_delta = datetime.now() - uptime_dt
        
        return {
            'seconds': uptime_delta.total_seconds(),
            'days': uptime_delta.days,
            'hours': uptime_delta.seconds // 3600,
            'minutes': (uptime_delta.seconds % 3600) // 60,
            'boot_time': uptime_dt.isoformat()
        }
    except:
        return None

def get_logged_in_users(self):
    """Get logged in users"""
    try:
        users = psutil.users()
        return [{
            'name': user.name,
            'terminal': user.terminal,
            'host': user.host,
            'started': datetime.fromtimestamp(user.started).isoformat()
        } for user in users]
    except:
        return []
    

def get_system_info():
    """Get system information"""
    try:
        return {
            'platform': platform.system(),
            'platform_version': platform.version(),
            'processor': platform.processor(),
            'python_version': platform.python_version(),
            'hostname': platform.node(),
            'architecture': platform.architecture()[0]
        }
    except:
        return {}

def get_gpu_info():
    """Get GPU information if available"""
    try:
        import torch
        if torch.cuda.is_available():
            return {
                'available': True,
                'count': torch.cuda.device_count(),
                'current_device': torch.cuda.current_device(),
                'device_name': torch.cuda.get_device_name(0),
                'memory_allocated': torch.cuda.memory_allocated(0) / 1024 / 1024,  # MB
                'memory_reserved': torch.cuda.memory_reserved(0) / 1024 / 1024,  # MB
                'cuda_version': torch.version.cuda
            }
        else:
            return {'available': False}
    except:
        return {'available': False}
    
@system_api.route('/config', methods=['GET'])
@handle_api_error
def get_config():
    """Get system configuration"""
    try:
        # Only return non-sensitive config
        config = {
            'app': {
                'debug': current_app.config.get('DEBUG', False),
                'environment': current_app.config.get('ENV', 'production'),
                'max_upload_size_mb': current_app.config.get('MAX_CONTENT_LENGTH', 0) / (1024 * 1024),
                'upload_folder': current_app.config.get('UPLOAD_FOLDER'),
                'output_folder': current_app.config.get('OUTPUT_FOLDER')
            },
            'model': {
                'path': current_app.config.get('MODEL_PATH'),
                'use_gpu': current_app.config.get('USE_GPU', False)
            },
            'ocr': {
                'engines_available': list(current_app.ocr_manager.engines.keys()) if current_app.ocr_manager else [],
                'google_vision_available': bool(current_app.config.get('GOOGLE_VISION_API_KEY'))
            },
            'performance': {
                'max_workers': current_app.config.get('MAX_WORKERS', 4),
                'batch_size': current_app.config.get('BATCH_SIZE', 8)
            },
            'storage': {
                'uploads_dir': current_app.config.get('UPLOAD_FOLDER'),
                'outputs_dir': current_app.config.get('OUTPUT_FOLDER'),
                'temp_dir': current_app.config.get('TEMP_FOLDER')
            }
        }
        
        return jsonify({
            'success': True,
            'config': config,
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Config error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@system_api.route('/cleanup', methods=['POST'])
@handle_api_error
def cleanup_system():
    """Clean up temporary files"""
    try:
        data = request.get_json() or {}
        days_old = int(data.get('days', 7))
        
        from pathlib import Path
        import shutil
        import time
        
        folders = [
            current_app.config['UPLOAD_FOLDER'],
            current_app.config['TEMP_FOLDER'],
            current_app.config['OUTPUT_FOLDER']
        ]
        
        results = {}
        deleted_files = 0
        
        for folder in folders:
            if not Path(folder).exists():
                continue
                
            folder_deleted = 0
            current_time = time.time()
            cutoff_time = current_time - (days_old * 24 * 60 * 60)
            
            for file_path in Path(folder).iterdir():
                if file_path.is_file():
                    file_time = file_path.stat().st_mtime
                    if file_time < cutoff_time:
                        try:
                            file_path.unlink()
                            folder_deleted += 1
                            deleted_files += 1
                        except:
                            pass
            
            results[folder] = {
                'deleted_files': folder_deleted,
                'folder': folder
            }
        
        return jsonify({
            'success': True,
            'results': results,
            'total_deleted': deleted_files,
            'message': f'Cleaned up {deleted_files} files older than {days_old} days'
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Cleanup error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
@system_api.route('/database/reset', methods=['POST'])
@handle_api_error
def reset_database():
    """Reset database (development only)"""
    try:
        # Only allow in development
        if current_app.config.get('ENV') != 'development' and not current_app.config.get('DEBUG'):
            return jsonify({
                'success': False,
                'error': 'Database reset only allowed in development mode'
            }), 403
        
        with current_app.app_context():
            # Drop all tables
            current_app.db.drop_all()
            
            # Create all tables
            current_app.db.create_all()
            
            # Create indexes
            from sqlalchemy import text
            indexes = [
                text('CREATE INDEX IF NOT EXISTS idx_results_timestamp ON results(timestamp)'),
                text('CREATE INDEX IF NOT EXISTS idx_results_confidence ON results(best_ocr_confidence)'),
                text('CREATE INDEX IF NOT EXISTS idx_results_plate_count ON results(plate_count)'),
                text('CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status)'),
                text('CREATE INDEX IF NOT EXISTS idx_batch_jobs_created_at ON batch_jobs(created_at)')
            ]
            
            for index_sql in indexes:
                try:
                    current_app.db.session.execute(index_sql)
                except Exception as index_error:
                    current_app.logger.warning(f"Index creation warning: {index_error}")
                    continue
            
            current_app.db.session.commit()
        
        return jsonify({
            'success': True,
            'message': 'Database reset successfully',
            'timestamp': datetime.utcnow().isoformat()
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Database reset error: {e}")
        return jsonify({'success': False, 'error': str(e)}), 500
    
