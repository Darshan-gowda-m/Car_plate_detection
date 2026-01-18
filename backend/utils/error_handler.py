"""
Error handling utilities
"""
import traceback
from functools import wraps
from flask import jsonify, current_app

class AppError(Exception):
    """Custom application error"""
    def __init__(self, message, status_code=500, error_code=None):
        super().__init__(message)
        self.message = message
        self.status_code = status_code
        self.error_code = error_code

def handle_api_error(func):
    """Decorator for handling API errors"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except AppError as e:
            current_app.logger.error(f"AppError in {func.__name__}: {e.message}")
            return jsonify({
                'success': False,
                'error': e.message,
                'error_code': e.error_code,
                'status_code': e.status_code
            }), e.status_code
        except Exception as e:
            current_app.logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            current_app.logger.error(traceback.format_exc())
            
            # Don't expose internal errors in production
            if current_app.config.get('DEBUG', False):
                details = traceback.format_exc()
            else:
                details = None
            
            return jsonify({
                'success': False,
                'error': 'Internal server error',
                'details': details,
                'status_code': 500
            }), 500
    return wrapper

def setup_error_handlers(app):
    """Setup global error handlers"""
    
    @app.errorhandler(404)
    def not_found_error(error):
        return jsonify({
            'success': False,
            'error': 'Resource not found',
            'status_code': 404
        }), 404
    
    @app.errorhandler(405)
    def method_not_allowed(error):
        return jsonify({
            'success': False,
            'error': 'Method not allowed',
            'status_code': 405
        }), 405
    
    @app.errorhandler(413)
    def request_entity_too_large(error):
        max_size = app.config.get('MAX_CONTENT_LENGTH', 0) / (1024 * 1024)
        return jsonify({
            'success': False,
            'error': f'File too large (max {max_size:.1f}MB)',
            'status_code': 413
        }), 413
    
    return app