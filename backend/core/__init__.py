"""
Core functionality package - UPDATED
"""
from .detector import PlateDetector
from .ocr_engines import OCREngineManager
from .preprocessor import ImagePreprocessor
from .validator import Validator

# Don't import from database here - models are defined in app.py
# Create placeholders instead
db = None
Result = None
BatchJob = None
SystemMetrics = None

__all__ = [
    'PlateDetector',
    'OCREngineManager', 
    'ImagePreprocessor',
    'Validator',
    'db',
    'Result',
    'BatchJob',
    'SystemMetrics'
]