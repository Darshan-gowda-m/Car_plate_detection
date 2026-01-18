"""
API endpoints package
"""
from .process import process_api
from .upload import upload_api
from .results import results_api
from .system import system_api

__all__ = ['process_api', 'upload_api', 'results_api', 'system_api']