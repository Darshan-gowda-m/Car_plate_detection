# backend/core/database.py
from datetime import datetime
import json
import numpy as np

# DON'T create SQLAlchemy instance here
# Just define the Base model that other models can inherit from

class BaseModel:
    """Base model with common functionality"""
    
    def to_dict(self):
        """Convert model to dictionary"""
        result = {}
        for column in self.__table__.columns:
            value = getattr(self, column.name)
            result[column.name] = self._make_json_serializable(value)
        return result
    
    def _make_json_serializable(self, obj):
        """Recursively convert numpy types to Python native types"""
        if isinstance(obj, (np.integer, np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, (np.floating, np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.bool_):
            return bool(obj)
        elif isinstance(obj, datetime):
            return obj.isoformat()
        elif isinstance(obj, dict):
            return {k: self._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._make_json_serializable(item) for item in obj]
        else:
            return obj

# Models will be defined in app.py or imported elsewhere