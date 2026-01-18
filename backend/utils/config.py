"""
Configuration management
"""
import os
import json
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv
from loguru import logger

class ConfigManager:
    """Manage application configuration"""
    
    def __init__(self, env_file: str = '.env'):
        self.env_file = env_file
        self.config = {}
        
        self.load_configuration()
    
    def load_configuration(self):
        """Load configuration from environment and config files"""
        # Load .env file
        if Path(self.env_file).exists():
            load_dotenv(self.env_file)
            logger.info(f"Loaded environment from {self.env_file}")
        
        # Load config.json if exists
        config_file = Path('config.json')
        if config_file.exists():
            try:
                with open(config_file, 'r') as f:
                    self.config = json.load(f)
                logger.info(f"Loaded configuration from config.json")
            except Exception as e:
                logger.error(f"Failed to load config.json: {e}")
        
        # Set default values
        self._set_defaults()
        
        # Override with environment variables
        self._override_with_env()
    
    def _set_defaults(self):
        """Set default configuration values"""
        defaults = {
            'app': {
                'name': 'PlateDetect Pro',
                'version': '1.0.0',
                'debug': False,
                'secret_key': 'dev-secret-key-change-in-production',
                'host': '0.0.0.0',
                'port': 5000,
                'max_upload_size_mb': 16
            },
            'database': {
                'url': 'sqlite:///plate_detection.db',
                'echo': False
            },
            'model': {
                'path': 'models/yolov11n.pt',
                'use_gpu': False,
                'confidence_threshold': 0.25
            },
            'ocr': {
                'engines': ['easyocr', 'tesseract', 'google'],
                'preprocess': True,
                'confidence_threshold': 0.5
            },
            'storage': {
                'upload_folder': 'uploads',
                'output_folder': 'outputs',
                'temp_folder': 'temp',
                'log_folder': 'logs',
                'max_age_days': 7
            },
            'performance': {
                'max_workers': 4,
                'batch_size': 8,
                'cache_enabled': True,
                'cache_ttl': 3600
            },
            'monitoring': {
                'enabled': True,
                'interval': 60,
                'retention_days': 30
            }
        }
        
        # Merge with existing config
        for key, value in defaults.items():
            if key not in self.config:
                self.config[key] = value
            else:
                # Deep merge
                if isinstance(value, dict):
                    for subkey, subvalue in value.items():
                        if subkey not in self.config[key]:
                            self.config[key][subkey] = subvalue
    
    def _override_with_env(self):
        """Override configuration with environment variables"""
        env_mappings = {
            'DEBUG': ('app', 'debug', lambda x: x.lower() == 'true'),
            'SECRET_KEY': ('app', 'secret_key', str),
            'WEB_HOST': ('app', 'host', str),
            'WEB_PORT': ('app', 'port', int),
            'MAX_UPLOAD_SIZE': ('app', 'max_upload_size_mb', int),
            
            'DATABASE_URL': ('database', 'url', str),
            
            'MODEL_PATH': ('model', 'path', str),
            'USE_GPU': ('model', 'use_gpu', lambda x: x.lower() == 'true'),
            
            'UPLOAD_FOLDER': ('storage', 'upload_folder', str),
            'OUTPUT_FOLDER': ('storage', 'output_folder', str),
            
            'MAX_WORKERS': ('performance', 'max_workers', int),
            'BATCH_SIZE': ('performance', 'batch_size', int),
            
            'GOOGLE_VISION_API_KEY': ('ocr', 'google_api_key', str)
        }
        
        for env_var, (section, key, converter) in env_mappings.items():
            value = os.getenv(env_var)
            if value is not None:
                try:
                    converted_value = converter(value)
                    self.config[section][key] = converted_value
                    logger.debug(f"Set {section}.{key} from environment: {converted_value}")
                except Exception as e:
                    logger.warning(f"Failed to convert {env_var}={value}: {e}")
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        keys = key.split('.')
        value = self.config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return value
    
    def set(self, key: str, value: Any):
        """Set configuration value using dot notation"""
        keys = key.split('.')
        config = self.config
        
        # Navigate to the parent
        for k in keys[:-1]:
            if k not in config:
                config[k] = {}
            config = config[k]
        
        # Set the value
        config[keys[-1]] = value
    
    def save_to_file(self, filepath: str = 'config.json'):
        """Save configuration to file"""
        try:
            with open(filepath, 'w') as f:
                json.dump(self.config, f, indent=2, default=str)
            logger.info(f"Configuration saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save configuration: {e}")
            return False
    
    def validate(self) -> Dict[str, Any]:
        """Validate configuration"""
        errors = []
        warnings = []
        
        # Check required paths
        required_folders = [
            ('storage.upload_folder', 'Upload folder'),
            ('storage.output_folder', 'Output folder'),
            ('storage.temp_folder', 'Temp folder')
        ]
        
        for config_key, name in required_folders:
            path = self.get(config_key)
            if not path:
                errors.append(f"{name} not configured")
            else:
                try:
                    Path(path).mkdir(parents=True, exist_ok=True)
                except Exception as e:
                    errors.append(f"Cannot create {name} at {path}: {e}")
        
        # Check model path
        model_path = self.get('model.path')
        if model_path and not Path(model_path).exists():
            warnings.append(f"Model file not found at {model_path}")
        
        # Check Google Vision API key if Google OCR is enabled
        ocr_engines = self.get('ocr.engines', [])
        if 'google' in ocr_engines and not self.get('ocr.google_api_key'):
            warnings.append("Google Vision API key not set (Google OCR may not work)")
        
        return {
            'valid': len(errors) == 0,
            'errors': errors,
            'warnings': warnings,
            'config_keys': list(self._flatten_config())
        }
    
    def _flatten_config(self) -> Dict[str, Any]:
        """Flatten configuration for display"""
        def flatten_dict(d, parent_key='', sep='.'):
            items = []
            for k, v in d.items():
                new_key = f"{parent_key}{sep}{k}" if parent_key else k
                if isinstance(v, dict):
                    items.extend(flatten_dict(v, new_key, sep=sep).items())
                else:
                    items.append((new_key, v))
            return dict(items)
        
        return flatten_dict(self.config)
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary"""
        return self.config.copy()

# Global configuration instance
config = ConfigManager()