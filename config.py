"""
Configuration Management for Vehicle Plate Detection System
"""
import os
from pathlib import Path
from typing import Optional, Dict, Any
from pydantic_settings import BaseSettings
from pydantic import Field, validator
import platform


class Settings(BaseSettings):
    """Application settings"""
    
    # Application
    APP_NAME: str = "Vehicle Plate Detection System"
    VERSION: str = "1.0.0"
    DEBUG: bool = Field(default=False, env="DEBUG")
    SECRET_KEY: str = Field(default="dev-secret-key-change-in-production", env="SECRET_KEY")
    
    # Paths
    BASE_DIR: Path = Path(__file__).parent.parent
    UPLOAD_FOLDER: Path = BASE_DIR / "uploads"
    OUTPUT_FOLDER: Path = BASE_DIR / "outputs"
    MODEL_FOLDER: Path = BASE_DIR / "models"
    LOG_FOLDER: Path = BASE_DIR / "logs"
    TEMP_FOLDER: Path = BASE_DIR / "temp"
    
    # Web Server
    HOST: str = Field(default="0.0.0.0", env="HOST")
    PORT: int = Field(default=5000, env="PORT")
    WORKERS: int = Field(default=4, env="WORKERS")
    
    # File Upload
    MAX_UPLOAD_SIZE: int = Field(default=50, env="MAX_UPLOAD_SIZE")  # MB
    ALLOWED_EXTENSIONS: set = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.mp4', '.avi', '.mov'}
    
    # Database
    DATABASE_URL: str = Field(
        default=f"sqlite:///{BASE_DIR / 'data' / 'plates.db'}",
        env="DATABASE_URL"
    )
    
    # Redis (for caching/queue)
    REDIS_URL: str = Field(default="redis://localhost:6379/0", env="REDIS_URL")
    
    # Model Configuration
    MODEL_PATH: str = Field(default="yolov11n.pt", env="MODEL_PATH")
    USE_GPU: bool = Field(default=True, env="USE_GPU")
    CONFIDENCE_THRESHOLD: float = Field(default=0.25, ge=0.0, le=1.0)
    DETECTION_TIMEOUT: int = Field(default=30, ge=1)  # seconds
    
    # OCR Configuration
    ENABLE_EASYOCR: bool = Field(default=True, env="ENABLE_EASYOCR")
    ENABLE_TESSERACT: bool = Field(default=True, env="ENABLE_TESSERACT")
    ENABLE_GOOGLE_VISION: bool = Field(default=False, env="ENABLE_GOOGLE_VISION")
    
    # Tesseract Path (auto-detect)
    TESSERACT_CMD: Optional[str] = Field(default=None, env="TESSERACT_CMD")
    
    # Google Vision API
    GOOGLE_VISION_API_KEY: Optional[str] = Field(default=None, env="GOOGLE_VISION_API_KEY")
    GOOGLE_APPLICATION_CREDENTIALS: Optional[Path] = Field(default=None, env="GOOGLE_APPLICATION_CREDENTIALS")
    
    # Performance
    BATCH_SIZE: int = Field(default=8, ge=1)
    MAX_WORKERS: int = Field(default=4, ge=1)
    CACHE_TTL: int = Field(default=300, ge=0)  # seconds
    
    # Logging
    LOG_LEVEL: str = Field(default="INFO", env="LOG_LEVEL")
    LOG_FORMAT: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    LOG_FILE: Optional[Path] = None
    
    # Security
    RATE_LIMIT_ENABLED: bool = Field(default=True, env="RATE_LIMIT_ENABLED")
    MAX_REQUESTS_PER_MINUTE: int = Field(default=60, ge=1)
    
    # Monitoring
    ENABLE_METRICS: bool = Field(default=True, env="ENABLE_METRICS")
    METRICS_PORT: int = Field(default=9090, env="METRICS_PORT")
    
    class Config:
        env_file = ".env"
        case_sensitive = False
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._setup_directories()
        self._auto_detect_tesseract()
        self._setup_logging()
    
    def _setup_directories(self):
        """Create necessary directories"""
        directories = [
            self.UPLOAD_FOLDER,
            self.OUTPUT_FOLDER,
            self.MODEL_FOLDER,
            self.LOG_FOLDER,
            self.TEMP_FOLDER,
            self.BASE_DIR / "data"
        ]
        
        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)
        
        # Set log file path
        self.LOG_FILE = self.LOG_FOLDER / "app.log"
    
    def _auto_detect_tesseract(self):
        """Auto-detect Tesseract installation path"""
        if self.TESSERACT_CMD and Path(self.TESSERACT_CMD).exists():
            return
        
        if platform.system() == "Windows":
            possible_paths = [
                r"C:\Program Files\Tesseract-OCR\tesseract.exe",
                r"C:\Program Files (x86)\Tesseract-OCR\tesseract.exe",
                r"C:\Users\{}\AppData\Local\Tesseract-OCR\tesseract.exe".format(
                    os.getenv('USERNAME', '')
                ),
                r"C:\Tesseract-OCR\tesseract.exe",
            ]
        elif platform.system() == "Darwin":  # macOS
            possible_paths = [
                "/usr/local/bin/tesseract",
                "/opt/homebrew/bin/tesseract",
                "/usr/bin/tesseract",
            ]
        else:  # Linux
            possible_paths = [
                "/usr/bin/tesseract",
                "/usr/local/bin/tesseract",
                "/opt/bin/tesseract",
            ]
        
        for path in possible_paths:
            if Path(path).exists():
                self.TESSERACT_CMD = path
                break
        
        if not self.TESSERACT_CMD:
            print("⚠️ Warning: Tesseract OCR not found. Please install Tesseract or set TESSERACT_CMD")
    
    def _setup_logging(self):
        """Setup logging configuration"""
        import logging
        from logging.handlers import RotatingFileHandler
        
        # Create logger
        logger = logging.getLogger()
        logger.setLevel(getattr(logging, self.LOG_LEVEL))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(getattr(logging, self.LOG_LEVEL))
        console_formatter = logging.Formatter(self.LOG_FORMAT)
        console_handler.setFormatter(console_formatter)
        logger.addHandler(console_handler)
        
        # File handler
        if self.LOG_FILE:
            file_handler = RotatingFileHandler(
                self.LOG_FILE,
                maxBytes=10*1024*1024,  # 10MB
                backupCount=5
            )
            file_handler.setLevel(getattr(logging, self.LOG_LEVEL))
            file_formatter = logging.Formatter(self.LOG_FORMAT)
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
    
    @property
    def full_model_path(self) -> Path:
        """Get full path to model file"""
        if Path(self.MODEL_PATH).is_absolute():
            return Path(self.MODEL_PATH)
        return self.MODEL_FOLDER / self.MODEL_PATH
    
    @validator('DATABASE_URL', pre=True)
    def validate_database_url(cls, v):
        """Ensure SQLite database directory exists"""
        if v.startswith('sqlite:///'):
            db_path = v.replace('sqlite:///', '')
            db_dir = Path(db_path).parent
            db_dir.mkdir(parents=True, exist_ok=True)
        return v
    
    @validator('UPLOAD_FOLDER', 'OUTPUT_FOLDER', 'TEMP_FOLDER')
    def validate_folders(cls, v, values):
        """Ensure folders are writable"""
        if isinstance(v, str):
            v = Path(v)
        
        # Create folder if it doesn't exist
        v.mkdir(parents=True, exist_ok=True)
        
        # Check if writable
        test_file = v / '.write_test'
        try:
            test_file.touch()
            test_file.unlink()
        except (PermissionError, OSError):
            raise ValueError(f"Directory {v} is not writable")
        
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert settings to dictionary"""
        return {
            k: str(v) if isinstance(v, Path) else v
            for k, v in self.__dict__.items()
            if not k.startswith('_')
        }


# Global settings instance
settings = Settings()