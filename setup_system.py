#!/usr/bin/env python3
"""
Setup script for Vehicle Plate Detection System
"""
import os
import sys
import subprocess
from pathlib import Path

def print_header(text):
    """Print formatted header"""
    print("\n" + "="*80)
    print(f" {text}")
    print("="*80)

def run_command(command, description):
    """Run shell command with error handling"""
    print(f"\n🔧 {description}")
    print(f"   Command: {command}")
    
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True)
        if result.returncode == 0:
            print(f"   ✅ Success")
            if result.stdout:
                print(f"   Output: {result.stdout[:200]}...")
        else:
            print(f"   ❌ Failed: {result.stderr}")
            return False
        return True
    except Exception as e:
        print(f"   ❌ Error: {e}")
        return False

def check_python_version():
    """Check Python version"""
    print_header("CHECKING PYTHON VERSION")
    
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher required")
        return False
    
    print("✅ Python version OK")
    return True

def create_directories():
    """Create required directories"""
    print_header("CREATING DIRECTORIES")
    
    directories = [
        'uploads',
        'outputs',
        'logs',
        'temp',
        'models',
        'data',
        'frontend/templates',
        'frontend/static',
        'backend',
        'backend/core',
        'backend/api',
        'backend/utils',
        'backend/workers',
        'backend/models'
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print(f"📁 Created: {directory}")
    
    return True

def install_requirements():
    """Install Python packages"""
    print_header("INSTALLING REQUIREMENTS")
    
    requirements = [
        "flask",
        "flask-cors",
        "flask-socketio",
        "flask-sqlalchemy",
        "python-dotenv",
        "opencv-python",
        "numpy",
        "pillow",
        "psutil",
        "ultralytics",  # YOLO
        "torch",        # PyTorch
        "torchvision",
        "easyocr",
        "pytesseract",
        "google-cloud-vision",
        "pandas",
        "werkzeug",
        "eventlet"
    ]
    
    for package in requirements:
        success = run_command(f"pip install {package}", f"Installing {package}")
        if not success:
            print(f"⚠️ Failed to install {package}")
    
    return True

def download_models():
    """Download pre-trained models"""
    print_header("DOWNLOADING MODELS")
    
    models = {
        'yolov8n.pt': 'https://github.com/ultralytics/assets/releases/download/v0.0.0/yolov8n.pt',
        'haarcascade_russian_plate_number.xml': 'https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_russian_plate_number.xml'
    }
    
    import urllib.request
    
    for filename, url in models.items():
        model_path = Path('models') / filename
        if not model_path.exists():
            print(f"\n📥 Downloading {filename}...")
            try:
                urllib.request.urlretrieve(url, str(model_path))
                print(f"✅ Downloaded: {filename}")
            except Exception as e:
                print(f"❌ Failed to download {filename}: {e}")
        else:
            print(f"✅ Model exists: {filename}")
    
    return True

def setup_database():
    """Setup SQLite database"""
    print_header("SETTING UP DATABASE")
    
    import sqlite3
    
    db_path = Path('data') / 'plates_prod.db'
    
    if db_path.exists():
        print(f"✅ Database exists: {db_path}")
        return True
    
    print(f"📊 Creating database: {db_path}")
    
    try:
        conn = sqlite3.connect(str(db_path))
        cursor = conn.cursor()
        
        # Create tables
        tables = [
            '''
            CREATE TABLE IF NOT EXISTS results (
                id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                filepath TEXT,
                detections TEXT,
                detection_method TEXT,
                detection_confidence REAL,
                ocr_results TEXT,
                best_ocr_engine TEXT,
                best_ocr_text TEXT,
                best_ocr_confidence REAL,
                processing_time REAL,
                detection_time REAL,
                ocr_time REAL,
                cropped_path TEXT,
                visualization_path TEXT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                user_id TEXT,
                session_id TEXT,
                status TEXT DEFAULT 'completed',
                error_message TEXT,
                plate_count INTEGER DEFAULT 0,
                plate_locations TEXT,
                vehicle_type TEXT,
                plate_type TEXT,
                country_code TEXT
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS batch_jobs (
                id TEXT PRIMARY KEY,
                name TEXT,
                total_files INTEGER DEFAULT 0,
                processed_files INTEGER DEFAULT 0,
                failed_files INTEGER DEFAULT 0,
                options TEXT,
                results TEXT,
                summary TEXT,
                status TEXT DEFAULT 'pending',
                progress REAL DEFAULT 0.0,
                started_at DATETIME,
                completed_at DATETIME,
                elapsed_time REAL,
                created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
                created_by TEXT
            )
            ''',
            '''
            CREATE TABLE IF NOT EXISTS system_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                detection_count INTEGER DEFAULT 0,
                avg_detection_confidence REAL DEFAULT 0.0,
                avg_detection_time REAL DEFAULT 0.0,
                ocr_count INTEGER DEFAULT 0,
                avg_ocr_confidence REAL DEFAULT 0.0,
                avg_ocr_time REAL DEFAULT 0.0,
                easyocr_stats TEXT,
                tesseract_stats TEXT,
                google_vision_stats TEXT,
                cpu_usage REAL,
                memory_usage REAL,
                gpu_usage REAL
            )
            '''
        ]
        
        for table_sql in tables:
            cursor.execute(table_sql)
        
        # Create indexes
        indexes = [
            'CREATE INDEX IF NOT EXISTS idx_results_timestamp ON results(timestamp)',
            'CREATE INDEX IF NOT EXISTS idx_results_confidence ON results(best_ocr_confidence)',
            'CREATE INDEX IF NOT EXISTS idx_results_status ON results(status)',
            'CREATE INDEX IF NOT EXISTS idx_batch_jobs_status ON batch_jobs(status)',
            'CREATE INDEX IF NOT EXISTS idx_batch_jobs_created ON batch_jobs(created_at)'
        ]
        
        for index_sql in indexes:
            cursor.execute(index_sql)
        
        conn.commit()
        conn.close()
        
        print("✅ Database created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Database setup failed: {e}")
        return False

def create_config_file():
    """Create configuration file"""
    print_header("CREATING CONFIGURATION")
    
    config_content = '''# Vehicle Plate Detection System Configuration
# Copy this to .env file

# Application
SECRET_KEY=your-secret-key-change-in-production
DEBUG=False

# Server
HOST=0.0.0.0
PORT=5000

# Database
DATABASE_URL=sqlite:///data/plates_prod.db

# AI Models
MODEL_PATH=models/yolov8n.pt
USE_GPU=True

# OCR Engines
GOOGLE_VISION_API_KEY=your-google-api-key-here
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Performance
MAX_WORKERS=4
BATCH_SIZE=8

# Storage
MAX_UPLOAD_SIZE_MB=100
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs

# Logging
LOG_LEVEL=INFO
LOG_FILE=logs/app.log
'''

    config_path = Path('config.env')
    config_path.write_text(config_content)
    
    print(f"✅ Configuration template created: {config_path}")
    print("\n📝 Next steps:")
    print("   1. Rename config.env to .env")
    print("   2. Update the values in .env file")
    print("   3. For Google Vision, get API key from: https://console.cloud.google.com/")
    
    return True

def test_installation():
    """Test the installation"""
    print_header("TESTING INSTALLATION")
    
    tests = [
        ("python -c \"import flask; print(f'Flask {flask.__version__}')\"", "Flask"),
        ("python -c \"import cv2; print(f'OpenCV {cv2.__version__}')\"", "OpenCV"),
        ("python -c \"import torch; print(f'PyTorch {torch.__version__}')\"", "PyTorch"),
        ("python -c \"import ultralytics; print('YOLO OK')\"", "YOLO"),
        ("python -c \"import easyocr; print('EasyOCR OK')\"", "EasyOCR"),
        ("python -c \"import pytesseract; print('Tesseract OK')\"", "Tesseract")
    ]
    
    all_passed = True
    
    for command, test_name in tests:
        success = run_command(command, f"Testing {test_name}")
        if not success:
            all_passed = False
    
    return all_passed

def main():
    """Main setup function"""
    print_header("🚀 VEHICLE PLATE DETECTION SYSTEM SETUP")
    
    steps = [
        ("Python Version Check", check_python_version),
        ("Creating Directories", create_directories),
        ("Installing Requirements", install_requirements),
        ("Downloading Models", download_models),
        ("Setting Up Database", setup_database),
        ("Creating Configuration", create_config_file),
        ("Testing Installation", test_installation)
    ]
    
    results = []
    
    for step_name, step_func in steps:
        print(f"\n▶️ {step_name}")
        try:
            success = step_func()
            results.append((step_name, success))
            if not success:
                print(f"❌ {step_name} failed")
                # Continue anyway
        except Exception as e:
            print(f"❌ {step_name} error: {e}")
            results.append((step_name, False))
    
    # Summary
    print_header("SETUP SUMMARY")
    
    for step_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {step_name}")
    
    failed_steps = [name for name, success in results if not success]
    
    if failed_steps:
        print(f"\n⚠️  Failed steps: {', '.join(failed_steps)}")
        print("   Some features may not work properly")
    else:
        print("\n🎉 All steps completed successfully!")
    
    print("\n" + "="*80)
    print("🚀 SETUP COMPLETE")
    print("="*80)
    print("\nTo start the system:")
    print("   1. Update the .env file with your configuration")
    print("   2. Run: python app.py")
    print("\nAccess the system at: http://localhost:5000")
    print("\nFor issues, check the logs in the 'logs' directory")
    print("="*80)

if __name__ == "__main__":
    main()