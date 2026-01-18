#!/usr/bin/env python3
"""
Simplified runner for Plate Detection System
"""
import os
import sys
from pathlib import Path

# Add project root to path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))

def main():
    """Main entry point"""
    print("🚗 Starting Vehicle Plate Detection System...")
    print("=" * 50)
    
    # Create necessary directories
    directories = ['uploads', 'outputs', 'logs', 'temp', 'models', 
                   'frontend/templates', 'frontend/static', 'data']
    
    for dir_name in directories:
        Path(dir_name).mkdir(parents=True, exist_ok=True)
        print(f"✓ Created/verified: {dir_name}")
    
    # Check for .env file
    if not Path('.env').exists():
        print("\n⚠️ Warning: .env file not found. Creating default...")
        with open('.env', 'w') as f:
            f.write("""# Application Settings
DEBUG=True
SECRET_KEY=dev-secret-key-change-in-production
WEB_HOST=0.0.0.0
WEB_PORT=5000
MAX_CONTENT_LENGTH=50

# Database
DATABASE_URL=sqlite:///data/plates.db

# Model Settings
MODEL_PATH=models/yolov11n.pt
USE_GPU=False

# OCR Settings
GOOGLE_VISION_API_KEY=

# Storage
UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
TEMP_FOLDER=temp

# Performance
MAX_WORKERS=4
BATCH_SIZE=8

# Logging
LOG_LEVEL=INFO
""")
        print("✓ Created .env file with default settings")
    
    print("\n" + "=" * 50)
    print("🔧 Initializing Application...")
    
    # Import and run app
    try:
        from app import app, socketio
        
        # Get configuration
        host = os.getenv('WEB_HOST', '0.0.0.0')
        port = int(os.getenv('WEB_PORT', 5000))
        debug = os.getenv('DEBUG', 'True').lower() == 'true'
        
        print(f"\n✅ Application initialized successfully!")
        print(f"📱 Server URL: http://{host}:{port}")
        print(f"🔧 Debug Mode: {debug}")
        print(f"📁 Uploads: {os.getenv('UPLOAD_FOLDER', 'uploads')}")
        print(f"📁 Outputs: {os.getenv('OUTPUT_FOLDER', 'outputs')}")
        print(f"🤖 Model: {os.getenv('MODEL_PATH', 'models/yolov11n.pt')}")
        print(f"⚡ GPU Acceleration: {os.getenv('USE_GPU', 'False')}")
        print("\n" + "=" * 50)
        print("🚀 Starting server...")
        print("Press Ctrl+C to stop\n")
        
        # Remove the problematic parameter and use simpler run configuration
        try:
            # Try with default parameters first
            socketio.run(
                app,
                host=host,
                port=port,
                debug=debug,
                log_output=True
            )
        except TypeError as te:
            print(f"⚠️ SocketIO configuration issue: {te}")
            print("🔄 Trying alternative configuration...")
            
            # Fallback to simpler configuration
            if 'allow_unsafe_werkzeug' in str(te):
                socketio.run(
                    app,
                    host=host,
                    port=port,
                    debug=debug
                )
            else:
                raise te
                
    except ImportError as e:
        print(f"\n❌ Import error: {e}")
        print("\n📦 Please install required dependencies:")
        print("pip install flask flask-socketio flask-cors flask-sqlalchemy loguru ultralytics opencv-python")
        print("\nFor OCR support:")
        print("pip install easyocr pytesseract pillow")
        print("\nFor system monitoring:")
        print("pip install psutil")
        sys.exit(1)
        
    except KeyboardInterrupt:
        print("\n\n🛑 Server stopped by user")
        
    except Exception as e:
        print(f"\n❌ Error starting application: {e}")
        import traceback
        traceback.print_exc()
        
        # Try to run Flask directly as last resort
        print("\n🔄 Attempting to run Flask directly (without SocketIO)...")
        try:
            app.run(host=host, port=port, debug=debug)
        except Exception as flask_error:
            print(f"❌ Flask also failed: {flask_error}")
            sys.exit(1)

if __name__ == '__main__':
    main()