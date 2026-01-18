# __init__.py in project root (vehicle-plate-detector/)
"""
Vehicle Plate Detection System
"""

import sys
from pathlib import Path

# Add the current directory to Python path
current_dir = Path(__file__).parent
sys.path.insert(0, str(current_dir))