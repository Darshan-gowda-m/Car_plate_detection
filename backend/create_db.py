#!/usr/bin/env python3
"""
Create database tables manually
"""
import os
import sys
import sqlite3
from pathlib import Path

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def create_database():
    """Create database and tables manually"""
    
    # Create data directory
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    
    db_path = data_dir / "plates.db"
    
    print(f"📁 Creating database at: {db_path}")
    
    # Connect to SQLite
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    
    # Check if tables exist
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = cursor.fetchall()
    
    if tables:
        print("📊 Existing tables:")
        for table in tables:
            print(f"  - {table[0]}")
    else:
        print("📝 No tables found. Creating them...")
    
    # Create results table
    cursor.execute('''
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
        error_message TEXT
    )
    ''')
    print("✅ Created 'results' table")
    
    # Create batch_jobs table
    cursor.execute('''
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
    ''')
    print("✅ Created 'batch_jobs' table")
    
    # Create system_metrics table
    cursor.execute('''
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
    ''')
    print("✅ Created 'system_metrics' table")
    
    # Create indexes
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_timestamp ON results(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_confidence ON results(best_ocr_confidence)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_results_engine ON results(best_ocr_engine)')
    
    print("✅ Created indexes")
    
    conn.commit()
    
    # Verify tables
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name")
    tables = cursor.fetchall()
    
    print("\n📊 Database structure:")
    for table in tables:
        print(f"\nTable: {table[0]}")
        cursor.execute(f"PRAGMA table_info({table[0]})")
        columns = cursor.fetchall()
        for col in columns:
            print(f"  - {col[1]} ({col[2]})")
    
    conn.close()
    print("\n✅ Database created successfully!")
    
    return True

if __name__ == '__main__':
    create_database()