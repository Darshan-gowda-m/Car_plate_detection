"""
Database verification utilities
"""
import sqlite3
from pathlib import Path
from sqlalchemy import text
import json

def verify_database_connection(app):
    """Verify database connection and tables"""
    try:
        with app.app_context():
            # Test basic connection
            db.session.execute(text('SELECT 1'))
            print("✅ Database connection: OK")
            
            # Check if tables exist
            from backend.core.database import Result, BatchJob
            
            # Check Result table
            result_count = db.session.query(Result).count()
            print(f"✅ Results table: {result_count} records")
            
            # Check BatchJob table
            batch_count = db.session.query(BatchJob).count()
            print(f"✅ BatchJobs table: {batch_count} records")
            
            # Check indexes
            try:
                # SQLite specific check
                db_path = Path(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
                if db_path.exists():
                    conn = sqlite3.connect(str(db_path))
                    cursor = conn.cursor()
                    
                    # Get index info
                    cursor.execute("SELECT name FROM sqlite_master WHERE type='index';")
                    indexes = cursor.fetchall()
                    print(f"✅ Database indexes: {len(indexes)} found")
                    
                    conn.close()
            except:
                print("⚠️ Could not verify indexes")
            
            return True
            
    except Exception as e:
        print(f"❌ Database verification failed: {e}")
        return False

def repair_database(app):
    """Attempt to repair database issues"""
    print("\n🛠️ Attempting database repair...")
    
    try:
        db_path = Path(app.config['SQLALCHEMY_DATABASE_URI'].replace('sqlite:///', ''))
        
        if not db_path.exists():
            print("❌ Database file does not exist")
            return False
        
        # Create backup
        backup_path = db_path.with_suffix('.backup.db')
        import shutil
        shutil.copy2(db_path, backup_path)
        print(f"📁 Backup created: {backup_path}")
        
        with app.app_context():
            # Drop and recreate tables
            print("🔄 Recreating tables...")
            db.drop_all()
            db.create_all()
            
            # Create indexes
            from sqlalchemy import text
            
            indexes = [
                text('CREATE INDEX IF NOT EXISTS idx_results_timestamp ON results(timestamp)'),
                text('CREATE INDEX IF NOT EXISTS idx_results_confidence ON results(best_ocr_confidence)'),
                text('CREATE INDEX IF NOT EXISTS idx_results_filename ON results(filename)'),
                text('CREATE INDEX IF NOT EXISTS idx_batch_status ON batch_jobs(status)')
            ]
            
            for index_sql in indexes:
                try:
                    db.session.execute(index_sql)
                    print(f"✅ Index created: {index_sql}")
                except Exception as e:
                    print(f"⚠️ Index creation warning: {e}")
            
            db.session.commit()
            
            print("✅ Database repair completed")
            return True
            
    except Exception as e:
        print(f"❌ Database repair failed: {e}")
        return False