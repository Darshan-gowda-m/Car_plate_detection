#!/usr/bin/env python3
"""
Database initialization script
"""
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Create a separate Flask app instance for database initialization
from flask import Flask
from backend.core.database import db

def init_database():
    """Initialize database tables"""
    # Create a minimal Flask app for initialization
    app = Flask(__name__)
    app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///./data/plates.db'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Initialize database with this app
    db.init_app(app)
    
    with app.app_context():
        try:
            # Create all tables
            db.create_all()
            
            print("✅ Database tables created successfully!")
            
            # Test database connection
            from sqlalchemy import text
            try:
                db.session.execute(text("SELECT 1"))
                print("✅ Database connection test successful")
            except Exception as e:
                print(f"❌ Database connection failed: {e}")
                
        except Exception as e:
            print(f"❌ Error creating database: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    print("🚀 Initializing Database...")
    init_database()
    print("✅ Database initialization complete!")