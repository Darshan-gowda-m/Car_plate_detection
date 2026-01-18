"""
Results API endpoints
"""
import json
from datetime import datetime, timedelta
from flask import Blueprint, request, jsonify, current_app
from sqlalchemy import desc, func, and_, text
import pandas as pd
import csv
import io
from sqlalchemy import text
from flask import make_response

from backend.utils.error_handler import handle_api_error

results_api = Blueprint('results_api', __name__)

@results_api.route('/', methods=['GET'])
@handle_api_error
def get_results():
    """Get paginated results"""
    try:
        # Get query parameters
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 20))
        engine = request.args.get('engine')
        min_confidence = request.args.get('min_confidence', type=float)
        start_date = request.args.get('start_date')
        end_date = request.args.get('end_date')
        search = request.args.get('search')
        
        # Get database components from current_app
        Result = current_app.Result
        db = current_app.db
        
        # Use db.session.query
        query = db.session.query(Result)
        
        # Apply filters
        if engine:
            query = query.filter(Result.best_ocr_engine == engine)
        
        if min_confidence:
            query = query.filter(Result.best_ocr_confidence >= min_confidence)
        
        if start_date:
            try:
                start_dt = datetime.fromisoformat(start_date.replace('Z', '+00:00'))
                query = query.filter(Result.timestamp >= start_dt)
            except:
                pass
        
        if end_date:
            try:
                end_dt = datetime.fromisoformat(end_date.replace('Z', '+00:00'))
                query = query.filter(Result.timestamp <= end_dt)
            except:
                pass
        
        if search:
            query = query.filter(
                Result.filename.ilike(f'%{search}%') |
                Result.best_ocr_text.ilike(f'%{search}%')
            )
        
        # Order by timestamp
        query = query.order_by(desc(Result.timestamp))
        
        # Paginate
        pagination = query.paginate(page=page, per_page=per_page, error_out=False)
        
        results = [result.to_dict() for result in pagination.items]
        
        return jsonify({
            'success': True,
            'results': results,
            'pagination': {
                'page': pagination.page,
                'per_page': pagination.per_page,
                'total': pagination.total,
                'pages': pagination.pages,
                'has_next': pagination.has_next,
                'has_prev': pagination.has_prev
            }
        }), 200
        
    except Exception as e:
        current_app.logger.error(f"Error getting results: {e}")
        return jsonify({
            'success': False,
            'error': str(e),
            'results': []
        }), 500
