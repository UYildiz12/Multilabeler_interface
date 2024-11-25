import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime, timedelta
from typing import Dict, Any
from threading import Lock
import logging
import psycopg2
from psycopg2.extras import Json
from urllib.parse import urlparse

app = Flask(__name__)
CORS(app)

class CategoryLock:
    def __init__(self):
        self.locks: Dict[str, Dict[str, Any]] = {}
        self.lock = Lock()
        
    def acquire_lock(self, category: str, user_id: str) -> bool:
        with self.lock:
            current_time = datetime.now()
            # Clean up expired locks
            self._cleanup_expired_locks()
            
            # Check if category is locked by another user
            if category in self.locks:
                lock_info = self.locks[category]
                if lock_info['user_id'] != user_id:
                    return False
                
            # Create or update lock
            self.locks[category] = {
                'user_id': user_id,
                'timestamp': current_time,
                'expires': current_time + timedelta(minutes=30)  # Lock expires after 30 minutes
            }
            return True
            
    def release_lock(self, category: str, user_id: str) -> bool:
        with self.lock:
            if category in self.locks and self.locks[category]['user_id'] == user_id:
                del self.locks[category]
                return True
            return False
            
    def _cleanup_expired_locks(self):
        current_time = datetime.now()
        expired = [cat for cat, info in self.locks.items() 
                  if info['expires'] < current_time]
        for cat in expired:
            del self.locks[cat]
            
    def get_locked_categories(self) -> Dict[str, str]:
        with self.lock:
            self._cleanup_expired_locks()
            return {cat: info['user_id'] 
                   for cat, info in self.locks.items()}

class ProgressStore:
    def __init__(self):
        self.database_url = os.environ.get('DATABASE_URL')
        if self.database_url and self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)
        self.init_db()
        self.progress_data = self.load_from_db()
        self.last_labeled_indices = self.load_last_labeled_indices()

    def init_db(self):
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS progress (
                        category TEXT PRIMARY KEY,
                        data JSONB
                    )
                """)
                conn.commit()

    def get_db_connection(self):
        return psycopg2.connect(self.database_url)

    def load_from_db(self) -> Dict[str, Dict[str, Any]]:
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT category, data FROM progress")
                    results = cur.fetchall()
                    return {row[0]: row[1] for row in results}
        except Exception as e:
            logging.error(f"Database load error: {e}")
            return {}

    def save_to_db(self):
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    for category, data in self.progress_data.items():
                        cur.execute("""
                            INSERT INTO progress (category, data)
                            VALUES (%s, %s)
                            ON CONFLICT (category) 
                            DO UPDATE SET data = %s
                        """, (category, Json(data), Json(data)))
                    conn.commit()
        except Exception as e:
            logging.error(f"Database save error: {e}")

    def update(self, category: str, index: str, data: Dict[str, Any]):
        if category not in self.progress_data:
            self.progress_data[category] = {}
        
        self.progress_data[category][str(index)] = data
        
        # Update last labeled index
        current_index = int(index)
        self.last_labeled_indices[category] = max(
            self.last_labeled_indices.get(category, -1),
            current_index
        )
        
        self.save_to_db()

    def load_last_labeled_indices(self) -> Dict[str, int]:
        last_indices = {}
        for category, labels in self.progress_data.items():
            if labels:
                # Only consider keys that can be converted to integers
                valid_keys = [int(index) for index in labels.keys() if index.isdigit()]
                last_indices[category] = max(valid_keys) if valid_keys else -1
            else:
                last_indices[category] = -1
        return last_indices


    def save_backup(self):
        with open(self.backup_file, 'w') as f:
            json.dump(self.progress_data, f, indent=2)

    def get_progress(self, category: str) -> Dict[str, Any]:
        return self.progress_data.get(category, {})

    def get_last_labeled_index(self, category: str) -> int:
        return self.last_labeled_indices.get(category, -1)




# Initialize stores
progress_store = ProgressStore()
category_lock = CategoryLock()

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

@app.route('/acquire_lock', methods=['POST'])
def acquire_lock():
    data = request.json
    logging.debug(f"Acquire Lock Request: {data}")
    if not data or 'category' not in data or 'user_id' not in data:
        return jsonify({"error": "Missing required fields"}), 400
        
    success = category_lock.acquire_lock(data['category'], data['user_id'])
    if success:
        logging.info(f"Lock acquired for category {data['category']} by user {data['user_id']}")
        return jsonify({"message": "Lock acquired"}), 200
    else:
        logging.warning(f"Failed to acquire lock for category {data['category']}, locked by another user")
        return jsonify({"error": "Category is locked by another user"}), 409

@app.route('/release_lock', methods=['POST'])
def release_lock():
    data = request.json
    if not data or 'category' not in data or 'user_id' not in data:
        return jsonify({"error": "Missing required fields"}), 400
        
    success = category_lock.release_lock(data['category'], data['user_id'])
    return jsonify({"message": "Lock released"}), 200

@app.route('/get_last_labeled_index', methods=['GET'])
def get_last_labeled_index():
    category = request.args.get('category')
    if not category:
        return jsonify({"error": "Missing category parameter"}), 400

    last_index = progress_store.get_last_labeled_index(category)
    return jsonify({"last_labeled_index": last_index}), 200

@app.route('/get_locked_categories', methods=['GET'])
def get_locked_categories():
    return jsonify(category_lock.get_locked_categories()), 200

@app.route('/save_progress', methods=['POST'])
def save_progress():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        required_fields = ['category', 'index', 'label', 'confidence', 'timestamp']
        if not all(field in data for field in required_fields):
            return jsonify({"error": "Missing required fields"}), 400

        progress_store.update(
            data['category'],
            str(data['index']),
            {
                "label": data["label"],
                "confidence": data["confidence"],
                "timestamp": data["timestamp"]
            }
        )
        return jsonify({"message": "Progress saved"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/get_progress', methods=['GET'])
def get_progress():
    category = request.args.get('category')
    
    if not category:
        return jsonify({"error": "Missing required parameters"}), 400

    return jsonify(progress_store.get_progress(category)), 200


@app.route('/get_all_progress', methods=['GET'])
def get_all_progress():
    return jsonify(progress_store.progress_data), 200

@app.route('/upload_progress', methods=['POST'])
def upload_progress():
    data = request.json
    if not data:
        return jsonify({"error": "No data provided"}), 400

    try:
        for category, labels in data.items():
            for index, label_data in labels.items():
                # Ensure the index is numeric
                if not str(index).isdigit():
                    continue
                progress_store.update(category, index, label_data)
        return jsonify({"message": "Progress uploaded successfully"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/')
def home():
    return jsonify({"status": "healthy"}), 200

@app.route('/health')
def health():
    return jsonify({
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }), 200

# Add port configuration
PORT = int(os.environ.get('PORT', 5000))  # Set default port to 8080
HOST = '0.0.0.0'

# Update the run command at the bottom
if __name__ == "__main__":
    app.run(host=HOST, port=PORT)