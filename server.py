import os
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
from datetime import datetime, timedelta
from typing import Dict, Any
from threading import Lock
import logging

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
    def __init__(self, backup_file: str = "server_progress_backup.json"):
        self.backup_file = backup_file
        self.progress_data: Dict[str, Dict[str, Any]] = self.load_backup()
        self.last_labeled_indices: Dict[str, int] = self.load_last_labeled_indices()

    def load_backup(self) -> Dict[str, Dict[str, Any]]:
        try:
            with open(self.backup_file, 'r') as f:
                return json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

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

    def update(self, category: str, index: str, data: Dict[str, Any]):
        if category not in self.progress_data:
            self.progress_data[category] = {}
            
        # Store the label data
        self.progress_data[category][str(index)] = data
        
        # Update last labeled index and count labeled items
        valid_indices = [i for i, d in self.progress_data[category].items() 
                        if str(i).isdigit() and d.get("label") != "unlabeled"]
        
        self.last_labeled_indices[category] = max(
            [int(i) for i in valid_indices] + [-1]
        )
        
        self.save_backup()


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
PORT = int(os.environ.get('PORT', 8080))
HOST = '0.0.0.0'

# Update the run command at the bottom
if __name__ == "__main__":
    app.run(host=HOST, port=PORT)