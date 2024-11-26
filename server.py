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
        logging.debug("=== Initializing ProgressStore ===")
        self.progress_data = self.load_from_db()
        logging.debug(f"Initial progress data: {json.dumps(self.progress_data, indent=2)}")
        logging.debug(f"Categories loaded: {list(self.progress_data.keys())}")
        self.last_labeled_indices = self.load_last_labeled_indices()
        logging.debug(f"Last labeled indices: {self.last_labeled_indices}")

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
        logging.debug("\n=== Loading from Database ===")
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT category, data FROM progress")
                    results = cur.fetchall()
                    loaded_data = {row[0]: row[1] for row in results}
                    logging.debug(f"Loaded categories: {list(loaded_data.keys())}")
                    for cat, data in loaded_data.items():
                        logging.debug(f"\nCategory: {cat}")
                        logging.debug(f"Number of items: {len(data)}")
                        logging.debug(f"Sample items: {dict(list(data.items())[:2])}")
                    return loaded_data
        except Exception as e:
            logging.error(f"Database load error: {e}", exc_info=True)
            return {}

    def save_to_db(self):
        logging.debug("\n=== Saving to Database ===")
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    for category, data in self.progress_data.items():
                        logging.debug(f"\nCategory: {category}")
                        logging.debug(f"Data size: {len(data)} items")
                        logging.debug(f"Sample of data being saved: {dict(list(data.items())[:2])}")
                        
                        cur.execute("""
                            INSERT INTO progress (category, data)
                            VALUES (%s, %s)
                            ON CONFLICT (category) 
                            DO UPDATE SET data = %s
                        """, (category, Json(data), Json(data)))
                    conn.commit()
                    logging.debug("Database commit successful")
        except Exception as e:
            logging.error(f"Database save error: {e}", exc_info=True)
            raise

    def update(self, category: str, index: str, data: Dict[str, Any]):
        logging.debug(f"\n=== Updating Progress ===")
        logging.debug(f"Category: {category}")
        logging.debug(f"Index: {index}")
        logging.debug(f"Data: {json.dumps(data, indent=2)}")
        logging.debug(f"Previous state for this index: {self.progress_data.get(category, {}).get(index, 'Not found')}")
        
        if category not in self.progress_data:
            logging.debug(f"Creating new category: {category}")
            self.progress_data[category] = {}
        
        # Store previous state for verification
        prev_state = self.progress_data[category].copy()
        
        self.progress_data[category][str(index)] = data
        
        # Verify update
        logging.debug("\n=== Verifying Update ===")
        logging.debug(f"Previous category state: {json.dumps(prev_state, indent=2)}")
        logging.debug(f"New category state: {json.dumps(self.progress_data[category], indent=2)}")
        logging.debug(f"Verification - data at index {index}: {self.progress_data[category].get(str(index))}")
        
        try:
            self.save_to_db()
            logging.debug("Update successfully saved to database")
        except Exception as e:
            logging.error(f"Failed to save update: {e}")
            # Rollback in memory if DB save failed
            self.progress_data[category] = prev_state
            logging.debug("Rolled back to previous state due to save failure")
            raise

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
        logging.debug(f"\n=== Getting Progress for {category} ===")
        progress = self.progress_data.get(category, {})
        logging.debug(f"Found {len(progress)} items")
        logging.debug(f"Keys present: {list(progress.keys())[:5]}")
        logging.debug(f"Sample data: {dict(list(progress.items())[:2])}")
        return progress

    def get_last_labeled_index(self, category: str) -> int:
        return self.last_labeled_indices.get(category, -1)

    def get_category_stats(self, category: str) -> Dict[str, Any]:
        """Get accurate statistics for a category"""
        try:
            logging.debug(f"=== Getting stats for category: {category} ===")
            category_data = self.progress_data.get(category, {})
            logging.debug(f"Raw category data length: {len(category_data)}")
            logging.debug(f"Category data keys: {list(category_data.keys())}")

            # Debug print first few items
            sample_items = dict(list(category_data.items())[:5])
            logging.debug(f"Sample data items: {json.dumps(sample_items, indent=2)}")

            total_labeled = 0
            label_distribution = {}
            
            logging.debug("Starting label counting...")
            for idx, data in category_data.items():
                logging.debug(f"\nProcessing index {idx}:")
                logging.debug(f"Data type: {type(data)}")
                logging.debug(f"Data content: {data}")
                
                if not isinstance(data, dict):
                    logging.warning(f"Skipping non-dict data at index {idx}")
                    continue

                label = data.get('label', '')
                logging.debug(f"Found label: '{label}'")
                
                if label and label != 'unlabeled':
                    total_labeled += 1
                    label_distribution[label] = label_distribution.get(label, 0) + 1
                    logging.debug(f"Valid label found. Running total: {total_labeled}")
                else:
                    logging.debug("Skipping empty or unlabeled item")

            logging.debug("\n=== Final Statistics ===")
            logging.debug(f"Total labeled items: {total_labeled}")
            logging.debug(f"Label distribution: {label_distribution}")
            
            stats = {
                'total_labeled': total_labeled,
                'label_distribution': label_distribution
            }
            logging.debug(f"Returning stats: {json.dumps(stats, indent=2)}")
            return stats

        except Exception as e:
            logging.error(f"Error getting category stats: {str(e)}", exc_info=True)
            return {'total_labeled': 0, 'label_distribution': {}}

progress_store = ProgressStore()
category_lock = CategoryLock()

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - [%(filename)s:%(lineno)d] - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('server_debug.log')
    ]
)

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
    logging.debug("\n=== Save Progress Request ===")
    data = request.json
    logging.debug(f"Received data: {json.dumps(data, indent=2)}")
    
    if not data:
        logging.error("No data provided in request")
        return jsonify({"error": "No data provided"}), 400

    try:
        required_fields = ['category', 'index', 'label', 'confidence', 'timestamp']
        if not all(field in data for field in required_fields):
            missing = [f for f in required_fields if f not in data]
            logging.error(f"Missing required fields: {missing}")
            return jsonify({"error": "Missing required fields"}), 400

        logging.debug("\n=== Before Update ===")
        logging.debug(f"Current state for category {data['category']}: {progress_store.get_progress(data['category'])}")
        
        progress_store.update(
            data['category'],
            str(data['index']),
            {
                "label": data["label"],
                "confidence": data["confidence"],
                "timestamp": data["timestamp"]
            }
        )
        
        logging.debug("\n=== After Update ===")
        logging.debug(f"New state for category {data['category']}: {progress_store.get_progress(data['category'])}")
        
        return jsonify({"message": "Progress saved"}), 200
    except Exception as e:
        logging.error(f"Error in save_progress: {str(e)}", exc_info=True)
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

@app.route('/get_category_stats', methods=['GET'])
def get_category_stats():
    category = request.args.get('category')
    logging.debug(f"\n=== Category Stats Request ===")
    logging.debug(f"Category: {category}")
    
    if not category:
        logging.error("Missing category parameter")
        return jsonify({"error": "Category parameter required"}), 400
    
    stats = progress_store.get_category_stats(category)
    logging.debug(f"Returning stats: {json.dumps(stats, indent=2)}")
    return jsonify(stats), 200

# Add port configuration
PORT = int(os.environ.get('PORT', 5000))  # Set default port to 8080
HOST = '0.0.0.0'

# Update the run command at the bottom
if __name__ == "__main__":
    app.run(host=HOST, port=PORT)