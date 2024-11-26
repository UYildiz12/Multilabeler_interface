import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
from psycopg2 import pool
import time
from flask import Flask, request, jsonify

# Initialize Flask app
app = Flask(__name__)

class ProgressStore:
    def __init__(self):
        # Use environment variable with fallback
        self.database_url = os.environ.get('DATABASE_URL', 
            'postgres://udmo7ja1aie1mf:p2ea685957fb88a9561d081dcceb42a8e4469378e158cb2c005d42fda2b45459f@c67okggoj39697.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d3f0rp0gu9odeo')
            
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)

        # Configure logging
        logging.basicConfig(level=logging.INFO)
        
        # Reduce max connections and add cleanup mechanism
        self.pool = None  # Initialize as None
        self._initialize_pool()
        self.is_shutting_down = False  # Add shutdown flag
            
        # Initialize dictionaries before calling refresh_state
        self.progress_data = {}
        self.last_labeled_indices = {}
        self.locked_categories = {}
        
        # Initialize database
        self.init_db()
        
        # Load initial state
        try:
            self._refresh_state()
            logging.info("Initial state loaded successfully")
        except Exception as e:
            logging.error(f"Failed to load initial state: {e}")
            # Initialize empty state as fallback
            self.progress_data = {}
            self.last_labeled_indices = {}

    def _initialize_pool(self):
        """Initialize or reinitialize the connection pool"""
        try:
            if self.pool:
                try:
                    self.pool.closeall()
                except:
                    pass
            
            self.pool = psycopg2.pool.ThreadedConnectionPool(  # Change to ThreadedConnectionPool
                minconn=1,
                maxconn=5,  # Reduced further to prevent connection exhaustion
                dsn=self.database_url,
                connect_timeout=3
            )
            logging.info("Database connection pool initialized successfully")
        except Exception as e:
            logging.error(f"Failed to create connection pool: {e}")
            raise

    def init_db(self):
        """Initialize database tables"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Modify the table schema to use TEXT for confidence
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS progress (
                        category TEXT NOT NULL,
                        index_id TEXT NOT NULL,
                        label TEXT,
                        confidence TEXT,  # Changed from FLOAT to TEXT
                        timestamp TIMESTAMP,
                        PRIMARY KEY (category, index_id)
                    )
                """)
                # Try to alter existing table if it exists
                try:
                    cur.execute("""
                        ALTER TABLE progress 
                        ALTER COLUMN confidence TYPE TEXT
                        USING confidence::TEXT
                    """)
                except Exception as e:
                    logging.warning(f"Column modification failed (might already be TEXT): {e}")
                
                # Create indices for better query performance 
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_progress_category 
                    ON progress(category)
                """)
                conn.commit()

    def get_db_connection(self):
        """Get database connection with better error handling"""
        if self.is_shutting_down:
            raise Exception("Service is shutting down")
            
        max_retries = 3
        retry_count = 0
        last_error = None
        
        while retry_count < max_retries:
            try:
                if not self.pool or self.pool.closed:
                    self._initialize_pool()
                conn = self.pool.getconn()
                if conn:
                    if conn.closed:
                        self.pool.putconn(conn)
                        raise Exception("Retrieved closed connection")
                    return conn
            except Exception as e:
                last_error = e
                retry_count += 1
                if retry_count < max_retries:
                    time.sleep(1)
                    logging.warning(f"Retrying database connection ({retry_count}/{max_retries})")
                else:
                    logging.error(f"Failed to get connection after {max_retries} attempts: {last_error}")
                    raise

    def return_db_connection(self, conn):
        """Return connection to pool with error handling"""
        if conn:
            try:
                self.pool.putconn(conn)
            except Exception as e:
                logging.error(f"Error returning connection to pool: {e}")
                try:
                    conn.close()
                except:
                    pass

    def __del__(self):
        """Ensure all connections are closed on deletion"""
        try:
            if hasattr(self, 'pool'):
                self.pool.closeall()
                logging.info("Database connection pool closed successfully")
        except Exception as e:
            logging.error(f"Error closing connection pool: {e}")

    def _refresh_state(self):
        """Refresh in-memory state from database with proper error handling"""
        # Create temporary dictionaries to avoid partial updates
        temp_progress_data = {}
        temp_last_labeled_indices = {}
        
        conn = None
        try:
            conn = self.get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    SELECT category, index_id, label, confidence, timestamp 
                    FROM progress
                """)
                rows = cur.fetchall()
                
                for row in rows:
                    category, index_id, label, confidence, timestamp = row
                    
                    if category not in temp_progress_data:
                        temp_progress_data[category] = {}
                        
                    temp_progress_data[category][index_id] = {
                        "label": label,
                        "confidence": confidence,
                        "timestamp": timestamp.isoformat() if timestamp else None
                    }
                    
                    # Update last labeled index
                    if label and label != "unlabeled":
                        current_index = int(index_id)
                        temp_last_labeled_indices[category] = max(
                            temp_last_labeled_indices.get(category, -1),
                            current_index
                        )
                
                # Only update main dictionaries after successful fetch
                self.progress_data = temp_progress_data
                self.last_labeled_indices = temp_last_labeled_indices
                
        except Exception as e:
            logging.error(f"Error refreshing state from database: {e}")
            raise
        finally:
            if conn:
                self.return_db_connection(conn)

    def update(self, category: str, index: str, data: Dict[str, Any]):
        """Update progress with confidence as text"""
        conn = None
        try:
            conn = self.get_db_connection()
            with conn.cursor() as cur:
                # No change needed here as the confidence will be stored as TEXT
                cur.execute("""
                    INSERT INTO progress (category, index_id, label, confidence, timestamp)
                    VALUES (%s, %s, %s, %s, %s)
                    ON CONFLICT (category, index_id) 
                    DO UPDATE SET 
                        label = EXCLUDED.label,
                        confidence = EXCLUDED.confidence,
                        timestamp = EXCLUDED.timestamp
                """, (
                    category,
                    str(index),
                    data.get("label"),
                    str(data.get("confidence")),  # Ensure confidence is string
                    datetime.fromisoformat(data.get("timestamp")) if data.get("timestamp") else None
                ))
                conn.commit()
                    
            # Refresh state after successful update
            self._refresh_state()
            
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Error updating progress: {e}")
            raise
        finally:
            if conn:
                self.return_db_connection(conn)

    def get_progress(self, category: str) -> Dict[str, Any]:
        """Get progress with fresh database state"""
        self._refresh_state()
        return self.progress_data.get(category, {})

    def get_last_labeled_index(self, category: str) -> int:
        """Get last labeled index with fresh database state"""
        self._refresh_state()
        return self.last_labeled_indices.get(category, -1)

    def get_category_stats(self, category: str) -> Dict[str, Any]:
        """Get category statistics directly from database"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    # Get total labeled count
                    cur.execute("""
                        SELECT COUNT(*) 
                        FROM progress 
                        WHERE category = %s 
                        AND label IS NOT NULL 
                        AND label != 'unlabeled'
                    """, (category,))
                    total_labeled = cur.fetchone()[0]
                    
                    # Get label distribution
                    cur.execute("""
                        SELECT label, COUNT(*) 
                        FROM progress 
                        WHERE category = %s 
                        AND label IS NOT NULL 
                        AND label != 'unlabeled'
                        GROUP BY label
                    """, (category,))
                    label_distribution = dict(cur.fetchall())
                    
                    return {
                        'total_labeled': total_labeled,
                        'label_distribution': label_distribution
                    }
                    
        except Exception as e:
            logging.error(f"Error getting category stats: {e}")
            return {'total_labeled': 0, 'label_distribution': {}}

    def get_locked_categories(self) -> Dict[str, str]:
        """Return the currently locked categories."""
        return self.locked_categories

    def acquire_lock(self, user_id: str, category: str) -> bool:
        """Try to acquire a lock for a category."""
        if category in self.locked_categories:
            return False  # Lock already acquired
        self.locked_categories[category] = user_id
        return True

    def release_lock(self, user_id: str, category: str) -> bool:
        """Release the lock for a category."""
        if self.locked_categories.get(category) == user_id:
            del self.locked_categories[category]
            return True
        return False

    def cleanup(self):
        """Proper cleanup method"""
        self.is_shutting_down = True
        if self.pool:
            try:
                self.pool.closeall()
            except:
                pass

# Add Flask routes
@app.route('/get_progress', methods=['GET'])
def get_progress():
    try:
        category = request.args.get('category')
        if not category:
            return jsonify({"error": "Category parameter is required"}), 400
            
        # Convert category back from URL-safe format
        category = category.replace('_', '/')
            
        progress = store.get_progress(category)
        if progress is None:
            return jsonify({}), 200  # Return empty dict if no progress found
            
        return jsonify(progress)
    except Exception as e:
        logging.error(f"Error in get_progress: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/get_all_progress', methods=['GET'])
def get_all_progress():
    try:
        all_progress = {}
        categories = set()
        
        # Get all categories from existing progress
        with store.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT DISTINCT category FROM progress")
                categories = {row[0] for row in cur.fetchall()}
                
        # Get progress for each category
        for category in categories:
            progress = store.get_progress(category)
            if progress:
                all_progress[category] = progress
                
        return jsonify(all_progress)
    except Exception as e:
        logging.error(f"Error in get_all_progress: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/save_progress', methods=['POST'])
def save_progress():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        category = data.get('category')
        index = data.get('index')
        label_data = {k: v for k, v in data.items() if k not in ['category', 'index']}
        
        store.update(category, str(index), label_data)
        return jsonify({"status": "success"})
    except Exception as e:
        logging.error(f"Error in save_progress: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    try:
        # Try to connect to the database
        with store.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
        return jsonify({"status": "healthy"}), 200
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({"status": "unhealthy", "error": str(e)}), 503

# Add a route to handle get_locked_categories
@app.route('/get_locked_categories', methods=['GET'])
def get_locked_categories():
    try:
        locked_categories = store.get_locked_categories()
        return jsonify(locked_categories), 200
    except Exception as e:
        logging.error(f"Error in get_locked_categories: {e}")
        return jsonify({"error": str(e)}), 500

# Add routes for acquire_lock and release_lock
@app.route('/acquire_lock', methods=['POST'])
def acquire_lock():
    data = request.get_json()
    user_id = data.get('user_id')
    category = data.get('category')
    if not user_id or not category:
        return jsonify({"error": "user_id and category are required"}), 400
    success = store.acquire_lock(user_id, category)
    if success:
        return jsonify({"status": "lock acquired"}), 200
    else:
        return jsonify({"error": "lock already acquired"}), 409

@app.route('/release_lock', methods=['POST'])
def release_lock():
    data = request.get_json()
    user_id = data.get('user_id')
    category = data.get('category')
    if not user_id or not category:
        return jsonify({"error": "user_id and category are required"}), 400
    success = store.release_lock(user_id, category)
    if success:
        return jsonify({"status": "lock released"}), 200
    else:
        return jsonify({"error": "failed to release lock"}), 400

@app.route('/upload_progress', methods=['POST'])
def upload_progress():
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
        
        # Assume data is a dictionary with category as keys
        for category, labels in data.items():
            for index, label_data in labels.items():
                store.update(category, index, label_data)
                
        return jsonify({"status": "success"}), 200
    except Exception as e:
        logging.error(f"Error in upload_progress: {e}")
        return jsonify({"error": str(e)}), 500

# Initialize store
store = ProgressStore()

@app.teardown_appcontext
def shutdown_session(exception=None):
    """Only log errors, don't close connections"""
    if exception:
        logging.error(f"Error in request: {str(exception)}")

# Add proper shutdown handling
import atexit
atexit.register(lambda: store.cleanup() if 'store' in globals() else None)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)