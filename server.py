import os
import json
import logging
from typing import Dict, Any
from datetime import datetime, timedelta
import psycopg2
from psycopg2.extras import Json
from psycopg2 import pool
import time
from flask import Flask, request, jsonify
from typing import Dict, Any, Callable, List

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
        self.locked_categories = {}  # {category: {"user": username, "timestamp": datetime}}
        
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

        self.active_users = {}  # Track active users and their last activity
        self.user_session_timeout = timedelta(hours=2)

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
        """Initialize database tables with modified schema"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                # First check if we need to modify the confidence column
                cur.execute("""
                    SELECT data_type 
                    FROM information_schema.columns 
                    WHERE table_name = 'progress' 
                    AND column_name = 'confidence'
                """)
                result = cur.fetchone()
                
                if result and result[0] != 'text':
                    # If table exists with wrong type, alter it
                    cur.execute("""
                        ALTER TABLE progress 
                        ALTER COLUMN confidence TYPE TEXT
                    """)
                    conn.commit()
                else:
                    # If table doesn't exist, create it
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS progress (
                            category TEXT NOT NULL,
                            index_id TEXT NOT NULL,
                            label TEXT,
                            confidence TEXT,
                            timestamp TIMESTAMP,
                            PRIMARY KEY (category, index_id)
                        )
                    """)
                    # Create indices for better query performance 
                    cur.execute("""
                        CREATE INDEX IF NOT EXISTS idx_progress_category 
                        ON progress(category)
                    """)
                    conn.commit()
                
                # Create locks table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS locks (
                        id SERIAL PRIMARY KEY,
                        category TEXT NOT NULL,
                        user_id TEXT NOT NULL,
                        action TEXT NOT NULL,
                        timestamp TIMESTAMP NOT NULL,
                        reason TEXT
                    )
                """)
                # Create index for better query performance
                cur.execute("""
                    CREATE INDEX IF NOT EXISTS idx_locks_category 
                    ON locks(category)
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
        """Update progress with connection pooling and error handling"""
        conn = None
        try:
            conn = self.get_db_connection()
            with conn.cursor() as cur:
                confidence = data.get("confidence")
                # Ensure confidence is stored as text
                if confidence is not None:
                    confidence = str(confidence)
                
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
                    confidence,
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

    def get_locked_categories(self) -> Dict[str, Dict[str, Any]]:
        """Return currently locked categories with lock info."""
        now = datetime.now()
        active_locks = {}
        
        # Filter out expired locks and format response
        for category, lock_info in self.locked_categories.items():
            if (now - lock_info["timestamp"]) <= timedelta(minutes=30):
                active_locks[category] = {
                    "user": lock_info["user"],
                    "locked_since": lock_info["timestamp"].isoformat()
                }
            else:
                # Remove expired lock
                del self.locked_categories[category]
        
        return active_locks

    def _cleanup_inactive_sessions(self):
        """Remove inactive user sessions"""
        now = datetime.now()
        expired_users = [
            user for user, last_active in self.active_users.items()
            if (now - last_active) > self.user_session_timeout
        ]
        for user in expired_users:
            if user in self.active_users:
                del self.active_users[user]
                # Release any locks held by expired user
                self._release_user_locks(user)

    def _release_user_locks(self, user_id: str):
        """Release all locks held by a user"""
        expired_categories = [
            category for category, lock_info in self.locked_categories.items()
            if lock_info["user"] == user_id
        ]
        for category in expired_categories:
            del self.locked_categories[category]
            self._log_lock_action(category, user_id, "expired", "User session expired")

    def acquire_lock(self, user_id: str, category: str) -> bool:
        """Enhanced lock acquisition with user session tracking"""
        now = datetime.now()
        self._cleanup_inactive_sessions()
        
        # Update user's last activity
        self.active_users[user_id] = now
        
        # Check if category is locked
        if category in self.locked_categories:
            lock_info = self.locked_categories[category]
            # Check if lock is by same user
            if lock_info["user"] == user_id:
                # Refresh lock timestamp
                lock_info["timestamp"] = now
                self._log_lock_action(category, user_id, "refresh", "User refreshed existing lock")
                return True
            # Check if lock has expired
            if (now - lock_info["timestamp"]) > timedelta(minutes=30):
                # Lock expired, remove it
                self._log_lock_action(category, lock_info["user"], "expired", 
                                    f"Lock expired after {(now - lock_info['timestamp']).seconds} seconds")
                del self.locked_categories[category]
            else:
                # Lock is valid and held by another user
                self._log_lock_action(category, user_id, "denied", 
                                    f"Lock held by {lock_info['user']}")
                return False
        
        # Acquire new lock
        self.locked_categories[category] = {
            "user": user_id,
            "timestamp": now
        }
        self._log_lock_action(category, user_id, "acquire", "New lock acquired")
        return True

    def release_lock(self, user_id: str, category: str) -> bool:
        """Enhanced lock release with session validation"""
        if user_id not in self.active_users:
            return False
            
        # Update user's last activity
        self.active_users[user_id] = datetime.now()
        
        if category in self.locked_categories:
            lock_info = self.locked_categories[category]
            if lock_info["user"] == user_id:
                del self.locked_categories[category]
                self._log_lock_action(category, user_id, "release", "User released lock")
                return True
            else:
                self._log_lock_action(category, user_id, "release_denied", 
                                    f"Attempted to release lock held by {lock_info['user']}")
        return False

    def _log_lock_action(self, category: str, user_id: str, action: str, reason: str = None):
        """Log lock-related actions to database"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        INSERT INTO locks (category, user_id, action, timestamp, reason)
                        VALUES (%s, %s, %s, %s, %s)
                    """, (category, user_id, action, datetime.now(), reason))
                    conn.commit()
        except Exception as e:
            logging.error(f"Failed to log lock action: {e}")

    def get_lock_history(self, category: str = None, limit: int = 100) -> List[Dict[str, Any]]:
        """Get lock history from database"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    if category:
                        cur.execute("""
                            SELECT category, user_id, action, timestamp, reason
                            FROM locks 
                            WHERE category = %s
                            ORDER BY timestamp DESC
                            LIMIT %s
                        """, (category, limit))
                    else:
                        cur.execute("""
                            SELECT category, user_id, action, timestamp, reason
                            FROM locks 
                            ORDER BY timestamp DESC
                            LIMIT %s
                        """, (limit,))
                    
                    rows = cur.fetchall()
                    return [{
                        "category": row[0],
                        "user_id": row[1],
                        "action": row[2],
                        "timestamp": row[3].isoformat(),
                        "reason": row[4]
                    } for row in rows]
        except Exception as e:
            logging.error(f"Error getting lock history: {e}")
            return []

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

# Add new endpoint for lock history
@app.route('/get_lock_history', methods=['GET'])
def get_lock_history():
    try:
        category = request.args.get('category')
        limit = request.args.get('limit', default=100, type=int)
        history = store.get_lock_history(category, limit)
        return jsonify(history), 200
    except Exception as e:
        logging.error(f"Error in get_lock_history: {e}")
        return jsonify({"error": str(e)}), 500

# Add new routes for user session management
@app.route('/user/heartbeat', methods=['POST'])
def user_heartbeat():
    try:
        data = request.get_json()
        user_id = data.get('user_id')
        if not user_id:
            return jsonify({"error": "user_id required"}), 400
            
        store.active_users[user_id] = datetime.now()
        return jsonify({"status": "ok"})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/user/active', methods=['GET'])
def get_active_users():
    try:
        store._cleanup_inactive_sessions()
        active_users = list(store.active_users.keys())
        return jsonify({"active_users": active_users})
    except Exception as e:
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