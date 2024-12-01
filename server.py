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
from threading import Lock
import threading

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
        
        # Initialize all attributes first
        self.pool = None
        self.is_shutting_down = False
        self._pool_lock = Lock()  # Initialize lock first
        self._active_connections = set()
        self._last_pool_reset = datetime.now()
        self._pool_healthy = True
        self._last_health_check = datetime.now()
        self._health_check_interval = timedelta(seconds=30)
        
        # Initialize state containers
        self.progress_data = {}
        self.last_labeled_indices = {}
        self.locked_categories = {}
        
        # Now initialize the pool
        self._initialize_pool()
        
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
            
        # Start pool monitor after everything is initialized
        self._pool_monitor = threading.Thread(target=self._monitor_pool, daemon=True)
        self._pool_monitor.start()
        
    def _monitor_pool(self):
        """Monitor pool health and recover if needed"""
        while not self.is_shutting_down:
            try:
                time.sleep(30)  # Check every 30 seconds
                if (datetime.now() - self._last_health_check) > self._health_check_interval:
                    self._check_pool_health()
            except Exception as e:
                logging.error(f"Pool monitor error: {e}")

    def _check_pool_health(self):
        """Check pool health and reinitialize if needed"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute('SELECT 1')
            self._pool_healthy = True
        except Exception as e:
            logging.error(f"Pool health check failed: {e}")
            self._pool_healthy = False
            self._initialize_pool()  # Reinitialize pool
        finally:
            self._last_health_check = datetime.now()

    def _initialize_pool(self):
        """Initialize or reinitialize the connection pool with backoff"""
        max_retries = 5
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self._pool_lock:
                    if self.pool:
                        self._close_all_connections()
                    
                    self.pool = psycopg2.pool.ThreadedConnectionPool(
                        minconn=2,  # Increased minimum connections
                        maxconn=20,  # Increased maximum connections
                        dsn=self.database_url,
                        connect_timeout=5
                    )
                    self._last_pool_reset = datetime.now()
                    self._pool_healthy = True
                    logging.info("Database connection pool initialized successfully")
                    return
            except Exception as e:
                retry_count += 1
                delay = 2 ** retry_count  # Exponential backoff
                logging.error(f"Failed to create connection pool (attempt {retry_count}/{max_retries}): {e}")
                if retry_count < max_retries:
                    time.sleep(delay)
                else:
                    raise

    def _close_all_connections(self):
        """Safely close all connections"""
        try:
            if self.pool:
                # Close tracked connections
                for conn in list(self._active_connections):
                    try:
                        if not conn.closed:
                            conn.close()
                        self._active_connections.remove(conn)
                    except:
                        pass
                # Close pool
                self.pool.closeall()
        except:
            pass

    def get_db_connection(self):
        """Get database connection with improved error handling"""
        if not self._pool_healthy:
            self._check_pool_health()
            
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                with self._pool_lock:
                    conn = self.pool.getconn()
                    if conn and not conn.closed:
                        self._active_connections.add(conn)
                        return conn
                    raise Exception("Invalid connection from pool")
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    delay = 2 ** retry_count
                    logging.warning(f"Retrying connection ({retry_count}/{max_retries}) after {delay}s")
                    time.sleep(delay)
                    if not self._pool_healthy:
                        self._initialize_pool()
                else:
                    raise

    def return_db_connection(self, conn):
        """Return connection to pool with improved cleanup"""
        if conn:
            try:
                if conn in self._active_connections:
                    self._active_connections.remove(conn)
                    if not conn.closed:
                        with self._pool_lock:
                            self.pool.putconn(conn)
            except Exception as e:
                logging.error(f"Error returning connection to pool: {e}")
                try:
                    if not conn.closed:
                        conn.close()
                except:
                    pass

    def __del__(self):
        """Enhanced cleanup on deletion"""
        self.is_shutting_down = True
        self._close_all_connections()

    def init_db(self):
        """Initialize database tables with modified schema"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                try:
                    # First create the active_locks table
                    cur.execute("""
                        CREATE TABLE IF NOT EXISTS active_locks (
                            category TEXT PRIMARY KEY,
                            user_id TEXT NOT NULL,
                            timestamp TIMESTAMP NOT NULL
                        )
                    """)
                    conn.commit()
                except Exception as e:
                    logging.error(f"Error creating active_locks table: {e}")
                    conn.rollback()

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
                try:
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
                except Exception as e:
                    logging.error(f"Error creating locks table: {e}")
                    conn.rollback()

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
            if conn:
                try:
                    conn.rollback()  # Rollback any pending transaction
                except:
                    pass  # Ignore rollback errors
            raise  # Re-raise the exception after cleanup
        finally:
            if conn:
                try:
                    self.return_db_connection(conn)  # Return connection to pool
                except Exception as e:
                    logging.error(f"Error returning connection to pool: {e}")
                    try:
                        conn.close()  # Force close if return fails
                    except:
                        pass  # Ignore close errors

    def update(self, category: str, index: str, data: Dict[str, Any]):
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
        """Get locked categories from database"""
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        DELETE FROM active_locks 
                        WHERE timestamp < NOW() - INTERVAL '30 minutes'
                    """)
                    
                    cur.execute("""
                        SELECT category, user_id, timestamp 
                        FROM active_locks
                    """)
                    
                    locks = {}
                    for row in cur.fetchall():
                        category, user_id, timestamp = row
                        locks[category] = {
                            "user": user_id,
                            "locked_since": timestamp.isoformat()
                        }
                    conn.commit()
                    return locks
        except Exception as e:
            logging.error(f"Error getting locked categories: {e}")
            return {}

    def acquire_lock(self, user_id: str, category: str) -> bool:
        """Acquire lock with database-level locking"""
        now = datetime.now()
        conn = None
        try:
            conn = self.get_db_connection()
            with conn.cursor() as cur:
                # First try to update existing lock if it's ours or expired
                cur.execute("""
                    DELETE FROM active_locks 
                    WHERE category = %s 
                    AND (
                        user_id = %s 
                        OR timestamp < %s - INTERVAL '30 minutes'
                    )
                """, (category, user_id, now))

                # Try to insert new lock
                cur.execute("""
                    INSERT INTO active_locks (category, user_id, timestamp)
                    VALUES (%s, %s, %s)
                    ON CONFLICT (category) DO NOTHING
                    RETURNING user_id
                """, (category, user_id, now))
                
                result = cur.fetchone()
                if result:
                    conn.commit()
                    self._log_lock_action(category, user_id, "acquire", "New lock acquired")
                    return True
                    
                # Check if someone else holds the lock
                cur.execute("""
                    SELECT user_id, timestamp 
                    FROM active_locks 
                    WHERE category = %s
                """, (category,))
                current_lock = cur.fetchone()
                if current_lock:
                    self._log_lock_action(category, user_id, "denied", 
                        f"Lock held by {current_lock[0]} since {current_lock[1]}")
                return False
                    
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Error acquiring lock: {e}")
            return False
        finally:
            if conn:
                self.return_db_connection(conn)

    def release_lock(self, user_id: str, category: str) -> bool:
        """Release lock with database-level locking"""
        conn = None
        try:
            conn = self.get_db_connection()
            with conn.cursor() as cur:
                cur.execute("""
                    DELETE FROM active_locks 
                    WHERE category = %s 
                    AND user_id = %s
                    RETURNING user_id
                """, (category, user_id))
                
                if cur.fetchone():
                    conn.commit()
                    self._log_lock_action(category, user_id, "release", "Lock released")
                    return True
                    
                # Check if lock exists but is owned by someone else
                cur.execute("""
                    SELECT user_id 
                    FROM active_locks 
                    WHERE category = %s
                """, (category,))
                result = cur.fetchone()
                if result:
                    self._log_lock_action(category, user_id, "release_denied", 
                        f"Attempted to release lock held by {result[0]}")
                return False
                
        except Exception as e:
            if conn:
                conn.rollback()
            logging.error(f"Error releasing lock: {e}")
            return False
        finally:
            if conn:
                self.return_db_connection(conn)

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
        # Check pool health
        if not store._pool_healthy:
            store._check_pool_health()
            
        if not store._pool_healthy:
            return jsonify({
                "status": "unhealthy",
                "error": "Database pool is unhealthy"
            }), 503
            
        # Test database connection
        with store.get_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute('SELECT 1')
                
        return jsonify({
            "status": "healthy",
            "pool_size": len(store._active_connections),
            "last_reset": store._last_pool_reset.isoformat()
        }), 200
        
    except Exception as e:
        logging.error(f"Health check failed: {e}")
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 503

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
    try:
        data = request.get_json()
        if not data:
            return jsonify({"error": "No data provided"}), 400
            
        user_id = data.get('user_id')
        category = data.get('category')
        
        if not user_id or not category:
            return jsonify({
                "error": "Missing required fields",
                "details": {
                    "user_id": "missing" if not user_id else "present",
                    "category": "missing" if not category else "present"
                }
            }), 400

        # Convert sanitized category back to original format if needed
        category = category.replace('_', '/')
            
        success = store.release_lock(user_id, category)
        if success:
            return jsonify({"status": "lock released"}), 200
        else:
            return jsonify({
                "error": "failed to release lock",
                "details": "Lock may be held by another user or already released"
            }), 400
    except Exception as e:
        logging.error(f"Error in release_lock endpoint: {str(e)}")
        return jsonify({"error": str(e)}), 500

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

# Initialize store
store = ProgressStore()

@app.teardown_appcontext
def shutdown_session(exception=None):
    """Improved cleanup on request end"""
    if exception:
        logging.error(f"Error in request: {str(exception)}")
    if 'store' in globals():
        try:
            store._close_all_connections()
        except:
            pass

# Add proper shutdown handling
import atexit
atexit.register(lambda: store.cleanup() if 'store' in globals() else None)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 8080))
    app.run(host='0.0.0.0', port=port)