import os
import json
import logging
from typing import Dict, Any
from datetime import datetime
import psycopg2
from psycopg2.extras import Json
from psycopg2 import pool

class ProgressStore:
    def __init__(self):
        self.database_url = os.environ.get('postgres://udmo7ja1aie1mf:p2ea685957fb88a9561d081dcceb42a8e4469378e158cb2c005d42fda2b45459f@c67okggoj39697.cluster-czrs8kj4isg7.us-east-1.rds.amazonaws.com:5432/d3f0rp0gu9odeo')
        if not self.database_url:
            raise ValueError("DATABASE_URL environment variable is not set")
            
        if self.database_url.startswith("postgres://"):
            self.database_url = self.database_url.replace("postgres://", "postgresql://", 1)
            
        # Add connection pool
        try:
            self.pool = psycopg2.pool.SimpleConnectionPool(
                minconn=1,
                maxconn=10,
                dsn=self.database_url
            )
        except Exception as e:
            logging.error(f"Failed to create connection pool: {e}")
            raise
            
        # Initialize database
        self.init_db()
        
        # Load initial state
        self._refresh_state()
        
    def init_db(self):
        """Initialize database tables"""
        with self.get_db_connection() as conn:
            with conn.cursor() as cur:
                # Create progress table
                cur.execute("""
                    CREATE TABLE IF NOT EXISTS progress (
                        category TEXT NOT NULL,
                        index_id TEXT NOT NULL,
                        label TEXT,
                        confidence FLOAT,
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

    def get_db_connection(self):
        """Get database connection from pool with retry logic"""
        max_retries = 3
        retry_count = 0
        
        while retry_count < max_retries:
            try:
                conn = self.pool.getconn()
                return conn
            except Exception as e:
                retry_count += 1
                if retry_count == max_retries:
                    logging.error(f"Failed to get connection after {max_retries} attempts: {e}")
                    raise
                logging.warning(f"Database connection attempt {retry_count} failed, retrying...")
                time.sleep(1)  # Add delay between retries

    def return_db_connection(self, conn):
        """Return connection to pool"""
        try:
            self.pool.putconn(conn)
        except Exception as e:
            logging.error(f"Error returning connection to pool: {e}")

    def __del__(self):
        """Clean up connection pool on deletion"""
        if hasattr(self, 'pool'):
            self.pool.closeall()

    def _refresh_state(self):
        """Refresh in-memory state from database"""
        self.progress_data = {}
        self.last_labeled_indices = {}
        
        try:
            with self.get_db_connection() as conn:
                with conn.cursor() as cur:
                    cur.execute("""
                        SELECT category, index_id, label, confidence, timestamp 
                        FROM progress
                    """)
                    rows = cur.fetchall()
                    
                    for row in rows:
                        category, index_id, label, confidence, timestamp = row
                        
                        if category not in self.progress_data:
                            self.progress_data[category] = {}
                            
                        self.progress_data[category][index_id] = {
                            "label": label,
                            "confidence": confidence,
                            "timestamp": timestamp.isoformat() if timestamp else None
                        }
                        
                        # Update last labeled index
                        if label and label != "unlabeled":
                            current_index = int(index_id)
                            self.last_labeled_indices[category] = max(
                                self.last_labeled_indices.get(category, -1),
                                current_index
                            )
                            
        except Exception as e:
            logging.error(f"Error refreshing state from database: {e}")
            raise

    def update(self, category: str, index: str, data: Dict[str, Any]):
        """Update progress with connection pooling and error handling"""
        conn = None
        try:
            conn = self.get_db_connection()
            with conn.cursor() as cur:
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
                    data.get("confidence"),
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