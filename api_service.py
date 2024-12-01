import requests
from typing import Dict, Any
import streamlit as st
import logging
import os
from datetime import datetime, timedelta
import backoff  # Add this dependency
import time
from weakref import WeakValueDictionary  # Add this import

class APIService:
    def __init__(self, base_url: str = None):
        # Normalize base URL by removing trailing slash
        self.base_url = (base_url or os.getenv('API_URL', 'https://multilabeler-interface-d9bb61fef429.herokuapp.com')).rstrip('/')
        self.last_sync_time = datetime.now()
        self.sync_interval = timedelta(seconds=30)  # Reduced from 30 to 10 seconds
        self.session = self._create_session()
        self._error_count = 0
        self._max_errors = 3
        self._reset_time = datetime.now()
        self._response_cache = WeakValueDictionary()  # Use weak references for caching
        self._cache_timeout = timedelta(minutes=5)

    def _create_session(self):
        """Create a new session with improved retry configuration"""
        session = requests.Session()
        retry_strategy = requests.adapters.Retry(
            total=5,  # Increased retries
            backoff_factor=1,  # Increased backoff
            status_forcelist=[500, 502, 503, 504, 429],  # Added 429 for rate limiting
            allowed_methods=["HEAD", "GET", "POST"],  # Specify allowed methods
            raise_on_status=True
        )
        adapter = requests.adapters.HTTPAdapter(
            max_retries=retry_strategy,
            pool_connections=10,
            pool_maxsize=10
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session

    def _handle_request_error(self, e: Exception):
        """Central error handling logic"""
        self._error_count += 1
        logging.error(f"Request error ({self._error_count}/{self._max_errors}): {str(e)}")
        
        # Reset error count after an hour
        if (datetime.now() - self._reset_time).seconds > 3600:
            self._error_count = 0
            self._reset_time = datetime.now()
        
        # Recreate session if too many errors
        if self._error_count >= self._max_errors:
            logging.warning("Too many errors, recreating session...")
            self.session.close()
            self.session = self._create_session()
            self._error_count = 0

    @backoff.on_exception(
        backoff.expo, 
        (requests.exceptions.RequestException, ConnectionError), 
        max_tries=5,  # Increased retries
        max_time=30  # Maximum time to try
    )
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        try:
            # Clear old cache entries
            self._clear_old_cache()
            
            # Generate cache key
            cache_key = f"{method}:{endpoint}:{str(kwargs)}"
            
            # Check cache
            if method == 'GET' and cache_key in self._response_cache:
                return self._response_cache[cache_key]
            
            kwargs.setdefault('timeout', (5, 15))  # (connect, read) timeouts
            # Normalize endpoint by removing leading/trailing slashes
            endpoint = endpoint.strip('/')
            url = f"{self.base_url}/{endpoint}"
            response = self.session.request(method, url, **kwargs)
            response.raise_for_status()
            
            # Cache GET responses
            if method == 'GET':
                self._response_cache[cache_key] = response
            
            return response
        except Exception as e:
            self._handle_request_error(e)
            raise

    def _clear_old_cache(self):
        """Clear expired cache entries"""
        try:
            current_time = datetime.now()
            expired_keys = [
                key for key, value in self._response_cache.items()
                if hasattr(value, '_cached_time') and 
                current_time - value._cached_time > self._cache_timeout
            ]
            for key in expired_keys:
                del self._response_cache[key]
        except:
            self._response_cache.clear()

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, ConnectionError),
        max_tries=5,
        max_time=30
    )
    def acquire_lock(self, user_id: str, category: str) -> bool:
        """Acquire lock with retries and validation"""
        try:
            if not user_id:
                logging.error("Cannot acquire lock: No user ID provided")
                return False
                
            # Check current lock status first
            locked_categories = self.get_locked_categories()
            if category in locked_categories:
                lock_info = locked_categories[category]
                if lock_info["user"] != user_id:
                    logging.info(f"Category {category} is locked by {lock_info['user']}")
                    return False
            
            sanitized_category = category.replace('/', '_')
            response = self._make_request(
                'POST',
                'acquire_lock',
                json={"user_id": user_id, "category": sanitized_category}
            )
            
            if response.status_code == 200:
                # Verify lock was acquired
                locked_categories = self.get_locked_categories()
                return (category in locked_categories and 
                        locked_categories[category]["user"] == user_id)
            return False
            
        except requests.RequestException as e:
            logging.error(f"Failed to acquire lock: {str(e)}")
            return False

    @backoff.on_exception(
        backoff.expo,
        (requests.exceptions.RequestException, ConnectionError),
        max_tries=5,
        max_time=30
    )
    def release_lock(self, user_id: str, category: str) -> bool:
        try:
            if not user_id or not category:
                logging.error("user_id and category are required to release lock")
                return False
                
            sanitized_category = category.replace('/', '_')
            response = self._make_request(
                'POST',
                'release_lock',
                json={"user_id": user_id, "category": sanitized_category}
            )
            
            if response.status_code == 200:
                # Verify lock was released
                locked_categories = self.get_locked_categories()
                return category not in locked_categories
            return False
            
        except requests.RequestException as e:
            logging.error(f"Failed to release lock: {str(e)}")
            return False

    def get_locked_categories(self) -> Dict[str, Dict[str, Any]]:
        """Get all currently locked categories with detailed info"""
        try:
            response = self._make_request('GET', 'get_locked_categories')
            if response.status_code == 200:
                locked_categories = response.json()
                # Sanitize the response
                return {k.replace('_', '/'): v for k, v in locked_categories.items()}
            return {}
        except requests.RequestException as e:
            logging.error(f"Failed to get locked categories: {str(e)}")
            return {}

    def save_progress(self, category: str, index: int, label_data: Dict[str, Any]) -> bool:
        try:
            # Sanitize category name
            sanitized_category = category.replace('/', '_')
            payload = {
                "category": sanitized_category,
                "index": index,
                **label_data  
            }
            response = self._make_request('POST', 'save_progress', json=payload)
            try:
                response_data = response.json()  # Attempt to parse JSON
                logging.debug(f"Save Progress Response: {response_data}")
            except ValueError:  # Catch non-JSON responses
                logging.error(f"Non-JSON response received: {response.text}")
                st.error(f"Unexpected response from server: {response.text}")
                return False
            
            return response.status_code == 200
        except requests.RequestException as e:
            logging.error(f"Failed to save progress: {str(e)}")
            st.error(f"Failed to save progress: {str(e)}")
            return False

    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all progress from the server."""
        try:
            response = self._make_request('GET', 'get_all_progress')
            if response.status_code == 200:
                return response.json()
            return {}
        except requests.RequestException as e:
            logging.error(f"Failed to fetch all progress: {str(e)}")
            return {}

    def get_progress(self, category: str) -> Dict[str, Any]:
        """Get progress with better error handling"""
        # Sanitize category name
        sanitized_category = category.replace('/', '_')
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = self._make_request(
                    'GET',
                    'get_progress',
                    params={"category": sanitized_category}
                )
                if response.status_code == 200:
                    return response.json()
                time.sleep(1)  # Add delay between retries
            except Exception as e:
                if attempt == max_retries - 1:
                    logging.error(f"Final attempt failed for get_progress: {str(e)}")
                    return {}
                logging.warning(f"Retrying get_progress ({attempt + 1}/{max_retries})")
                time.sleep(1)
        return {}

    def upload_progress(self, progress_data: Dict[str, Dict[str, Any]]) -> bool:
        """Upload all progress to the server."""
        try:
            response = self._make_request('POST', 'upload_progress', json=progress_data)
            return response.status_code == 200
        except requests.RequestException as e:
            logging.error(f"Failed to upload progress: {str(e)}")
            return False

    def get_last_labeled_index(self, category: str) -> int:
        # Sanitize category name
        sanitized_category = category.replace('/', '_')
        try:
            response = self._make_request('GET', 'get_last_labeled_index', params={"category": sanitized_category})
            if response.status_code == 200:
                return response.json().get("last_labeled_index", -1)
            return -1
        except requests.RequestException as e:
            logging.error(f"Failed to get last labeled index for category {category}: {str(e)}")
            st.error(f"Failed to get last labeled index for category {category}: {str(e)}")
            return -1

    def should_sync(self) -> bool:
        """Check if it's time to sync with the server"""
        return datetime.now() - self.last_sync_time > self.sync_interval

    def sync_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """Get all progress from server and update last sync time"""
        try:
            response = self._make_request('GET', 'get_all_progress')
            self.last_sync_time = datetime.now()
            return response.json()
        except requests.RequestException as e:
            logging.error(f"Failed to sync progress: {str(e)}")
            return {}

    def __del__(self):
        """Enhanced cleanup"""
        try:
            self._response_cache.clear()
            if hasattr(self, 'session'):
                self.session.close()
        except:
            pass
