import requests
from typing import Dict, Any
import streamlit as st
import logging
import os
from datetime import datetime, timedelta
import backoff  # Add this dependency

class APIService:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('API_URL', 'https://multilabeler-interface-d9bb61fef429.herokuapp.com/')
        self.last_sync_time = datetime.now()
        self.sync_interval = timedelta(seconds=10)  # Reduced from 30 to 10 seconds
        self.session = requests.Session()  # Use session for connection pooling
        self.cache_timeout = timedelta(seconds=5)  # Short cache timeout
        self._progress_cache = {}
        self._last_cache_update = datetime.min  # Global cache timestamp
        self._category_cache_update = {}  # Per-category cache timestamps

    @backoff.on_exception(backoff.expo, requests.exceptions.RequestException, max_tries=3)
    def _make_request(self, method: str, endpoint: str, **kwargs) -> requests.Response:
        url = f"{self.base_url}/{endpoint}"
        response = self.session.request(method, url, **kwargs)
        response.raise_for_status()
        return response

    def _is_cache_valid(self) -> bool:
        """Check if global cache is valid"""
        return datetime.now() - self._last_cache_update < self.cache_timeout

    def _is_category_cache_valid(self, category: str) -> bool:
        """Check if category specific cache is valid"""
        if category not in self._category_cache_update:
            return False
        return datetime.now() - self._category_cache_update[category] < self.cache_timeout

    def acquire_lock(self, user_id: str, category: str) -> bool:
        try:
            if not user_id:  # Add validation
                logging.error("Cannot acquire lock: No user ID provided")
                return False
                
            response = self._make_request(
                'POST',
                'acquire_lock',
                json={"user_id": user_id, "category": category}
            )
            return response.status_code == 200
        except requests.RequestException as e:
            logging.error(f"Failed to acquire lock: {str(e)}")
            return False

    def release_lock(self, user_id: str, category: str) -> bool:
        """Release lock for a category"""
        try:
            response = self._make_request(
                'POST',
                'release_lock',
                json={"user_id": user_id, "category": category}
            )
            if response.status_code == 200:
                # Force immediate sync after release
                self.sync_all_progress()
            return response.status_code == 200
        except requests.RequestException as e:
            st.error(f"Failed to release lock: {str(e)}")
            return False

    def get_locked_categories(self) -> Dict[str, str]:
        """Get all currently locked categories and their users"""
        try:
            response = self._make_request('GET', 'get_locked_categories')
            if response.status_code == 200:
                return response.json()
            return {}
        except requests.RequestException as e:
            st.error(f"Failed to get locked categories: {str(e)}")
            return {}

    def save_progress(self, category: str, index: int, label_data: Dict[str, Any]) -> bool:
        try:
            payload = {
                "category": category,
                "index": index,
                **label_data  
            }
            response = self._make_request('POST', 'save_progress', json=payload)
            if response.status_code == 200:
                # Invalidate both global and category-specific cache
                self._last_cache_update = datetime.min
                if category in self._category_cache_update:
                    del self._category_cache_update[category]
                return True
            return False
        except requests.RequestException as e:
            logging.error(f"Failed to save progress: {str(e)}")
            st.error(f"Failed to save progress: {str(e)}")
            return False

    def get_all_progress(self) -> Dict[str, Dict[str, Any]]:
        """Fetch all progress with minimal caching"""
        try:
            if not self._is_cache_valid():
                response = self._make_request('GET', 'get_all_progress')
                if response.status_code == 200:
                    self._progress_cache = response.json()
                    self._last_cache_update = datetime.now()
            return self._progress_cache
        except requests.RequestException as e:
            logging.error(f"Failed to fetch all progress: {str(e)}")
            return {}

    def get_progress(self, category: str) -> Dict[str, Any]:
        try:
            # Check category-specific cache
            if self._is_category_cache_valid(category):
                return self._progress_cache.get(category, {})

            response = self._make_request('GET', 'get_progress', params={"category": category})
            if response.status_code == 200:
                data = response.json()
                self._progress_cache[category] = data
                self._category_cache_update[category] = datetime.now()
                return data
            return {}
        except requests.RequestException as e:
            logging.error(f"Failed to get progress: {str(e)}")
            st.error(f"Failed to get progress: {str(e)}")
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
        try:
            response = self._make_request('GET', 'get_last_labeled_index', params={"category": category})
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

    def get_category_stats(self, category: str) -> Dict[str, Any]:
        """Get accurate statistics from server"""
        try:
            response = self._make_request('GET', 'get_category_stats', params={"category": category})
            if response.status_code == 200:
                return response.json()
            return {'total_labeled': 0, 'label_distribution': {}}
        except requests.RequestException as e:
            logging.error(f"Failed to get category stats: {str(e)}")
            return {'total_labeled': 0, 'label_distribution': {}}
