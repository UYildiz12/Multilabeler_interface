import requests
from typing import Dict, Any
import streamlit as st
import logging
import os

class APIService:
    def __init__(self, base_url: str = None):
        self.base_url = base_url or os.getenv('API_URL', 'https://multilabeler-interface-d9bb61fef429.herokuapp.com')
        
    logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

    def acquire_lock(self, user_id: str, category: str) -> bool:
        try:
            response = requests.post(f"{self.base_url}/acquire_lock", json={"user_id": user_id, "category": category})
            try:
                response_data = response.json()  # Attempt to parse JSON
                logging.debug(f"Acquire Lock Response: {response_data}")
            except ValueError:  # Catch non-JSON responses
                logging.error(f"Non-JSON response received: {response.text}")
                st.error(f"Unexpected response from server: {response.text}")
                return False
            
            return response.status_code == 200
        except requests.RequestException as e:
            logging.error(f"Failed to acquire lock: {str(e)}")
            st.error(f"Failed to acquire lock: {str(e)}")
            return False

            
    def release_lock(self, user_id: str, category: str) -> bool:
        """Release lock for a category"""
        try:
            response = requests.post(f"{self.base_url}/release_lock", 
                                   json={"user_id": user_id, "category": category})
            return response.status_code == 200
        except requests.RequestException as e:
            st.error(f"Failed to release lock: {str(e)}")
            return False
            
    def get_locked_categories(self) -> Dict[str, str]:
        """Get all currently locked categories and their users"""
        try:
            response = requests.get(f"{self.base_url}/get_locked_categories")
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
            response = requests.post(f"{self.base_url}/save_progress", json=payload)
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
            response = requests.get(f"{self.base_url}/get_all_progress")
            if response.status_code == 200:
                return response.json()
            return {}
        except requests.RequestException as e:
            logging.error(f"Failed to fetch all progress: {str(e)}")
            return {}
    def sync_progress(self, category: str):
        """Sync progress for a specific category from server"""
        try:
            progress = self.get_progress(category)
            if not progress:
                return
                
            # Update session state with server progress
            for index, label_data in progress.items():
                if str(index).isdigit():
                    idx = int(index)
                    if idx < len(st.session_state.labels[category]):
                        st.session_state.labels[category][idx] = label_data
                        
        except Exception as e:
            logging.error(f"Failed to sync progress: {str(e)}")
    def upload_progress(self, progress_data: Dict[str, Dict[str, Any]]) -> bool:
        """Upload all progress to the server."""
        try:
            response = requests.post(f"{self.base_url}/upload_progress", json=progress_data)
            return response.status_code == 200
        except requests.RequestException as e:
            logging.error(f"Failed to upload progress: {str(e)}")
            return False

    def get_last_labeled_index(self, category: str) -> int:
        try:
            response = requests.get(f"{self.base_url}/get_last_labeled_index", params={"category": category})
            if response.status_code == 200:
                return response.json().get("last_labeled_index", -1)
            return -1
        except requests.RequestException as e:
            logging.error(f"Failed to get last labeled index for category {category}: {str(e)}")
            st.error(f"Failed to get last labeled index for category {category}: {str(e)}")
            return -1



    def get_progress(self, category: str) -> Dict[str, Any]:
        try:
            response = requests.get(f"{self.base_url}/get_progress", params={"category": category})
            if response.status_code == 200:
                return response.json()
            return {}
        except requests.RequestException as e:
            logging.error(f"Failed to get progress: {str(e)}")
            st.error(f"Failed to get progress: {str(e)}")
            return {}

    def sync_category_progress(self, category: str) -> bool:
        """Sync progress for a specific category"""
        try:
            progress = self.get_progress(category)
            if not progress:
                return False
                
            if category in st.session_state.labels:
                for index, label_data in progress.items():
                    if str(index).isdigit():
                        idx = int(index)
                        if idx < len(st.session_state.labels[category]):
                            st.session_state.labels[category][idx] = label_data
            return True
        except Exception as e:
            logging.error(f"Failed to sync category progress: {str(e)}")
            return False
