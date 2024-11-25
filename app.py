import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import json
import os
import requests
from io import BytesIO
from queue import Queue
from threading import Thread, Lock
from typing import Dict, Any, Callable, List
import time
from api_service import APIService
from dataloader import DataLoader
import atexit
import logging
import uuid

class StreamlitImageLabeler:
    CATEGORIES = [
        "animate/inanimate",
        "natural/human-made",
        "face present/face absent",
        "human face present/human face absent"
    ]

    def __init__(self, images):
        self.images = images
        self.api_service = APIService()
        self.sync_lock = Lock()
        
        # Increase sync frequency
        self.sync_interval = timedelta(seconds=10)
        self.last_sync = datetime.now()
        
        # Register cleanup handler
        atexit.register(self.cleanup)
        
        self.initialize_session_state()

        # Initialize label buffer for batch saving
        if 'label_buffer' not in st.session_state:
            st.session_state.label_buffer = []

        # Define batch size
        self.batch_size = 25
    
    def initialize_session_state(self):
        # First page - Username input only if not set
        if 'user_id' not in st.session_state:
            st.session_state.user_id = None

        if not st.session_state.user_id:
            st.title("Multi-Category Image Labeling Tool")
            username = st.text_input("Enter your name:", key="username_input")
            if st.button("Start Labeling"):
                if username:
                    st.session_state.user_id = username
                    st.experimental_rerun()
                else:
                    st.error("Please enter your name to continue")
            st.stop()
            return

        # Initialize core session state variables only once
        if 'active_category' not in st.session_state:
            st.session_state.active_category = None
            
        if 'labels' not in st.session_state:
            # Fetch fresh data from server just once
            all_progress = self.api_service.get_all_progress()
            st.session_state.labels = {category: [] for category in self.CATEGORIES}
            
            # Initialize with default placeholders and overlay server progress
            for category in self.CATEGORIES:
                st.session_state.labels[category] = [
                    {"label": "unlabeled", "confidence": "N/A", "timestamp": None}
                    for _ in range(len(self.images))
                ]
                
                if category in all_progress:
                    for index, label_data in all_progress[category].items():
                        if str(index).isdigit():
                            idx = int(index)
                            if idx < len(st.session_state.labels[category]):
                                st.session_state.labels[category][idx] = label_data

        # Initialize other required state variables only if they don't exist
        if 'locked_categories' not in st.session_state:
            st.session_state.locked_categories = {}
        
        if 'show_reset_confirm' not in st.session_state:
            st.session_state.show_reset_confirm = False
        
        if 'category_progress' not in st.session_state:
            st.session_state.category_progress = {
                category: {"last_labeled_index": -1}
                for category in self.CATEGORIES
            }
        
        if 'category_indices' not in st.session_state:
            st.session_state.category_indices = {
                category: self.api_service.get_last_labeled_index(category) + 1
                for category in self.CATEGORIES
            }
    
    def load_all_progress(self):
        """Load progress for all categories"""
        for category in self.CATEGORIES:
            progress = self.api_service.get_progress(category)  # Removed user_id
            st.session_state.category_progress[category] = progress

    
    def select_category(self, category: str) -> bool:
        logging.debug(f"Attempting to acquire lock for category {category}")
        
        # First try to acquire the lock
        if not self.api_service.acquire_lock(st.session_state.user_id, category):
            return False
                
        # Get fresh progress from server
        category_progress = self.api_service.get_progress(category)
        
        # Update labels with server data
        for index, label_data in category_progress.items():
            if str(index).isdigit():
                idx = int(index)
                if idx < len(st.session_state.labels[category]):
                    st.session_state.labels[category][idx] = label_data
        
        # Find the first unlabeled image
        current_index = 0
        for idx, label_data in enumerate(st.session_state.labels[category]):
            if label_data['label'] == 'unlabeled':
                current_index = idx
                break
        else:
            # If all images are labeled, start from the last image
            current_index = len(self.images) - 1
        
        # Set active category and indices
        st.session_state.active_category = category
        st.session_state.current_index = current_index
        st.session_state.category_indices[category] = current_index
        
        # Initialize form state
        st.session_state.radio_value = self._get_label_options()[0]
        st.session_state.confidence_value = "medium"
        
        return True

    def run(self):
        # Sync progress more frequently
        self.sync_progress()
        
        # Only show username input at the very beginning
        if 'user_id' not in st.session_state or not st.session_state.user_id:
            self.initialize_session_state()
            return
        
        st.title("Multi-Category Image Labeling Tool")
        
        # Category selection if none active
        if not st.session_state.active_category:
            self.render_category_selection()
            return
        
        # Main layout
        self.render_sidebar()
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            self.render_image()
        
        with col2:
            self.render_controls()
        
        self.render_navigation()
    
    def render_category_selection(self):
        st.header("Select a Category to Label")
        st.session_state.locked_categories = self.api_service.get_locked_categories()

        # Get all progress data once to avoid multiple server calls
        all_progress_data = self.api_service.get_all_progress()

        for category in self.CATEGORIES:
            # Get current progress for this category
            category_data = all_progress_data.get(category, {})
            
            # Count total labeled (excluding 'unlabeled' entries)
            labeled_count = sum(1 for label_data in category_data.values() 
                              if isinstance(label_data, dict) and 
                              label_data.get('label', 'unlabeled') != 'unlabeled')
            
            total_images = len(self.images)
            progress_pct = (labeled_count / total_images) * 100 if total_images > 0 else 0

            # Display progress bar and stats
            st.write(f"**{category}**")
            st.progress(progress_pct / 100)
            st.write(f"Labeled: {labeled_count}/{total_images} ({progress_pct:.1f}%)")
            
            # Show label distribution
            distribution = {}
            for label_data in category_data.values():
                if isinstance(label_data, dict):
                    label = label_data.get('label')
                    if label and label != 'unlabeled':
                        distribution[label] = distribution.get(label, 0) + 1
            
            if distribution:
                st.write("Label Distribution:")
                for label, count in distribution.items():
                    st.write(f"- {label}: {count}")

            # Check if category is locked
            locked_by = st.session_state.locked_categories.get(category)
            if locked_by:
                if locked_by == st.session_state.user_id:
                    st.info("Currently labeling")
                else:
                    st.warning(f"In use by {locked_by}")
                    continue

            # Show button to select category
            if st.button(f"Label {category}", key=f"select_{category}"):
                success = self.select_category(category)
                if success:
                    st.rerun()

    
    def render_sidebar(self):
        with st.sidebar:
            st.header(f"Category: {st.session_state.active_category}")
            
            # Switch category button
            if st.button("End Labeling Session"):
                self.release_lock()
            
            st.markdown("---")
            
            st.header("Statistics")
            self.display_statistics()
            
            st.markdown("---")
            
            # Manual save button
            if st.button("Save Progress"):
                self.save_progress()
            
            # Export button
            if st.button("Export Labels"):
                self.export_labels()
            
            st.markdown("---")
            
            # Reset button with confirmation
            if 'show_reset_confirm' not in st.session_state:
                st.session_state.show_reset_confirm = False
                
            if not st.session_state.show_reset_confirm:
                if st.button("Reset Category Labels", type="secondary"):
                    st.session_state.show_reset_confirm = True
                    st.rerun()
            else:
                st.warning("Are you sure? This cannot be undone!")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Reset"):
                        self.reset_labels()
                        st.session_state.show_reset_confirm = False
                        st.rerun()
                with col2:
                    if st.button("Cancel"):
                        st.session_state.show_reset_confirm = False
                        st.rerun()
            
            st.markdown("---")
        
            # Add view labels button
            if st.button("View Current Labels"):
                # Get current labels for active category
                if st.session_state.active_category:
                    current_labels = {
                        "category": st.session_state.active_category,
                        "labels": st.session_state.labels[st.session_state.active_category]
                    }
                    # Display as JSON
                    st.json(current_labels)
                else:
                    st.warning("No category selected")

    def render_image(self):
        st.subheader(f"Image {st.session_state.current_index + 1}/{len(self.images)}")
        
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(self.images[st.session_state.current_index], cmap='gray')
        ax.axis('off')
        st.pyplot(fig)
        plt.close()

    def render_controls(self):
        st.subheader("Labeling Controls")
        
        current_labels = st.session_state.labels[st.session_state.active_category]
        current_label = current_labels[st.session_state.current_index]

        # Initialize session state for radio selection if needed
        if "radio_value" not in st.session_state:
            # If current label is unlabeled, set to first option in label options
            if current_label["label"] == "unlabeled":
                st.session_state.radio_value = self._get_label_options()[0]
            else:
                st.session_state.radio_value = current_label["label"]
            
        if "confidence_value" not in st.session_state:
            st.session_state.confidence_value = current_label["confidence"]

        # Create containers for different parts of the form
        radio_container = st.container()
        confidence_container = st.container()
        form_container = st.container()

        # Callback for radio button changes
        def on_radio_change():
            st.session_state.radio_value = st.session_state.label_radio
            st.session_state.confidence_key = datetime.now().isoformat()

        # Radio button in its own container
        with radio_container:
            options = self._get_label_options()
            selected_label = st.radio(
                "Select Label",
                options,
                key="label_radio",
                on_change=on_radio_change,
                index=options.index(st.session_state.radio_value) if st.session_state.radio_value in options else 0
            )

        # Confidence slider in its own container
        with confidence_container:
            confidence = "N/A"
            if st.session_state.radio_value != "ambiguous":
                confidence = st.select_slider(
                    "Confidence Level",
                    options=["low", "medium", "high"],
                    value=st.session_state.confidence_value if st.session_state.confidence_value != "N/A" else "medium",
                    key=f"confidence_slider_{getattr(st.session_state, 'confidence_key', 'default')}"
                )
                st.session_state.confidence_value = confidence

        # Form in its own container
        with form_container:
            with st.form(key=f"label_form_{st.session_state.current_index}"):
                submitted = st.form_submit_button("Submit Label", type="primary", use_container_width=True)
                if submitted:
                    self.save_label(st.session_state.radio_value, 
                                  "N/A" if st.session_state.radio_value == "ambiguous" else st.session_state.confidence_value)
                    
                    # Only advance if not at the end
                    if st.session_state.current_index < len(self.images) - 1:
                        st.session_state.current_index += 1
                        
                    # Reset form state
                    st.session_state.radio_value = self._get_label_options()[0]
                    st.session_state.confidence_value = "medium"
                    st.rerun()

    def render_navigation(self):
        st.markdown("---")
        
        # Progress bar
        progress = (st.session_state.current_index + 1) / len(self.images)
        st.progress(progress)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("← Previous", use_container_width=True):
                self.previous_image()
        
        with col2:
            st.markdown(
                f"<h4 style='text-align: center;'>{st.session_state.current_index + 1} / {len(self.images)}</h4>",
                unsafe_allow_html=True
            )
        
        with col3:
            if st.button("Next →", use_container_width=True):
                self.next_image()

    def _get_label_options(self) -> List[str]:
        """Get label options based on active category"""
        category = st.session_state.active_category
        if category == "animate/inanimate":
            return ["animate", "inanimate", "ambiguous"]
        elif category == "natural/human-made":
            return ["natural", "human-made", "ambiguous"]
        elif category == "face present/face absent":
            return ["face present", "face absent", "ambiguous"]
        elif category == "human face present/human face absent":
            return ["human face present", "human face absent", "ambiguous"]
        return ["ambiguous"]

    def display_statistics(self):
        if not st.session_state.active_category:
            st.warning("No active category selected")
            return

        # Get stats directly from server
        stats = self.api_service.get_category_stats(st.session_state.active_category)
        total_images = len(self.images)
        total_labeled = stats['total_labeled']
        completion_percentage = (total_labeled / total_images) * 100 if total_images > 0 else 0

        # Display metrics
        st.metric("Total Images", total_images)
        st.metric("Labeled Images", total_labeled)
        st.metric("Completion", f"{completion_percentage:.1f}%")

        # Display label distribution
        if stats['label_distribution']:
            st.subheader("Label Distribution")
            st.bar_chart(stats['label_distribution'])

    def save_label(self, label: str, confidence: str):
        # Immediate server update
        label_data = {
            "label": label,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }

        # Update server first
        success = self.api_service.save_progress(
            st.session_state.active_category,
            st.session_state.current_index,
            label_data
        )

        if success:
            # Update local state only after successful server update
            st.session_state.labels[st.session_state.active_category][st.session_state.current_index] = label_data
            
            # Clear any cached data to force fresh fetch next time
            if 'cached_progress' in st.session_state:
                del st.session_state['cached_progress']

        # Update statistics with fresh data
        self.display_statistics()

    def save_progress(self):
        """Saves accumulated labels to the server and clears the buffer"""
        if st.session_state.label_buffer:
            try:
                # Prepare data to send
                progress_data = {}
                for item in st.session_state.label_buffer:
                    category = item['category']
                    index = item['index']
                    label_data = item['label_data']
                    if category not in progress_data:
                        progress_data[category] = {}
                    progress_data[category][str(index)] = label_data

                # Upload progress to server
                success = self.api_service.upload_progress(progress_data)
                if success:
                    st.sidebar.success("Progress saved to server.")
                    # Clear the buffer
                    st.session_state.label_buffer.clear()
                else:
                    st.sidebar.error("Failed to save progress to server.")
            except Exception as e:
                st.sidebar.error(f"Error saving progress: {str(e)}")

    def release_lock(self):
        """Release category lock and save any remaining progress"""
        # Save any remaining labels before releasing the lock
        self.save_progress()
        # Release the lock on the server
        self.api_service.release_lock(
            st.session_state.user_id,
            st.session_state.active_category
        )
        st.session_state.active_category = None
        st.rerun()

    def next_image(self):
        if st.session_state.current_index < len(self.images) - 1:
            # Update both indices
            st.session_state.current_index += 1
            st.session_state.category_indices[st.session_state.active_category] = st.session_state.current_index
            
            # Reset form state for new image
            st.session_state.radio_value = self._get_label_options()[0]
            st.session_state.confidence_value = "medium"
            
            # Force page rerun to update UI
            st.rerun()

    def previous_image(self):
        if st.session_state.current_index > 0:
            # Update both indices
            st.session_state.current_index -= 1
            st.session_state.category_indices[st.session_state.active_category] = st.session_state.current_index
            
            # Load existing label for previous image
            current_label = st.session_state.labels[st.session_state.active_category][st.session_state.current_index]
            st.session_state.radio_value = current_label.get("label", self._get_label_options()[0])
            st.session_state.confidence_value = current_label.get("confidence", "medium")
            
            # Force page rerun to update UI
            st.rerun()


    def export_labels(self):
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_data = {
                "user_id": st.session_state.user_id,
                "timestamp": datetime.now().isoformat(),
                "categories": {
                    cat: labels for cat, labels in st.session_state.labels.items()
                    if cat in st.session_state.category_progress
                }
            }
            
            filename = f"labels_export_{timestamp}.json"
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2)
            st.sidebar.success(f"Labels exported to {filename}")
        except Exception as e:
            st.sidebar.error(f"Error exporting labels: {str(e)}")

    def reset_labels(self):
        logging.debug("Starting reset_labels process")
        if st.session_state.active_category:
            current_category = st.session_state.active_category
            logging.debug(f"Resetting category: {current_category}")
            
            # Create reset data
            reset_data = [
                {"label": "unlabeled", "confidence": "N/A", "timestamp": datetime.now().isoformat()}
                for _ in range(len(self.images))
            ]
            logging.debug(f"Created reset data for {len(reset_data)} images")
            
            try:
                # Update server first
                category_data = {
                    current_category: {
                        str(i): data for i, data in enumerate(reset_data)
                    }
                }
                logging.debug("Attempting to upload reset data to server")
                
                if self.api_service.upload_progress(category_data):
                    logging.debug("Server update successful, updating local state")
                    # Reset all state at once
                    st.session_state.labels[current_category] = reset_data
                    st.session_state.category_indices[current_category] = 0
                    st.session_state.current_index = 0
                    st.session_state.category_progress[current_category] = {"last_labeled_index": -1}
                    st.session_state.radio_value = self._get_label_options()[0]
                    st.session_state.confidence_value = "medium"
                    st.session_state.show_reset_confirm = False
                    
                    logging.debug("Local state updated, initiating rerun")
                    st.success("Category reset successful")
                    time.sleep(0.5)  # Give time for success message to display
                    st.rerun()
                else:
                    logging.error("Failed to update server with reset data")
                    st.error("Failed to reset labels on server")
                    st.session_state.show_reset_confirm = False
            except Exception as e:
                logging.error(f"Error in reset_labels: {str(e)}")
                st.error(f"Error resetting labels: {str(e)}")
                st.session_state.show_reset_confirm = False
        else:
            logging.debug("No active category to reset")

    def cleanup(self):
        """Release category lock and save progress when the app closes"""
        if st.session_state.get('active_category'):
            # Save any remaining progress
            self.save_progress()
            # Release the lock
            self.api_service.release_lock(
                st.session_state.user_id,
                st.session_state.active_category
            )

    def update_shared_progress(self, latest_progress: Dict[str, Dict[str, Any]]):
        """Update local progress with latest server data"""
        for category, labels in latest_progress.items():
            if category in self.CATEGORIES:
                # Don't update active category to avoid conflicts
                if category != st.session_state.active_category:
                    if category not in st.session_state.labels:
                        st.session_state.labels[category] = [
                            {"label": "unlabeled", "confidence": "N/A", "timestamp": None}
                            for _ in range(len(self.images))
                        ]
                    for index, label_data in labels.items():
                        if str(index).isdigit():
                            idx = int(index)
                            if idx < len(st.session_state.labels[category]):
                                st.session_state.labels[category][idx] = label_data
        # Update last sync time
        self.api_service.last_sync_time = datetime.now()

    def should_sync(self) -> bool:
        return datetime.now() - self.last_sync >= self.sync_interval

    def sync_progress(self):
        with self.sync_lock:
            if self.should_sync():
                latest_progress = self.api_service.sync_all_progress()
                if latest_progress:
                    self.update_shared_progress(latest_progress)
                self.last_sync = datetime.now()

def main():
    if "loading_state" not in st.session_state:
        st.session_state.loading_state = "not_started"
    
    if st.session_state.loading_state == "not_started":
        with st.spinner("Loading data and previous progress..."):
            # Set include_train=True to load both training and test sets
            images = DataLoader.load_data(include_train=True)
            if images is not None:
                st.session_state.images = images
                st.session_state.loading_state = "completed"
            else:
                st.error("Failed to load data. Please refresh the page to try again.")
                return
    
    # Ensure images are loaded before creating the labeler
    if st.session_state.loading_state == "completed":
        if 'images' not in st.session_state:
            st.session_state.images = images  # Assign loaded images to session state
        labeler = StreamlitImageLabeler(st.session_state.images)
        labeler.run()

if __name__ == "__main__":
    main()

def update(self, category: str, index: str, data: Dict[str, Any]):
    if category not in self.progress_data:
        self.progress_data[category] = {}
        
    # Store the label data
    self.progress_data[category][str(index)] = data
    
    # Update last labeled index
    current_index = int(index)
    self.last_labeled_indices[category] = max(
        self.last_labeled_indices.get(category, -1),
        current_index
    )
    
    # Count total labeled
    labeled_indices = [i for i, d in self.progress_data[category].items() 
                      if str(i).isdigit() and d.get("label") != "unlabeled"]
    
    self.save_backup()  # Save immediately to persist progress

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

def release_lock(self, user_id: str, category: str) -> bool:
    """Release lock and sync progress"""
    try:
        response = requests.post(f"{self.base_url}/release_lock", 
                               json={"user_id": user_id, "category": category})
        if (response.status_code == 200):
            # Force immediate sync after release
            self.sync_all_progress()
        return response.status_code == 200
    except requests.RequestException as e:
        st.error(f"Failed to release lock: {str(e)}")
        return False