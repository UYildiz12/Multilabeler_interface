import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import json
import os
import requests
from io import BytesIO
from queue import Queue
from threading import Thread
from typing import Dict, Any, Callable, List
import time
from queue_service import BackgroundQueue
from api_service import APIService
from dataloader import DataLoader
import atexit
import logging

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
        
        # Initialize background queue for label submissions
        self.submission_queue = BackgroundQueue(
            process_func=self._process_label_submission
        )
        
        # Register cleanup function
        atexit.register(self.cleanup)
        
        self.initialize_session_state()
    
    def initialize_session_state(self):
        if 'user_id' not in st.session_state:
            st.session_state.user_id = st.text_input("Enter Your Name:", key="user_id_input")
            if not st.session_state.user_id:
                st.stop()

        if 'category_progress' not in st.session_state:
            # Initialize with default values for all categories
            st.session_state.category_progress = {
                category: {
                    "last_labeled_index": -1
                }
                for category in self.CATEGORIES
            }
            
            # Then update with server progress if available
            all_progress = self.api_service.get_all_progress()
            for category, labels in all_progress.items():
                if category in self.CATEGORIES:
                    st.session_state.category_progress[category] = {
                        "last_labeled_index": max(
                            [int(index) for index in labels.keys() if index.isdigit()],
                            default=-1
                        )
                    }

        if 'labels' not in st.session_state:
            all_progress = self.api_service.get_all_progress()
            st.session_state.labels = {category: [] for category in self.CATEGORIES}
            
            # First initialize with unlabeled placeholders
            for category in self.CATEGORIES:
                st.session_state.labels[category] = [
                    {"label": "unlabeled", "confidence": "N/A", "timestamp": None}
                    for _ in range(len(self.images))  # Use len(self.images) instead of len(all_progress)
                ]
            
            # Then overlay existing progress
            for category, labels in all_progress.items():
                if category in self.CATEGORIES:  # Only process known categories
                    for index, label_data in labels.items():
                        if str(index).isdigit():
                            idx = int(index)
                            if idx < len(st.session_state.labels[category]):
                                st.session_state.labels[category][idx] = label_data


        # Get locked categories
        if 'locked_categories' not in st.session_state:
            st.session_state.locked_categories = self.api_service.get_locked_categories()
        
        # Category selection
        if 'active_category' not in st.session_state:
            st.session_state.active_category = None
        
        # Initialize current_index
        if 'current_index' not in st.session_state:
            st.session_state.current_index = 0
        
        # Initialize labels for each category
        if 'labels' not in st.session_state:
            st.session_state.labels = {}
            
        if 'show_reset_confirm' not in st.session_state:
            st.session_state.show_reset_confirm = False
    
    def load_all_progress(self):
        """Load progress for all categories"""
        for category in self.CATEGORIES:
            progress = self.api_service.get_progress(category)  # Removed user_id
            st.session_state.category_progress[category] = progress

    
    def select_category(self, category: str):
        if category in st.session_state.locked_categories:
            locker = st.session_state.locked_categories[category]
            if locker != st.session_state.user_id:
                st.error(f"This category is currently being labeled by {locker}")
                return False

        # Release previous lock if exists
        if st.session_state.active_category:
            self.api_service.release_lock(st.session_state.user_id, st.session_state.active_category)

        # Try to acquire lock for the new category
        if self.api_service.acquire_lock(st.session_state.user_id, category):
            st.session_state.active_category = category
            st.session_state.locked_categories = self.api_service.get_locked_categories()

            # Sync progress from server for this category
            self.api_service.sync_progress(category)

            # Set the current index based on the last labeled index
            last_index = self.api_service.get_last_labeled_index(category)
            st.session_state.current_index = last_index + 1 if last_index >= 0 else 0

            st.rerun()
            return True
        return False


    def run(self):
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

        for category in self.CATEGORIES:
            progress = len([
            l for l in st.session_state.labels.get(category, [])
            if isinstance(l, dict) and "label" in l and l["label"] != "unlabeled"
        ])

            total = len(self.images)
            progress_pct = (progress / total) * 100 if total > 0 else 0

            col1, col2 = st.columns([3, 1])
            with col1:
                st.progress(progress_pct / 100, f"Progress: {progress_pct:.1f}%")
            with col2:
                locked_by = st.session_state.locked_categories.get(category)
                if locked_by:
                    if locked_by == st.session_state.user_id:
                        st.info("Currently working")
                    else:
                        st.warning(f"Locked by {locked_by}")
                elif st.button(f"Label {category}", key=f"select_{category}"):
                    if self.select_category(category):
                        st.rerun()  # Ensure the page is rerun to reflect the new state

    
    def render_sidebar(self):
        with st.sidebar:
            st.header(f"Category: {st.session_state.active_category}")
            
            # Switch category button
            if st.button("Switch Category"):
                self.api_service.release_lock(
                    st.session_state.user_id,
                    st.session_state.active_category
                )
                st.session_state.active_category = None
                st.rerun()
            
            st.markdown("---")
            
            st.header("Statistics")
            self.display_statistics()
            
            st.markdown("---")
            
            # Auto-save toggle
            st.checkbox("Auto-save", value=True, key="auto_save")
            
            # Manual save button
            if st.button("Save Progress"):
                self.save_progress()
            
            # Export button
            if st.button("Export Labels"):
                self.export_labels()
            
            st.markdown("---")
            
            # Reset button with confirmation
            if st.button("Reset Category Labels", type="secondary"):
                st.session_state.show_reset_confirm = True
            
            if st.session_state.show_reset_confirm:
                st.warning("Are you sure? This cannot be undone!")
                col1, col2 = st.columns(2)
                with col1:
                    if st.button("Yes, Reset"):
                        self.reset_labels()
                with col2:
                    if st.button("Cancel"):
                        st.session_state.show_reset_confirm = False

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
                    self.next_image()
                    # Reset the state for the next image
                    st.session_state.radio_value = "unlabeled"
                    st.session_state.confidence_value = "N/A"
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

        # Get current category's labels
        labels = st.session_state.labels[st.session_state.active_category]
        
        # Count labeled images (excluding unlabeled)
        labeled = sum(1 for l in labels if l.get("label") != "unlabeled")
        total = len(labels)
        
        # Calculate completion percentage
        completion_percentage = (labeled / total) * 100 if total > 0 else 0

        # Display metrics
        st.metric("Total Images", total)
        st.metric("Labeled Images", labeled)
        st.metric("Completion", f"{completion_percentage:.1f}%")

        # Label distribution
        label_counts = {}
        for l in labels:
            label = l.get("label", "unlabeled")
            label_counts[label] = label_counts.get(label, 0) + 1

        if label_counts:
            st.bar_chart(label_counts)


    def save_label(self, label: str, confidence: str):
        # Ensure category exists in progress
        if st.session_state.active_category not in st.session_state.category_progress:
            st.session_state.category_progress[st.session_state.active_category] = {
                "last_labeled_index": -1
            }

        label_data = {
            "label": label,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        }
        
        # Update labels and progress
        st.session_state.labels[st.session_state.active_category][st.session_state.current_index] = label_data
        st.session_state.category_progress[st.session_state.active_category]["last_labeled_index"] = st.session_state.current_index

        # Add to submission queue
        if st.session_state.get("auto_save", True):
            self.submission_queue.add_task({
                'category': st.session_state.active_category,
                'index': st.session_state.current_index,
                'label_data': label_data
            })





    def _process_label_submission(self, task: Dict[str, Any]):
        try:
            success = self.api_service.save_progress(
                task['category'],  # Pass the category
                task['index'],     # Pass the index
                task['label_data'] # Pass the label data
            )
            if not success:
                logging.error(f"Failed to save label for index {task['index']}")
        except Exception as e:
            logging.error(f"Error processing label submission: {str(e)}")


    def next_image(self):
        # Move to the next image in the current category
        if st.session_state.current_index < len(self.images) - 1:
            st.session_state.current_index += 1

    def previous_image(self):
        # Move to the previous image in the current category
        if st.session_state.current_index > 0:
            st.session_state.current_index -= 1



    def save_progress(self):
        try:
            # Upload all progress to the server
            success = self.api_service.upload_progress(st.session_state.labels)
            if success:
                st.sidebar.success("Progress saved to the server!")
            else:
                st.sidebar.error("Failed to save progress to the server.")
        except Exception as e:
            st.sidebar.error(f"Error uploading progress: {str(e)}")


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
        if st.session_state.active_category:
            # Create reset data
            reset_data = [
                {"label": "unlabeled", "confidence": "N/A", "timestamp": None}
                for _ in range(len(self.images))
            ]
            
            # Update local state
            st.session_state.labels[st.session_state.active_category] = reset_data
            
            # Update server
            try:
                # Create a dictionary with the reset data
                category_data = {
                    st.session_state.active_category: {
                        str(i): data for i, data in enumerate(reset_data)
                    }
                }
                
                # Upload to server
                if self.api_service.upload_progress(category_data):
                    st.success("Category labels reset successfully on server")
                else:
                    st.error("Failed to reset labels on server")
                    
            except Exception as e:
                st.error(f"Error resetting labels on server: {str(e)}")
                
            # Reset current index and confirmation dialog
            st.session_state.current_index = 0
            st.session_state.show_reset_confirm = False
            st.rerun()

    def cleanup(self):
        """Release category lock when the app closes"""
        if st.session_state.get('active_category'):
            self.api_service.release_lock(
                st.session_state.user_id,
                st.session_state.active_category
            )

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
    
    # Run the labeling tool
    if st.session_state.loading_state == "completed":
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