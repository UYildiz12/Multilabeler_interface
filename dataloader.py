import numpy as np
import requests
import os
import streamlit as st
from typing import Optional, Dict, Any

class DataLoader:
    """Class for downloading and loading the Kay dataset"""
    
    @staticmethod
    def download_file(url: str, filename: str) -> bool:
        """
        Download a file from a URL if it doesn't already exist
        
        Args:
            url (str): URL to download from
            filename (str): Name to save the file as
            
        Returns:
            bool: True if download successful or file exists, False otherwise
        """
        try:
            if os.path.exists(filename):
                st.info(f"File {filename} already exists, skipping download.")
                return True
                
            st.info(f"Downloading {filename}...")
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            block_size = 1024  # 1 KB
            progress_bar = st.progress(0)
            
            with open(filename, 'wb') as f:
                downloaded = 0
                for data in response.iter_content(block_size):
                    f.write(data)
                    downloaded += len(data)
                    if total_size:
                        progress = downloaded / total_size
                        progress_bar.progress(progress)
            
            st.success(f"Successfully downloaded {filename}")
            return True
            
        except Exception as e:
            st.error(f"Error downloading {filename}: {str(e)}")
            if os.path.exists(filename):
                os.remove(filename)  # Remove partially downloaded file
            return False
    
    @staticmethod
    def load_data(include_train: bool = True) -> Optional[np.ndarray]:
        """
        Download and load the Kay dataset
        
        Args:
            include_train: If True, loads both training and test sets. If False, only loads test set.
            
        Returns:
            Optional[np.ndarray]: Array of images, or None if loading fails
        """
        # Define files and URLs
        files = {
            "kay_images.npz": "https://osf.io/ymnjv/download",
        }
        
        try:
            # Create data directory if it doesn't exist
            os.makedirs("data", exist_ok=True)
            
            # Change to data directory
            original_dir = os.getcwd()
            os.chdir("data")
            
            # Download required files
            for fname, url in files.items():
                if not DataLoader.download_file(url, fname):
                    os.chdir(original_dir)
                    return None
            
            # Load the image data
            try:
                with np.load("kay_images.npz") as dobj:
                    data = dict(**dobj)
                    
                    if include_train:
                        # Combine training and test sets
                        train_stimuli = data["stimuli"]  # 1750 images
                        test_stimuli = data["stimuli_test"]  # 120 images
                        stimuli = np.concatenate([train_stimuli, test_stimuli], axis=0)
                        st.info(f"Loaded {len(train_stimuli)} training images and {len(test_stimuli)} test images")
                    else:
                        # Only load test set
                        stimuli = data["stimuli_test"]
                        st.info(f"Loaded {len(stimuli)} test images")

                    # Change back to original directory
                    os.chdir(original_dir)
                    
                    if stimuli.ndim == 3:  # Shape is (n_images, height, width)
                        stimuli = np.expand_dims(stimuli, axis=-1)
                    if stimuli.ndim != 4:
                        raise ValueError(f"Unexpected data shape: {stimuli.shape}")
                        
                    return stimuli
                    
            except Exception as e:
                st.error(f"Error loading data from NPZ file: {str(e)}")
                os.chdir(original_dir)
                return None
                
        except Exception as e:
            st.error(f"Error in data loading process: {str(e)}")
            if 'original_dir' in locals():
                os.chdir(original_dir)
            return None

def download_data():
    url = "https://osf.io/ymnjv/download"
    filename = "data/kay_images.npz"
    os.makedirs("data", exist_ok=True)
    if not os.path.exists(filename):
        response = requests.get(url, stream=True)
        with open(filename, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)