"""Utilities for file operations."""
import os
import re
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Set, List

def sanitize_name(name: str) -> str:
    """Convert a string to a safe filename/directory name.
    
    Args:
        name: The string to sanitize
        
    Returns:
        A sanitized string safe for use as a filename or directory name
    """
    return re.sub(r'[<>:"/\\|?*]', '_', name)

def download_file(url: str, output_path: str) -> bool:
    """Download a file from a URL to the specified path.
    
    Args:
        url: The URL to download from
        output_path: The path to save the file to
        
    Returns:
        True if download was successful, False otherwise
    """
    try:
        if not os.path.exists(output_path):
            print(f"Downloading {os.path.basename(output_path)}...")
            urllib.request.urlretrieve(url, output_path)
            print(f"Download complete: {output_path}")
        return True
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        return False

def find_images(directory: str, supported_formats: Set[str]) -> List[Path]:
    """Find all images with supported formats in a directory recursively.
    
    Args:
        directory: Directory to search
        supported_formats: Set of supported file extensions
        
    Returns:
        List of Path objects to images
    """
    image_paths = []
    for root, _, files in os.walk(directory):
        for file in files:
            if Path(file).suffix.lower() in supported_formats:
                image_paths.append(Path(root) / file)
    return image_paths

def copy_image(src_path: str, dest_dir: str) -> str:
    """Copy an image to a destination directory, avoiding filename conflicts.
    
    Args:
        src_path: Source path of the image
        dest_dir: Destination directory
        
    Returns:
        Path where the image was copied to
    """
    os.makedirs(dest_dir, exist_ok=True)
    
    filename = os.path.basename(src_path)
    dest_path = os.path.join(dest_dir, filename)
    
    # Avoid filename conflicts
    if os.path.exists(dest_path):
        base, ext = os.path.splitext(filename)
        dest_path = os.path.join(dest_dir, f"{base}_{int(time.time())}{ext}")
    
    shutil.copy2(src_path, dest_path)
    return dest_path