"""Module for collecting faces from a directory of images."""
import os
import concurrent.futures
from pathlib import Path
from typing import List, Tuple, Any, Dict

from src.utils.file_utils import find_images
from src.models.face_analyzer import FaceAnalyzer


class FaceCollector:
    """Class for collecting faces from images in a directory."""
    
    def __init__(self, face_analyzer: FaceAnalyzer, max_workers: int = 8):
        """Initialize the face collector.
        
        Args:
            face_analyzer: Face analyzer instance
            max_workers: Number of parallel workers
        """
        self.face_analyzer = face_analyzer
        self.max_workers = max_workers
    
    def collect_faces(self, source_dir: str, supported_formats: set) -> List[Tuple[str, List[Any]]]:
        """Collect faces from all images in a directory.
        
        Args:
            source_dir: Source directory with images
            supported_formats: Set of supported image formats
            
        Returns:
            List of tuples (image_path, faces)
        """
        all_faces = []
        
        # Find all image files
        print("Finding all images...")
        image_paths = find_images(source_dir, supported_formats)
        print(f"Found {len(image_paths)} images. Extracting faces...")
        
        # Process images in parallel
        total_faces = 0
        processed = 0
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Create a mapping of futures to image paths
            future_to_path = {}
            for path in image_paths:
                future = executor.submit(self._process_image, str(path))
                future_to_path[future] = path
            
            # Process results as they complete
            for future in concurrent.futures.as_completed(future_to_path):
                path = future_to_path[future]
                faces = future.result()
                processed += 1
                
                if faces:
                    total_faces += len(faces)
                    all_faces.append((str(path), faces))
                
                # Print progress
                if processed % 100 == 0 or processed == len(image_paths):
                    print(f"Processed {processed}/{len(image_paths)} images, found {total_faces} faces so far")
        
        return all_faces
    
    def _process_image(self, image_path: str) -> List[Any]:
        """Process a single image to extract faces.
        
        Args:
            image_path: Path to the image
            
        Returns:
            List of face objects
        """
        return self.face_analyzer.extract_faces(image_path)