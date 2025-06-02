"""Module for organizing photos into person folders."""
import os
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any, Set

from src.utils.file_utils import sanitize_name, copy_image
from src.utils.image_utils import create_face_preview, SuperResolution


class PhotoOrganizer:
    """Class for organizing photos into person folders."""
    
    def __init__(self, output_dir: str, sr_model: SuperResolution = None):
        """Initialize the photo organizer.
        
        Args:
            output_dir: Directory to organize photos into
            sr_model: Optional super-resolution model for previews
        """
        self.output_dir = output_dir
        self.sr_model = sr_model
        self.processed_images = set()  # Track which images have been organized
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(os.path.join(output_dir, "unknown"), exist_ok=True)
    
    def organize_photos(self, people_dict: Dict[str, List]) -> Set[str]:
        """Organize photos into person folders.
        
        Args:
            people_dict: Dictionary mapping person names to face data
            
        Returns:
            Set of processed image paths
        """
        print("Organizing photos into folders...")
        
        # Process each person's photos
        for person_name, face_list in people_dict.items():
            # Skip if no faces for this person
            if not face_list:
                continue
                
            # Create directory for this person - sanitize name for safe filesystem use
            safe_person_name = sanitize_name(person_name)
            person_dir = os.path.join(self.output_dir, safe_person_name)
            os.makedirs(person_dir, exist_ok=True)
            
            print(f"Processing {person_name} with {len(face_list)} faces")
            
            # Group by original image path
            images_by_path = defaultdict(list)
            for item in face_list:
                if len(item) == 3:  # For known people: (image_path, face, score)
                    image_path, face, score = item
                    images_by_path[image_path].append((face, score))
                else:  # For unknown clusters: (image_path, embedding, face)
                    image_path, _, face = item
                    images_by_path[image_path].append((face, 1.0))  # Default score for unknown clusters
            
            # Copy each image to the person's directory
            for image_path, faces_with_scores in images_by_path.items():
                # Mark this image as processed
                self.processed_images.add(str(image_path))
                
                # Copy the original image
                dest_path = copy_image(image_path, person_dir)
                
                # Create a preview with face rectangles for each face
                preview_dir = os.path.join(person_dir, "previews")
                os.makedirs(preview_dir, exist_ok=True)
                
                for face, score in faces_with_scores:
                    filename = f"preview_{os.path.basename(image_path)}"
                    preview_path = os.path.join(preview_dir, filename)
                    create_face_preview(
                        image_path, face, person_name, score, 
                        preview_path, self.sr_model
                    )
        
        return self.processed_images
    
    def handle_unprocessed_images(
        self, 
        source_dir: str, 
        supported_formats: Set[str]
    ) -> int:
        """Handle images with no detected faces.
        
        Args:
            source_dir: Source directory with images
            supported_formats: Set of supported image formats
            
        Returns:
            Number of images copied to no_faces directory
        """
        print("Handling images with no faces...")
        no_face_dir = os.path.join(self.output_dir, "no_faces")
        os.makedirs(no_face_dir, exist_ok=True)
        
        count = 0
        for root, _, files in os.walk(source_dir):
            for file in files:
                if Path(file).suffix.lower() in supported_formats:
                    image_path = os.path.join(root, file)
                    
                    if image_path not in self.processed_images:
                        copy_image(image_path, no_face_dir)
                        count += 1
        
        return count