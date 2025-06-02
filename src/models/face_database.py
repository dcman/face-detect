"""Face database for storing and matching face embeddings."""
import os
import pickle
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Set
from scipy.spatial.distance import cosine

from src.utils.file_utils import find_images
# Ensure FaceAnalyzer is importable (it was in the original)
from src.models.face_analyzer import FaceAnalyzer


class FaceDatabase:
    """
    Database for storing face embeddings and matching faces to known people.

    Attributes:
        analyzer: An instance of FaceAnalyzer used for extracting embeddings.
        db_path: Path to the file where the database is stored.
        similarity_threshold: Threshold for considering a face match (higher is stricter).
        reference_db: A dictionary mapping person names to lists of their face embeddings.
    """

    def __init__(self, face_analyzer: FaceAnalyzer, db_path: str, similarity_threshold: float = 0.6):
        """
        Initialize the face database.

        Args:
            face_analyzer: Face analyzer instance for embedding extraction.
            db_path: Path to save/load the database file.
            similarity_threshold: Threshold for face similarity (higher = stricter).
        """
        if not isinstance(face_analyzer, FaceAnalyzer):
             raise TypeError("face_analyzer must be an instance of FaceAnalyzer")
        if not db_path or not isinstance(db_path, str):
             raise ValueError("db_path must be a non-empty string")

        self.analyzer: FaceAnalyzer = face_analyzer
        self.db_path: str = db_path
        self.similarity_threshold: float = similarity_threshold
        self.reference_db: Dict[str, List[np.ndarray]] = {} # Person name -> List of face embeddings
        print(f"FaceDatabase initialized. DB path: '{self.db_path}', Threshold: {self.similarity_threshold}")

    def build_from_directory(self, known_people_dir: str, supported_formats: Optional[Set[str]] = None) -> int:
        """
        Build reference database from a directory of known people using the stored analyzer.

        Args:
            known_people_dir: Directory with subfolders for each person.
            supported_formats: Optional set of supported image file extensions (e.g., {'.jpg', '.png'}).
                               If None, allows any file type (less safe).

        Returns:
            Number of people added to the database.
        """
        print(f"Building reference database from {known_people_dir}...")
        self.reference_db = {}  # Clear existing data before building

        # Check if known people directory exists
        if not os.path.isdir(known_people_dir):
            print(f"Warning: Known people directory '{known_people_dir}' not found or not a directory!")
            return 0

        people_processed_count = 0
        # Process each item in the known people directory
        for person_name in os.listdir(known_people_dir):
            person_dir = os.path.join(known_people_dir, person_name)

            if os.path.isdir(person_dir):
                print(f"Processing reference images for '{person_name}'...")
                embeddings = []

                # Find all images in the person's folder
                image_paths = find_images(person_dir, supported_formats if supported_formats else None)
                if not image_paths:
                    print(f"  No images found in directory: {person_dir}")
                    continue  # Skip this directory if no images found

                # Process each image using the stored face_analyzer
                for image_path in image_paths:
                    # Use self.analyzer stored during __init__
                    faces = self.analyzer.extract_faces(str(image_path))

                    # Skip the image if more than one face is detected
                    if len(faces) > 1:
                        print(f"  Skipping {image_path}: More than one face detected.")
                        continue

                    if faces:
                        for face in faces:
                            if hasattr(face, 'embedding') and face.embedding is not None:
                                embeddings.append(face.embedding)
                            else:
                                print(f"  Warning: Face found in {image_path} but embedding is missing.")

                # Store all valid embeddings for this person
                if embeddings:
                    self.reference_db[person_name] = embeddings
                    print(f"  Added '{person_name}' to reference database with {len(embeddings)} face(s).")
                    people_processed_count += 1
                else:
                    print(f"  Warning: No valid faces/embeddings found for '{person_name}' in {len(image_paths)} image(s), skipping.")

        print(f"Reference database built with {people_processed_count} known people.")
        return people_processed_count

    def identify_person(self, embedding: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Identify a person from a face embedding using the stored threshold.

        Args:
            embedding: Face embedding vector (NumPy array).

        Returns:
            Tuple of (person_name, confidence_score) or (None, 0.0) if no match above threshold.
        """
        best_match_person: Optional[str] = None
        best_score: float = 0.0 # Use float for consistency

        if not isinstance(embedding, np.ndarray):
             print("Warning: Invalid embedding provided to identify_person (must be numpy array).")
             return None, 0.0
        if not self.reference_db:
             # print("Warning: Reference database is empty. Cannot identify person.") # Can be verbose
             return None, 0.0

        for person_name, ref_embeddings in self.reference_db.items():
            if not ref_embeddings: # Skip if a person somehow has no embeddings listed
                 continue

            # Calculate cosine similarity (1 - cosine distance)
            # Handle potential errors during cosine calculation
            try:
                person_scores = [1.0 - cosine(embedding, ref_emb) for ref_emb in ref_embeddings if ref_emb is not None]
                if not person_scores: # Skip if no valid embeddings for comparison
                     continue
                best_person_score = max(person_scores)
            except Exception as e:
                 print(f"Error calculating similarity for {person_name}: {e}")
                 continue # Skip this person if similarity calculation fails

            # Check against threshold and update best match
            # Use self.similarity_threshold stored during __init__
            if best_person_score > self.similarity_threshold and best_person_score > best_score:
                best_score = best_person_score
                best_match_person = person_name

        return best_match_person, best_score

    def save(self) -> bool:
        """
        Save the face database to the path specified during initialization.

        Returns:
            True if save was successful, False otherwise.
        """
        print(f"Saving face database to '{self.db_path}'...")
        try:
            # Ensure the directory exists
            os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
            # Use self.db_path stored during __init__
            with open(self.db_path, 'wb') as f:
                pickle.dump(self.reference_db, f)
            print(f"Database saved successfully with {len(self.reference_db)} people.")
            return True
        except (pickle.PicklingError, OSError, Exception) as e:
            print(f"Error saving face database to '{self.db_path}': {e}")
            return False

    def load(self) -> bool:
        """
        Load the face database from the path specified during initialization.

        Returns:
            True if load was successful, False otherwise.
        """
        # Use self.db_path stored during __init__
        db_file_path = self.db_path
        print(f"Attempting to load face database from '{db_file_path}'...")
        try:
            if os.path.exists(db_file_path) and os.path.getsize(db_file_path) > 0:
                with open(db_file_path, 'rb') as f:
                    loaded_db = pickle.load(f)
                    if isinstance(loaded_db, dict):
                         self.reference_db = loaded_db
                         print(f"Loaded face database with {len(self.reference_db)} people.")
                         return True
                    else:
                         print(f"Error: Loaded file '{db_file_path}' is not a dictionary.")
                         self.reference_db = {} # Reset to empty if load fails
                         return False
            else:
                print(f"Database file not found or empty at '{db_file_path}'. Starting with an empty database.")
                self.reference_db = {} # Ensure it's empty if file doesn't exist
                return False # Return False indicating nothing was loaded
        except (pickle.UnpicklingError, EOFError, OSError, Exception) as e:
            print(f"Error loading face database from '{db_file_path}': {e}")
            self.reference_db = {} # Reset to empty if load fails
            return False

    # Added property for easier access if needed elsewhere, consistent with main.py usage
    @property
    def known_embeddings(self) -> Dict[str, List[np.ndarray]]:
        """Returns the dictionary of known embeddings."""
        return self.reference_db

