"""Module for clustering unknown faces."""
import numpy as np
from typing import Dict, List, Tuple, Any
from sklearn.cluster import DBSCAN

from src.models.face_database import FaceDatabase


class FaceClustering:
    """Class for clustering unknown faces."""
    
    def __init__(self, eps: float = 0.4, min_samples: int = 3):
        """Initialize the face clustering.
        
        Args:
            eps: Maximum distance between samples for DBSCAN
            min_samples: Minimum samples per cluster for DBSCAN
        """
        self.eps = eps
        self.min_samples = min_samples
    
    def organize_faces(
        self, 
        all_faces_data: List[Tuple[str, List[Any]]], 
        face_db: FaceDatabase
    ) -> Dict[str, List]:
        """Organize faces into known people or clusters.
        
        Args:
            all_faces_data: List of tuples (image_path, faces)
            face_db: Face database instance
            
        Returns:
            Dictionary mapping person/cluster names to face data
        """
        print("Organizing faces...")
        
        # Dictionary to store faces by person
        known_people = {name: [] for name in face_db.reference_db.keys()}
        unknown_faces = []  # Faces that don't match any known person
        
        # Process each face
        for image_path, faces in all_faces_data:
            for face in faces:
                # Try to match with a known person
                person_name, score = face_db.identify_person(face.embedding)
                
                if person_name:
                    known_people[person_name].append((image_path, face, score))
                    print(f"Matched face in {os.path.basename(image_path)} to {person_name} with score {score:.2f}")
                else:
                    unknown_faces.append((image_path, face.embedding, face))
        
        # Cluster remaining unknown faces
        unknown_clusters = self._cluster_unknown_faces(unknown_faces)
        
        # Combine all person data
        all_people = {**known_people, **unknown_clusters}
        return all_people
    
    def _cluster_unknown_faces(self, unknown_faces: List[Tuple]) -> Dict[str, List]:
        """Cluster unknown faces using DBSCAN.
        
        Args:
            unknown_faces: List of tuples (image_path, embedding, face)
            
        Returns:
            Dictionary mapping cluster names to face data
        """
        print(f"Clustering {len(unknown_faces)} unknown faces using DBSCAN...")
        unknown_clusters = {}
        
        if not unknown_faces:
            return unknown_clusters
            
        # Extract just the embeddings for clustering
        embeddings = np.array([face[1] for face in unknown_faces])
        
        # Cluster using DBSCAN
        clustering = DBSCAN(
            eps=self.eps, 
            min_samples=self.min_samples, 
            metric='cosine'
        ).fit(embeddings)
        
        # Get cluster labels (-1 is noise/outliers)
        labels = clustering.labels_
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        
        print(f"DBSCAN clustering found {n_clusters} unknown people clusters")
        
        # Group faces by cluster
        for i, label in enumerate(labels):
            if label == -1:
                # Handle outliers (faces that don't cluster well)
                cluster_name = f"unknown_single_{i}"
                unknown_clusters[cluster_name] = [unknown_faces[i]]
            else:
                cluster_name = f"unknown_person_{label}"
                if cluster_name not in unknown_clusters:
                    unknown_clusters[cluster_name] = []
                unknown_clusters[cluster_name].append(unknown_faces[i])
                
        return unknown_clusters