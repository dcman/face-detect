"""Tests for model modules."""
import os
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import pickle
import cv2

from src.models.face_analyzer import FaceAnalyzer
from src.models.face_database import FaceDatabase


class TestFaceAnalyzer(unittest.TestCase):
    """Tests for FaceAnalyzer class."""
    
    @classmethod
    def setUpClass(cls):
        """Set up test fixtures that are used for all tests."""
        # Skip these tests if insightface is not available
        try:
            import insightface
            cls.skip_tests = False
        except ImportError:
            cls.skip_tests = True
    
    def setUp(self):
        """Set up test fixtures for each test."""
        if self.skip_tests:
            self.skipTest("InsightFace not available")
        
        # Initialize face analyzer with minimal setup
        self.face_analyzer = FaceAnalyzer()
    
    def test_extract_faces_empty_image(self):
        """Test extracting faces from a blank image."""
        # Create a temporary blank image
        with tempfile.NamedTemporaryFile(suffix='.jpg') as temp_file:
            # Create a blank test image
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(temp_file.name, test_img)
            
            # Should find no faces
            faces = self.face_analyzer.extract_faces(temp_file.name)
            self.assertEqual(len(faces), 0)
    
    @unittest.skip("Requires a test image with faces")
    def test_extract_faces_with_face(self):
        """Test extracting faces from an image with a face."""
        # This test requires a test image with a face
        # Use a path to a test image with a face
        test_image = "path/to/test/image/with/face.jpg"
        
        # Check if file exists
        if not os.path.exists(test_image):
            self.skipTest(f"Test image {test_image} not found")
        
        # Should find at least one face
        faces = self.face_analyzer.extract_faces(test_image)
        self.assertGreater(len(faces), 0)
        
        # Check that face has expected attributes
        face = faces[0]
        self.assertTrue(hasattr(face, 'bbox'))
        self.assertTrue(hasattr(face, 'embedding'))
        self.assertIsInstance(face.embedding, np.ndarray)


class TestFaceDatabase(unittest.TestCase):
    """Tests for FaceDatabase class."""
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.face_db = FaceDatabase(similarity_threshold=0.6)
    
    def test_identify_person_empty_db(self):
        """Test identifying a person with empty database."""
        # Create a random embedding
        embedding = np.random.rand(512)  # InsightFace typically uses 512 dimensions
        
        # Should return None with empty database
        person, score = self.face_db.identify_person(embedding)
        self.assertIsNone(person)
        self.assertEqual(score, 0)
    
    def test_identify_person_with_matches(self):
        """Test identifying a person with matches in database."""
        # Create some test embeddings
        embedding1 = np.random.rand(512)
        embedding2 = embedding1 * 0.9 + np.random.rand(512) * 0.1  # Similar to embedding1
        embedding3 = np.random.rand(512)  # Different embedding
        
        # Add to database
        self.face_db.reference_db = {
            "Person1": [embedding1],
            "Person2": [embedding3]
        }
        
        # Test with similar embedding to Person1
        person, score = self.face_db.identify_person(embedding2)
        self.assertEqual(person, "Person1")
        self.assertGreater(score, 0.6)
        
        # Test with a new random embedding (should not match)
        person, score = self.face_db.identify_person(np.random.rand(512))
        self.assertIsNone(person)
    
    def test_save_load_database(self):
        """Test saving and loading the face database."""
        # Create a test database
        self.face_db.reference_db = {
            "Person1": [np.random.rand(512)],
            "Person2": [np.random.rand(512)]
        }
        
        # Create a temporary file
        with tempfile.NamedTemporaryFile(suffix='.pkl') as temp_file:
            # Save database
            self.assertTrue(self.face_db.save(temp_file.name))
            
            # Create a new database and load
            new_db = FaceDatabase()
            self.assertTrue(new_db.load(temp_file.name))
            
            # Check if data was loaded correctly
            self.assertEqual(len(new_db.reference_db), 2)
            self.assertIn("Person1", new_db.reference_db)
            self.assertIn("Person2", new_db.reference_db)


if __name__ == '__main__':
    unittest.main()