"""Tests for processor modules."""
import os
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2
from unittest.mock import MagicMock, patch

from src.processors.face_collector import FaceCollector
from src.processors.face_clustering import FaceClustering
from src.processors.photo_organizer import PhotoOrganizer
from src.models.face_database import FaceDatabase


class MockFace:
    """Mock face object for testing."""
    def __init__(self, embedding):
        self.embedding = embedding
        self.bbox = np.array([10, 10, 50, 50])


class TestFaceCollector(unittest.TestCase):
    """Tests for FaceCollector class."""
    
    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a mock face analyzer
        self.mock_analyzer = MagicMock()
        self.collector = FaceCollector(self.mock_analyzer, max_workers=2)
    
    def test_collect_faces_empty_dir(self):
        """Test collecting faces from an empty directory."""
        # Create an empty temporary directory
        with tempfile.TemporaryDirectory() as temp_dir:
            # Should find no faces
            results = self.collector.collect_faces(temp_dir, {'.jpg', '.png'})
            self.assertEqual(len(results), 0)
    
    def test_collect_faces(self):
        """Test collecting faces from a directory with images."""
        # Create a temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test images
            test_files = ['image1.jpg', 'image2.png', 'image3.txt']  # Last one is not an image
            
            for file_path in test_files:
                full_path = os.path.join(temp_dir, file_path)
                with open(full_path, 'w') as f:
                    f.write('')
            
            # Set up the mock analyzer to return faces for valid images
            def mock_extract(path):
                if path.endswith(('.jpg', '.png')):
                    return [MockFace(np.random.rand(512))]
                return []
            
            self.mock_analyzer.extract_faces.side_effect = mock_extract
            
            # Run the collector
            results = self.collector.collect_faces(temp_dir, {'.jpg', '.png'})
            
            # Should find faces in 2 images (.jpg and .png)
            self.assertEqual(len(results), 2)
            
            # Check that the analyzer was called for each image
            self.assertEqual(self.mock_analyzer.extract_faces.call_count, 2)


class TestFaceClustering(unittest.TestCase):
    """Tests for FaceClustering class."""
    
    def setUp(self):
        """Set up test fixtures for each test."""
        self.clustering = FaceClustering(eps=0.4, min_samples=2)
        
        # Create a mock face database
        self.mock_db = MagicMock(spec=FaceDatabase)
        self.mock_db.reference_db = {"Person1": [], "Person2": []}
    
    def test_organize_faces_empty(self):
        """Test organizing faces with no faces."""
        # Should return empty dictionaries for each person
        results = self.clustering.organize_faces([], self.mock_db)
        self.assertEqual(len(results), 2)  # Two people in reference DB
        self.assertEqual(sum(len(faces) for faces in results.values()), 0)
    
    def test_organize_faces_with_matches(self):
        """Test organizing faces with matches to known people."""
        # Create mock data
        face1 = MockFace(np.random.rand(512))
        face2 = MockFace(np.random.rand(512))
        all_faces = [
            ("image1.jpg", [face1]),
            ("image2.jpg", [face2])
        ]
        
        # Set up the mock database to return matches
        def mock_identify(embedding):
            if np.array_equal(embedding, face1.embedding):
                return "Person1", 0.9
            return None, 0
        
        self.mock_db.identify_person.side_effect = mock_identify
        
        # Run the clustering
        results = self.clustering.organize_faces(all_faces, self.mock_db)
        
        # Should have matched face1 to Person1
        self.assertEqual(len(results["Person1"]), 1)
        self.assertEqual(results["Person1"][0][0], "image1.jpg")
        
        # Should have at least one unknown cluster with face2
        unknown_clusters = [k for k in results.keys() if k.startswith("unknown")]
        self.assertGreaterEqual(len(unknown_clusters), 1)


class TestPhotoOrganizer(unittest.TestCase):
    """Tests for PhotoOrganizer class."""
    
    def setUp(self):
        """Set up test fixtures for each test."""
        # Create a temporary output directory
        self.temp_dir = tempfile.mkdtemp()
        self.organizer = PhotoOrganizer(self.temp_dir)
    
    def tearDown(self):
        """Clean up after each test."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    def test_organize_photos_empty(self):
        """Test organizing photos with empty data."""
        # Should return empty set of processed images
        processed = self.organizer.organize_photos({})
        self.assertEqual(len(processed), 0)
    
    def test_organize_photos(self):
        """Test organizing photos with face data."""
        # Create temporary source directory and test image
        with tempfile.TemporaryDirectory() as src_dir:
            # Create a test image
            img_path = os.path.join(src_dir, "test.jpg")
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(img_path, test_img)
            
            # Create mock face data
            face = MockFace(np.random.rand(512))
            people_dict = {
                "Person1": [(img_path, face, 0.9)]
            }
            
            # Run the organizer
            processed = self.organizer.organize_photos(people_dict)
            
            # Should have processed one image
            self.assertEqual(len(processed), 1)
            self.assertIn(img_path, processed)
            
            # Should have created a directory for Person1
            person_dir = os.path.join(self.temp_dir, "Person1")
            self.assertTrue(os.path.exists(person_dir))
            
            # Should have copied the image to the person directory
            copied_images = [f for f in os.listdir(person_dir) if f.endswith('.jpg')]
            self.assertEqual(len(copied_images), 1)
            
            # Should have created a preview directory
            preview_dir = os.path.join(person_dir, "previews")
            self.assertTrue(os.path.exists(preview_dir))


if __name__ == '__main__':
    unittest.main()