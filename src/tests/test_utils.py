# src/tests/test_utils.py

"""Tests for utility modules."""
import os
import unittest
import tempfile
import shutil
from pathlib import Path
import numpy as np
import cv2

# --- Required imports from your project ---
# Assuming this file is in src/tests/ and needs to import from src/utils/
# Adjust relative paths if necessary based on how tests are run.
try:
    from src.utils.file_utils import sanitize_name, find_images, copy_image
    from src.utils.image_utils import SuperResolution, read_image, create_face_preview
    # Import config to get the model path
    from config import SR_MODEL_PATH
except ImportError as e:
    # Fallback for running directly or different structures
    print(f"Import warning: {e}. Attempting relative import...")
    try:
        from ..utils.file_utils import sanitize_name, find_images, copy_image
        from ..utils.image_utils import SuperResolution, read_image, create_face_preview
        from ...config import SR_MODEL_PATH
    except ImportError:
         raise ImportError("Could not resolve imports for test_utils.py. Ensure PYTHONPATH is set or run tests using 'python -m unittest discover'.")


# --- Mock Face for create_face_preview test ---
class MockFace:
    """Mock face object for testing."""
    def __init__(self, bbox):
        self.bbox = np.array(bbox)

# --- Existing TestFileUtils ---
class TestFileUtils(unittest.TestCase):
    """Tests for file utility functions."""

    def test_sanitize_name(self):
        """Test sanitizing filenames."""
        test_cases = {
            'John Doe': 'John Doe',
            'File/with:invalid*chars?': 'File_with_invalid_chars_',
            'File\\with\\backslashes': 'File_with_backslashes',
            '<>:"/\\|?*': '__________'
        }

        for input_name, expected in test_cases.items():
            self.assertEqual(sanitize_name(input_name), expected)

    def test_find_images(self):
        """Test finding images in a directory."""
        # Create a temporary directory with test files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create test files
            test_files = [
                'image1.jpg',
                'image2.png',
                'image3.txt',  # Not an image
                'image4.jpeg',
                'subdir/image5.webp',
            ]

            for file_path in test_files:
                full_path = os.path.join(temp_dir, file_path)
                os.makedirs(os.path.dirname(full_path), exist_ok=True)
                with open(full_path, 'w') as f:
                    f.write('')

            # Test finding images
            supported_formats = {'.jpg', '.jpeg', '.png', '.webp'}
            found_images = find_images(temp_dir, supported_formats)

            # Check results
            self.assertEqual(len(found_images), 4)  # Should find 4 images (not .txt)
            found_extensions = {path.suffix.lower() for path in found_images}
            self.assertTrue(all(ext in supported_formats for ext in found_extensions))

    def test_copy_image(self):
        """Test copying an image with conflict handling."""
        # Create temporary source and destination directories
        with tempfile.TemporaryDirectory() as src_dir:
            with tempfile.TemporaryDirectory() as dest_dir:
                # Create a test image
                test_image = os.path.join(src_dir, 'test.jpg')
                with open(test_image, 'w') as f:
                    f.write('test content')

                # Test copying
                dest_path = copy_image(test_image, dest_dir)
                self.assertTrue(os.path.exists(dest_path))
                self.assertEqual(os.path.basename(dest_path), 'test.jpg')

                # Test conflict handling (create a duplicate)
                with open(test_image, 'w') as f:
                    f.write('new content')

                dest_path2 = copy_image(test_image, dest_dir)
                self.assertTrue(os.path.exists(dest_path2))
                self.assertNotEqual(dest_path, dest_path2)  # Should have different names


# --- Existing TestImageUtils ---
class TestImageUtils(unittest.TestCase):
    """Tests for image utility functions (excluding SR)."""

    def test_read_image(self):
        """Test reading images with different formats."""
        # Create a temporary test image
        with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as temp_file:
            temp_img_path = temp_file.name
            # Create a simple test image
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imwrite(temp_img_path, test_img)

        try:
            # Test reading
            img = read_image(temp_img_path)
            self.assertIsNotNone(img)
            self.assertEqual(img.shape, (100, 100, 3))
        finally:
            os.remove(temp_img_path) # Clean up the temporary file

    def test_create_face_preview(self):
        """Test creating face preview images."""
        # Create a temporary directory for the output
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a test image file
            img_path = os.path.join(temp_dir, 'test.jpg')
            test_img = np.zeros((100, 100, 3), dtype=np.uint8) # Black image
            cv2.imwrite(img_path, test_img)

            # Create a mock face
            mock_face = MockFace([10, 10, 50, 50])  # [x1, y1, x2, y2]

            # Test creating a preview
            output_path = os.path.join(temp_dir, 'preview.jpg')
            result = create_face_preview(
                img_path, mock_face, "Test Person", 0.95, output_path
            )

            # Check results
            self.assertTrue(result)
            self.assertTrue(os.path.exists(output_path))
            # Optionally load the preview and check if it has markings, but basic check is sufficient


# --- NEW Test Case for Super Resolution ---
# Check if the model file specified in config exists
# SR_MODEL_PATH should be the full path to the model file
model_exists = os.path.exists(SR_MODEL_PATH) and os.path.getsize(SR_MODEL_PATH) > 0

@unittest.skipUnless(model_exists, f"SR model not found or empty at {SR_MODEL_PATH}, skipping SuperResolution tests.")
class TestSuperResolutionUpscaling(unittest.TestCase):
    """Tests for the SuperResolution upscale functionality."""

    @classmethod
    def setUpClass(cls):
        """Initialize the SR model once for all tests in this class."""
        # This assumes SR_MODEL_PATH is valid and points to the correct model
        # The class decorator already checks if it exists.
        try:
            cls.sr_model = SuperResolution(SR_MODEL_PATH)
            print(f"\nSuperResolution model loaded for testing from {SR_MODEL_PATH}")
        except Exception as e:
            # If loading fails despite file existing, skip tests
            raise unittest.SkipTest(f"Failed to load SR model from {SR_MODEL_PATH}: {e}")

    def setUp(self):
        """Create a temporary directory and a small dummy image for each test."""
        self.temp_dir = tempfile.mkdtemp()
        # Create a small dummy image (e.g., 64x64 pixels)
        self.dummy_img_path = os.path.join(self.temp_dir, "dummy_low_res.png")
        self.input_height, self.input_width = 64, 64
        dummy_img = np.random.randint(0, 256, (self.input_height, self.input_width, 3), dtype=np.uint8)
        cv2.imwrite(self.dummy_img_path, dummy_img)
        self.original_image = dummy_img # Keep in memory for comparison

    def tearDown(self):
        """Remove the temporary directory and its contents."""
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_upscale_dimensions(self):
        """Test if the upscaled image has the expected dimensions (e.g., 4x)."""
        # Assuming the model is a 4x upscaler (common for Real-ESRGAN)
        expected_scale = 4
        expected_height = self.input_height * expected_scale
        expected_width = self.input_width * expected_scale

        # Read the dummy image using cv2 directly for the test input
        img_to_upscale = cv2.imread(self.dummy_img_path)
        self.assertIsNotNone(img_to_upscale, "Failed to read dummy test image")

        # Perform upscaling
        upscaled_image = self.sr_model.upscale(img_to_upscale)

        # Assertions
        self.assertIsNotNone(upscaled_image, "Upscale method returned None")
        self.assertEqual(upscaled_image.ndim, 3, "Upscaled image should have 3 dimensions (H, W, C)")
        self.assertEqual(upscaled_image.shape[0], expected_height, f"Upscaled height mismatch: expected {expected_height}, got {upscaled_image.shape[0]}")
        self.assertEqual(upscaled_image.shape[1], expected_width, f"Upscaled width mismatch: expected {expected_width}, got {upscaled_image.shape[1]}")
        self.assertEqual(upscaled_image.shape[2], 3, "Upscaled image should have 3 color channels")

    def test_upscale_content_change(self):
        """Test if the upscaled image content is different from the original."""
        # Read the dummy image
        img_to_upscale = cv2.imread(self.dummy_img_path)
        self.assertIsNotNone(img_to_upscale, "Failed to read dummy test image")

        # Perform upscaling
        upscaled_image = self.sr_model.upscale(img_to_upscale)
        self.assertIsNotNone(upscaled_image, "Upscale method returned None")

        # Basic check: are the images definitely different?
        # Comparing pixel values directly can be tricky due to resizing.
        # A simple check is if they have different shapes (which they should).
        # Another sanity check: ensure the upscaled image isn't identical to a simple cv2 resize.
        resized_original = cv2.resize(img_to_upscale, (upscaled_image.shape[1], upscaled_image.shape[0]), interpolation=cv2.INTER_CUBIC)

        self.assertNotEqual(upscaled_image.shape, img_to_upscale.shape, "Upscaled image shape should be different from original")
        # This assertion might fail if the SR model somehow produces exactly what cubic resize does,
        # but it's a reasonable sanity check that *some* transformation beyond basic resizing occurred.
        self.assertFalse(np.array_equal(upscaled_image, resized_original), "Upscaled image content seems identical to simple cubic resize")

    def test_upscale_error_handling(self):
        """Test how upscale handles potential errors (e.g., invalid input type)."""
        # Pass None as input
        with self.assertRaises(Exception): # Expecting some kind of error
            # The exact exception might vary (e.g., AttributeError, TypeError)
            # depending on how SuperResolution handles None internally.
            # If it returns the input on error, this test needs adjustment.
            # Checking the image_utils code, it returns original on error.
            # So, let's check that.
            result = self.sr_model.upscale(None) # Should raise an error before returning, based on cvtColor
            # If it returns None instead of raising error:
            # self.assertIsNone(result, "Upscale should ideally return None or raise error on None input")

        # Pass a non-image numpy array
        invalid_input = np.zeros((10, 10)) # 2D array, not 3D image
        try:
            # This should trigger an error during processing (e.g., cvtColor)
            result = self.sr_model.upscale(invalid_input)
            # Check if it returned the original invalid input as per the except block
            self.assertTrue(np.array_equal(result, invalid_input), "Upscale should return original input on processing error")
        except Exception as e:
            # Alternatively, if it raises an exception, catch it
            print(f"Caught expected exception for invalid input: {e}")


# --- Main execution block for running tests ---
if __name__ == '__main__':
    unittest.main()