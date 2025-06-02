"""Face detection and analysis module."""
import os
import numpy as np
import cv2
from pathlib import Path
from typing import Optional, List, Tuple, Dict, Any

import insightface
from insightface.app import FaceAnalysis

# Import the specific config variable
from config import INSIGHTFACE_MODELS_DIR, MIN_FACE_SIZE # Import INSIGHTFACE_MODELS_DIR and MIN_FACE_SIZE

# Assuming read_image is correctly imported from src.utils.image_utils
# Make sure the import path is correct based on your project structure
try:
    from src.utils.image_utils import read_image, SuperResolution
except ImportError:
    # Fallback or alternative import if needed
    from ..utils.image_utils import read_image, SuperResolution


class FaceAnalyzer:
    """Class for detecting and analyzing faces in images."""

    # Use MIN_FACE_SIZE from config as default
    def __init__(self, det_size: Tuple[int, int] = (640, 640), ctx_id: int = 0, min_face_size: int = MIN_FACE_SIZE):
        """Initialize the face analyzer."""
        print(f"Initializing face analyzer, models expected at: {INSIGHTFACE_MODELS_DIR}")
        os.makedirs(INSIGHTFACE_MODELS_DIR, exist_ok=True)

        self.app = FaceAnalysis(
            name='buffalo_l',
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            root=INSIGHTFACE_MODELS_DIR
        )
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.min_face_size = min_face_size
        self.sr_model: Optional[SuperResolution] = None

    def set_super_resolution(self, sr_model: Optional[SuperResolution]):
        """Set super-resolution model for image enhancement."""
        self.sr_model = sr_model

    # *** NEW/REFACTORED Internal Method ***
    def _analyze_image_array(self, img_bgr: np.ndarray) -> List[Any]:
        """
        Internal method to run face analysis on a NumPy array (BGR format).

        Args:
            img_bgr: The image data as a NumPy array (BGR).

        Returns:
            List of valid face objects found in the image array.
        """
        if img_bgr is None:
            print("Warning: Received None image array for analysis.")
            return []
        try:
            # Convert to RGB for InsightFace
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

            # Detect faces using insightface
            faces = self.app.get(img_rgb) # 'faces' is a list of Face objects

            # Filter faces based on minimum size and presence of embedding
            valid_faces = []
            for face in faces:
                # 1. Check if the face object itself exists and has the bbox attribute
                if face and hasattr(face, 'bbox') and face.bbox is not None:
                    # 2. Check if the bounding box meets the minimum size criteria
                    try:
                        bbox_width = float(face.bbox[2]) - float(face.bbox[0])
                        bbox_height = float(face.bbox[3]) - float(face.bbox[1])

                        # Use self.min_face_size set during initialization
                        if bbox_width >= self.min_face_size and bbox_height >= self.min_face_size:
                            # 3. Check if embedding exists (though not strictly needed for just detection/bbox)
                            # Keep this check if you might use the embedding later from these results
                            if hasattr(face, 'embedding') and face.embedding is not None:
                                 valid_faces.append(face)
                            # If you ONLY need bbox, you could skip the embedding check,
                            # but it's safer to keep it aligned with the original logic.
                            # else:
                            #    print(f"Debug: Face found with valid bbox but no embedding in analyzed array.")
                    except (IndexError, TypeError, ValueError) as bbox_err:
                         print(f"Debug: Error accessing/calculating bbox size [{face.bbox}] for analyzed array: {bbox_err}")

            return valid_faces
        except Exception as e:
            print(f"Error analyzing image array: {e}")
            # import traceback # Optional for more detailed debugging
            # traceback.print_exc()
            return []

    # *** MODIFIED: Now calls the internal method ***
    def extract_faces(self, image_path: str) -> List[Any]:
        """
        Extract faces from an image file path. Reads the image and analyzes it.

        Args:
            image_path: Path to the image file.

        Returns:
            List of valid face objects found in the image file.
        """
        print(f"Reading and analyzing image file: {os.path.basename(image_path)}")
        try:
            # Use read_image utility - it handles non-ASCII paths and
            # can optionally apply SR based on its *own* logic if sr_model is set,
            # although for images from test-data, SR shouldn't trigger here.
            # Pass self.sr_model in case read_image logic needs it (though unlikely here).
            img = read_image(image_path, self.sr_model)
            if img is None:
                print(f"Failed to read image: {image_path}")
                return []
            # Call the internal analysis method that works on the image array
            return self._analyze_image_array(img)
        except Exception as e:
            print(f"Error reading or processing image path {image_path}: {e}")
            # import traceback
            # traceback.print_exc()
            return []

