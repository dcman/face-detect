"""Utilities for image loading, processing, and face preview generation."""

import cv2
import numpy as np
import torch
import os
import basicsr
import realesrgan
from PIL import Image
from typing import Optional, Tuple, Any

# --- Placeholder for Real-ESRGAN Model Architecture ---
# TODO: You MUST replace this with the actual import for the RealESRGAN model
#       architecture you are using. This might come from the 'realesrgan'
#       package, 'basicsr', or the specific repository you obtained the
#       model from.
# Example (adjust based on your library):
try:
    # Try importing from a common location (adjust as needed)
    from basicsr.archs.rrdbnet_arch import RRDBNet
except ImportError:
    print("WARNING: Failed to import RRDBNet from basicsr.archs.rrdbnet_arch.")
    print("         Please ensure the required library (e.g., basicsr, realesrgan) is installed")
    print("         and update the import statement in image_utils.py accordingly.")
    # Define a dummy class to avoid crashing if import fails, but SR won't work
    class RRDBNet:
        def __init__(self, *args, **kwargs):
            raise NotImplementedError("RRDBNet class not found. Please install requirements and fix import.")
        def load_state_dict(self, *args, **kwargs):
            pass
        def to(self, *args, **kwargs):
            return self
        def eval(self):
            pass
        def __call__(self, *args, **kwargs):
            raise NotImplementedError("RRDBNet class not found.")

# Suppress PIL DecompressionBombError warnings for large images if needed
# Image.MAX_IMAGE_PIXELS = None

class SuperResolution:
    """Handles image upscaling using a pre-trained Real-ESRGAN model."""

    def __init__(self, model_path: str):
        """
        Initializes the SuperResolution model by loading its state dictionary.

        Args:
            model_path: Path to the pre-trained Real-ESRGAN model state_dict (.pth).
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"SuperResolution using device: {self.device}")

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Super Resolution model not found at: {model_path}")
        if os.path.getsize(model_path) == 0:
             raise ValueError(f"Super Resolution model file is empty: {model_path}. Please check the download.")

        try:
            # --- Load Model using State Dictionary ---
            print(f"Loading SR model state dictionary from: {model_path}")
            # Instantiate the model architecture (adjust params if needed)
            model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
            # Load state dict (CPU first)
            loadnet = torch.load(model_path, map_location=torch.device('cpu'))
            # Handle different key names ('params_ema' or 'params')
            if 'params_ema' in loadnet: keyname = 'params_ema'
            elif 'params' in loadnet: keyname = 'params'
            else: keyname = None

            if keyname: model.load_state_dict(loadnet[keyname], strict=True)
            else: model.load_state_dict(loadnet, strict=True)

            model.eval()
            self.model = model.to(self.device)
            print("SuperResolution model loaded successfully using state dictionary.")

        except (RuntimeError, FileNotFoundError, ValueError, NotImplementedError, KeyError) as e:
            print(f"Error loading SuperResolution model state dictionary from {model_path}: {e}")
            print("Check model path, file integrity, library imports (RRDBNet), and state_dict keys.")
            raise
        except Exception as e:
            print(f"An unexpected error occurred loading the SuperResolution model: {e}")
            raise

    @torch.no_grad()
    def upscale(self, img: np.ndarray) -> np.ndarray:
        """Upscales an image using the Real-ESRGAN model."""
        try:
            if not hasattr(self, 'model'): raise RuntimeError("SuperResolution model is not loaded.")
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor = torch.from_numpy(np.transpose(img_rgb, (2, 0, 1))).float()
            img_tensor = img_tensor.unsqueeze(0).to(self.device) / 255.0
            output_tensor = self.model(img_tensor)
            output = output_tensor.squeeze().float().cpu().clamp_(0, 1) * 255.0
            output_np = np.transpose(output.numpy(), (1, 2, 0)).astype(np.uint8)
            output_bgr = cv2.cvtColor(output_np, cv2.COLOR_RGB2BGR)
            return output_bgr
        except Exception as e:
            print(f"Error during image upscaling: {e}")
            return img # Return original image on error


def read_image(image_path: str, sr_model: Optional[SuperResolution] = None, min_size: int = 100, sr_upscale_factor: int = 4) -> Optional[np.ndarray]:
    """
    Reads an image file robustly (handling non-ASCII paths),
    optionally applying super-resolution if it's too small.
    Uses the default min_size=100 to trigger SR.
    """
    img: Optional[np.ndarray] = None
    try:
        # Read the file as bytes and decode with OpenCV - more robust for non-ASCII paths
        with open(image_path, "rb") as f:
            file_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR) # Use IMREAD_COLOR for BGR

        if img is None:
             print(f"Failed to decode image {image_path} with cv2.imdecode.")
             # Optional: Add fallback to PIL if imdecode fails, though less likely now
             # try:
             #     from PIL import Image
             #     pil_img = Image.open(image_path).convert('RGB')
             #     img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
             # except Exception as pil_err:
             #     print(f"Also failed to read image {image_path} with PIL: {pil_err}")
             #     return None
             # if img is None: # Check again after PIL attempt
             #      print(f"Image data is None after read attempts for: {image_path}")
             #      return None
             return None # Return None if imdecode failed and no fallback used

        # Apply Super Resolution if image is small and model is available
        if sr_model:
            h, w = img.shape[:2]
            if min(h, w) < min_size:
                 reasonable_max_dim = 4000 # Avoid excessive upscaling
                 if (h * sr_upscale_factor < reasonable_max_dim and
                     w * sr_upscale_factor < reasonable_max_dim):
                    print(f"Upscaling image: {image_path} (Original size: {w}x{h})")
                    # Apply SR model
                    img = sr_model.upscale(img)
                 else:
                    print(f"Skipping SR for {image_path}: Original size {w}x{h} is small, but upscaled size might be too large.")
        return img
    except FileNotFoundError:
        print(f"Error: File not found at path {image_path}")
        return None
    except Exception as e:
        print(f"Error reading or processing image {image_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


# --- Updated create_face_preview function ---
def create_face_preview(
    image_path: str,
    face: Any, # Use Any since the exact type from insightface might vary
    person_name: str,
    score: float,
    output_path: str,
    sr_model: Optional[SuperResolution] = None
) -> bool:
    """
    Creates and saves an annotated preview image showing a detected face.

    Reads the original image, draws a bounding box around the specified face,
    adds text with the person's name and score, and saves the result.

    Args:
        image_path: Path to the original image file.
        face: The detected face object (must have a 'bbox' attribute).
        person_name: The identified name or cluster name for the face.
        score: The confidence score associated with the identification/match.
        output_path: The full path where the preview image should be saved.
        sr_model: Optional SuperResolution model instance (passed to read_image).

    Returns:
        True if the preview was created and saved successfully, False otherwise.
    """
    try:
        # Read the image (potentially applying SR based on read_image's logic)
        img = read_image(image_path, sr_model)
        if img is None:
            print(f"Failed to read image {image_path} for preview generation.")
            return False

        # Check if face object is valid and has bounding box
        if face is None or not hasattr(face, 'bbox'):
            #print(f"Invalid face object or missing bbox for preview in {image_path}.")
            return False

        # Get bounding box coordinates and ensure they are integers
        bbox = face.bbox.astype(int)
        x1, y1, x2, y2 = bbox

        # Ensure coordinates are within image bounds (simple clamp)
        img_h, img_w = img.shape[:2]
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(img_w, x2), min(img_h, y2)

        # Draw rectangle around the face (Green color: BGR)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

        # Prepare text label
        label = f"{person_name}: {score:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1 # Use thickness 1 for smaller font scale
        text_color = (0, 255, 0) # Green

        # Calculate text size to position it above the box
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)

        # Put text background rectangle (optional, for better visibility)
        # Adjust y-coordinate for background, ensure it's not negative
        bg_y1 = max(y1 - text_height - baseline - 2, 0)
        bg_y2 = y1 - baseline + 2
        cv2.rectangle(img, (x1, bg_y1), (x1 + text_width, bg_y2), (0, 0, 0), cv2.FILLED) # Black background

        # Put text label above the bounding box
        # Adjust y-coordinate for text, ensure it's not negative
        text_y = max(y1 - baseline - 2, text_height) # Position text baseline slightly above box or at min height
        cv2.putText(img, label, (x1, text_y), font, font_scale, text_color, thickness)

        # Ensure output directory exists
        output_dir = os.path.dirname(output_path)
        os.makedirs(output_dir, exist_ok=True)

        # Save the annotated image
        success = cv2.imwrite(output_path, img)
        if not success:
             print(f"Failed to save preview image to {output_path}")
             return False

        # print(f"Saved preview to {output_path}") # Optional: uncomment for verbose logging
        return True

    except Exception as e:
        print(f"Error creating face preview for {image_path} -> {output_path}: {e}")
        import traceback
        traceback.print_exc() # Print detailed traceback for debugging
        return False

