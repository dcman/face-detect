"""Configuration settings for the face recognition system."""
import os
from typing import Set, Optional

# --- Directory Paths ---
# These paths are expected to be mounted into the Docker container.
# They can be overridden using environment variables.

# Directory containing the images to be processed.
SOURCE_DIR: str = os.getenv("SOURCE_DIR", "./input")

# Directory where organized photos and previews will be saved.
OUTPUT_DIR: str = os.getenv("OUTPUT_DIR", "./output")

# Directory containing subdirectories, each named after a known person
# and containing their reference images.
KNOWN_PEOPLE_DIR: str = os.getenv("KNOWN_PEOPLE_DIR", "./known_people")

# Directory where models are stored (used for SR model path).
MODELS_DIR: str = os.getenv("MODELS_DIR", "./models")

# --- Image Processing Parameters ---
SR_UPSCALING_THRESHOLD: int = 150  # Minimum width or height (in pixels) for SR upscaling.

# Set of supported image file extensions (lowercase).
SUPPORTED_FORMATS: Set[str] = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}

# Minimum size (height or width in pixels) of a face to be considered valid.
MIN_FACE_SIZE: int = 32

# --- Face Matching and Clustering Parameters ---

# Cosine similarity threshold for matching a face to a known person.
# Higher values mean stricter matching (faces must be more similar).
SIMILARITY_THRESHOLD: float = 0.6

# DBSCAN clustering parameter: Maximum distance between samples for one to be
# considered as in the neighborhood of the other. Used for clustering unknown faces.
# Lower values lead to smaller, denser clusters.
CLUSTERING_EPS: float = 0.4

# DBSCAN clustering parameter: The number of samples in a neighborhood for a
# point to be considered as a core point. This includes the point itself.
# Higher values mean fewer clusters are formed.
CLUSTERING_MIN_SAMPLES: int = 3

# --- Performance Settings ---

# Maximum number of worker threads to use for parallel processing tasks
# like face collection. Adjust based on your system's CPU cores.
MAX_WORKERS: int = int(os.getenv("MAX_WORKERS", "20"))

# --- Model and Database Paths ---

# Path to the serialized face database file (stores known face embeddings).
# Can be overridden using the FACE_DB_PATH environment variable.
FACE_DB_PATH: str = os.getenv("FACE_DB_PATH", os.path.join(OUTPUT_DIR, "face_database.pkl"))

# Path to the Super Resolution model file within the container.
SR_MODEL_PATH: str = os.path.join(MODELS_DIR, "4xESRGAN.pth")

# URL to download the Super Resolution model if it doesn't exist locally.
# Verify this URL points to the correct 'RealESRGAN_x4plus.pth' model.
# The original URL seemed to point to a different SR model.
SR_MODEL_URL: str = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.1/ESRGAN_SRx4_DF2KOST_official-ff704c30.pth" # Placeholder - replace with correct URL for RealESRGAN_x4plus.pth if needed

# Directory where InsightFace models are stored/downloaded inside the container.
# This should match the target path of the Docker volume mount.
INSIGHTFACE_MODELS_DIR: str = os.getenv("INSIGHTFACE_MODELS_DIR", "/root/.insightface/models/")