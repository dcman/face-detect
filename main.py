"""Main script to run the face recognition and photo organization pipeline."""

import os
import time
import pickle
from pathlib import Path
from collections import defaultdict
from config import (
    SOURCE_DIR, OUTPUT_DIR, KNOWN_PEOPLE_DIR, FACE_DB_PATH,
    SR_MODEL_PATH, SR_MODEL_URL, # SR settings are needed again
    MAX_WORKERS, SIMILARITY_THRESHOLD,
    CLUSTERING_EPS, CLUSTERING_MIN_SAMPLES, SUPPORTED_FORMATS,
    SR_UPSCALING_THRESHOLD # Threshold used by read_image
)
# Need download_file again for SR model
from src.utils.file_utils import download_file
# Need SuperResolution class again
from src.utils.image_utils import SuperResolution
from src.models.face_analyzer import FaceAnalyzer
from src.models.face_database import FaceDatabase
from src.processors.face_collector import FaceCollector
from src.processors.face_clustering import FaceClustering
from src.processors.photo_organizer import PhotoOrganizer

def main():
    """Runs the complete face recognition and organization pipeline."""
    start_time = time.time()

    # --- Ensure Output Directories Exist ---
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    Path(KNOWN_PEOPLE_DIR).mkdir(parents=True, exist_ok=True)
    db_dir = os.path.dirname(FACE_DB_PATH)
    if db_dir: Path(db_dir).mkdir(parents=True, exist_ok=True)
    # Ensure SR model directory exists if path is specified
    if SR_MODEL_PATH:
        sr_dir = os.path.dirname(SR_MODEL_PATH)
        if sr_dir: Path(sr_dir).mkdir(parents=True, exist_ok=True)

    print(f"Source directory: {SOURCE_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
    print(f"Known people directory: {KNOWN_PEOPLE_DIR}")
    print(f"Face database path: {FACE_DB_PATH}")
    print(f"Supported image formats: {SUPPORTED_FORMATS}")
    print(f"SR Upscaling Threshold: {SR_UPSCALING_THRESHOLD}px")

    # --- Download and Initialize Super Resolution Model ---
    # This section is needed again to potentially initialize SR
    sr_model = None
    if SR_MODEL_URL and SR_MODEL_PATH: # Check if path is defined
        if not os.path.exists(SR_MODEL_PATH):
            print(f"Downloading Super Resolution model from {SR_MODEL_URL}...")
            if download_file(SR_MODEL_URL, SR_MODEL_PATH):
                 print("Super Resolution model downloaded.")
            else:
                 print(f"Warning: Failed to download Super Resolution model. SR will be disabled.")
        # Check again after potential download, and also check size
        if os.path.exists(SR_MODEL_PATH) and os.path.getsize(SR_MODEL_PATH) > 0:
            try:
                print("Initializing SuperResolution model...")
                # Pass scale if needed by your specific SR model/class implementation
                sr_model = SuperResolution(SR_MODEL_PATH)
                print("SuperResolution model initialized.")
            except Exception as e:
                print(f"Warning: Failed to initialize SuperResolution model: {e}. SR will be disabled.")
                sr_model = None # Ensure it's None if init fails
        elif os.path.exists(SR_MODEL_PATH):
             print(f"Warning: Super Resolution model file exists but is empty: {SR_MODEL_PATH}. SR will be disabled.")
        else:
             print("Super Resolution model not found. SR will be disabled.")
    else:
        print("SR_MODEL_PATH not configured. SR will be disabled.")


    # --- Initialize TWO Face Analyzers ---
    # One without SR, one potentially with SR
    print("Initializing Face Analyzers...")
    try:
        # Analyzer for KNOWN_PEOPLE (Database building) - NO SR
        face_analyzer_no_sr = FaceAnalyzer()
        print(" -> Face Analyzer (No SR) initialized for Database.")

        # Analyzer for SOURCE_DIR (Collection) - potentially WITH SR
        face_analyzer_with_sr = FaceAnalyzer()
        if sr_model:
            # If SR model loaded successfully, set it on this analyzer
            face_analyzer_with_sr.set_super_resolution(sr_model)
            print(" -> Face Analyzer (With SR if needed) initialized for Source Dir.")
        else:
            print(" -> Face Analyzer (SR Disabled) initialized for Source Dir.")

    except Exception as e:
        print(f"Fatal Error: Failed to initialize one or more Face Analyzers: {e}")
        exit(1)

    # --- Build or Load Face Database ---
    print("Loading/Building Face Database...")
    # *** Use the analyzer WITHOUT SR for building the database ***
    face_db = FaceDatabase(face_analyzer_no_sr, FACE_DB_PATH, SIMILARITY_THRESHOLD)
    if not face_db.load():
        print(f"No existing or valid database found at {FACE_DB_PATH}. Building from {KNOWN_PEOPLE_DIR}...")
        # build_from_directory uses the analyzer passed during init (face_analyzer_no_sr)
        # read_image called inside will NOT receive an sr_model, so no upscaling happens here.
        face_db.build_from_directory(KNOWN_PEOPLE_DIR, SUPPORTED_FORMATS)
        face_db.save()
    print(f"Database contains {len(face_db.known_embeddings)} known people.")

    # --- Collect Faces from Source Directory ---
    print(f"Collecting faces from {SOURCE_DIR}...")
    # *** Use the analyzer potentially WITH SR for collecting from source ***
    collector = FaceCollector(face_analyzer_with_sr, max_workers=MAX_WORKERS)
    # collect_faces uses the analyzer passed during init (face_analyzer_with_sr)
    # read_image called inside WILL receive the sr_model (if loaded), enabling
    # conditional upscaling based on SR_UPSCALING_THRESHOLD for small images here.
    collected_faces_data = collector.collect_faces(SOURCE_DIR, SUPPORTED_FORMATS)
    print(f"Collected face data from {len(collected_faces_data)} images containing faces (after potential SR).")


    # --- Identify Known / Collect Unknown Faces (Single Pass) ---
    # This part uses the collected_faces_data (potentially from upscaled images)
    # and compares against the face_db (built from non-upscaled images).
    # This comparison logic remains the same.
    known_faces_for_organizer = defaultdict(list)
    unknown_faces_for_clustering = []
    processed_images_with_unknowns = set()
    print("Identifying known faces and collecting unknown faces...")
    total_faces_processed_in_loop = 0
    total_known_matches = 0
    total_unknowns_collected = 0
    for image_path, faces in collected_faces_data:
        if not faces: continue
        for face in faces:
            total_faces_processed_in_loop += 1
            if hasattr(face, 'embedding') and face.embedding is not None:
                # Use the single face_db (built without SR) for identification
                person_name, score = face_db.identify_person(face.embedding)
                if person_name:
                    known_faces_for_organizer[person_name].append((image_path, face, score))
                    total_known_matches += 1
                else:
                    unknown_faces_for_clustering.append((image_path, face.embedding, face))
                    processed_images_with_unknowns.add(image_path)
                    total_unknowns_collected += 1

    print(f"Processed {total_faces_processed_in_loop} valid faces.")
    print(f"Found {total_known_matches} known face instances.")
    print(f"Collected {total_unknowns_collected} unknown face instances for clustering.")


    # --- Cluster Unknown Faces ---
    clustered_unknowns = {}
    if unknown_faces_for_clustering:
        print("Clustering unknown faces...")
        clustering = FaceClustering(eps=CLUSTERING_EPS, min_samples=CLUSTERING_MIN_SAMPLES)
        # clustered_unknowns maps cluster_name -> list of (path, embedding, face_obj)
        clustered_unknowns = clustering._cluster_unknown_faces(unknown_faces_for_clustering)

        # *** MODIFIED: Save only filenames for standalone analysis ***
        clusters_save_path = os.path.join(OUTPUT_DIR, "clustered_unknowns.pkl")
        print(f"Saving clustered unknown filenames data to: {clusters_save_path}")
        try:
            # Create a simplified dictionary with just image paths
            clusters_for_filename_analysis: Dict[str, List[str]] = {}
            for cluster_name, face_data_list in clustered_unknowns.items():
                # Extract only the first element (image path) from each tuple
                paths_only = [data[0] for data in face_data_list]
                clusters_for_filename_analysis[cluster_name] = paths_only

            # Save the simplified dictionary
            with open(clusters_save_path, 'wb') as f_clusters:
                pickle.dump(clusters_for_filename_analysis, f_clusters) # Save paths only

        except Exception as e_save:
            print(f"Warning: Failed to save clustered unknown data: {e_save}")
        # *** End of modified section ***

    else:
        print("No unknown faces found to cluster.")
        # (Keep the logic to remove the old file if no clusters are found)
        clusters_save_path = os.path.join(OUTPUT_DIR, "clustered_unknowns.pkl")
        if os.path.exists(clusters_save_path):
             try: os.remove(clusters_save_path); print(f"Removed old cluster file: {clusters_save_path}")
             except OSError: pass


    # --- Analyze Filenames (Optional: Call function here if integrating later) ---
    # if clustered_unknowns: # Still use original clustered_unknowns if integrating here
    #    known_names_list = list(face_db.reference_db.keys())
    #    analyze_unknown_cluster_filenames(clustered_unknowns, known_names_list, common_names)


    # --- Organize Photos ---
    print(f"Organizing photos into {OUTPUT_DIR}...")
    people_to_organize = {**known_faces_for_organizer, **clustered_unknowns}
    # Pass sr_model to organizer ONLY if create_face_preview needs it internally.
    # If create_face_preview takes the already processed image array, sr_model isn't needed here.
    # Let's assume it doesn't need it for simplicity based on previous implementation.
    organizer = PhotoOrganizer(OUTPUT_DIR) # Pass sr_model=sr_model if needed by organizer internals
    processed_images_set = organizer.organize_photos(people_to_organize)
    print(f"Organized photos for {len(people_to_organize)} people/clusters.")
    print(f"A total of {len(processed_images_set)} unique images were processed and moved/copied.")


    # --- Handle Unprocessed Images ---
    # (Logic remains the same)
    print("Handling images that were not processed...")
    count_no_faces = organizer.handle_unprocessed_images(SOURCE_DIR, SUPPORTED_FORMATS)
    print(f"Moved {count_no_faces} images with no detected faces to the 'no_faces' directory.")


    end_time = time.time()
    print(f"\nPipeline finished in {end_time - start_time:.2f} seconds.")


if __name__ == "__main__":
    main()
