# Face Recognition and Photo Organizer

This project performs automatic face recognition, clustering of unknown individuals, and photo organization. It leverages `insightface` for face analysis, optionally applies super-resolution for small images using `Real-ESRGAN`, and organizes images by known individuals or clustered unknowns.

---

## Features

- Detect and identify known individuals based on reference images.
- Cluster unknown faces using DBSCAN.
- Optionally apply super-resolution to low-resolution images.
- Organize photos into folders per person/cluster.
- Generate preview images with bounding boxes and labels.
- Supports `.jpg`, `.jpeg`, `.png`, `.bmp`, `.webp`, and `.tiff`.

---

## Setup Instructions

### 🧰 Dependencies (macOS & Windows)

Install Python 3.8 or later, then install the required packages:

```bash
pip install -r requirements.txt
```

Your `requirements.txt` should include:

```txt
opencv-python
insightface
scikit-learn
numpy
torch
pillow
realesrgan
basicsr
```

> **Note:** On Windows, you may need to install `torch` with a compatible CUDA version or fallback to CPU.

### 📦 External Models

This project requires:

1. **Real-ESRGAN x4+ model**  
   Download automatically on first run or manually from:  
   https://github.com/xinntao/Real-ESRGAN/releases  
   Save as:  
   ```
   ./models/4xESRGAN.pth
   ```

2. **InsightFace models**  
   These are downloaded automatically on first run and stored in:
   ```
   ~/.insightface/models/
   ```

---

## File Structure

```
project/
│
├── main.py                      # Main pipeline entry point
├── config.py                    # Configuration and paths
│
├── models/
│   └── 4xESRGAN.pth             # Super-resolution model (Real-ESRGAN)
│
├── known_people/               # Folder of known people with reference images
│   ├── Alice/
│   │   └── image1.jpg
│   └── Bob/
│       └── image1.jpg
│
├── input/                      # Source directory for unorganized photos
│   └── *.jpg/*.png/etc.
│
├── output/                     # Output organized photos
│   ├── Alice/
│   │   ├── image.jpg
│   │   └── previews/
│   │       └── preview_image.jpg
│   ├── unknown_person_0/
│   ├── no_faces/
│   └── clustered_unknowns.pkl
│
├── src/
│   ├── models/
│   │   ├── face_analyzer.py
│   │   └── face_database.py
│   ├── processors/
│   │   ├── face_collector.py
│   │   ├── face_clustering.py
│   │   └── photo_organizer.py
│   └── utils/
│       ├── image_utils.py
│       └── file_utils.py
```

---

## Usage

Once all files are organized as shown above:

```bash
python main.py
```

You’ll see console logs detailing every stage, from face detection to organization and preview generation.

---

## Customization

Modify thresholds or paths via `config.py`:

- `SOURCE_DIR`, `OUTPUT_DIR`, `KNOWN_PEOPLE_DIR` — Input/output directories.
- `SIMILARITY_THRESHOLD` — Face match strictness.
- `CLUSTERING_EPS`, `CLUSTERING_MIN_SAMPLES` — DBSCAN parameters.
- `SR_UPSCALING_THRESHOLD` — Minimum image size before SR is applied.

---

## Notes

- This script runs best with a GPU (for InsightFace and Super-Resolution).
- For large photo collections, set `MAX_WORKERS` in `config.py` to match your CPU core count.

---

## License

MIT License. See LICENSE file for details.
