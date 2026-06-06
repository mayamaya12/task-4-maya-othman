# Project 4 — The Machine's Optic Nerve
### Vision Pipeline: OCR + Object Detection

> **DecodeLabs Industrial Training Kit · Batch 2026**
> Built as part of a structured AI/ML training programme — one project in a four-project portfolio demonstrating real applied AI skills.

---

## What This Does

A single Python script that gives a machine two forms of vision:

**Pipeline A — Document OCR**
Synthesises a medical prescription image, runs it through a full OpenCV pre-processing chain (grayscale conversion + adaptive Gaussian thresholding), then extracts structured text using Google's Tesseract 5 LSTM engine.

**Pipeline B — Object Detection**
Loads MobileNet-SSD via `cv2.dnn`, runs a forward pass on a real scene image, and produces annotated bounding boxes for every detection that exceeds an 80% confidence threshold.

Both pipelines pass all four DecodeLabs validation gates in a single run.

---

## Demo Output

```
══════════════════════════════════════════════════════════════════════
  PIPELINE A  ─  OCR: Medical Prescription Reader
══════════════════════════════════════════════════════════════════════
  [1/5] Generating synthetic medical prescription …
  ✅  GATE 2 PASSED — Grayscale Conversion
  ✅  GATE 2 PASSED — Adaptive Thresholding  (blockSize=25, C=8)
  ✅  GATE 1 PASSED — Library Integration — pytesseract  (OCR in 142ms)
  ✅  GATE 3 PASSED — Accuracy Benchmarking  (8/8 medical terms · 100%)
  ✅  GATE 4 PASSED — Visual Confirmation — OCR Text Output

══════════════════════════════════════════════════════════════════════
  PIPELINE B  ─  Object Detection: MobileNet-SSD
══════════════════════════════════════════════════════════════════════
  ✅  GATE 1 PASSED — Library Integration — cv2.dnn  (189 layers loaded)
  ✅  GATE 2 PASSED — Pre-Processing Integrity
    • person          conf=99.51%   box=(52,12)→(405,466)
    • person          conf=87.32%   box=(294,18)→(616,477)
  ✅  GATE 3 PASSED — Accuracy Benchmarking ≥ 80%
  ✅  GATE 4 PASSED — Visual Confirmation — Annotated Bounding Boxes

  🏆  ALL GATES PASSED — CERTIFICATE ELIGIBLE
```

---

## Tech Stack

| Layer | Tool | Role |
|---|---|---|
| Computer Vision | OpenCV 4.x (`cv2`) | Image pre-processing + DNN inference |
| OCR Engine | pytesseract + Tesseract 5 | Text extraction (LSTM, PSM 6) |
| Deep Learning | MobileNet-SSD (Caffe) | Object detection, PASCAL VOC classes |
| Image Processing | Pillow | Synthetic document generation |
| Numerical | NumPy | Tensor manipulation, coordinate scaling |

---

## Architecture

```
project4_vision_pipeline.py
│
├── PIPELINE A — OCR
│   ├── generate_prescription_image()   Pillow → synthetic 620×480 prescription
│   ├── cv2.cvtColor(BGR2GRAY)          Grayscale conversion
│   ├── cv2.adaptiveThreshold()         Adaptive Gaussian thresholding
│   ├── cv2.morphologyEx(MORPH_CLOSE)   Noise reduction
│   └── pytesseract.image_to_string()   Tesseract 5 LSTM inference
│
└── PIPELINE B — Object Detection
    ├── cv2.dnn.readNetFromCaffe()      Load MobileNet-SSD (189 layers)
    ├── cv2.cvtColor + adaptiveThresh   Same pre-processing chain as Pipeline A
    ├── cv2.dnn.blobFromImage()         Normalise + resize to 300×300
    ├── net.forward()                   Forward pass → (1,1,100,7) detections
    └── confidence filter ≥ 0.80        Bounding box annotation + JSON export
```

---

## Validation Gates

| Gate | Criterion | Result |
|---|---|---|
| 1 | Library Integration (`pytesseract` + `cv2.dnn`) | ✅ Pass |
| 2 | Pre-Processing: Grayscale + Adaptive Threshold | ✅ Pass |
| 3 | Confidence ≥ 80% on all kept detections | ✅ Pass (99.5% max) |
| 4 | Visual output: OCR string + annotated bounding boxes | ✅ Pass |

---

## Quickstart

### Prerequisites

```bash
# System (Linux)
sudo apt install tesseract-ocr fonts-dejavu-core

# macOS
brew install tesseract

# Python packages
pip install opencv-python pytesseract pillow numpy
```

### Run

```bash
python3 project4_vision_pipeline.py
```

First run downloads the MobileNet-SSD weights (~23 MB) and caches them automatically. All subsequent runs are offline.

**Run time:** ~30s first run · ~3s subsequent runs

---

## Output Files

All outputs land in `project4_assets/output/` after a successful run.

```
project4_assets/
├── models/
│   ├── MobileNetSSD_deploy.prototxt       downloaded on first run
│   └── MobileNetSSD_deploy.caffemodel     downloaded on first run (~23 MB)
├── scene.png                              downloaded on first run
└── output/
    ├── A1_prescription_raw.png            synthetic prescription (original)
    ├── A2_prescription_gray.png           after grayscale conversion
    ├── A3_prescription_threshold.png      after adaptive thresholding
    ├── A4_extracted_text.txt              full OCR output + accuracy stats
    ├── A5_pipeline_mosaic.png             raw | gray | threshold side-by-side
    ├── B1_scene_gray.png                  scene in grayscale
    ├── B2_scene_threshold.png             scene after adaptive thresholding
    ├── B3_detections_annotated.png        bounding boxes + confidence labels
    └── B4_detection_results.json         structured detection data
```

**`B4_detection_results.json` schema:**
```json
{
  "pipeline": "MobileNet-SSD Object Detection",
  "model": "MobileNetSSD_deploy.caffemodel",
  "threshold": 0.8,
  "detections": [
    {
      "label": "person",
      "confidence": 0.9951,
      "bbox": [52, 12, 405, 466]
    }
  ]
}
```

---

## Key Technical Decisions

**Why Adaptive Thresholding instead of global thresholding?**
Global thresholding applies a single cutoff to the entire image. Adaptive thresholding computes a separate threshold per 25×25 pixel neighbourhood — essential for real-world scanned documents where illumination is uneven. This is the same technique used in production document digitisation pipelines.

**Why MobileNet-SSD over a heavier model?**
MobileNet-SSD runs inference on CPU in under 100ms per frame, making it suitable for edge devices and embedded clinical systems. The trade-off (fewer parameters than YOLO/ResNet variants) is intentional — this project demonstrates architectural awareness, not just accuracy maximisation.

**Why a single script?**
Both pipelines share the same pre-processing chain (grayscale → adaptive threshold) deliberately. This demonstrates that the same foundational CV steps apply across both vision tasks — a key concept in production vision systems where preprocessing modules are shared and reused.

---

## Narrative Context

This is the **vision layer** for the Medical Patient Simulator built in Project 2.

- **Pipeline A** reads patient prescriptions and clinical reports, transforming raw scanned images into structured text the simulator's NLP engine can ingest.
- **Pipeline B** enables scene understanding — detecting patients, equipment, and clinical objects to give the simulator spatial awareness of its environment.

Together: Projects 2 and 4 form a two-layer architecture. Project 2 handles language and logic. Project 4 handles vision and perception. This mirrors the design of production medical AI systems.

---

## Project Series

This is Project 4 of 4 in the DecodeLabs Industrial Training Kit (Batch 2026).

| # | Project | Core Skill |
|---|---|---|
| 1 | Rule-Based Personality Chatbot | NLP logic, dialogue flow |
| 2 | Medical Patient Simulator (KNN + TF-IDF) | Supervised ML, text classification |
| 3 | AI Career Path Recommender | Hybrid ML pipeline, cosine similarity, K-Means |
| 4 | **Vision Pipeline — OCR + Object Detection** | Computer vision, pre-trained models |

---

## Author

**Maya** — AI/CS Student, Nile University Cairo · AI Automation Developer · Founder, AURA4AI
