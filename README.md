#  AI-Powered Intelligent Surveillance System

An end-to-end, modular CCTV analytics pipeline for **real-time person detection, multi-object tracking, face recognition, and open-set identification**.

Unlike traditional CCTV systems that simply record video for later review, this system transforms raw surveillance footage into **structured, searchable intelligence** by automatically detecting, tracking, identifying, and logging individuals across video streams.

---

##  Features

-  Real-time person detection using **YOLOv8**
-  Face detection with **SCRFD**
-  Persistent multi-object tracking using **OC-SORT**
-  Face alignment using facial landmarks
-  Face recognition using **AdaFace / ArcFace**
-  Fast similarity search with **FAISS**
-  Open-set recognition (`UNKNOWN` identities)
-  Annotated video generation
-  Audit logs and run manifests
-  Fully modular and production-inspired architecture

---

# System Architecture

```
                Input Video
                     │
                     ▼
              FrameStream (PyAV)
                     │
                     ▼
          ┌─────────────────────┐
          │ Combined Detector   │
          │ YOLOv8 + SCRFD      │
          └─────────────────────┘
                     │
                     ▼
               OC-SORT Tracker
                     │
                     ▼
         Face-Body Association
                     │
                     ▼
              Face Alignment
                     │
                     ▼
          AdaFace / ArcFace
                     │
                     ▼
              Quality Filtering
                     │
                     ▼
             Temporal Fusion
                     │
                     ▼
         OpenSetIdentifier (FAISS)
                     │
                     ▼
         Annotator + Video Writer
                     │
                     ▼
        annotated.mp4
        audit.jsonl
        run_manifest.json
```

---

# Motivation

Traditional surveillance systems primarily act as recording devices, requiring manual monitoring and time-consuming review of footage.

This project aims to automate that workflow by enabling the system to:

- Detect people automatically
- Track individuals across frames
- Recognise authorised personnel
- Identify unknown visitors
- Generate searchable security logs
- Produce annotated surveillance videos

---

# Project Structure

```
surveillance-system/
│
├── configs/          # Configuration files
├── datasets/         # Input videos & gallery images
├── detection/        # YOLOv8 + SCRFD
├── tracking/         # OC-SORT
├── association/      # Face ↔ Body association
├── alignment/        # Face alignment
├── recognition/      # AdaFace / ArcFace
├── search/           # FAISS search engine
├── rendering/        # Video annotation
├── pipeline/         # Pipeline orchestration
├── scripts/          # Gallery enrolment utilities
├── outputs/          # Generated outputs
├── weights/          # Pretrained models
├── tests/            # Unit tests
├── docs/
├── main.py
└── requirements.txt
```

---

# Pipeline Overview

## 1. Video Decoding

Frames are decoded using **PyAV**, preserving presentation timestamps for accurate processing.

---

## 2. Person & Face Detection

The pipeline performs dual detection:

- **YOLOv8** → Person Detection
- **SCRFD** → Face Detection + Facial Landmarks

---

## 3. Multi-Object Tracking

Detected people are assigned persistent IDs using **OC-SORT**, which combines:

- Kalman Filter
- Hungarian Algorithm

This maintains stable identities across consecutive frames.

---

## 4. Face Alignment

Detected faces are geometrically aligned using facial landmarks to minimise pose variation before recognition.

---

## 5. Face Recognition

Aligned faces are converted into **512-dimensional embeddings** using **AdaFace**.

These embeddings represent identity rather than directly predicting names.

---

## 6. Open-Set Identification

Embeddings are searched against an enrolled gallery using **FAISS**.

If the similarity exceeds the configured threshold:

```
Known Person
```

Otherwise:

```
UNKNOWN
```

This prevents incorrect identity assignments.

---

## 7. Output Generation

The pipeline generates:

```
annotated.mp4
```

Annotated surveillance footage.

```
audit.jsonl
```

Structured event log.

```
run_manifest.json
```

Execution metadata.

---

# Offline Gallery Enrolment

Gallery images are processed once before inference.

```
Gallery Images
       │
       ▼
 Face Detection
       │
       ▼
 Face Alignment
       │
       ▼
 AdaFace Embedding
       │
       ▼
 FAISS Index
```

During runtime only similarity search is performed.

---

# Tech Stack

| Component | Technology |
|------------|------------|
| Language | Python |
| Detection | YOLOv8 |
| Face Detection | SCRFD |
| Tracking | OC-SORT |
| Recognition | AdaFace / ArcFace |
| Similarity Search | FAISS |
| Video Processing | PyAV |
| Numerical Computing | NumPy |
| Deep Learning | PyTorch |

---

# Design Principles

- Modular Architecture
- Single Responsibility per Module
- Open-Set Recognition
- Production-Oriented Pipeline
- Easy Model Replacement
- Separation of Offline Enrolment and Online Inference

---

# Example Output

✔ Person Detection

✔ Persistent Tracking

✔ Face Recognition

✔ UNKNOWN Classification

✔ Annotated Video

✔ Audit Logging

---


