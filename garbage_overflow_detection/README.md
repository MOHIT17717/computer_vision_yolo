# 🗑️ Garbage Overflow Detection — Smart Cities CV Project

> Real-time garbage bin detection, fill-level estimation, and overflow alerting
> using YOLOv8, OpenCV, and Flask.

---

## 📋 Project Overview

This system detects garbage bins in real-time video or webcam feeds, estimates
their fill level, classifies each bin as **OK** / **Near Full** / **Overflow**,
and triggers visual alerts. It includes a Flask-based web dashboard for remote
monitoring.

### Key Features

- **Real-time Object Detection** — YOLOv8 nano model fine-tuned on garbage bin data
- **Fill-Level Estimation** — Heuristic algorithm using bounding box geometry
- **Visual Alerts** — Colour-coded bounding boxes + overflow banner alerts
- **Event Logging** — CSV-based logging of all overflow events with timestamps
- **Web Dashboard** — Live MJPEG stream, stats cards, and event history table
- **CPU-Friendly** — Runs on CPU; auto-detects and uses GPU if available

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    INPUT SOURCES                                │
│   ┌──────────┐    ┌──────────────┐    ┌────────────────┐       │
│   │  Webcam   │    │  Video File   │    │  RTSP Stream   │       │
│   └─────┬────┘    └──────┬───────┘    └───────┬────────┘       │
│         └────────────────┼────────────────────┘                │
│                          ▼                                     │
│              ┌──────────────────────┐                          │
│              │   YOLOv8n Detector    │ ◄── models/best.pt      │
│              │  (Ultralytics YOLO)   │                          │
│              └──────────┬───────────┘                          │
│                         ▼                                      │
│              ┌──────────────────────┐                          │
│              │  Fill Level Estimator │                          │
│              │  (fill_estimator.py)  │                          │
│              └──────────┬───────────┘                          │
│                         ▼                                      │
│         ┌───────────────┼───────────────┐                      │
│         ▼               ▼               ▼                      │
│  ┌────────────┐  ┌────────────┐  ┌────────────────┐           │
│  │  OpenCV UI  │  │  CSV Logger │  │  Flask Dashboard│           │
│  │  (detect.py)│  │(logs/*.csv) │  │   (app.py)     │           │
│  └────────────┘  └────────────┘  └────────────────┘           │
│                                         │                      │
│                                         ▼                      │
│                                  ┌────────────┐               │
│                                  │  Browser UI │               │
│                                  │ :5000       │               │
│                                  └────────────┘               │
└─────────────────────────────────────────────────────────────────┘
```

---

## ⚙️ Installation

### Prerequisites

- Python 3.10 or higher
- pip (Python package manager)
- Webcam or video file for testing

### Setup

```bash
# 1. Clone / navigate to the project
cd garbage_overflow_detection

# 2. Create a virtual environment
python -m venv venv

# 3. Activate the virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# 4. Install dependencies
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. Train the Model

```bash
python train.py
```

This will:
- Attempt to download the dataset from Roboflow
- Fall back to synthetic dataset generation if download fails
- Train YOLOv8n for 50 epochs
- Auto-retrain with tuned hyperparameters if mAP50 < 0.5
- Save the best model to `models/best.pt`

### 2. Run Real-Time Detection

```bash
# Using webcam
python detect.py --source 0

# Using a video file
python detect.py --source path/to/video.mp4

# Headless mode (no GUI, for Flask integration)
python detect.py --source 0 --headless
```

**Controls:**
| Key | Action |
|-----|--------|
| `q` | Quit detection |
| `s` | Save screenshot |

### 3. Launch Web Dashboard

```bash
# Dashboard only (connect to a running detect.py)
python app.py

# Dashboard + built-in detection
python app.py --with-detection --source 0
```

Then open **http://localhost:5000** in your browser.

---

## 📊 Model Performance

| Metric | Value |
|--------|-------|
| mAP@50 | 0.7419 |
| mAP@50-95 | 0.6164 |
| Precision | 0.7311 |
| Recall | 0.6972 |

### Per-Class Results

| Class | mAP@50 | mAP@50-95 |
|-------|--------|-----------|
| trash_bin | 0.836 | 0.697 |
| garbage | 0.381 | 0.354 |
| overflow | 0.943 | 0.819 |
| lid_open | 0.808 | 0.572 |

*Trained on synthetic dataset (250 images) with YOLOv8n (nano) architecture, 50 epochs on CPU.*

---

## 📂 Project Structure

```
garbage_overflow_detection/
├── data/                    # Dataset (auto-generated or downloaded)
│   ├── train/
│   │   ├── images/
│   │   └── labels/
│   ├── valid/
│   │   ├── images/
│   │   └── labels/
│   └── test/
│       ├── images/
│       └── labels/
├── models/                  # Trained model weights
│   ├── best.pt
│   └── metrics.yaml
├── logs/                    # Event logs
│   ├── events.csv
│   └── detection.log
├── runs/                    # YOLOv8 training runs
├── screenshots/             # Saved screenshots
├── static/                  # Flask static assets
├── templates/
│   └── dashboard.html       # Web dashboard template
├── fill_estimator.py        # Fill level estimation & visualisation
├── detect.py                # Real-time detection script
├── train.py                 # Model training pipeline
├── app.py                   # Flask web dashboard
├── utils.py                 # Shared utility functions
├── data.yaml                # YOLOv8 dataset configuration
├── requirements.txt         # Python dependencies
└── README.md                # This file
```

---

## 🏷️ Detection Classes

| ID | Class Name | Description |
|----|------------|-------------|
| 0 | `trash_bin` | Normal garbage bin |
| 1 | `garbage` | Loose garbage / debris |
| 2 | `overflow` | Overflowing bin |
| 3 | `lid_open` | Bin with lid open |

---

## 🎨 Fill Level Classification

| Status | Fill Range | Colour | Action |
|--------|-----------|--------|--------|
| ✅ OK | < 65% | 🟢 Green | No action needed |
| ⚠️ NEAR_FULL | 65% – 80% | 🟠 Orange | Schedule pickup |
| 🔴 OVERFLOW | > 80% | 🔴 Red | Immediate attention |

---

## 🛠️ Technologies Used

| Technology | Purpose |
|------------|---------|
| **Python 3.10+** | Core programming language |
| **YOLOv8 (Ultralytics)** | Object detection model |
| **OpenCV** | Video processing & visualisation |
| **Flask** | Web dashboard backend |
| **NumPy** | Numerical computations |
| **Matplotlib** | Training metrics plots |
| **Roboflow** | Dataset management |
| **Pillow** | Image processing |
| **Bootstrap 5** | Dashboard UI framework |

---

## 🔮 Future Improvements

- [ ] **Real Dataset** — Train on TACO or custom-labelled garbage bin images
- [ ] **GPS Integration** — Map bin locations for city-wide monitoring
- [ ] **Alert System** — Email / SMS notifications on overflow events
- [ ] **Historical Analytics** — Time-series charts of fill levels over time
- [ ] **Multi-Camera** — Support multiple camera feeds simultaneously
- [ ] **Edge Deployment** — Optimise for Raspberry Pi / Jetson Nano
- [ ] **RTSP Streams** — Support IP camera feeds directly
- [ ] **Database Backend** — Replace CSV logging with PostgreSQL/SQLite
- [ ] **Mobile App** — React Native companion app for field workers
- [ ] **Predictive Model** — Predict when bins will overflow using time-series ML

---

## 📄 License

This project is open-source and available for educational and research purposes.

---

*Built with ❤️ for Smart Cities — making urban waste management smarter.*
