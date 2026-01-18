# 🚗 Vehicle Plate Detection System

## Overview

Advanced AI-powered vehicle license plate detection and recognition system with multiple detection algorithms (**YOLO + Transformer**) and OCR engines (**EasyOCR, Tesseract, Google Vision, Deep OCR**).

---

## 🚀 Features

### Core Detection

* **YOLOv8 / YOLOv11** – Primary detection with multiple strategies
* **Transformer (DETR)** – Secondary detection for improved accuracy
* **Computer Vision Fallback** – Haar Cascade, edge detection, color segmentation
* **Multi-scale Detection** – Detects plates at various scales
* **Non-Maximum Suppression** – Removes duplicate detections

### OCR Capabilities

* **EasyOCR** – Fast and accurate text recognition
* **Tesseract OCR** – Open-source OCR engine
* **Google Vision API** – Cloud-based OCR (billing required)
* **Deep OCR (TrOCR)** – Transformer-based OCR
* **Consensus Voting** – Combines multiple OCR results for higher accuracy

### Advanced Features

* Batch processing of images
* Live camera (real-time detection)
* Video file processing
* Image & detection quality assessment
* WebSocket-based progress tracking

---

## 🌐 Web Interface

* **Dashboard** – System overview & stats
* **Upload Interface** – Single & batch uploads
* **Results Management** – Filter, view, export
* **Live Camera** – Real-time detection
* **API Docs** – Interactive API documentation

---

## 📋 Requirements

### Python

* **Python 3.11 (Recommended)**
* Python 3.8+ (Minimum)

### System

* RAM: 4GB minimum (8GB+ recommended)
* Storage: 2GB+ free space
* GPU: Optional (CUDA-compatible recommended)
* Webcam: Required for live camera processing

---

## 🛠️ Installation

### 1️⃣ Clone Repository

```bash
git clone <repository-url>
cd vehicle-plate-detection
```

### 2️⃣ Run Setup Script

```bash
python setup.py
```

### OR Manual Setup

```bash
# Create virtual environment
python -m venv venv

# Activate venv
# Windows
venv\Scripts\activate
# Linux / macOS
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Create directories
mkdir uploads outputs logs temp models data
```

### 3️⃣ Download Models

Automatically downloads:

* YOLOv8 model
* Haar Cascade classifier
* EasyOCR language models

---

## ⚙️ Configuration

### Environment Variables (`.env`)

```env
# ============================================
# VEHICLE PLATE DETECTION SYSTEM - PRODUCTION
# ============================================

APP_NAME=Vehicle Plate Detection Pro
VERSION=3.0.0
DEBUG=False
SECRET_KEY=change-this-secret-key

HOST=0.0.0.0
PORT=5000
WORKERS=4

MAX_UPLOAD_SIZE=100
MAX_CONTENT_LENGTH=104857600

UPLOAD_FOLDER=uploads
OUTPUT_FOLDER=outputs
TEMP_FOLDER=temp
LOG_FOLDER=logs
MODEL_FOLDER=models
DATA_FOLDER=data

DATABASE_URL=sqlite:///data/plates_prod.db

MODEL_PATH=models/yolov8n.pt
USE_GPU=True
CONFIDENCE_THRESHOLD=0.25
USE_TRANSFORMER=True

ENABLE_EASYOCR=True
ENABLE_TESSERACT=True
ENABLE_GOOGLE_VISION=False
ENABLE_DEEP_OCR=True
OCR_LANGUAGES=en

GOOGLE_VISION_API_KEY=your-google-api-key
GOOGLE_APPLICATION_CREDENTIALS=path/to/credentials.json

# Tesseract Path
# Windows
TESSERACT_CMD=C:\Program Files\Tesseract-OCR\tesseract.exe
# Linux / macOS
# TESSERACT_CMD=/usr/bin/tesseract

MAX_WORKERS=4
BATCH_SIZE=8

LOG_LEVEL=INFO
```

---

## ☁️ Google Vision Setup

1. Enable **Google Vision API** in Google Cloud Console
2. Enable **Billing** (mandatory)
3. Create API credentials
4. Download credentials JSON
5. Update `.env`

---

## 🏃‍♂️ Running the Application

### Development Mode

```bash
python app.py
```

OR

```bash
flask run --host=0.0.0.0 --port=5000 --debug
```

---

## 📦 Included Models

### Detection

* YOLOv8n
* DETR Transformer
* Haar Cascade

### OCR

* EasyOCR
* Tesseract OCR
* TrOCR
* Google Vision API

---

## 🌍 Access Web App

Open browser:

```
http://localhost:5000
```

### Pages

* Dashboard
* Upload
* Batch Processing
* Live Camera
* Results
* Analytics
* API Docs

---

## 🔧 API Endpoints

### Core APIs

* `POST /api/process/single`
* `POST /api/process/batch`
* `GET /api/process/batch/<job_id>/status`
* `GET /api/results`

### Upload

* `POST /api/upload/single`
* `POST /api/upload/multiple`

### System

* `GET /api/health`
* `GET /api/system/config`
* `GET /api/system/metrics`
* `POST /api/system/cleanup`

---

## 📊 Export Formats

* JSON
* CSV
* Excel

---

## 🐛 Troubleshooting

### Google Vision Not Working

* Ensure billing is enabled
* Check API permissions
* Verify credentials path

### Tesseract Not Found

* Install Tesseract OCR
* Set correct `TESSERACT_CMD`

### GPU Issues

* Install CUDA toolkit
* Install PyTorch with CUDA
* Set `USE_GPU=False` for CPU-only

### Out of Memory

* Reduce batch size
* Use smaller YOLO model
* Increase system swap

---

## 📁 Logs

* `logs/app.log`
* `logs/error.log`
* `logs/access.log`

---

## 🔒 Security Notes

* Change `SECRET_KEY` in production
* Use HTTPS
* Restrict CORS
* Enable rate limiting
* Keep dependencies updated

---

## 📈 Performance Tips

* Enable GPU acceleration
* Use batch processing
* Resize large images
* Scale workers per CPU cores

---

## 🤝 Contributing

1. Fork repository
2. Create feature branch
3. Add tests
4. Submit pull request

---

## 📄 License

MIT License

---

## 🚀 Quick Start

```bash
git clone https://github.com/Darshan-gowda-m/Car_plate_detection
cd vehicle-plate-detection
python setup.py
python app.py
```

Test API:

```bash
curl -X POST -F "image=@test.jpg" http://localhost:5000/api/process/single
```

---

⚠️ **Note:** Google Vision API requires billing enabled. Free tier may not work for this application.
