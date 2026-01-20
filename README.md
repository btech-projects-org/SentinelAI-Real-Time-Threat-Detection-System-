# SentinelAI - Real-Time Threat Detection System

SentinelAI is an advanced autonomous security system leveraging Deep Learning (YOLOv8, DeepFace) to detect threats (weapons, violence) and identify known criminals in real-time. It provides instant alerts via Telegram/Websockets and logs incidents to a secure database.

## 🚀 Features

*   **Real-Time Threat Detection**: Detects guns, knives, and violent behavior using custom-trained YOLOv8 models.
*   **Facial Recognition**: Identifies known criminals using DeepFace (FaceNet512) and matches them against a MongoDB database.
*   **Instant Alerts**: Sends screenshot-accompanied alerts to Telegram and the Frontend Dashboard immediately upon detection.
*   **Live Dashboard**: React-based frontend for monitoring live video feeds and viewing alert history.
*   **Secure & Scalable**: Built with FastAPI, MongoDB, and optimizes performance for edge deployment.

## 🛠️ Tech Stack

*   **Frontend**: React, TypeScript, Vite, TailwindCSS
*   **Backend**: Python, FastAPI, Uvicorn
*   **Deep Learning**: YOLOv8 (Ultralytics), DeepFace, TensorFlow, PyTorch
*   **Database**: MongoDB
*   **Notifications**: Telegram Bot API

## 📦 Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/btech-projects-org/SentinelAI-Real-Time-Threat-Detection-System-.git
    cd SentinelAI-Real-Time-Threat-Detection-System-
    ```

2.  **Environment Setup:**
    Create a `.env` file in the root directory (see `.env.example` if available, or ask administrator for keys).
    Required keys: `MONGO_URI`, `DB_NAME`, `TELEGRAM_BOT_TOKEN`, `GEMINI_API_KEY`.

3.  **Run Verification & Setup:**
    We have included a robust setup script to validate your environment and install dependencies automatically.
    ```bash
    python setup_verify.py
    ```

4.  **Start the Application:**
    *   **Backend**: `python backend/main.py`
    *   **Frontend**: `npm run dev`

## 🛡️ Usage

*   Access the dashboard at `http://localhost:5173`.
*   The system will automatically connect to the configured camera source and begin analysis.
*   Alerts will appear in the "Live Alerts" panel and be sent to the configured Telegram chat.

## 📄 License

[MIT License](LICENSE)

---

## 📘 Repository Architecture & System Documentation

### 1. Project Overview
SentinelAI is a production-grade **Autonomous Threat Response System** designed for high-security environments (banks, airports, public spaces). It bridges the gap between passive surveillance and active intervention by using **Computer Vision** to detect specific threat signatures (weapons, fighting) and **Biometric Identification** to flag blacklisted individuals.

Unlike standard CCTV, SentinelAI processes video streams in real-time (edge-compatible), making instantaneous decisions to notify security personnel via **Telegram** and a **Central Command Dashboard**. The system integrates **YOLOv8** for object detection and **DeepFace (FaceNet512)** for state-of-the-art facial recognition.

**Key Technologies:**
*   **Computer Vision**: YOLOv8 (Object Detection), OpenCV (Image Processing), DeepFace (Biometrics).
*   **Backend Architecture**: FastAPI (High-performance Async I/O), WebSockets (Real-time streaming).
*   **Frontend Interface**: React + Vite (Low-latency dashboard updates).
*   **Infrastructure**: MongoDB (NoSQL Persistence), Docker-ready structure.

### 2. Repository Structure & File Responsibilities

```
SentinelAI-Real-Time-Threat-Detection-System/
│
├── backend/                       # Core Python Backend & AI Engine
│   ├── auth/                      # Authentication & Security logic
│   ├── config/                    # Application Configuration (pydantic)
│   ├── database/                  # MongoDB Connection & ODM layer
│   ├── models/                    # Pydantic data models (validation)
│   ├── schemas/                   # API Request/Response schemas
│   ├── services/
│   │   ├── dl_threat_engine.py    # MAIN AI ENGINE: YOLOv8 + DeepFace Logic
│   │   ├── telegram_service.py    # Telegram Bot API integration
│   │   └── threat_engine.py       # Legacy/heuristic engine (supporting logic)
│   ├── main.py                    # Application Entry Point & WebSocket Handler
│   └── yolov8n.pt                 # Pre-trained YOLOv8 Nano model (auto-downloaded)
│
├── components/                    # React UI Components (VideoFeed, Alerts)
├── hooks/                         # Custom React Hooks (State management)
├── pages/                         # Main Frontend Application Views
├── utils/                         # Frontend utility functions
│
├── setup_verify.py                # Automated Environment Validation Script
├── requirements.txt               # Python Dependencies (Strict Versioning)
├── .env                           # Secrets & Config (Excluded from Git)
├── package.json                   # Frontend Dependencies & Scripts
├── vite.config.ts                 # Vite Build Configuration
└── README.md                      # Project Documentation
```

### 3. Environment Variables & Configuration
The system complies with **12-Factor App** principles. All configuration is managed via `.env`.

| Variable | Type | Required | Description |
| :--- | :--- | :--- | :--- |
| **Database** | | | |
| `MONGO_URI` | Connection String | **YES** | MongoDB connection URI (e.g., `mongodb://localhost:27017`) |
| `DB_NAME` | String | **YES** | Target database name (e.g., `sentinel_core`) |
| **Security** | | | |
| `JWT_SECRET` | String | **YES** | Cryptographic key for signing Auth Tokens. **MUST** be strong. |
| `ADMIN_PASSWORD` | String | **YES** | Password for the root `admin` user. |
| **Notifications** | | | |
| `TELEGRAM_BOT_TOKEN` | String | **YES** | BotFather token for alert delivery. |
| `TELEGRAM_CHAT_ID` | String | **YES** | Target Chat/Channel ID for receiving alerts. |
| **AI Services** | | | |
| `GEMINI_API_KEY` | Key | **YES** | Google Gemini API Key for advanced threat analysis. |
| `CONFIDENCE_THRESHOLD` | Float | No | Global AI confidence minimum (Default: `0.60`). |
| `YOLO_MODEL_PATH` | Path | No | Path to custom YOLO model (Default: `./models/yolov8n.pt`). |
| `DEEPFACE_MODEL` | String | No | Face Recognition model name (Default: `Facenet512`). |

### 4. Dependency Analysis
The `requirements.txt` file defines the precise runtime environment.

**Core Backend:**
*   `fastapi`, `uvicorn`: High-performance ASGI web server framework.
*   `pydantic`, `pydantic-settings`: Data validation and settings management.

**Deep Learning & Vision:**
*   `ultralytics`: YOLOv8 implementation for weapon detection (Gun, Knife).
*   `deepface`: Wrapper for state-of-the-art Face Recognition models (FaceNet, VGG-Face).
*   `tensorflow`: Backend engine for DeepFace tensor operations.
*   `opencv-python-headless`: Core image processing (reading frames, resizing, headers).
*   `torch`, `torchvision`: PyTorch runtime for YOLOv8 execution.

**Infrastructure:**
*   `pymongo`, `motor`: Async MongoDB drivers for non-blocking database I/O.
*   `python-telegram-bot`: Async wrapper for Telegram API interactions.
*   `python-jose`, `passlib`: Cryptography for JWT tokens and password hashing.

### 5. System & Laptop Configuration Requirements

**Minimum Requirements (CPU Inference):**
*   **OS**: Windows 10/11, Linux (Ubuntu 20.04+), or macOS.
*   **CPU**: Intel Core i5 (8th Gen) / AMD Ryzen 5 or better.
*   **RAM**: 8 GB (DeepFace models load into memory).
*   **Storage**: 2 GB free space (for models and dependencies).
*   **Python**: 3.10, 3.11, or 3.12 (Validated).

**Recommended Requirements (GPU Accelerated):**
*   **GPU**: NVIDIA GTX 1060 (6GB) or higher.
*   **CUDA**: Version 11.8 or 12.1.
*   **RAM**: 16 GB DDR4.
*   **Performance**: ~30 FPS on high-res streams (vs ~5-10 FPS on CPU).

**Windows-Specific Notes:**
*   **Long Paths**: Deep learning libraries like TensorFlow often exceed the 260-character path limit. The `setup_verify.py` script includes an auto-fix for this Registry setting.

### 6. Application Execution Flow

1.  **Bootstrapping**:
    *   `main.py` initializes. It first checks for a Virtual Environment (`.venv`) and auto-switches if running globally.
    *   Configuration (`config.py`) loads `.env` variables and validates presence of critical keys (`CRITICAL` check).

2.  **Resource Initialization**:
    *   **Database**: Connection pool to MongoDB is established.
    *   **AI Models**:
        *   Checks for `yolov8n.pt`. Auto-downloads from Ultralytics if missing.
        *   Initializes `DeepFace` engine.
    *   **Biometric Data**: Loads all registered "Criminal" profiles from MongoDB, generating/caching 512-D embeddings for fast searching.

3.  **Server Startup**:
    *   Uvicorn starts the ASGI server on `0.0.0.0:8000`.
    *   WebSocket endpoint `/ws/video-feed` becomes active.

4.  **Runtime Loop (Per Frame)**:
    *   Frontend sends video frame via WebSocket.
    *   **Layer 1 (Object)**: YOLOv8 scans for `weapon`, `knife`, `gun`. If Conf > Threshold -> **ALERT**.
    *   **Layer 2 (Behavior)**: System checks for `fighting` based on Person-to-Person IoU overlap.
    *   **Layer 3 (Biometric)**: DeepFace extracts faces and compares embeddings against the loaded criminal database. If Match -> **ALERT**.
    *   **Response**: Frame metadata (boxes, labels) sent back to Frontend; Alerts dispatched to Telegram.

### 7. Setup & Installation Best Practices
*   **Virtual Environment**: Always use the provided `setup_verify.py` to ensure you are running in the isolated `.env`.
*   **Database**: Ensure MongoDB is running locally (`localhost:27017`) or update `MONGO_URI` to point to Atlas.
*   **Windows User**: If you encounter `OSError: [Errno 2] No such file` during install, run `python setup_verify.py` which detects and fixes Windows Long Path issues.

### 8. Security & Best Practices
*   **Secret Management**: never commit `.env` to version control. The repository includes `.env.example` safe for public viewing.
*   **Authentication**: The dashboard requires a JWT token login (`admin` / `ADMIN_PASSWORD`).
*   **Rate Limiting**: The API implements rate limiting (e.g., 60 req/min) to prevent abuse.
*   **HTTPS**: Middleware for HTTPS redirection is included in `main.py` (enabled in production mode).

---
📄 Documentation auto-generated by repository analysis.
Existing content preserved intentionally.
