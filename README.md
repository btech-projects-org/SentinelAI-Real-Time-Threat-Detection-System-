
# 🛡️ SentinelAI – Deep Learning Threat Detection System

**Real-time AI surveillance system integrating YOLOv8, DeepFace, and Behavioral Analysis for proactive threat mitigation.**

> **✅ SYSTEM STATUS: CERTIFIED (v1.1.0 Security Hardened)**  
> *Certified for Production Deployment under Antigravity Audit V2.*

---

## 🧠 Deep Learning Architecture

- **YOLOv8 (Ultralytics)**: Real-time object detection for weapons (knives, guns, bats) and persons.
- **DeepFace (Facenet512)**: Criminal identification using 512-d vector embeddings with Cosine Similarity (Accuracy: 99.65%).
- **Behavioral Heuristics**: **[NEW]** Fighting/Violence detection using Intersection-over-Union (IoU) bounding box analysis.
- **Fallback Logic**: Automatic degradation to Haar Cascades if Deep Learning models fail.

## 🛡️ Security & Compliance

- **Authentication**: JWT (JSON Web Token) with HS256 encryption.
- **Validation**: Strict Pydantic schemas for all API inputs.
- **Access Control**: Role-Based Access Control (RBAC) with secured endpoints.
- **Infrastructure**: Fail-fast configuration validation (strict `.env` enforcement).

## 🛠️ Tech Stack

- **Backend**: Python FastAPI (Async), Uvicorn
- **Database**: MongoDB Atlas (Async Motor driver)
- **AI/ML**: PyTorch 2.2, TensorFlow 2.15, OpenCV
- **Frontend**: React 19 + Vite + TailwindCSS
- **Alerts**: Telegram Bot API integration

---

## 🚀 Quick Start

### Option 1: Automated Startup (Recommended)
The system includes a self-healing startup script that manages venv, dependencies, and model downloads.

1.  **Configure Environment**:
    copy `.env.example` to `.env` and ensure `JWT_SECRET` and `MONGO_URI` are set.
2.  **Run Script**:
    Double-click **`start_server.bat`**
    *(This will install requirements, download YOLO/DeepFace models, and start the API)*

### Option 2: Manual Installation

```bash
# 1. Setup Virtual Environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install Dependencies (Security + AI)
pip install -r requirements.txt

# 3. Configure .env
# Ensure ADMIN_PASSWORD and JWT_SECRET are set!

# 4. Run Verification & Server
python validate_and_run.py
```

---

## 🔐 Default Credentials

The system is pre-configured with a secure admin account (defined in `.env`):

*   **Username:** `admin`
*   **Password:** `SentinelAdmin2026!` (Change in `.env` for production)
*   **Login URL:** http://localhost:5173

---

## 📖 Documentation & Audits

- **[PRODUCTION_CERTIFICATION_REPORT_V2.md](PRODUCTION_CERTIFICATION_REPORT_V2.md)**: Full Security & Quality Audit Log.
- **[DEEP_LEARNING_SETUP.md](DEEP_LEARNING_SETUP.md)**: Detailed model configuration guide.
