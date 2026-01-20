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
