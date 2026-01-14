
# 🛡️ SentinelAI – Deep Learning Threat Detection System

**Real-time AI surveillance using YOLOv8 and DeepFace neural networks**

## 🧠 Deep Learning Architecture

- **YOLOv8** (Ultralytics) - Weapon/threat detection (80 object classes)
- **DeepFace (Facenet512)** - Face recognition (99.65% LFW accuracy)
- **PyTorch 2.2** - Neural network inference
- **Real-time processing** - 10-20 FPS (CPU), 50-100 FPS (GPU)

## Tech Stack

- **Backend**: Python FastAPI (Async)
- **Database**: MongoDB Atlas
- **Alerts**: Telegram Bot
- **Frontend**: React 19 + Vite + TailwindCSS
- **Deep Learning**: PyTorch, YOLOv8, DeepFace, TensorFlow
- **Deployment**: Local (CPU) or GPU-accelerated

## 🚀 Quick Start

### 1. Install Deep Learning Dependencies

```bash
# Activate virtual environment
.venv\Scripts\activate

# Install PyTorch (CPU version)
pip install torch==2.2.0 torchvision==0.17.0 --index-url https://download.pytorch.org/whl/cpu

# Install deep learning frameworks
pip install ultralytics==8.1.24 deepface==0.0.91 tensorflow==2.15.0

# Or install all dependencies
pip install -r requirements.txt
```

### 2. Configure Environment

Update `.env` with your credentials:

```dotenv
MONGO_URI=mongodb://localhost:27017
JWT_SECRET=your-strong-secret-key-minimum-32-characters

# Optional: AI enhancements
GEMINI_API_KEY=your-gemini-api-key
TELEGRAM_BOT_TOKEN=your-telegram-bot-token
TELEGRAM_CHAT_ID=your-telegram-chat-id
```

### 3. Start Backend (Models Auto-Download)

```bash
cd backend
python main.py
```

**First run:** YOLOv8 and DeepFace models will auto-download (~650MB)

### 4. Start Frontend

```bash
npm install
npm run dev
```

**Access:** http://localhost:5173

---

## 📖 Full Documentation

- **[DEEP_LEARNING_SETUP.md](DEEP_LEARNING_SETUP.md)** - Complete setup guide
- **[PRODUCTION_CERTIFICATION_REPORT.md](PRODUCTION_CERTIFICATION_REPORT.md)** - QA audit report
- **[TESTING_HISTORY.md](TESTING_HISTORY.md)** - Testing records
