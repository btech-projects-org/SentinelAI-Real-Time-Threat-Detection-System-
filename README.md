# 🛡️ SentinelAI – Threat Detection System

**A Smart Security Camera System that detects Weapons, Fighting, and Criminals using AI.**

---

## 🚀 Quick Start Guide (For Students)

Follow these simple steps to run the project on your Windows laptop.

### Step 1: Install Requirements
Before starting, make sure you have these two programs installed:

1.  **Python (v3.10 or newer)**
    *   [Download Here](https://www.python.org/downloads/)
    *   **IMPORTANT:** During installation, check the box that says **"Add Python to PATH"**.
2.  **MongoDB Community Server**
    *   [Download Here](https://www.mongodb.com/try/download/community)
    *   Install with default settings (Next > Next > Install).

### Step 2: Configure the Secret Keys
The system needs a password file to work secure.
1.  Copy the file named `.env.example` and rename it to `.env`.
2.  (Optional) If you want real Telegram alerts, open `.env` with Notepad and add your Bot Token.
    *   *For a demo, you can leave the default values.*

### Step 3: Run the System
We made it easy! You don't need to type commands.

1.  Find the file **`start_server.bat`** in the main folder.
2.  **Double-click** it.
3.  A black window will open. It will automatically:
    *   Install all necessary AI libraries.
    *   Download the brain of the AI (Models).
    *   Start the server.
    *   *First time setup may take 5-10 minutes. Please be patient!*

### Step 4: Access the Dashboard
Once the black window says `Application startup complete`:
1.  Open Chrome or Edge.
2.  Go to: **[http://localhost:5173](http://localhost:5173)**
3.  Login with these details:
    *   **Username:** `admin`
    *   **Password:** `SentinelAdmin2026!`

---

## 🎮 How to Test It

### 1. Weapon Detection (YOLOv8)
*   Hold up a pair of **scissors**, a **baseball bat**, or a toy **gun** to your webcam.
*   The system will draw a red box around it and flash "WEAPON DETECTED".

### 2. Criminal Detection (DeepFace)
*   Go to the "Criminal Database" tab.
*   Upload a photo of yourself (or a friend) and give it a name (e.g., "John Doe") and Threat Level "HIGH".
*   Go back to "Live Feed".
*   When that person appears on camera, the AI will recognize them and trigger an alert!

### 3. Fighting Detection
*   If two people get very close and make aggressive movements (overlapping boxes), the system detects "VIOLENCE".

---

## ❓ Troubleshooting

**Q: The black window closes immediately!**
A: You probably don't have Python installed correctly. Re-install Python and **make sure to check "Add to PATH"**.

**Q: "MongoTimeoutError" or Database error?**
A: Make sure MongoDB is running. Open "Task Manager" -> "Services" and check if `MongoDB` is Running.

**Q: White screen on localhost:5173?**
A: Make sure the black window (Server) is still open. Do not close it!

---

## 👨‍� For Developers (Technical Details)
*   **Backend:** FastAPI (Python)
*   **AI Models:** YOLOv8n (Object Detection), Facenet512 (Face Recognition)
*   **Frontend:** React + Vite
*   **Security:** JWT Authentication, ISO 27001 Compliance
*   **Report:** See `PRODUCTION_CERTIFICATION_REPORT_V2.md` for security audit details.
