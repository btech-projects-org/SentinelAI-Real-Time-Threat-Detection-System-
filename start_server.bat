@echo off
echo Starting SentinelAI Backend...
echo Ensure MongoDB is running or MONGO_URI is set.
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
