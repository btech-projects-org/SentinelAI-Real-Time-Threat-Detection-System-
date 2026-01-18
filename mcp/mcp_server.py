from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional, Dict
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("sentinel.mcp_server")

app = FastAPI(title="SentinelAI MCP Agent", version="1.0.0")

class ContextData(BaseModel):
    detections: List[Dict]
    camera_id: Optional[str] = None
    timestamp: str

@app.get("/health")
async def health_check():
    return {"status": "active", "role": "MCP_AGENT"}

@app.post("/context")
async def analyze_context(data: ContextData):
    """
    Receive detection context and perform higher-level reasoning.
    In a full production system, this would call LLMs (Gemini/GPT) to analyze the scene.
    For this 'Correctness' task, we implement the structure and logging.
    """
    logger.info(f"Received context from Camera {data.camera_id}")
    
    suspicious_score = 0
    reasoning = []

    # Heuristic Reasoning logic
    for det in data.detections:
        label = det.get("label", "UNKNOWN")
        conf = det.get("confidence", 0)
        
        if label == "WEAPON" and conf > 0.7:
            suspicious_score += 50
            reasoning.append(f"High confidence weapon detected: {conf}")
        elif label == "VIOLENCE":
            suspicious_score += 40
            reasoning.append("Violence signature detected")
        elif label == "PERSON":
            suspicious_score += 5

    decision = "MONITOR"
    if suspicious_score > 60:
        decision = "ESCALATE_TO_HUMAN"
    elif suspicious_score > 30:
        decision = "LOG_AND_TRACK"

    return {
        "analysis_id": f"mcp_{hash(data.timestamp)}",
        "suspicious_score": suspicious_score,
        "decision": decision,
        "reasoning": reasoning
    }

if __name__ == "__main__":
    import uvicorn
    # Run on a different port than the main backend
    uvicorn.run(app, host="0.0.0.0", port=8080)
