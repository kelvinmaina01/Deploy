"""
TuneKit API
===========
FastAPI backend for the TuneKit pipeline.
"""

import os
import sys
import shutil
import uuid
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tunekit import (
    TuneKitState,
    ingest_data,
    validate_quality,
    analyze_dataset,
    planning_agent,
    generate_package,
)

# ============================================================================
# APP SETUP
# ============================================================================

app = FastAPI(
    title="TuneKit API",
    description="Automated LLM Fine-Tuning Pipeline",
    version="0.1.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Store sessions in memory (use Redis in production)
sessions: dict[str, dict] = {}

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ============================================================================
# MODELS
# ============================================================================

class AnalyzeRequest(BaseModel):
    session_id: str
    user_description: str


class PlanRequest(BaseModel):
    session_id: str


class GenerateRequest(BaseModel):
    session_id: str


class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str


class AnalyzeResponse(BaseModel):
    session_id: str
    num_rows: int
    columns: list[str]
    quality_score: float
    quality_issues: list[str]
    inferred_task_type: str
    task_confidence: float
    num_classes: Optional[int]
    column_candidates: dict


class PlanResponse(BaseModel):
    session_id: str
    final_task_type: str
    base_model: str
    reasoning: str
    training_config: dict


class GenerateResponse(BaseModel):
    session_id: str
    package_path: str
    download_url: str


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def create_empty_state(file_path: str, user_description: str = "") -> TuneKitState:
    """Create an empty TuneKitState."""
    return {
        "file_path": file_path,
        "user_description": user_description,
        "raw_data": None, "columns": None, "sample_rows": None, "num_rows": None,
        "quality_score": None, "quality_issues": None,
        "inferred_task_type": None, "task_confidence": None, "num_classes": None,
        "label_list": None, "column_analysis": None, "column_candidates": None, "dataset_stats": None,
        "final_task_type": None, "base_model": None, "training_config": None, "planning_reasoning": None,
        "est_cost": None, "est_time_minutes": None, "user_approved": None,
        "job_id": None, "job_status": None, "current_epoch": None, "metrics": None,
        "error_msg": None, "retry_count": None,
        "final_report": None, "next_steps": None, "model_download_url": None,
    }


# ============================================================================
# ENDPOINTS
# ============================================================================

@app.get("/")
async def root():
    """Health check."""
    return {"status": "ok", "message": "TuneKit API is running"}


@app.post("/upload", response_model=SessionResponse)
async def upload_file(file: UploadFile = File(...)):
    """
    Upload a dataset file (CSV, JSON, JSONL).
    Returns a session_id for subsequent requests.
    """
    # Validate file type
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided")
    
    ext = os.path.splitext(file.filename)[1].lower()
    if ext not in [".csv", ".json", ".jsonl"]:
        raise HTTPException(status_code=400, detail="Unsupported file type. Use .csv, .json, or .jsonl")
    
    # Create session
    session_id = str(uuid.uuid4())[:8]
    
    # Save file
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{file.filename}")
    with open(file_path, "wb") as f:
        content = await file.read()
        f.write(content)
    
    # Initialize session
    sessions[session_id] = {
        "file_path": file_path,
        "filename": file.filename,
        "state": None,
        "created_at": datetime.now().isoformat(),
    }
    
    return SessionResponse(
        session_id=session_id,
        status="uploaded",
        message=f"File '{file.filename}' uploaded successfully",
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze(request: AnalyzeRequest):
    """
    Run ingestion, validation, and analysis on the uploaded file.
    """
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")
    
    session = sessions[session_id]
    file_path = session["file_path"]
    
    # Create state
    state = create_empty_state(file_path, request.user_description)
    
    # Step 1: Ingest
    result = ingest_data(state)
    state.update(result)
    
    if state.get("error_msg"):
        raise HTTPException(status_code=400, detail=state["error_msg"])
    
    # Step 2: Validate
    result = validate_quality(state)
    state.update(result)
    
    # Step 3: Analyze
    result = analyze_dataset(state)
    state.update(result)
    
    # Save state to session
    sessions[session_id]["state"] = state
    
    return AnalyzeResponse(
        session_id=session_id,
        num_rows=state["num_rows"],
        columns=state["columns"],
        quality_score=state["quality_score"],
        quality_issues=state["quality_issues"],
        inferred_task_type=state["inferred_task_type"],
        task_confidence=state["task_confidence"],
        num_classes=state.get("num_classes"),
        column_candidates=state["column_candidates"],
    )


@app.post("/plan", response_model=PlanResponse)
async def plan(request: PlanRequest):
    """
    Run the planning agent to decide task type, model, and config.
    """
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        raise HTTPException(status_code=400, detail="Run /analyze first")
    
    # Run planning agent
    result = planning_agent(state)
    state.update(result)
    
    # Save updated state
    sessions[session_id]["state"] = state
    
    return PlanResponse(
        session_id=session_id,
        final_task_type=state["final_task_type"],
        base_model=state["base_model"],
        reasoning=state["planning_reasoning"],
        training_config=state["training_config"],
    )


@app.post("/generate", response_model=GenerateResponse)
async def generate(request: GenerateRequest):
    """
    Generate the training package (5 files).
    """
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state or not state.get("final_task_type"):
        raise HTTPException(status_code=400, detail="Run /plan first")
    
    # Generate package
    result = generate_package(state)
    state.update(result)
    
    # Save updated state
    sessions[session_id]["state"] = state
    
    package_path = state["package_path"]
    
    return GenerateResponse(
        session_id=session_id,
        package_path=package_path,
        download_url=f"/download/{session_id}",
    )


@app.get("/download/{session_id}")
async def download(session_id: str):
    """
    Download the generated training package as a ZIP file.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state or not state.get("package_path"):
        raise HTTPException(status_code=400, detail="Package not generated yet")
    
    package_path = state["package_path"]
    
    if not os.path.exists(package_path):
        raise HTTPException(status_code=404, detail="Package folder not found")
    
    # Create ZIP file
    zip_path = f"{package_path}.zip"
    if not os.path.exists(zip_path):
        shutil.make_archive(package_path, "zip", package_path)
    
    return FileResponse(
        zip_path,
        media_type="application/zip",
        filename=f"tunekit_package_{session_id}.zip",
    )


@app.get("/session/{session_id}")
async def get_session(session_id: str):
    """
    Get the current state of a session.
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state", {})
    
    return {
        "session_id": session_id,
        "filename": session.get("filename"),
        "created_at": session.get("created_at"),
        "has_state": state is not None,
        "current_step": _get_current_step(state),
    }


def _get_current_step(state: dict) -> str:
    """Determine which step the session is at."""
    if not state:
        return "uploaded"
    if state.get("package_path"):
        return "generated"
    if state.get("final_task_type"):
        return "planned"
    if state.get("inferred_task_type"):
        return "analyzed"
    if state.get("raw_data"):
        return "ingested"
    return "uploaded"


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

