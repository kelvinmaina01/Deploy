"""
TuneKit API
===========
FastAPI backend for the TuneKit pipeline.
"""

import os
import sys
import shutil
import uuid
import json
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tunekit import (
    TuneKitState,
    ingest_data,
    validate_quality,
    analyze_dataset,
    generate_package,
    recommend_model,
)
from tunekit.training import (
    start_training,
    get_training_status,
    get_model_download_url,
    compare_models,
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
# In production, set ALLOWED_ORIGINS environment variable
allowed_origins_str = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000,http://127.0.0.1:8000")
allowed_origins = [origin.strip() for origin in allowed_origins_str.split(",")]
app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

# Store sessions in memory (use Redis in production)
sessions: dict[str, dict] = {}

# Upload directory
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Serve static files (frontend)
FRONTEND_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "frontend")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


# ============================================================================
# MODELS
# ============================================================================

class AnalyzeRequest(BaseModel):
    session_id: str
    user_description: str


class RecommendRequest(BaseModel):
    session_id: str
    user_task: str  # classify, qa, conversation, generation, extraction
    deployment_target: str = 'not_sure'  # cloud_api, mobile_app, edge_device, web_browser, desktop_app, not_sure


class PlanRequest(BaseModel):
    session_id: str
    user_task: Optional[str] = None  # classify, qa, conversation, generation, extraction
    deployment_target: Optional[str] = None  # cloud_api, mobile_app, etc.


class GenerateRequest(BaseModel):
    session_id: str


class SessionResponse(BaseModel):
    session_id: str
    status: str
    message: str
    rows: Optional[int] = 0
    stats: Optional[dict] = None


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
    sample_rows: Optional[list[dict]] = None  # First 5 rows for preview


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


class TrainRequest(BaseModel):
    session_id: str


class TrainResponse(BaseModel):
    session_id: str
    job_id: str
    status: str
    message: str


class TrainingStatusResponse(BaseModel):
    job_id: str
    status: str
    metrics: Optional[dict] = None
    base_metrics: Optional[dict] = None
    finetuned_metrics: Optional[dict] = None
    improvement: Optional[dict] = None
    base_model: Optional[str] = None
    task_type: Optional[str] = None
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    message: Optional[str] = None


class CompareRequest(BaseModel):
    job_id: str
    text: str


class CompareResponse(BaseModel):
    base: Optional[dict] = None
    finetuned: Optional[dict] = None
    error: Optional[str] = None


class ModelDownloadResponse(BaseModel):
    job_id: str
    status: str
    files: Optional[list] = None
    download_endpoint: Optional[str] = None
    error: Optional[str] = None
    message: Optional[str] = None


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

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the landing page."""
    landing_path = os.path.join(FRONTEND_DIR, "landing.html")
    with open(landing_path, "r") as f:
        return f.read()


@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard():
    """Serve the main dashboard/app."""
    index_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(index_path, "r") as f:
        return f.read()


@app.get("/favicon.ico")
async def favicon():
    """Serve the favicon."""
    favicon_path = os.path.join(FRONTEND_DIR, "favicon.svg")
    if os.path.exists(favicon_path):
        return FileResponse(favicon_path, media_type="image/svg+xml")
    logo_path = os.path.join(FRONTEND_DIR, "logo.png")
    if os.path.exists(logo_path):
        return FileResponse(logo_path, media_type="image/png")
    raise HTTPException(status_code=404)


@app.get("/health")
async def health():
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
    
    # Security: Sanitize filename to prevent path traversal
    from pathlib import Path
    filename = os.path.basename(file.filename)  # Remove any path components
    # Only allow alphanumeric, dots, dashes, underscores
    filename = "".join(c for c in filename if c.isalnum() or c in "._-")
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    ext = os.path.splitext(filename)[1].lower()
    if ext != ".jsonl":
        raise HTTPException(status_code=400, detail="Only JSONL format is supported. Please upload a .jsonl file.")
    
    # File size limit (100MB)
    MAX_FILE_SIZE = 100 * 1024 * 1024
    
    # Create session
    session_id = str(uuid.uuid4())[:8]
    
    # Save file
    file_path = os.path.join(UPLOAD_DIR, f"{session_id}_{filename}")
    
    # Ensure path is within UPLOAD_DIR (prevent path traversal)
    resolved_path = Path(file_path).resolve()
    resolved_upload_dir = Path(UPLOAD_DIR).resolve()
    try:
        resolved_path.relative_to(resolved_upload_dir)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid file path")
    
    # Read and validate size
    content = await file.read()
    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(status_code=413, detail=f"File too large. Maximum size: {MAX_FILE_SIZE // (1024*1024)}MB")
    
    with open(file_path, "wb") as f:
        f.write(content)
    
    # Initialize session
    sessions[session_id] = {
        "file_path": file_path,
        "filename": file.filename,
        "original_filename": file.filename,  # Store original for download
        "state": None,
        "created_at": datetime.now().isoformat(),
    }
    
    # Run initial ingestion to validate format and get stats
    try:
        state = create_empty_state(file_path, "")
        result = ingest_data(state)
        state.update(result)
        
        print(f"[Upload] Ingest result keys: {result.keys() if isinstance(result, dict) else 'not a dict'}")
        print(f"[Upload] State num_rows: {state.get('num_rows')}")
        print(f"[Upload] State stats: {state.get('stats')}")
        
        if state.get("error_msg"):
            # Clean up file on error
            if os.path.exists(file_path):
                os.remove(file_path)
            del sessions[session_id]
            raise HTTPException(status_code=400, detail=state["error_msg"])
        
        # Store state with ingestion results
        sessions[session_id]["state"] = state
        
        # Get number of rows/examples - check both num_rows and stats
        num_rows = state.get("num_rows")
        stats = state.get("stats", {})
        
        print(f"[Upload] After state update - num_rows: {num_rows}, type: {type(num_rows)}")
        print(f"[Upload] Stats total_examples: {stats.get('total_examples') if stats else 'no stats'}")
        
        # If num_rows is None or 0, try to get from stats
        if not num_rows or num_rows == 0:
            if stats and stats.get("total_examples"):
                num_rows = stats.get("total_examples")
                print(f"[Upload] Using total_examples from stats: {num_rows}")
        
        # Ensure it's an integer
        num_rows = int(num_rows) if num_rows else 0
        
        print(f"[Upload] Final num_rows: {num_rows} for session {session_id}")
        
        return {
            "session_id": session_id,
            "status": "uploaded",
            "message": f"File '{file.filename}' uploaded successfully",
            "rows": num_rows,
            "stats": stats,
        }
    except HTTPException:
        raise
    except Exception as e:
        # Clean up on error
        if os.path.exists(file_path):
            os.remove(file_path)
        if session_id in sessions:
            del sessions[session_id]
        raise HTTPException(status_code=400, detail=f"Failed to parse file: {str(e)}")


@app.post("/recommend")
async def recommend(request: RecommendRequest):
    """
    Analyze dataset and recommend the best model based on user task and deployment target.
    
    Flow: Upload → Recommend (validate + analyze + recommend)
    """
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        raise HTTPException(status_code=400, detail="No data found. Please re-upload your file.")
    
    # Step 1: Validate quality (if not already done)
    if state.get("quality_score") is None:
        result = validate_quality(state)
        state.update(result)
    
    # Step 2: Analyze dataset (if not already done)
    if state.get("conversation_characteristics") is None:
        result = analyze_dataset(state)
        state.update(result)
    
    # Step 3: Get model recommendation
    conversation_chars = state.get("conversation_characteristics", {})
    num_examples = state.get("num_rows", 0)
    
    recommendation = recommend_model(
        user_task=request.user_task,
        conversation_characteristics=conversation_chars,
        num_examples=num_examples,
        deployment_target=request.deployment_target
    )
    
    # Store user selections in state
    state["user_task"] = request.user_task
    state["deployment_target"] = request.deployment_target
    state["recommendation"] = recommendation
    
    # Save updated state
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "recommendation": recommendation,
        "analysis": {
            "stats": state.get("stats", {}),
            "quality_score": state.get("quality_score", 0),
            "quality_issues": state.get("quality_issues", []),
            "conversation_characteristics": conversation_chars,
        }
    }


class AddSystemPromptRequest(BaseModel):
    session_id: str
    system_prompt: str


@app.post("/add-system-prompt")
async def add_system_prompt(request: AddSystemPromptRequest):
    """
    Add a system prompt to all conversations in the dataset.
    """
    session_id = request.session_id
    system_prompt = request.system_prompt.strip()
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if not system_prompt:
        raise HTTPException(status_code=400, detail="System prompt cannot be empty")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state or not state.get("raw_data"):
        raise HTTPException(status_code=400, detail="No data found. Please re-upload your file.")
    
    raw_data = state["raw_data"]
    modified_count = 0
    
    # Add system prompt to each conversation
    for entry in raw_data:
        messages = entry.get("messages", [])
        
        # Check if already has a system prompt
        has_system = any(msg.get("role") == "system" for msg in messages)
        
        if not has_system:
            # Insert system prompt at the beginning
            messages.insert(0, {
                "role": "system",
                "content": system_prompt
            })
            modified_count += 1
    
    # Save modified data back to file
    file_path = session["file_path"]
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            for entry in raw_data:
                f.write(json.dumps(entry) + '\n')
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save modified data: {str(e)}")
    
    # Re-run ingestion to update stats
    state["file_path"] = file_path
    result = ingest_data(state)
    state.update(result)
    
    if state.get("error_msg"):
        raise HTTPException(status_code=400, detail=state["error_msg"])
    
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "status": "success",
        "message": f"Added system prompt to {modified_count} conversations",
        "modified_count": modified_count,
        "stats": state.get("stats", {})
    }


@app.get("/download-dataset/{session_id}")
async def download_dataset(session_id: str):
    """
    Download the dataset file (with any modifications like system prompts).
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    file_path = session.get("file_path")
    
    if not file_path or not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    # Get original filename or generate one
    original_filename = session.get("original_filename", "dataset.jsonl")
    if not original_filename.endswith('.jsonl'):
        original_filename = original_filename.rsplit('.', 1)[0] + '.jsonl'
    
    # Add suffix if modified
    if session.get("state", {}).get("stats", {}).get("has_system_prompts"):
        name, ext = os.path.splitext(original_filename)
        original_filename = f"{name}_with_system_prompt{ext}"
    
    return FileResponse(
        path=file_path,
        filename=original_filename,
        media_type='application/jsonl'
    )


@app.post("/analyze")
async def analyze(request: AnalyzeRequest):
    """
    Legacy endpoint - Run ingestion, validation, and analysis on the uploaded file.
    """
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found. Upload a file first.")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        file_path = session["file_path"]
    state = create_empty_state(file_path, request.user_description)
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
    
    return {
        "session_id": session_id,
        "num_rows": state.get("num_rows", 0),
        "quality_score": state.get("quality_score", 0),
        "quality_issues": state.get("quality_issues", []),
        "conversation_characteristics": state.get("conversation_characteristics", {}),
        "stats": state.get("stats", {}),
    }


@app.get("/sample-data/{session_id}")
async def get_sample_data(session_id: str, limit: int = 100):
    """
    Get sample rows from the uploaded dataset.
    
    Args:
        session_id: Session identifier
        limit: Maximum number of rows to return (default: 100)
    
    Returns:
        List of sample rows
    """
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    # If we have state with raw_data, use it
    if session.get("state") and session["state"].get("raw_data"):
        raw_data = session["state"]["raw_data"]
        return {
            "rows": raw_data[:limit],
            "total_rows": len(raw_data),
            "returned_rows": min(limit, len(raw_data))
        }
    
    # Otherwise, try to read from file
    file_path = session["file_path"]
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")
    
    try:
        from tunekit.tools.ingest import ingest_data
        from tunekit.state import create_empty_state
        
        # Quick ingest to get data
        state = create_empty_state(file_path, "")
        result = ingest_data(state)
        state.update(result)
        
        if state.get("raw_data"):
            raw_data = state["raw_data"]
            return {
                "rows": raw_data[:limit],
                "total_rows": len(raw_data),
                "returned_rows": min(limit, len(raw_data))
            }
        else:
            raise HTTPException(status_code=500, detail="Could not read file data")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")


@app.post("/plan")
async def plan(request: PlanRequest):
    """
    Create training plan based on recommendation.
    Uses the model from the recommendation step.
    """
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state:
        raise HTTPException(status_code=400, detail="Run /recommend first")
    
    # Get recommendation from state
    recommendation = state.get("recommendation", {})
    primary_rec = recommendation.get("primary_recommendation", {})
    
    if not primary_rec:
        raise HTTPException(status_code=400, detail="No recommendation found. Run /recommend first.")
    
    # Apply user task and deployment target if provided
    user_task = request.user_task or state.get("user_task", "conversation")
    deployment_target = request.deployment_target or state.get("deployment_target", "not_sure")
    
    # Set up training configuration based on recommendation
    model_id = primary_rec.get("model_id", "microsoft/Phi-4-mini-instruct")
    
    state["final_task_type"] = user_task
    state["base_model"] = model_id
    state["planning_reasoning"] = f"Based on your {user_task} task and {deployment_target} deployment target, we recommend {primary_rec.get('model_name', 'Phi-4 Mini')}."
    
    # Create training config for chat-based fine-tuning with LoRA
    state["training_config"] = {
        "model_name": model_id,
        "task_type": "chat",
        "training_args": {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 4,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
            "fp16": True,
            "logging_steps": 10,
            "save_strategy": "epoch",
            "warmup_ratio": 0.1,
        },
        "lora_config": {
            "r": 16,
            "lora_alpha": 32,
            "lora_dropout": 0.05,
            "target_modules": ["q_proj", "v_proj", "k_proj", "o_proj"],
        },
    }
    
    # Save updated state
    sessions[session_id]["state"] = state
    
    return {
        "session_id": session_id,
        "final_task_type": state["final_task_type"],
        "base_model": state["base_model"],
        "reasoning": state["planning_reasoning"],
        "training_config": state["training_config"],
    }


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
    if state.get("job_status") == "completed":
        return "trained"
    if state.get("job_id"):
        return "training"
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
# TRAINING ENDPOINTS (Modal)
# ============================================================================

@app.post("/train", response_model=TrainResponse)
async def train(request: TrainRequest):
    """
    Start training on Modal cloud GPUs.
    
    Requires: /plan to be completed first.
    """
    session_id = request.session_id
    
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    state = session.get("state")
    
    if not state or not state.get("final_task_type"):
        raise HTTPException(status_code=400, detail="Run /plan first to configure training")
    
    # Check if already training
    if state.get("job_id") and state.get("job_status") == "running":
        return TrainResponse(
            session_id=session_id,
            job_id=state["job_id"],
            status="already_running",
            message="Training is already in progress",
        )
    
    # Start training on Modal
    result = start_training(
        task_type=state["final_task_type"],
        config=state["training_config"],
        data_path=state["file_path"],
    )
    
    if result["status"] == "failed":
        raise HTTPException(status_code=500, detail=result.get("error", "Training failed to start"))
    
    # Update state with job info
    state["job_id"] = result["job_id"]
    state["job_status"] = "running"
    sessions[session_id]["state"] = state
    
    return TrainResponse(
        session_id=session_id,
        job_id=result["job_id"],
        status="running",
        message=result.get("message", "Training started on Modal GPU"),
    )


@app.get("/training-status/{job_id}", response_model=TrainingStatusResponse)
async def training_status(job_id: str):
    """
    Check the status of a training job.
    """
    result = get_training_status(job_id)
    
    # Update session state if we have the job
    for session in sessions.values():
        state = session.get("state")
        if state and state.get("job_id") == job_id:
            state["job_status"] = result["status"]
            if result.get("metrics"):
                state["metrics"] = result["metrics"]
            break
    
    return TrainingStatusResponse(
        job_id=job_id,
        status=result["status"],
        metrics=result.get("metrics"),
        base_metrics=result.get("base_metrics"),
        finetuned_metrics=result.get("finetuned_metrics"),
        improvement=result.get("improvement"),
        base_model=result.get("base_model"),
        task_type=result.get("task_type"),
        error=result.get("error"),
        started_at=result.get("started_at"),
        completed_at=result.get("completed_at"),
        message=result.get("message"),
    )


@app.get("/download-model/{job_id}", response_model=ModelDownloadResponse)
async def download_model_info(job_id: str):
    """
    Get download information for a trained model.
    """
    result = get_model_download_url(job_id)
    
    return ModelDownloadResponse(
        job_id=job_id,
        status=result["status"],
        files=result.get("files"),
        download_endpoint=result.get("download_endpoint"),
        error=result.get("error"),
        message=result.get("message"),
    )


@app.get("/download-model/{job_id}/{filename:path}")
async def download_model_file(job_id: str, filename: str):
    """
    Download a specific file from the trained model.
    """
    from tunekit.training.modal_service import download_model_file_content
    from pathlib import Path
    
    # Security: Sanitize filename to prevent path traversal
    filename = os.path.basename(filename)  # Remove any path components
    # Only allow safe characters
    if not all(c.isalnum() or c in "._-/" for c in filename):
        raise HTTPException(status_code=400, detail="Invalid filename")
    
    content = download_model_file_content(job_id, filename)
    
    if content is None:
        raise HTTPException(status_code=404, detail=f"File not found: {filename}")
    
    # Determine media type
    if filename.endswith(".json"):
        media_type = "application/json"
    elif filename.endswith(".bin") or filename.endswith(".safetensors"):
        media_type = "application/octet-stream"
    else:
        media_type = "application/octet-stream"
    
    from fastapi.responses import Response
    return Response(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


@app.get("/download-model-zip/{job_id}")
async def download_model_zip(job_id: str):
    """
    Download all model files as a ZIP archive.
    Simple approach: Get file list from Modal, create ZIP here.
    """
    import io
    import zipfile
    
    print(f"[API] Download ZIP requested for job: {job_id}")
    
    try:
        # Get file list and content from Modal
        result = get_model_download_url(job_id)
        
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        
        files = result.get("files", [])
        if not files:
            raise HTTPException(status_code=404, detail="No model files found")
        
        print(f"[API] Found {len(files)} files, creating ZIP...")
        
        # Create ZIP in memory
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zf:
            for file_info in files:
                filename = file_info.get("name") or file_info.get("filename")  # Support both formats
                if not filename:
                    print(f"[API] Skipping file with no name: {file_info}")
                    continue
                
                # Skip checkpoint directories - users only need the final model
                if filename.startswith("checkpoint-"):
                    print(f"[API] Skipping checkpoint file: {filename}")
                    continue
                    
                # Download each file content
                from tunekit.training.modal_service import download_model_file_content
                content = download_model_file_content(job_id, filename)
                if content:
                    zf.writestr(filename, content)
                    print(f"[API] Added: {filename} ({len(content)} bytes)")
                else:
                    print(f"[API] Warning: Failed to download {filename}")
        
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.read()
        print(f"[API] ZIP created: {len(zip_bytes)} bytes")
        
        from fastapi.responses import Response
        return Response(
            content=zip_bytes,
            media_type="application/zip",
            headers={
                "Content-Disposition": f"attachment; filename=tunekit_model_{job_id}.zip",
                "Content-Length": str(len(zip_bytes)),
            }
        )
        
    except HTTPException:
        raise
    except KeyError as e:
        print(f"[API] Missing key in file info: {e}")
        print(f"[API] Files structure: {files[:2] if files else 'No files'}")
        raise HTTPException(
            status_code=500,
            detail=f"Download failed: Invalid file structure - {str(e)}"
        )
    except Exception as e:
        import traceback
        print(f"[API] Error downloading ZIP: {e}")
        print(f"[API] Traceback: {traceback.format_exc()}")
        raise HTTPException(
            status_code=500,
            detail=f"Download failed: {str(e)}"
        )


# ============================================================================
# MODEL COMPARISON
# ============================================================================

@app.post("/compare", response_model=CompareResponse)
async def compare_models_endpoint(request: CompareRequest):
    """
    Compare base model vs fine-tuned model predictions on custom input.
    """
    result = compare_models(request.job_id, request.text)
    
    if "error" in result:
        return CompareResponse(error=result["error"])
    
    return CompareResponse(
        base=result.get("base"),
        finetuned=result.get("finetuned"),
    )


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

