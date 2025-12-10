"""
Modal Service Wrapper
=====================
High-level interface for managing training jobs on Modal.
"""

import os
import uuid
from datetime import datetime
from typing import Optional
import modal

# In-memory job tracking (use Redis/DB in production)
_jobs: dict[str, dict] = {}


def start_training(
    task_type: str,
    config: dict,
    data_path: str,
) -> dict:
    """
    Start a training job on Modal.
    
    Args:
        task_type: "classification", "ner", or "instruction_tuning"
        config: Training configuration dict
        data_path: Path to the data file
    
    Returns:
        dict with job_id and status
    """
    # Generate unique job ID
    job_id = f"tk_{uuid.uuid4().hex[:12]}"
    
    # Read data file content
    with open(data_path, "r") as f:
        data_content = f.read()
    
    # Get the Modal app
    from tunekit.training.modal_train import (
        app,
        train_classification,
        train_ner,
        train_instruction,
    )
    
    # Select training function based on task type
    if task_type == "classification":
        train_fn = train_classification
    elif task_type == "ner":
        train_fn = train_ner
    elif task_type == "instruction_tuning":
        train_fn = train_instruction
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    # Spawn the training job asynchronously
    try:
        # Use Modal's spawn to run async
        function_call = train_fn.spawn(config, data_content, job_id)
        
        # Store job info
        _jobs[job_id] = {
            "job_id": job_id,
            "task_type": task_type,
            "config": config,
            "status": "running",
            "started_at": datetime.now().isoformat(),
            "modal_call_id": function_call.object_id,
            "function_call": function_call,
            "metrics": None,
            "error": None,
            "completed_at": None,
        }
        
        print(f"[TuneKit] Training job {job_id} started on Modal")
        
        return {
            "job_id": job_id,
            "status": "running",
            "message": f"Training started for {task_type} task",
        }
        
    except Exception as e:
        error_msg = str(e)
        _jobs[job_id] = {
            "job_id": job_id,
            "task_type": task_type,
            "status": "failed",
            "started_at": datetime.now().isoformat(),
            "error": error_msg,
        }
        
        return {
            "job_id": job_id,
            "status": "failed",
            "error": error_msg,
        }


def get_training_status(job_id: str) -> dict:
    """
    Get the status of a training job.
    
    Args:
        job_id: The job identifier
    
    Returns:
        dict with job status, metrics (if complete), and error (if failed)
    """
    if job_id not in _jobs:
        return {
            "job_id": job_id,
            "status": "not_found",
            "error": "Job not found",
        }
    
    job = _jobs[job_id]
    
    # If already completed or failed, return cached result
    if job["status"] in ["completed", "failed"]:
        return {
            "job_id": job_id,
            "status": job["status"],
            "metrics": job.get("metrics"),
            "error": job.get("error"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }
    
    # Check Modal for current status
    function_call = job.get("function_call")
    
    if function_call is None:
        return {
            "job_id": job_id,
            "status": job["status"],
            "error": "No Modal function call found",
        }
    
    try:
        # Try to get the result (non-blocking check)
        # Modal's FunctionCall has a .get() method
        result = function_call.get(timeout=0)
        
        # If we got here, the job is complete
        job["status"] = result.get("status", "completed")
        job["metrics"] = result.get("metrics")
        job["error"] = result.get("error")
        job["completed_at"] = datetime.now().isoformat()
        
        return {
            "job_id": job_id,
            "status": job["status"],
            "metrics": job.get("metrics"),
            "error": job.get("error"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }
        
    except modal.exception.FunctionTimeoutError:
        # Job is still running
        return {
            "job_id": job_id,
            "status": "running",
            "started_at": job.get("started_at"),
            "message": "Training in progress...",
        }
        
    except Exception as e:
        # Something went wrong
        job["status"] = "failed"
        job["error"] = str(e)
        job["completed_at"] = datetime.now().isoformat()
        
        return {
            "job_id": job_id,
            "status": "failed",
            "error": str(e),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }


def get_model_download_url(job_id: str) -> dict:
    """
    Get the download information for a trained model.
    
    Args:
        job_id: The job identifier
    
    Returns:
        dict with download URLs or error
    """
    if job_id not in _jobs:
        return {
            "job_id": job_id,
            "status": "not_found",
            "error": "Job not found",
        }
    
    job = _jobs[job_id]
    
    if job["status"] != "completed":
        return {
            "job_id": job_id,
            "status": job["status"],
            "error": "Model not ready. Training must be completed first.",
        }
    
    # Get model files from Modal volume
    try:
        from tunekit.training.modal_train import get_model_files
        
        result = get_model_files.remote(job_id)
        
        if not result.get("exists"):
            return {
                "job_id": job_id,
                "status": "error",
                "error": "Model files not found on Modal volume",
            }
        
        return {
            "job_id": job_id,
            "status": "ready",
            "files": result.get("files", []),
            "download_endpoint": f"/download-model/{job_id}",
            "message": "Model ready for download",
        }
        
    except Exception as e:
        return {
            "job_id": job_id,
            "status": "error",
            "error": f"Failed to get model info: {str(e)}",
        }


def download_model_file_content(job_id: str, filename: str) -> Optional[bytes]:
    """
    Download a specific model file.
    
    Args:
        job_id: The job identifier
        filename: The file to download
    
    Returns:
        File content as bytes, or None if not found
    """
    try:
        from tunekit.training.modal_train import download_model_file
        
        content = download_model_file.remote(job_id, filename)
        return content
        
    except Exception as e:
        print(f"[TuneKit] Error downloading file: {e}")
        return None


def list_jobs() -> list[dict]:
    """
    List all training jobs.
    
    Returns:
        List of job summaries
    """
    return [
        {
            "job_id": job["job_id"],
            "task_type": job["task_type"],
            "status": job["status"],
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }
        for job in _jobs.values()
    ]

