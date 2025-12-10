"""
Modal Service Wrapper
=====================
High-level interface for managing training jobs on Modal.
"""

import uuid
from datetime import datetime
from typing import Optional
import modal

# In-memory job tracking (use Redis/DB in production)
_jobs: dict[str, dict] = {}

# Modal app name (must match modal_train.py)
MODAL_APP_NAME = "tunekit-training"


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
    
    # Map task type to function name
    function_map = {
        "classification": "train_classification",
        "ner": "train_ner",
        "instruction_tuning": "train_instruction",
    }
    
    if task_type not in function_map:
        return {
            "job_id": job_id,
            "status": "failed",
            "error": f"Unknown task type: {task_type}",
        }
    
    function_name = function_map[task_type]
    
    # Spawn the training job asynchronously
    try:
        # Look up the deployed function from Modal
        train_fn = modal.Function.from_name(MODAL_APP_NAME, function_name)
        
        # Spawn async execution
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
        
    except modal.exception.NotFoundError:
        error_msg = (
            f"Modal app '{MODAL_APP_NAME}' not found. "
            "Please deploy first: modal deploy tunekit/training/modal_train.py"
        )
        return {
            "job_id": job_id,
            "status": "failed",
            "error": error_msg,
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
            "base_metrics": job.get("base_metrics"),
            "finetuned_metrics": job.get("finetuned_metrics"),
            "improvement": job.get("improvement"),
            "base_model": job.get("base_model"),
            "task_type": job.get("task_type"),
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
        # Try to get the result with a small timeout
        # timeout=0.5 gives Modal a moment to return cached results
        result = function_call.get(timeout=0.5)
        
        # If we got here, the job is complete
        job["status"] = result.get("status", "completed")
        job["metrics"] = result.get("metrics")
        job["base_metrics"] = result.get("base_metrics")
        job["finetuned_metrics"] = result.get("finetuned_metrics")
        job["improvement"] = result.get("improvement")
        job["base_model"] = result.get("base_model")
        job["task_type"] = result.get("task_type", job.get("task_type"))
        job["error"] = result.get("error")
        job["completed_at"] = datetime.now().isoformat()
        
        return {
            "job_id": job_id,
            "status": job["status"],
            "metrics": job.get("metrics"),
            "base_metrics": job.get("base_metrics"),
            "finetuned_metrics": job.get("finetuned_metrics"),
            "improvement": job.get("improvement"),
            "base_model": job.get("base_model"),
            "task_type": job.get("task_type"),
            "error": job.get("error"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }
        
    except (modal.exception.FunctionTimeoutError, TimeoutError):
        # Job is still running (timeout means result not ready yet)
        return {
            "job_id": job_id,
            "status": "running",
            "started_at": job.get("started_at"),
            "message": "Training in progress...",
        }
        
    except modal.exception.ExecutionError as e:
        # The function itself raised an error
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
        
    except Exception as e:
        # Unknown error - log it but don't mark as failed yet
        # It might just be a transient network issue
        print(f"[TuneKit] Status check error for {job_id}: {type(e).__name__}: {e}")
        return {
            "job_id": job_id,
            "status": "running",
            "started_at": job.get("started_at"),
            "message": f"Checking status... ({type(e).__name__})",
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
        get_model_files_fn = modal.Function.from_name(MODAL_APP_NAME, "get_model_files")
        result = get_model_files_fn.remote(job_id)
        
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
        
    except modal.exception.NotFoundError:
        return {
            "job_id": job_id,
            "status": "error",
            "error": f"Modal app '{MODAL_APP_NAME}' not deployed",
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
        download_fn = modal.Function.from_name(MODAL_APP_NAME, "download_model_file")
        content = download_fn.remote(job_id, filename)
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


def compare_models(job_id: str, text: str) -> dict:
    """
    Compare base model vs fine-tuned model predictions.
    
    Args:
        job_id: The job identifier for the fine-tuned model
        text: Input text to run inference on
    
    Returns:
        dict with base and finetuned predictions
    """
    if job_id not in _jobs:
        return {
            "error": "Job not found",
            "job_id": job_id,
        }
    
    job = _jobs[job_id]
    
    if job["status"] != "completed":
        return {
            "error": "Training not completed yet",
            "job_id": job_id,
        }
    
    task_type = job.get("task_type", "classification")
    
    try:
        compare_fn = modal.Function.from_name(MODAL_APP_NAME, "compare_inference")
        result = compare_fn.remote(job_id, text, task_type)
        return result
        
    except modal.exception.NotFoundError:
        return {
            "error": f"Comparison function not deployed. Run: modal deploy tunekit/training/modal_train.py",
            "job_id": job_id,
        }
        
    except Exception as e:
        return {
            "error": f"Comparison failed: {str(e)}",
            "job_id": job_id,
        }
