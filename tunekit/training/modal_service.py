"""
Modal Service Wrapper
=====================
High-level interface for managing training jobs on Modal.
Updated for Unsloth LoRA training.
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
    Start a training job on Modal using Unsloth LoRA.
    
    Args:
        task_type: Task type (now always "chat" for SLM fine-tuning)
        config: Training configuration dict containing:
            - model_name: HuggingFace model ID
            - training_args: Training parameters
            - lora_config: LoRA parameters (optional, auto-generated in package.py)
        data_path: Path to the JSONL data file
    
    Returns:
        dict with job_id and status
    """
    # Generate unique job ID
    job_id = f"tk_{uuid.uuid4().hex[:12]}"
    
    # Read data file content
    with open(data_path, "r") as f:
        data_content = f.read()
    
    # Prepare config for train_chat_lora
    training_config = {
        "base_model": config.get("model_name", "microsoft/Phi-4-mini-instruct"),
        "max_seq_length": config.get("max_seq_length", 2048),
        "training_args": config.get("training_args", {
            "num_train_epochs": 3,
            "per_device_train_batch_size": 2,
            "gradient_accumulation_steps": 4,
            "learning_rate": 2e-4,
        }),
    }
    
    # Add lora_config if provided (otherwise modal_train.py will auto-generate)
    if "lora_config" in config:
        training_config["lora_config"] = config["lora_config"]
    
    # Spawn the training job asynchronously
    try:
        # Always use train_chat_lora for SLM fine-tuning
        print(f"[TuneKit] Starting Unsloth LoRA training for job {job_id}")
        print(f"[TuneKit] Model: {training_config['base_model']}")
        
        train_fn = modal.Function.from_name(MODAL_APP_NAME, "train_chat_lora")
        
        # Spawn async execution
        function_call = train_fn.spawn(training_config, data_content, job_id)
        
        # Store job info
        _jobs[job_id] = {
            "job_id": job_id,
            "task_type": "chat",  # Always chat for SLM fine-tuning
            "config": training_config,
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
            "message": f"Unsloth LoRA training started for {training_config['base_model']}",
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
            "task_type": "chat",
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
            "base_model": job.get("base_model"),
            "task_type": job.get("task_type"),
            "trainable_params": job.get("trainable_params"),
            "total_params": job.get("total_params"),
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
        result = function_call.get(timeout=0.5)
        
        # If we got here, the job is complete
        job["status"] = result.get("status", "completed")
        job["metrics"] = result.get("metrics")
        job["base_model"] = result.get("base_model")
        job["task_type"] = result.get("task_type", "chat")
        job["trainable_params"] = result.get("trainable_params")
        job["total_params"] = result.get("total_params")
        job["error"] = result.get("error")
        job["completed_at"] = datetime.now().isoformat()
        
        return {
            "job_id": job_id,
            "status": job["status"],
            "metrics": job.get("metrics"),
            "base_model": job.get("base_model"),
            "task_type": job.get("task_type"),
            "trainable_params": job.get("trainable_params"),
            "total_params": job.get("total_params"),
            "error": job.get("error"),
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }
        
    except (modal.exception.FunctionTimeoutError, TimeoutError):
        # Job is still running
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
        # Unknown error - don't mark as failed yet (might be transient)
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
    print(f"[TuneKit] Getting download info for job: {job_id}")
    
    # Check local cache first
    if job_id in _jobs:
        job = _jobs[job_id]
        if job["status"] != "completed":
            print(f"[TuneKit] Job {job_id} not completed (status: {job['status']})")
            return {
                "job_id": job_id,
                "status": job["status"],
                "error": "Model not ready. Training must be completed first.",
            }
    
    # Try to get model files from Modal volume
    try:
        print(f"[TuneKit] Fetching model files from Modal for job: {job_id}")
        get_model_files_fn = modal.Function.from_name(MODAL_APP_NAME, "get_model_files")
        result = get_model_files_fn.remote(job_id)
        print(f"[TuneKit] Modal response: exists={result.get('exists')}, files={len(result.get('files', []))}")
        
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
            "message": "LoRA adapter ready for download",
        }
        
    except modal.exception.NotFoundError:
        print(f"[TuneKit] Modal app not found: {MODAL_APP_NAME}")
        return {
            "job_id": job_id,
            "status": "error",
            "error": f"Modal app '{MODAL_APP_NAME}' not deployed",
        }
        
    except Exception as e:
        print(f"[TuneKit] Error getting model files: {e}")
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
        print(f"[TuneKit] Downloading file: {filename} from job {job_id}")
        download_fn = modal.Function.from_name(MODAL_APP_NAME, "download_model_file")
        content = download_fn.remote(job_id, filename)
        print(f"[TuneKit] Downloaded {filename}: {len(content) if content else 0} bytes")
        return content
        
    except Exception as e:
        print(f"[TuneKit] Error downloading file {filename}: {e}")
        return None


def download_model_zip_from_modal(job_id: str) -> Optional[bytes]:
    """
    Download model ZIP from Modal.
    Step 1: Ensure ZIP exists (generate if needed for old models)
    Step 2: Download the ZIP
    
    Args:
        job_id: The job identifier
    
    Returns:
        ZIP file content as bytes, or None if failed
    """
    import time
    start_time = time.time()
    
    try:
        # Step 1: Ensure ZIP exists
        step1_start = time.time()
        print(f"[TuneKit] Step 1: Ensuring ZIP exists for job {job_id}...")
        generate_fn = modal.Function.from_name(MODAL_APP_NAME, "generate_model_zip")
        success = generate_fn.remote(job_id)
        step1_time = time.time() - step1_start
        
        if not success:
            print(f"[TuneKit] Failed to generate/find ZIP for {job_id} (took {step1_time:.1f}s)")
            return None
        
        print(f"[TuneKit] Step 1 complete: ZIP ready (took {step1_time:.1f}s)")
        
        # Step 2: Download the ZIP
        step2_start = time.time()
        print(f"[TuneKit] Step 2: Downloading ZIP...")
        download_fn = modal.Function.from_name(MODAL_APP_NAME, "download_model_zip")
        zip_bytes = download_fn.remote(job_id)
        step2_time = time.time() - step2_start
        
        total_elapsed = time.time() - start_time
        print(f"[TuneKit] Step 2 complete: Downloaded {len(zip_bytes) if zip_bytes else 0} bytes (took {step2_time:.1f}s)")
        print(f"[TuneKit] TOTAL TIME: {total_elapsed:.1f}s")
        return zip_bytes
        
    except modal.exception.NotFoundError as e:
        print(f"[TuneKit] Function not found: {e}")
        return None
        
    except FileNotFoundError as e:
        print(f"[TuneKit] Model not found: {e}")
        return None
        
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"[TuneKit] Error downloading model ZIP after {elapsed:.1f}s: {type(e).__name__}: {e}")
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
            "task_type": job.get("task_type", "chat"),
            "status": job["status"],
            "started_at": job.get("started_at"),
            "completed_at": job.get("completed_at"),
        }
        for job in _jobs.values()
    ]


def test_model_inference(job_id: str, messages: list) -> dict:
    """
    Test a fine-tuned model with a conversation.
    
    Args:
        job_id: The job identifier for the fine-tuned model
        messages: List of messages [{"role": "user", "content": "..."}]
    
    Returns:
        dict with generated response
    """
    print(f"[TuneKit] Testing model for job {job_id}")
    
    # Check if job is completed
    if job_id in _jobs:
        job = _jobs[job_id]
        if job["status"] != "completed":
            return {
                "error": "Training not completed yet",
                "job_id": job_id,
            }
    
    try:
        print(f"[TuneKit] Calling Modal test_model for job {job_id}")
        test_fn = modal.Function.from_name(MODAL_APP_NAME, "test_model")
        result = test_fn.remote(job_id, messages)
        print(f"[TuneKit] Test result: {result.get('status')}")
        return result
        
    except modal.exception.NotFoundError:
        return {
            "error": "Test function not deployed. Run: modal deploy tunekit/training/modal_train.py",
            "job_id": job_id,
        }
        
    except Exception as e:
        print(f"[TuneKit] Test error: {e}")
        return {
            "error": f"Model test failed: {str(e)}",
            "job_id": job_id,
        }


# Keep old function name for backward compatibility
def compare_models(job_id: str, text: str) -> dict:
    """
    DEPRECATED: Use test_model_inference instead.
    Kept for backward compatibility.
    """
    messages = [{"role": "user", "content": text}]
    return test_model_inference(job_id, messages)
