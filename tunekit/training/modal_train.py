"""
Modal Training Functions - Unsloth LoRA
========================================
GPU-accelerated LoRA fine-tuning using Unsloth on Modal's serverless infrastructure.
Supports all 5 TuneKit SLMs: Phi-4, Gemma-3, Llama-3.2, Qwen-2.5, Mistral-7B
"""

import modal

# =============================================================================
# MODAL APP & IMAGE SETUP
# =============================================================================

app = modal.App("tunekit-training")

# Unsloth-optimized image with all dependencies
training_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch>=2.1.0",
        "transformers>=4.45.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "peft>=0.7.0",
        "trl>=0.7.0",
        "bitsandbytes>=0.41.0",
        "xformers<0.0.27",
        "sentencepiece",
        "protobuf",
    )
    .pip_install(
        "unsloth[cu121-ampere-torch230] @ git+https://github.com/unslothai/unsloth.git"
    )
)

# Persistent volume for storing trained models
model_volume = modal.Volume.from_name("tunekit-models", create_if_missing=True)
MODEL_DIR = "/models"


# =============================================================================
# MODEL ARCHITECTURE DETECTION
# =============================================================================

def get_target_modules(model_id: str) -> list:
    """
    Get the correct LoRA target modules based on model architecture.
    
    - Phi models use fc1/fc2 for MLP layers
    - Llama, Mistral, Gemma, Qwen use gate_proj/up_proj/down_proj
    """
    model_id_lower = model_id.lower()
    
    if "phi" in model_id_lower:
        return ["q_proj", "k_proj", "v_proj", "o_proj", "fc1", "fc2"]
    else:
        # Llama, Mistral, Gemma, Qwen
        return ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def get_gpu_for_model(model_id: str) -> str:
    """
    Select appropriate GPU based on model size.
    
    - 2B-4B models: T4 (16GB) is sufficient
    - 7B models: A10G (24GB) recommended
    """
    model_id_lower = model_id.lower()
    
    if "mistral-7b" in model_id_lower or "7b" in model_id_lower:
        return "A10G"
    else:
        return "T4"


# =============================================================================
# UNSLOTH LORA TRAINING FOR CHAT MODELS
# =============================================================================

@app.function(
    image=training_image,
    gpu="A10G",  # Use A10G for all models to ensure enough VRAM
    timeout=7200,  # 2 hours max
    volumes={MODEL_DIR: model_volume},
    secrets=[modal.Secret.from_name("huggingface-secret", required=False)],
)
def train_chat_lora(config: dict, data_content: str, job_id: str) -> dict:
    """
    Train a chat model using Unsloth LoRA on Modal GPU.
    
    Args:
        config: Training configuration dict containing:
            - base_model: HuggingFace model ID (e.g., "microsoft/Phi-4-mini-instruct")
            - lora_config: LoRA parameters (r, lora_alpha, lora_dropout, target_modules)
            - training_args: Training parameters (epochs, batch_size, learning_rate, etc.)
            - max_seq_length: Maximum sequence length (default 2048)
        data_content: JSONL data as string (each line: {"messages": [...]})
        job_id: Unique job identifier
    
    Returns:
        dict with status, metrics, and model path
    """
    import json
    import os
    import tempfile
    import torch
    
    print(f"\n{'='*60}")
    print(f"🚀 TuneKit Unsloth LoRA Training")
    print(f"{'='*60}")
    print(f"Job ID: {job_id}")
    print(f"Model: {config.get('base_model', 'Unknown')}")
    print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"{'='*60}\n")
    
    # Write JSONL data to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
        f.write(data_content)
        data_path = f.name
    
    try:
        # =============================================
        # 1. LOAD MODEL WITH UNSLOTH
        # =============================================
        from unsloth import FastLanguageModel
        
        base_model = config.get("base_model", "microsoft/Phi-4-mini-instruct")
        max_seq_length = config.get("max_seq_length", 2048)
        
        print(f"[TuneKit] Loading model: {base_model}")
        print(f"[TuneKit] Max sequence length: {max_seq_length}")
        
        # Load with 4-bit quantization for memory efficiency
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,  # Auto-detect (bf16 if available, else fp16)
            load_in_4bit=True,
            trust_remote_code=True,
        )
        
        print(f"[TuneKit] Model loaded successfully!")
        
        # =============================================
        # 2. CONFIGURE LORA
        # =============================================
        lora_config = config.get("lora_config", {})
        
        # Get model-specific target modules if not provided
        target_modules = lora_config.get("target_modules")
        if not target_modules:
            target_modules = get_target_modules(base_model)
        
        r = lora_config.get("r", 16)
        lora_alpha = lora_config.get("lora_alpha", 16)
        lora_dropout = lora_config.get("lora_dropout", 0)
        
        print(f"\n[TuneKit] LoRA Configuration:")
        print(f"  - r (rank): {r}")
        print(f"  - lora_alpha: {lora_alpha}")
        print(f"  - lora_dropout: {lora_dropout}")
        print(f"  - target_modules: {target_modules}")
        
        model = FastLanguageModel.get_peft_model(
            model,
            r=r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            target_modules=target_modules,
            bias="none",
            use_gradient_checkpointing="unsloth",  # Optimized checkpointing
            random_state=42,
            use_rslora=False,
            loftq_config=None,
        )
        
        # Print trainable parameters
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        print(f"[TuneKit] Trainable params: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
        
        # =============================================
        # 3. LOAD AND FORMAT DATASET
        # =============================================
        from datasets import Dataset
        
        print(f"\n[TuneKit] Loading dataset from {data_path}")
        
        # Load JSONL data
        conversations = []
        with open(data_path, 'r') as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        entry = json.loads(line)
                        if "messages" in entry:
                            conversations.append(entry)
                    except json.JSONDecodeError:
                        continue
        
        print(f"[TuneKit] Loaded {len(conversations)} conversations")
        
        if len(conversations) == 0:
            raise ValueError("No valid conversations found in dataset")
        
        # Format conversations using the tokenizer's chat template
        def format_conversation(example):
            """Convert messages to the model's chat format."""
            messages = example["messages"]
            
            # Use tokenizer's chat template if available
            if hasattr(tokenizer, 'apply_chat_template'):
                try:
                    text = tokenizer.apply_chat_template(
                        messages,
                        tokenize=False,
                        add_generation_prompt=False
                    )
                except Exception:
                    # Fallback: manual formatting
                    text = format_messages_manual(messages)
            else:
                text = format_messages_manual(messages)
            
            return {"text": text}
        
        def format_messages_manual(messages):
            """Manual fallback formatting for chat messages."""
            formatted = ""
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                if role == "system":
                    formatted += f"<|system|>\n{content}\n"
                elif role == "user":
                    formatted += f"<|user|>\n{content}\n"
                elif role == "assistant":
                    formatted += f"<|assistant|>\n{content}\n"
            return formatted
        
        # Create dataset
        dataset = Dataset.from_list(conversations)
        dataset = dataset.map(format_conversation)
        
        # Split into train/eval
        if len(dataset) > 50:
            split = dataset.train_test_split(test_size=0.1, seed=42)
            train_dataset = split["train"]
            eval_dataset = split["test"]
            print(f"[TuneKit] Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")
        else:
            train_dataset = dataset
            eval_dataset = None
            print(f"[TuneKit] Train: {len(train_dataset)} (no eval split for small datasets)")
        
        # =============================================
        # 4. CONFIGURE TRAINING
        # =============================================
        from trl import SFTTrainer
        from transformers import TrainingArguments
        
        training_args_config = config.get("training_args", {})
        
        output_dir = f"{MODEL_DIR}/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        # Training arguments with Unsloth optimizations
        num_epochs = training_args_config.get("num_train_epochs", 3)
        batch_size = training_args_config.get("per_device_train_batch_size", 2)
        grad_accum = training_args_config.get("gradient_accumulation_steps", 4)
        learning_rate = training_args_config.get("learning_rate", 2e-4)
        warmup_steps = training_args_config.get("warmup_steps", 5)
        
        print(f"\n[TuneKit] Training Configuration:")
        print(f"  - Epochs: {num_epochs}")
        print(f"  - Batch size: {batch_size}")
        print(f"  - Gradient accumulation: {grad_accum}")
        print(f"  - Effective batch size: {batch_size * grad_accum}")
        print(f"  - Learning rate: {learning_rate}")
        print(f"  - Warmup steps: {warmup_steps}")
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            gradient_accumulation_steps=grad_accum,
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=training_args_config.get("weight_decay", 0.01),
            logging_steps=training_args_config.get("logging_steps", 10),
            save_strategy="epoch",
            eval_strategy="epoch" if eval_dataset else "no",
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            optim="adamw_8bit",  # Memory-efficient optimizer
            lr_scheduler_type="linear",
            seed=42,
            report_to="none",
        )
        
        # =============================================
        # 5. TRAIN WITH SFTTRAINER
        # =============================================
        print(f"\n[TuneKit] Starting training...")
        print(f"{'='*60}")
        
        trainer = SFTTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            dataset_text_field="text",
            max_seq_length=max_seq_length,
            args=training_args,
        )
        
        # Train!
        train_result = trainer.train()
        
        print(f"\n{'='*60}")
        print(f"[TuneKit] Training complete!")
        print(f"{'='*60}")
        
        # =============================================
        # 6. SAVE MODEL
        # =============================================
        print(f"\n[TuneKit] Saving LoRA adapter to {output_dir}")
        
        # Save LoRA adapter (small, ~50MB)
        model.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save training config
        with open(f"{output_dir}/tunekit_config.json", "w") as f:
            json.dump({
                "base_model": base_model,
                "max_seq_length": max_seq_length,
                "lora_config": {
                    "r": r,
                    "lora_alpha": lora_alpha,
                    "lora_dropout": lora_dropout,
                    "target_modules": target_modules,
                },
                "training_args": {
                    "num_train_epochs": num_epochs,
                    "per_device_train_batch_size": batch_size,
                    "gradient_accumulation_steps": grad_accum,
                    "learning_rate": learning_rate,
                },
                "dataset_size": len(conversations),
                "job_id": job_id,
            }, f, indent=2)
        
        # =============================================
        # 7. CREATE ZIP FOR DOWNLOAD
        # =============================================
        import zipfile
        
        zip_path = f"{output_dir}/model.zip"
        print(f"[TuneKit] Creating ZIP archive...")
        
        with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
            for root, dirs, files in os.walk(output_dir):
                for file in files:
                    if file != "model.zip":
                        filepath = os.path.join(root, file)
                        arcname = os.path.relpath(filepath, output_dir)
                        zf.write(filepath, arcname)
        
        # Commit to volume
        model_volume.commit()
        
        # =============================================
        # 8. COLLECT METRICS
        # =============================================
        metrics = {
            "train_loss": train_result.training_loss,
            "train_runtime": train_result.metrics.get("train_runtime", 0),
            "train_samples_per_second": train_result.metrics.get("train_samples_per_second", 0),
            "epochs": num_epochs,
        }
        
        # Eval metrics if available
        if eval_dataset:
            eval_results = trainer.evaluate()
            metrics["eval_loss"] = eval_results.get("eval_loss", 0)
        
        print(f"\n[TuneKit] Final Metrics:")
        for key, value in metrics.items():
            if isinstance(value, float):
                print(f"  - {key}: {value:.4f}")
            else:
                print(f"  - {key}: {value}")
        
        print(f"\n{'='*60}")
        print(f"✅ Training completed successfully!")
        print(f"📁 Model saved to: {output_dir}")
        print(f"{'='*60}\n")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "metrics": metrics,
            "model_path": output_dir,
            "base_model": base_model,
            "task_type": "chat",
            "trainable_params": trainable_params,
            "total_params": total_params,
        }
        
    except Exception as e:
        import traceback
        error_msg = str(e)
        error_traceback = traceback.format_exc()
        
        print(f"\n{'='*60}")
        print(f"❌ Training failed!")
        print(f"{'='*60}")
        print(f"Error: {error_msg}")
        print(f"\nTraceback:\n{error_traceback}")
        
        return {
            "status": "failed",
            "job_id": job_id,
            "error": error_msg,
            "traceback": error_traceback,
        }
    finally:
        # Cleanup temp file
        if os.path.exists(data_path):
            os.unlink(data_path)


# =============================================================================
# LEGACY TRAINING FUNCTIONS (Kept for backward compatibility)
# =============================================================================

@app.function(
    image=training_image,
    gpu="T4",
    timeout=3600,
    volumes={MODEL_DIR: model_volume},
)
def train_classification(config: dict, data_content: str, job_id: str) -> dict:
    """
    DEPRECATED: Use train_chat_lora instead.
    Kept for backward compatibility with existing jobs.
    """
    print(f"[TuneKit] WARNING: train_classification is deprecated. Use train_chat_lora instead.")
    
    # Redirect to chat training with converted config
    chat_config = {
        "base_model": config.get("base_model", "microsoft/Phi-4-mini-instruct"),
        "max_seq_length": config.get("max_length", 2048),
        "training_args": {
            "num_train_epochs": config.get("num_epochs", 3),
            "per_device_train_batch_size": config.get("batch_size", 2),
            "learning_rate": config.get("learning_rate", 2e-4),
        }
    }
    
    return train_chat_lora.local(chat_config, data_content, job_id)


@app.function(
    image=training_image,
    gpu="T4",
    timeout=3600,
    volumes={MODEL_DIR: model_volume},
)
def train_ner(config: dict, data_content: str, job_id: str) -> dict:
    """
    DEPRECATED: Use train_chat_lora instead.
    """
    print(f"[TuneKit] WARNING: train_ner is deprecated. Use train_chat_lora instead.")
    return {"status": "failed", "error": "NER training deprecated. Use chat-based fine-tuning."}


@app.function(
    image=training_image,
    gpu="A10G",
    timeout=7200,
    volumes={MODEL_DIR: model_volume},
)
def train_instruction(config: dict, data_content: str, job_id: str) -> dict:
    """
    DEPRECATED: Use train_chat_lora instead.
    """
    print(f"[TuneKit] WARNING: train_instruction is deprecated. Use train_chat_lora instead.")
    
    # Redirect to chat training
    chat_config = {
        "base_model": config.get("base_model", "microsoft/Phi-4-mini-instruct"),
        "max_seq_length": config.get("max_length", 2048),
        "lora_config": {
            "r": config.get("lora_r", 16),
            "lora_alpha": config.get("lora_alpha", 16),
            "lora_dropout": config.get("lora_dropout", 0),
        },
        "training_args": {
            "num_train_epochs": config.get("num_epochs", 3),
            "per_device_train_batch_size": config.get("batch_size", 2),
            "learning_rate": config.get("learning_rate", 2e-4),
            "gradient_accumulation_steps": config.get("gradient_accumulation_steps", 4),
        }
    }
    
    return train_chat_lora.local(chat_config, data_content, job_id)


# =============================================================================
# MODEL DOWNLOAD HELPERS
# =============================================================================

@app.function(
    image=modal.Image.debian_slim(),
    volumes={MODEL_DIR: model_volume},
)
def get_model_files(job_id: str) -> dict:
    """
    Get the list of model files for a completed training job.
    """
    import os
    
    model_path = f"{MODEL_DIR}/{job_id}"
    
    if not os.path.exists(model_path):
        return {"exists": False, "files": []}
    
    files = []
    for root, dirs, filenames in os.walk(model_path):
        for filename in filenames:
            filepath = os.path.join(root, filename)
            rel_path = os.path.relpath(filepath, model_path)
            files.append({
                "name": rel_path,
                "size": os.path.getsize(filepath),
            })
    
    return {"exists": True, "files": files, "path": model_path}


@app.function(
    image=modal.Image.debian_slim(),
    volumes={MODEL_DIR: model_volume},
)
def download_model_file(job_id: str, filename: str) -> bytes:
    """
    Download a specific file from a trained model.
    """
    import os
    
    filepath = f"{MODEL_DIR}/{job_id}/{filename}"
    
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"File not found: {filename}")
    
    with open(filepath, "rb") as f:
        return f.read()


@app.function(
    image=modal.Image.debian_slim(),
    volumes={MODEL_DIR: model_volume},
    timeout=120,
)
def generate_model_zip(job_id: str) -> bool:
    """
    Generate ZIP file for an existing model (for models trained before this feature).
    Called once, then download_model_zip can use the pre-generated file.
    """
    import os
    import zipfile
    
    model_path = f"{MODEL_DIR}/{job_id}"
    zip_path = f"{model_path}/model.zip"
    
    if not os.path.exists(model_path):
        print(f"[TuneKit] Model not found: {job_id}")
        return False
    
    if os.path.exists(zip_path):
        print(f"[TuneKit] ZIP already exists for {job_id}")
        return True
    
    print(f"[TuneKit] Generating ZIP for existing model {job_id}...")
    
    with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zf:
        for root, dirs, files in os.walk(model_path):
            for file in files:
                if file != "model.zip":
                    filepath = os.path.join(root, file)
                    arcname = os.path.relpath(filepath, model_path)
                    zf.write(filepath, arcname)
                    print(f"[TuneKit] Added: {arcname}")
    
    model_volume.commit()
    print(f"[TuneKit] ZIP generated: {os.path.getsize(zip_path)} bytes")
    return True


@app.function(
    image=modal.Image.debian_slim(),
    volumes={MODEL_DIR: model_volume},
    timeout=60,
)
def download_model_zip(job_id: str) -> bytes:
    """
    Return the pre-generated ZIP file (created during training).
    FAST because ZIP was already created, just reading a file.
    """
    import os
    
    model_path = f"{MODEL_DIR}/{job_id}"
    zip_path = f"{model_path}/model.zip"
    
    print(f"[TuneKit] Looking for pre-generated ZIP at: {zip_path}")
    
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {job_id}")
    
    # Check for pre-generated ZIP (created during training)
    if os.path.exists(zip_path):
        print(f"[TuneKit] Found pre-generated ZIP, reading...")
        with open(zip_path, "rb") as f:
            zip_bytes = f.read()
        print(f"[TuneKit] Read ZIP: {len(zip_bytes)} bytes")
        return zip_bytes
    
    # Fallback: create ZIP on the fly (for older models without pre-generated ZIP)
    print(f"[TuneKit] No pre-generated ZIP, creating on the fly...")
    import zipfile
    import io
    
    zip_buffer = io.BytesIO()
    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for root, dirs, files in os.walk(model_path):
            for filename in files:
                filepath = os.path.join(root, filename)
                arcname = os.path.relpath(filepath, model_path)
                zip_file.write(filepath, arcname)
    
    zip_buffer.seek(0)
    zip_bytes = zip_buffer.read()
    print(f"[TuneKit] Created ZIP: {len(zip_bytes)} bytes")
    
    return zip_bytes


# =============================================================================
# INFERENCE / TESTING
# =============================================================================

@app.function(
    image=training_image,
    gpu="T4",
    timeout=120,
    volumes={MODEL_DIR: model_volume},
)
def test_model(job_id: str, messages: list) -> dict:
    """
    Test a fine-tuned model with a conversation.
    
    Args:
        job_id: Job ID of the fine-tuned model
        messages: List of messages [{"role": "user", "content": "..."}]
    
    Returns:
        dict with generated response
    """
    import json
    import os
    import torch
    
    model_path = f"{MODEL_DIR}/{job_id}"
    config_path = f"{model_path}/tunekit_config.json"
    
    if not os.path.exists(model_path):
        return {"error": f"Model not found: {job_id}"}
    
    try:
        from unsloth import FastLanguageModel
        from peft import PeftModel
        
        # Load config
        with open(config_path) as f:
            config = json.load(f)
        
        base_model = config["base_model"]
        max_seq_length = config.get("max_seq_length", 2048)
        
        print(f"[TuneKit] Loading base model: {base_model}")
        
        # Load base model
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=base_model,
            max_seq_length=max_seq_length,
            dtype=None,
            load_in_4bit=True,
            trust_remote_code=True,
        )
        
        # Load LoRA adapter
        print(f"[TuneKit] Loading LoRA adapter from: {model_path}")
        model = PeftModel.from_pretrained(model, model_path)
        
        # Enable inference mode
        FastLanguageModel.for_inference(model)
        
        # Format input
        if hasattr(tokenizer, 'apply_chat_template'):
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            prompt = "\n".join([f"<|{m['role']}|>\n{m['content']}" for m in messages])
            prompt += "\n<|assistant|>\n"
        
        # Generate
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.7,
                top_p=0.9,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id,
            )
        
        # Decode response
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the assistant's response
        if "<|assistant|>" in response:
            response = response.split("<|assistant|>")[-1].strip()
        elif prompt in response:
            response = response[len(prompt):].strip()
        
        return {
            "status": "success",
            "response": response,
            "model": f"tunekit/{job_id}",
            "base_model": base_model,
        }
        
    except Exception as e:
        import traceback
        return {
            "status": "error",
            "error": str(e),
            "traceback": traceback.format_exc(),
        }


# =============================================================================
# MAIN ENTRY POINT (for CLI testing)
# =============================================================================

@app.local_entrypoint()
def main():
    """
    Test the training function locally.
    """
    print("TuneKit Modal Training - Unsloth LoRA")
    print("=====================================")
    print("Deploy with: modal deploy tunekit/training/modal_train.py")
    print("Run training via the TuneKit API at /train endpoint")
