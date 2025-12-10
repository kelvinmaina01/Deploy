"""
Modal Training Functions
========================
GPU-accelerated training functions that run on Modal's serverless infrastructure.
"""

import modal

# =============================================================================
# MODAL APP & IMAGE SETUP
# =============================================================================

app = modal.App("tunekit-training")

# Base image with ML dependencies
training_image = modal.Image.debian_slim(python_version="3.11").pip_install(
    "torch>=2.0.0",
    "transformers>=4.36.0",
    "datasets>=2.14.0",
    "accelerate>=0.24.0",
    "scikit-learn>=1.3.0",
    "peft>=0.7.0",
    "trl>=0.7.0",
    "bitsandbytes>=0.41.0",
    "seqeval>=1.2.2",
    "numpy<2",
)

# Persistent volume for storing trained models
model_volume = modal.Volume.from_name("tunekit-models", create_if_missing=True)
MODEL_DIR = "/models"


# =============================================================================
# CLASSIFICATION TRAINING
# =============================================================================

@app.function(
    image=training_image,
    gpu="T4",
    timeout=3600,
    volumes={MODEL_DIR: model_volume},
)
def train_classification(config: dict, data_content: str, job_id: str) -> dict:
    """
    Train a classification model on Modal GPU.
    
    Args:
        config: Training configuration dict
        data_content: CSV data as string
        job_id: Unique job identifier
    
    Returns:
        dict with status, metrics, and model path
    """
    import json
    import os
    import tempfile
    import numpy as np
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForSequenceClassification,
        TrainingArguments,
        Trainer,
        DataCollatorWithPadding,
    )
    from sklearn.metrics import accuracy_score, f1_score
    
    print(f"[TuneKit] Starting classification training for job {job_id}")
    print(f"[TuneKit] Model: {config['base_model']}")
    
    # Write data to temp file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(data_content)
        data_path = f.name
    
    try:
        # Load tokenizer and model
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        model = AutoModelForSequenceClassification.from_pretrained(
            config["base_model"],
            num_labels=config["num_labels"],
        )
        
        # Load dataset
        dataset = load_dataset("csv", data_files=data_path, split="train")
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        # Create label mapping
        labels = dataset["train"].unique(config["label_column"])
        label2id = {label: i for i, label in enumerate(labels)}
        id2label = {i: label for label, i in label2id.items()}
        
        model.config.label2id = label2id
        model.config.id2label = id2label
        
        def tokenize_function(examples):
            tokens = tokenizer(
                examples[config["text_column"]],
                padding="max_length",
                truncation=True,
                max_length=config["max_length"],
            )
            tokens["labels"] = [label2id[label] for label in examples[config["label_column"]]]
            return tokens
        
        tokenized = dataset.map(
            tokenize_function,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        def compute_metrics(eval_pred):
            logits, labels = eval_pred
            predictions = np.argmax(logits, axis=-1)
            return {
                "accuracy": accuracy_score(labels, predictions),
                "f1": f1_score(labels, predictions, average="weighted"),
            }
        
        # Output directory on the volume
        output_dir = f"{MODEL_DIR}/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            num_train_epochs=config["num_epochs"],
            warmup_ratio=config.get("warmup_ratio", 0.1),
            weight_decay=config.get("weight_decay", 0.01),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="accuracy",
            logging_steps=10,
            report_to="none",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
        
        print("[TuneKit] Training started...")
        trainer.train()
        
        # Save final model
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        # Save config for reference
        with open(f"{output_dir}/tunekit_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        # Commit to volume
        model_volume.commit()
        
        # Final evaluation
        results = trainer.evaluate()
        print(f"[TuneKit] Training complete! Results: {results}")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "metrics": results,
            "model_path": output_dir,
        }
        
    except Exception as e:
        print(f"[TuneKit] Training failed: {str(e)}")
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
        }
    finally:
        os.unlink(data_path)


# =============================================================================
# NER TRAINING
# =============================================================================

@app.function(
    image=training_image,
    gpu="T4",
    timeout=3600,
    volumes={MODEL_DIR: model_volume},
)
def train_ner(config: dict, data_content: str, job_id: str) -> dict:
    """
    Train a NER model on Modal GPU.
    """
    import json
    import os
    import tempfile
    import numpy as np
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
    )
    from seqeval.metrics import f1_score, precision_score, recall_score
    
    print(f"[TuneKit] Starting NER training for job {job_id}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(data_content)
        data_path = f.name
    
    try:
        label_list = config["label_list"]
        label2id = {label: i for i, label in enumerate(label_list)}
        id2label = {i: label for label, i in label2id.items()}
        
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        model = AutoModelForTokenClassification.from_pretrained(
            config["base_model"],
            num_labels=len(label_list),
            id2label=id2label,
            label2id=label2id,
        )
        
        dataset = load_dataset("csv", data_files=data_path, split="train")
        dataset = dataset.train_test_split(test_size=0.2, seed=42)
        
        def tokenize_and_align_labels(examples):
            tokenized_inputs = tokenizer(
                examples[config["text_column"]],
                truncation=True,
                is_split_into_words=True,
                max_length=config["max_length"],
            )
            
            labels = []
            for i, label in enumerate(examples[config["label_column"]]):
                word_ids = tokenized_inputs.word_ids(batch_index=i)
                label_ids = []
                previous_word_idx = None
                for word_idx in word_ids:
                    if word_idx is None:
                        label_ids.append(-100)
                    elif word_idx != previous_word_idx:
                        label_ids.append(label2id[label[word_idx]])
                    else:
                        label_ids.append(-100)
                    previous_word_idx = word_idx
                labels.append(label_ids)
            
            tokenized_inputs["labels"] = labels
            return tokenized_inputs
        
        tokenized = dataset.map(tokenize_and_align_labels, batched=True)
        
        def compute_metrics(eval_pred):
            predictions, labels = eval_pred
            predictions = np.argmax(predictions, axis=2)
            
            true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
            true_predictions = [
                [id2label[p] for (p, l) in zip(prediction, label) if l != -100]
                for prediction, label in zip(predictions, labels)
            ]
            
            return {
                "precision": precision_score(true_labels, true_predictions),
                "recall": recall_score(true_labels, true_predictions),
                "f1": f1_score(true_labels, true_predictions),
            }
        
        output_dir = f"{MODEL_DIR}/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            num_train_epochs=config["num_epochs"],
            warmup_ratio=config.get("warmup_ratio", 0.1),
            weight_decay=config.get("weight_decay", 0.01),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            logging_steps=10,
            report_to="none",
        )
        
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized["train"],
            eval_dataset=tokenized["test"],
            tokenizer=tokenizer,
            data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
            compute_metrics=compute_metrics,
        )
        
        print("[TuneKit] Training started...")
        trainer.train()
        
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        with open(f"{output_dir}/tunekit_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        model_volume.commit()
        
        results = trainer.evaluate()
        print(f"[TuneKit] Training complete! Results: {results}")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "metrics": results,
            "model_path": output_dir,
        }
        
    except Exception as e:
        print(f"[TuneKit] Training failed: {str(e)}")
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
        }
    finally:
        os.unlink(data_path)


# =============================================================================
# INSTRUCTION TUNING (QLoRA)
# =============================================================================

@app.function(
    image=training_image,
    gpu="A10G",  # Need more VRAM for LLMs
    timeout=7200,  # 2 hours for LLM training
    volumes={MODEL_DIR: model_volume},
)
def train_instruction(config: dict, data_content: str, job_id: str) -> dict:
    """
    Train an instruction-following model using QLoRA on Modal GPU.
    """
    import json
    import os
    import tempfile
    import torch
    from datasets import load_dataset
    from transformers import (
        AutoTokenizer,
        AutoModelForCausalLM,
        BitsAndBytesConfig,
        TrainingArguments,
    )
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from trl import SFTTrainer
    
    print(f"[TuneKit] Starting instruction tuning for job {job_id}")
    print(f"[TuneKit] Model: {config['base_model']}")
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        f.write(data_content)
        data_path = f.name
    
    try:
        # QLoRA quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        
        # Load model with quantization
        model = AutoModelForCausalLM.from_pretrained(
            config["base_model"],
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        model = prepare_model_for_kbit_training(model)
        
        # LoRA config
        lora_config = LoraConfig(
            r=config.get("lora_r", 8),
            lora_alpha=config.get("lora_alpha", 16),
            lora_dropout=config.get("lora_dropout", 0.1),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        
        # Tokenizer
        tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
        
        # Load dataset
        dataset = load_dataset("csv", data_files=data_path, split="train")
        
        # Format as chat
        def format_instruction(example):
            instruction = example[config["instruction_column"]]
            response = example[config["response_column"]]
            return {"text": f"### Instruction:\n{instruction}\n\n### Response:\n{response}"}
        
        dataset = dataset.map(format_instruction)
        dataset = dataset.train_test_split(test_size=0.1, seed=42)
        
        output_dir = f"{MODEL_DIR}/{job_id}"
        os.makedirs(output_dir, exist_ok=True)
        
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=config["learning_rate"],
            per_device_train_batch_size=config["batch_size"],
            per_device_eval_batch_size=config["batch_size"],
            num_train_epochs=config["num_epochs"],
            warmup_ratio=config.get("warmup_ratio", 0.1),
            weight_decay=config.get("weight_decay", 0.01),
            evaluation_strategy="epoch",
            save_strategy="epoch",
            gradient_accumulation_steps=config.get("gradient_accumulation_steps", 4),
            fp16=True,
            logging_steps=10,
            report_to="none",
        )
        
        trainer = SFTTrainer(
            model=model,
            args=training_args,
            train_dataset=dataset["train"],
            eval_dataset=dataset["test"],
            tokenizer=tokenizer,
            dataset_text_field="text",
            max_seq_length=config["max_length"],
        )
        
        print("[TuneKit] Training started...")
        trainer.train()
        
        # Save LoRA adapter
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        
        with open(f"{output_dir}/tunekit_config.json", "w") as f:
            json.dump(config, f, indent=2)
        
        model_volume.commit()
        
        print(f"[TuneKit] Training complete! Model saved to {output_dir}")
        
        return {
            "status": "completed",
            "job_id": job_id,
            "metrics": {"training_loss": trainer.state.log_history[-1].get("loss", 0)},
            "model_path": output_dir,
        }
        
    except Exception as e:
        print(f"[TuneKit] Training failed: {str(e)}")
        return {
            "status": "failed",
            "job_id": job_id,
            "error": str(e),
        }
    finally:
        os.unlink(data_path)


# =============================================================================
# MODEL DOWNLOAD HELPER
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

