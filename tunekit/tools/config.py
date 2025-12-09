"""
Training Config Tool
====================
Generate complete, validated training configurations.
"""


# ============================================================================
# HELPER FUNCTIONS (Adaptive Hyperparameters)
# ============================================================================

def get_learning_rate(task_type: str) -> float:
    """Learning rate based on task type (HF best practices)."""
    return 2e-4 if task_type == "instruction_tuning" else 2e-5


def get_batch_size(model_name: str) -> int:
    """Batch size based on model size/family."""
    name = model_name.lower()
    if "llama" in name or "7b" in name:
        return 4       # large LLMs
    if "large" in name or "deberta" in name:
        return 8       # large encoders
    if "distil" in name:
        return 32      # small models
    return 16          # default for BERT-size


def get_num_epochs(num_rows: int) -> int:
    """Epochs based on dataset size."""
    if num_rows < 500:
        return 5
    if num_rows < 2000:
        return 3
    return 2


def get_max_length(task_type: str) -> int:
    """Max sequence length by task type."""
    return 512 if task_type == "instruction_tuning" else 128


# ============================================================================
# VALIDATION
# ============================================================================

def validate_config(config: dict) -> None:
    """Manual validation for critical fields and ranges.

    Raises ValueError if invalid.
    """
    # Required base fields
    required = ["base_model", "learning_rate", "batch_size", "num_epochs"]
    for field in required:
        if field not in config:
            raise ValueError(f"Missing required field: {field}")

    lr = config["learning_rate"]
    if not (0 < lr <= 1):
        raise ValueError(f"Invalid learning_rate: {lr}")

    bs = config["batch_size"]
    if not (1 <= bs <= 128):
        raise ValueError(f"Invalid batch_size: {bs}")

    ne = config["num_epochs"]
    if not (1 <= ne <= 20):
        raise ValueError(f"Invalid num_epochs: {ne}")

    # Task-specific sanity
    task_type = config.get("task_type")

    if task_type == "classification":
        num_labels = config.get("num_labels")
        if num_labels is None or num_labels < 2:
            raise ValueError("Classification needs num_labels >= 2")

    if task_type == "ner":
        num_labels = config.get("num_labels")
        if num_labels is None or num_labels < 2:
            raise ValueError("NER needs num_labels >= 2")


# ============================================================================
# MASTER TRAINING CONFIG GENERATOR
# ============================================================================

def get_training_config(
    task_type: str,
    model_name: str,
    num_rows: int,
    columns: dict,
) -> dict:
    """
    Generate complete training config as a plain dict.
    
    Args:
        task_type: "classification", "ner", or "instruction_tuning"
        model_name: Model identifier (e.g., "bert-base-uncased")
        num_rows: Dataset size
        columns: Column mapping from Agent 1
    
    Returns:
        Complete config dict ready for training
    
    Raises:
        ValueError: If required columns are missing or config is invalid
    """
    # --- Validate required columns per task ---
    if task_type == "classification":
        if not columns.get("text_column") or not columns.get("label_column"):
            raise ValueError("Classification needs text_column and label_column")
    elif task_type == "ner":
        if not columns.get("text_column") or not columns.get("label_column"):
            raise ValueError("NER needs text_column and label_column")
    elif task_type == "instruction_tuning":
        if not columns.get("instruction_column") or not columns.get("response_column"):
            raise ValueError("Instruction tuning needs instruction_column and response_column")

    # Base config shared across tasks
    config: dict = {
        "task_type": task_type,
        "base_model": model_name,
        "learning_rate": get_learning_rate(task_type),
        "batch_size": get_batch_size(model_name),
        "num_epochs": get_num_epochs(num_rows),
        "max_length": get_max_length(task_type),
        "warmup_ratio": 0.1,
        "weight_decay": 0.01,
        "save_strategy": "epoch",
        "evaluation_strategy": "epoch",
        "output_dir": f"./output/{task_type}",
        "remove_unused_columns": True,
    }

    # Task-specific settings
    if task_type == "classification":
        config.update({
            "text_column": columns.get("text_column"),
            "label_column": columns.get("label_column"),
            "num_labels": columns.get("num_labels", 2),
            "metric_for_best_model": "accuracy",
        })

    elif task_type == "ner":
        label_list = columns.get("label_list") or []
        if label_list:
            num_labels = len(label_list)
        else:
            num_labels = columns.get("num_labels")
            if num_labels is None:
                raise ValueError("NER needs label_list or num_labels in columns dict")
        config.update({
            "text_column": columns.get("text_column"),
            "label_column": columns.get("label_column"),
            "label_list": label_list,
            "num_labels": num_labels,
            "metric_for_best_model": "f1",
        })

    elif task_type == "instruction_tuning":
        config.update({
            "instruction_column": columns.get("instruction_column"),
            "response_column": columns.get("response_column"),
            "use_qlora": True,
            "lora_r": 8,
            "lora_alpha": 16,
            "lora_dropout": 0.1,
            "gradient_accumulation_steps": 4,
        })

    else:
        raise ValueError(f"Unknown task_type: {task_type}")

    # Final validation
    validate_config(config)
    return config

