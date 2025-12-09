"""
Model Recommendation Tool
=========================
Rule-based model selection for different task types.

Key insight:
- BERT-sized models (66M-184M) -> Full fine-tuning (fits in consumer GPUs)
- LLMs (1.1B-7B) -> QLoRA (makes large models accessible)
- NER uses CASED models (capitalization matters for entities)
"""


def get_classification_model(num_rows: int, user_description: str) -> dict:
    """Pick best model for text classification."""
    wants_speed = any(w in user_description.lower() for w in ["fast", "quick", "speed", "efficient"])
    
    if wants_speed and num_rows < 2000:
        return {
            "model": "distilbert-base-uncased", 
            "method": "full",
            "reasoning": "Fast model for speed priority"
        }
    if num_rows < 1000:
        return {
            "model": "distilbert-base-uncased", 
            "method": "full",
            "reasoning": "Small dataset - lightweight model to avoid overfitting"
        }
    elif num_rows < 5000:
        return {
            "model": "bert-base-uncased", 
            "method": "full",
            "reasoning": "Medium dataset - balanced BERT model"
        }
    elif num_rows < 10000:
        return {
            "model": "roberta-base", 
            "method": "full",
            "reasoning": "Large dataset - RoBERTa for best quality"
        }
    else:
        return {
            "model": "microsoft/deberta-v3-base", 
            "method": "full",
            "reasoning": "Very large dataset - state-of-art DeBERTa model"
        }


def get_ner_model(num_rows: int, user_description: str) -> dict:
    """Pick best model for NER (uses CASED models - capitalization matters!)."""
    wants_speed = any(w in user_description.lower() for w in ["fast", "quick", "speed", "efficient"])
    
    if wants_speed and num_rows < 2000:
        return {
            "model": "distilbert-base-cased", 
            "method": "full",
            "reasoning": "Fast cased model for NER with speed priority"
        }
    if num_rows < 1000:
        return {
            "model": "distilbert-base-cased", 
            "method": "full",
            "reasoning": "Small dataset - lightweight cased model"
        }
    elif num_rows < 5000:
        return {
            "model": "bert-base-cased", 
            "method": "full",
            "reasoning": "Medium dataset - BERT-cased for entity recognition"
        }
    else:
        return {
            "model": "roberta-base", 
            "method": "full",
            "reasoning": "Large dataset - RoBERTa for best NER quality"
        }


def get_instruction_model(num_rows: int, user_description: str) -> dict:
    """Pick best model for instruction tuning (always QLoRA - memory efficient)."""
    wants_quality = any(w in user_description.lower() for w in ["accurate", "best", "quality", "production"])
    
    if num_rows < 5000 and not wants_quality:
        return {
            "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
            "method": "qlora",
            "reasoning": "TinyLlama with QLoRA - efficient for instruction tuning"
        }
    else:
        return {
            "model": "meta-llama/Llama-2-7b-chat-hf", 
            "method": "qlora",
            "reasoning": "Llama-2 7B with QLoRA - best quality for instruction tuning"
        }


def get_model_recommendation(task_type: str, num_rows: int, user_description: str) -> dict:
    """
    Master function: Returns {model, method, reasoning} for any task.
    
    Args:
        task_type: "classification", "ner", or "instruction_tuning"
        num_rows: Dataset size
        user_description: User's description (used to detect speed/quality preference)
    
    Returns:
        dict with keys: model, method, reasoning
    
    Rules:
    - BERT-sized models (66M-184M) -> Full fine-tuning (fits in consumer GPUs)
    - LLMs (1.1B-7B) -> QLoRA (makes large models accessible)
    - NER uses CASED models (capitalization matters for entities)
    """
    if task_type == "classification":
        return get_classification_model(num_rows, user_description)
    elif task_type == "ner":
        return get_ner_model(num_rows, user_description)
    elif task_type == "instruction_tuning":
        return get_instruction_model(num_rows, user_description)
    else:
        raise ValueError(f"Unknown task type: {task_type}")

