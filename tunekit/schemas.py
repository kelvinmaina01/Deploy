"""
TuneKit Schemas
===============
Pydantic models for data validation and API responses.
"""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field


class Message(BaseModel):
    """A single message in a conversation."""
    role: str = Field(description="Role: system, user, or assistant")
    content: str = Field(description="Message content")


class Conversation(BaseModel):
    """A single training example (one JSONL line)."""
    messages: List[Message] = Field(description="List of messages in the conversation")


class DatasetStats(BaseModel):
    """Statistics about the uploaded dataset."""
    total_examples: int = Field(description="Number of conversations")
    total_messages: int = Field(description="Total messages across all conversations")
    avg_messages_per_example: float = Field(description="Average messages per conversation")
    has_system_prompts: bool = Field(description="Whether dataset includes system prompts")
    avg_conversation_length: int = Field(description="Average character length per conversation")


class ValidationResult(BaseModel):
    """Result of dataset validation."""
    is_valid: bool = Field(description="Whether the dataset passed validation")
    quality_score: float = Field(description="Quality score from 0.0 to 1.0")
    quality_issues: List[str] = Field(default=[], description="List of quality issues found")
    stats: Optional[DatasetStats] = Field(default=None, description="Dataset statistics")
    error_message: Optional[str] = Field(default=None, description="Error message if validation failed")


class ModelRecommendation(BaseModel):
    """Recommended model for fine-tuning."""
    model_id: str = Field(description="HuggingFace model ID")
    model_name: str = Field(description="Human-readable model name")
    model_size: str = Field(description="Model size (e.g., '1.5B', '3.8B')")
    reason: str = Field(description="Why this model was recommended")
    estimated_time: str = Field(description="Estimated training time")
    tier: str = Field(description="Model tier: lite, balanced, smart, pro")


class TrainingConfig(BaseModel):
    """Configuration for fine-tuning."""
    base_model: str = Field(description="Base model to fine-tune")
    learning_rate: float = Field(default=2e-4, description="Learning rate")
    num_epochs: int = Field(default=3, description="Number of training epochs")
    batch_size: int = Field(default=4, description="Batch size")
    max_seq_length: int = Field(default=2048, description="Maximum sequence length")
    
    # LoRA config
    lora_r: int = Field(default=16, description="LoRA rank")
    lora_alpha: int = Field(default=32, description="LoRA alpha")
    lora_dropout: float = Field(default=0.05, description="LoRA dropout")
    
    # Output
    output_dir: str = Field(default="./output", description="Output directory")


class TrainingPlan(BaseModel):
    """Complete training plan returned to the user."""
    model: ModelRecommendation = Field(description="Recommended model")
    config: TrainingConfig = Field(description="Training configuration")
    dataset_stats: DatasetStats = Field(description="Dataset statistics")
    quality_score: float = Field(description="Data quality score")
    estimated_cost: Optional[str] = Field(default=None, description="Estimated cloud training cost")
