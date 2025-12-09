"""
TuneKit Schemas
===============
Pydantic models for structured LLM outputs.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


class AgentDecision(BaseModel):
    """
    Structured output from the LLM.
    Agent decides: task type + which columns to use for training.
    """
    final_task_type: Literal["classification", "ner", "instruction_tuning"] = Field(
        description="The resolved task type"
    )
    
    # Column mapping (varies by task type)
    text_column: Optional[str] = Field(
        default=None,
        description="Column containing input text (for classification/NER)"
    )
    label_column: Optional[str] = Field(
        default=None,
        description="Column containing labels (for classification)"
    )
    instruction_column: Optional[str] = Field(
        default=None,
        description="Column containing instructions (for instruction tuning)"
    )
    response_column: Optional[str] = Field(
        default=None,
        description="Column containing responses (for instruction tuning)"
    )
    tags_column: Optional[str] = Field(
        default=None,
        description="Column containing NER tags (for NER)"
    )
    
    reasoning: str = Field(
        description="3-4 sentences explaining task type and column choices"
    )

