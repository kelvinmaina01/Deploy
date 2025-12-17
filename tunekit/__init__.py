"""
TuneKit - Automated LLM Fine-Tuning Pipeline
=============================================

A LangGraph-powered workflow that automates the entire fine-tuning process:
- Ingest data (CSV, JSON, JSONL)
- Validate quality
- Analyze dataset and detect task type
- Select optimal model and training config
- (Coming soon) Estimate cost, human review, training, monitoring
"""

from .state import TuneKitState
from .schemas import AgentDecision
from .tools import (
    ingest_data,
    validate_quality,
    analyze_dataset,
    recommend_model,
    get_training_config,
    generate_package,
)
from .agents import planning_agent

__version__ = "0.1.0"

__all__ = [
    # State
    "TuneKitState",
    # Schemas
    "AgentDecision",
    # Tools
    "ingest_data",
    "validate_quality",
    "analyze_dataset",
    "recommend_model",
    "get_training_config",
    "generate_package",
    # Agents
    "planning_agent",
]
