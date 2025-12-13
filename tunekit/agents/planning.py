"""
Planning Agent
==============
Agent 1: LLM decides task type + column mapping.
Rules decide: model + method + training config.
"""

import os
from typing import TYPE_CHECKING

from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

from tunekit.schemas import AgentDecision
from tunekit.tools.model_rec import get_model_recommendation
from tunekit.tools.config import get_training_config

if TYPE_CHECKING:
    from tunekit.state import TuneKitState


# Load environment variables
load_dotenv()

# Initialize LangChain ChatOpenAI
llm = ChatOpenAI(model="gpt-4o", temperature=0)


def planning_agent(state: "TuneKitState") -> dict:
    """
    Agent 1: LLM decides task type + columns. Rules decide model + config.
    
    Inputs (from state):
        - user_description: str
        - inferred_task_type: str
        - task_confidence: float
        - num_rows: int
        - num_classes: int (optional)
        - column_candidates: Dict
        - sample_rows: List[Dict]
        - label_list: List[str] (optional)
    
    Outputs (to state):
        - final_task_type: str
        - base_model: str
        - training_config: Dict
        - planning_reasoning: str
    """
    
    # Extract inputs
    user_description = state["user_description"]
    inferred_task_type = state["inferred_task_type"]
    task_confidence = state["task_confidence"]
    num_rows = state["num_rows"]
    num_classes = state.get("num_classes")
    column_candidates = state.get("column_candidates", {})
    sample_rows = state.get("sample_rows", [])[:3]
    
    # Format candidates and samples
    candidates_str = "\n".join(f"  {role}: {cols}" for role, cols in column_candidates.items() if cols)
    sample_str = "\n".join(str(row) for row in sample_rows)
    
    # Build prompt
    prompt = f"""You are a Senior Machine Learning Engineer helping configure a fine-tuning job.

## CONTEXT
A user wants to fine-tune a language model. Your job is to:
1. Determine the correct task type based on their request and the data
2. Map the dataset columns to the appropriate roles for training

## USER'S REQUEST
"{user_description}"

## DATASET ANALYSIS
- Total rows: {num_rows}
- Auto-detected task type: {inferred_task_type} (confidence: {task_confidence:.0%})
- Number of unique labels/classes: {num_classes if num_classes else 'Not detected'}

## AVAILABLE COLUMNS
The following columns were detected in the dataset:
{candidates_str}

## SAMPLE DATA (First 3 rows)
{sample_str}

## TASK TYPE DEFINITIONS

**classification**: Assign a category/label to input text
- Examples: sentiment analysis, spam detection, topic classification, intent detection
- Required columns: text_column (input text), label_column (category/label)

**ner** (Named Entity Recognition): Extract and tag entities from text
- Examples: extracting names, locations, organizations, dates from text
- Required columns: text_column (input text), tags_column (BIO tags like B-PER, I-LOC, O)

**instruction_tuning**: Train model to follow instructions and generate responses
- Examples: chatbots, Q&A systems, instruction-following assistants
- Required columns: instruction_column (user prompt/question), response_column (expected output)

## YOUR DECISIONS

1. **final_task_type**: Choose - classification, ner, or instruction_tuning
   - Consider the user's request first - what are they trying to achieve?
   - If auto-detected type has high confidence (>80%) and aligns with user intent, use it
   - If there's a conflict, prioritize the user's stated goal

2. **Column Mapping**: Select the appropriate columns for the chosen task type
   - IMPORTANT: Use the EXACT column names from the candidates above
   - Pick the column that best matches each role based on the sample data
   - For classification: set text_column and label_column
   - For NER: set text_column and tags_column
   - For instruction_tuning: set instruction_column and response_column

3. **reasoning**: Write a clear, professional explanation (2-3 concise sentences)
   - First sentence: State the chosen task type and why it matches the user's goal
   - Second sentence: Identify the selected columns and their roles
   - Optional third sentence: Note any important considerations (e.g., class balance, data quality)
   - Be direct and technical - avoid filler words or repetition
   - Example format: "Selected classification task based on user's goal to categorize text. Using 'text' column as input and 'label' column for categories. Dataset contains 2 balanced classes suitable for training." """

    # Call LLM
    structured_llm = llm.with_structured_output(AgentDecision)
    decision = structured_llm.invoke(prompt)
    
    # Log
    print(f"Task: {decision.final_task_type}")
    for col in ["text_column", "label_column", "instruction_column", "response_column", "tags_column"]:
        if getattr(decision, col, None):
            print(f"   {col}: {getattr(decision, col)}")
    print(f"   Reasoning: {decision.reasoning[:80]}...")
    
    # Get model (rule-based)
    model_rec = get_model_recommendation(decision.final_task_type, num_rows, user_description)
    model, method = model_rec["model"], model_rec["method"]
    print(f"Model: {model} ({method})")
    
    # Build columns dict (only non-None values)
    columns = {
        "text_column": decision.text_column,
        "label_column": decision.label_column,
        "instruction_column": decision.instruction_column,
        "response_column": decision.response_column,
        "tags_column": decision.tags_column,
        "num_labels": num_classes,
        "label_list": state.get("label_list"),
    }
    columns = {k: v for k, v in columns.items() if v is not None}
    
    # Generate config
    config = get_training_config(
        task_type=decision.final_task_type,
        model_name=model,
        num_rows=num_rows,
        columns=columns,
    )
    print(f"Config: {len(config)} fields")
    
    return {
        "final_task_type": decision.final_task_type,
        "base_model": model,
        "training_config": config,
        "planning_reasoning": decision.reasoning,
    }

