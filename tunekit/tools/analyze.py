"""
Analyze Dataset Tool
====================
Smart dataset analysis that detects task type and finds column candidates.
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tunekit.state import TuneKitState


def _analyze_column(values: list) -> dict:
    """Analyze column values and return type + stats."""
    if not values:
        return {"type": "empty", "unique_count": 0, "avg_length": 0}
    
    # Handle list columns (NER tokens)
    if isinstance(values[0], list):
        flat = [s for v in values[:10] for s in v if isinstance(s, str)]
        has_bio = any("B-" in s or "I-" in s or s == "O" for s in flat)
        return {"type": "bio_tags" if has_bio else "list", "unique_count": 0, "avg_length": 0}
    
    # Convert to strings
    str_vals = [str(v) for v in values if v is not None]
    if not str_vals:
        return {"type": "empty", "unique_count": 0, "avg_length": 0}
    
    unique_count = len(set(str_vals))
    unique_ratio = unique_count / len(str_vals)
    avg_length = sum(len(v) for v in str_vals) / len(str_vals)
    
    # Determine type
    if any("B-" in v or "I-" in v for v in str_vals[:20]):
        col_type = "bio_tags"
    elif unique_ratio > 0.5 or avg_length > 50:
        col_type = "text"
    elif unique_count <= 30:
        col_type = "categorical"
    else:
        col_type = "text"
    
    return {"type": col_type, "unique_count": unique_count, "avg_length": avg_length}


def analyze_dataset(state: "TuneKitState") -> dict:
    """
    Detect task type and find column candidates for Agent 1.
    
    Inputs (from state):
        - raw_data: List[Dict]
        - columns: List[str]
        - num_rows: int
        - sample_rows: List[Dict]
    
    Outputs (to state):
        - inferred_task_type: str
        - task_confidence: float
        - num_classes: int (for classification)
        - label_list: List[str] (actual labels found)
        - column_analysis: Dict
        - column_candidates: Dict
        - dataset_stats: Dict
    """
    raw_data = state["raw_data"]
    columns = state["columns"]
    num_rows = state["num_rows"]
    
    if not raw_data or not columns:
        return {
            "inferred_task_type": None, 
            "task_confidence": 0.0, 
            "column_analysis": {}, 
            "column_candidates": {}
        }
    
    # Analyze each column
    col_info = {}
    for col in columns:
        values = [row.get(col) for row in raw_data if row.get(col) is not None]
        col_info[col] = _analyze_column(values)
    
    # Build lowercase lookup
    cols_lower = {c.lower(): c for c in columns}
    
    # --- Detect task type ---
    
    # 1. Instruction tuning (keyword match)
    inst_keys = ["instruction", "prompt", "input", "question", "query"]
    resp_keys = ["response", "completion", "output", "answer", "reply"]
    inst_cols = [cols_lower[k] for k in inst_keys if k in cols_lower]
    resp_cols = [cols_lower[k] for k in resp_keys if k in cols_lower]
    
    if inst_cols and resp_cols:
        task_type, confidence = "instruction_tuning", 0.9
        sys_cols = [cols_lower.get(k) for k in ["system", "system_prompt"]]
        candidates = {
            "instruction_column": inst_cols,
            "response_column": resp_cols,
            "system_column": [c for c in sys_cols if c]
        }
    
    # 2. NER (BIO tags detected)
    elif any(col_info[c]["type"] == "bio_tags" for c in columns):
        task_type, confidence = "ner", 0.85
        candidates = {
            "text_column": [c for c in columns if col_info[c]["type"] in ["text", "list"]] or [columns[0]],
            "tags_column": [c for c in columns if col_info[c]["type"] == "bio_tags"]
        }
    
    # 3. Classification (default)
    else:
        task_type = "classification"
        text_cols = [c for c in columns if col_info[c]["type"] == "text"]
        label_cols = [c for c in columns if col_info[c]["type"] == "categorical"]
        
        # Sort: longest text first, fewest unique labels first
        text_cols.sort(key=lambda c: col_info[c]["avg_length"], reverse=True)
        label_cols.sort(key=lambda c: col_info[c]["unique_count"])
        
        confidence = 0.8 if (text_cols and label_cols) else 0.4
        candidates = {"text_column": text_cols, "label_column": label_cols}
    
    # Count classes for classification (filter out None/empty)
    num_classes = None
    label_list = None
    if task_type == "classification" and candidates.get("label_column"):
        label_col = candidates["label_column"][0]
        labels = set()
        for row in raw_data:
            val = row.get(label_col)
            if val is not None and str(val).strip():
                labels.add(str(val).strip())
        if labels:
            num_classes = len(labels)
            label_list = sorted(list(labels))
    
    print(f"Analysis: {task_type} (confidence: {confidence:.0%})")
    print(f"Candidates: {candidates}")
    
    return {
        "inferred_task_type": task_type,
        "task_confidence": confidence,
        "num_classes": num_classes,
        "label_list": label_list,
        "column_analysis": col_info,
        "column_candidates": candidates,
        "dataset_stats": {
            "num_rows": num_rows,
            "num_columns": len(columns),
            "sample_rows": state.get("sample_rows", [])[:3]
        }
    }

