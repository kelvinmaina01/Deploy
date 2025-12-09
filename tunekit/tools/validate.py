"""
Validate Quality Tool
=====================
Quality gate BEFORE analysis/agent. Blocks bad data early.
Decision: score >= 0.6 -> proceed | score < 0.6 -> human review
"""

from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tunekit.state import TuneKitState


def validate_quality(state: "TuneKitState") -> dict:
    """
    Quality gate BEFORE analysis/agent. Blocks bad data early.
    
    Inputs (from state):
        - raw_data: List[Dict]
        - columns: List[str]
        - num_rows: int
    
    Outputs (to state):
        - quality_score: float (0.0 to 1.0)
        - quality_issues: List[str]
    
    Decision: score >= 0.6 -> proceed | score < 0.6 -> human review
    """
    raw_data = state["raw_data"]
    columns = state["columns"]
    num_rows = state["num_rows"]
    
    if not raw_data:
        return {"quality_score": 0.0, "quality_issues": ["No data available"]}
    
    score = 1.0
    issues = []
    
    # --- Check 1: Duplicates ---
    seen = set()
    duplicates = 0
    for row in raw_data:
        row_tuple = tuple(sorted(row.items()))
        if row_tuple in seen:
            duplicates += 1
        else:
            seen.add(row_tuple)
    
    if duplicates > 0:
        dup_ratio = duplicates / num_rows
        score -= min(dup_ratio * 0.3, 0.3)
        issues.append(f"{duplicates} duplicate rows ({dup_ratio:.0%})")
    
    # --- Check 2: Missing Values (per column) ---
    missing_by_col = {col: 0 for col in columns}
    for row in raw_data:
        for col in columns:
            val = row.get(col)
            if val is None or val == "" or (isinstance(val, float) and pd.isna(val)):
                missing_by_col[col] += 1
    
    total_missing = sum(missing_by_col.values())
    if total_missing > 0:
        miss_ratio = total_missing / (num_rows * len(columns))
        score -= min(miss_ratio * 0.5, 0.3)
        
        # Report which columns have missing values
        cols_with_missing = {col: count for col, count in missing_by_col.items() if count > 0}
        issues.append(f"{total_missing} missing values found:")
        for col, count in sorted(cols_with_missing.items(), key=lambda x: x[1], reverse=True):
            issues.append(f"  -> '{col}': {count} missing ({count/num_rows:.0%} of rows)")
    
    # --- Check 3: Dataset Size ---
    if num_rows < 50:
        score -= 0.4
        issues.append(f"Very small dataset ({num_rows} rows) - need 50+")
    elif num_rows < 100:
        score -= 0.2
        issues.append(f"Small dataset ({num_rows} rows) - 100+ recommended")
    
    # --- Finalize ---
    score = max(0.0, min(1.0, score))
    
    if score >= 0.8:
        level = "excellent"
    elif score >= 0.6:
        level = "acceptable"
    else:
        level = "poor"
    
    print(f"Quality: {score:.2f} ({level})")
    if issues:
        for issue in issues:
            print(f"   {issue}")
    
    return {
        "quality_score": score,
        "quality_issues": issues if issues else ["Data looks good!"]
    }

