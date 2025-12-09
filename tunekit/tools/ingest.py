"""
Ingest Data Tool
================
Load CSV, JSON, or JSONL files with encoding fallback.
"""

import json
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from tunekit.state import TuneKitState


# Supported encodings (in order of preference)
ENCODINGS = ["utf-8", "utf-8-sig", "latin-1", "cp1252"]


def _detect_encoding(file_path: str) -> str:
    """Detect working encoding by trying each one."""
    for enc in ENCODINGS:
        try:
            with open(file_path, "r", encoding=enc) as f:
                f.read(1024)  # Test with small chunk
            return enc
        except UnicodeDecodeError:
            continue
    raise ValueError("Unsupported encoding. Please re-save as UTF-8.")


def ingest_data(state: "TuneKitState") -> dict:
    """
    Load CSV, JSON, or JSONL files with encoding fallback.
    
    Inputs (from state):
        - file_path: str
    
    Outputs (to state):
        - raw_data: List[Dict]
        - columns: List[str]
        - sample_rows: List[Dict]
        - num_rows: int
        - error_msg: str (if failed)
    """
    file_path = state["file_path"]
    
    def err(msg): 
        return {
            "error_msg": msg, 
            "raw_data": None, 
            "columns": None, 
            "sample_rows": None, 
            "num_rows": None
        }
    
    try:
        path = Path(file_path)
        if not path.exists():
            return err(f"File not found: {file_path}")
        
        ext = path.suffix.lower()
        
        # --- CSV ---
        if ext == ".csv":
            enc = _detect_encoding(file_path)
            df = pd.read_csv(file_path, encoding=enc)
            raw_data = df.to_dict(orient="records")
            columns = df.columns.tolist()
        
        # --- JSONL (one object per line) ---
        elif ext == ".jsonl":
            enc = _detect_encoding(file_path)
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            raw_data = []
            for i, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if line:
                    try:
                        raw_data.append(json.loads(line))
                    except json.JSONDecodeError as e:
                        return err(f"Invalid JSON on line {i}: {e.msg}")
            columns = list(raw_data[0].keys()) if raw_data else []
        
        # --- JSON (array of objects) ---
        elif ext == ".json":
            enc = _detect_encoding(file_path)
            with open(file_path, "r", encoding=enc) as f:
                content = f.read()
            data = json.loads(content)
            if isinstance(data, list):
                raw_data = data
            elif isinstance(data, dict):
                raw_data = [data]
            else:
                return err("JSON must be an array or object")
            columns = list(raw_data[0].keys()) if raw_data else []
        
        else:
            return err(f"Unsupported: {ext}. Use .csv, .json, or .jsonl")
        
        if not raw_data:
            return err("File is empty")
        
        num_rows = len(raw_data)
        
        # ===== SANITY CHECKS (fail fast before LLM calls) =====
        if num_rows < 20:
            return err(f"Dataset too small ({num_rows} rows). Need at least 20 rows for fine-tuning.")
        if len(columns) < 2:
            return err(f"Need at least 2 columns (e.g., text + label). Found only {len(columns)}.")
        
        print(f"✅ Ingested {num_rows} rows, {len(columns)} columns")
        print(f"📊 Columns: {columns}")
        
        return {
            "raw_data": raw_data,
            "columns": columns,
            "sample_rows": raw_data[:5],
            "num_rows": num_rows,
            "error_msg": None
        }
        
    except json.JSONDecodeError as e:
        return err(f"Invalid JSON: {e}")
    except pd.errors.EmptyDataError:
        return err("CSV is empty")
    except ValueError as e:
        return err(str(e))
    except Exception as e:
        return err(f"Failed to load: {e}")

