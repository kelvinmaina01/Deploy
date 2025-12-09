"""
Package Generator Tool
======================
Generates a complete, ready-to-run training package with 5 files.
"""

import json
import os
from datetime import datetime
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from tunekit.state import TuneKitState


# ============================================================================
# TEMPLATE: train.py for CLASSIFICATION
# ============================================================================

TRAIN_CLASSIFICATION = '''"""
TuneKit Generated Training Script - Classification
===================================================
Run: python train.py
"""

import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
import numpy as np

# Load config
with open("config.json") as f:
    config = json.load(f)

print(f"Task: {config['task_type']}")
print(f"Model: {config['base_model']}")
print(f"Data: {config['data_path']}")

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
model = AutoModelForSequenceClassification.from_pretrained(
    config["base_model"],
    num_labels=config["num_labels"],
)

# Load and prepare dataset
dataset = load_dataset("csv", data_files=config["data_path"], split="train")
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

tokenized = dataset.map(tokenize_function, batched=True, remove_columns=dataset["train"].column_names)

# Metrics
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, predictions),
        "f1": f1_score(labels, predictions, average="weighted"),
    }

# Training arguments
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    num_train_epochs=config["num_epochs"],
    warmup_ratio=config["warmup_ratio"],
    weight_decay=config["weight_decay"],
    evaluation_strategy=config["evaluation_strategy"],
    save_strategy=config["save_strategy"],
    load_best_model_at_end=True,
    metric_for_best_model=config["metric_for_best_model"],
    push_to_hub=False,
    logging_steps=10,
    report_to="none",
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("\\nStarting training...")
trainer.train()

# Save
trainer.save_model(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])
print(f"\\nModel saved to {config['output_dir']}")

# Final evaluation
results = trainer.evaluate()
print(f"\\nFinal Results: {results}")
'''


# ============================================================================
# TEMPLATE: train.py for NER
# ============================================================================

TRAIN_NER = '''"""
TuneKit Generated Training Script - Named Entity Recognition
=============================================================
Run: python train.py
"""

import json
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import f1_score, precision_score, recall_score
import numpy as np

# Load config
with open("config.json") as f:
    config = json.load(f)

print(f"Task: {config['task_type']}")
print(f"Model: {config['base_model']}")
print(f"Data: {config['data_path']}")

# Label list
label_list = config["label_list"]
label2id = {label: i for i, label in enumerate(label_list)}
id2label = {i: label for label, i in label2id.items()}

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(config["base_model"])
model = AutoModelForTokenClassification.from_pretrained(
    config["base_model"],
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

# Load dataset
dataset = load_dataset("csv", data_files=config["data_path"], split="train")
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

# Metrics
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

# Training arguments
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    num_train_epochs=config["num_epochs"],
    warmup_ratio=config["warmup_ratio"],
    weight_decay=config["weight_decay"],
    evaluation_strategy=config["evaluation_strategy"],
    save_strategy=config["save_strategy"],
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    push_to_hub=False,
    logging_steps=10,
    report_to="none",
)

# Train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["test"],
    tokenizer=tokenizer,
    data_collator=DataCollatorForTokenClassification(tokenizer=tokenizer),
    compute_metrics=compute_metrics,
)

print("\\nStarting training...")
trainer.train()

# Save
trainer.save_model(config["output_dir"])
tokenizer.save_pretrained(config["output_dir"])
print(f"\\nModel saved to {config['output_dir']}")

# Final evaluation
results = trainer.evaluate()
print(f"\\nFinal Results: {results}")
'''


# ============================================================================
# TEMPLATE: train.py for INSTRUCTION TUNING (QLoRA)
# ============================================================================

TRAIN_INSTRUCTION = '''"""
TuneKit Generated Training Script - Instruction Tuning (QLoRA)
===============================================================
Run: python train.py
"""

import json
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

# Load config
with open("config.json") as f:
    config = json.load(f)

print(f"Task: {config['task_type']}")
print(f"Model: {config['base_model']}")
print(f"Data: {config['data_path']}")

# QLoRA config
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
    r=config["lora_r"],
    lora_alpha=config["lora_alpha"],
    lora_dropout=config["lora_dropout"],
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
dataset = load_dataset("csv", data_files=config["data_path"], split="train")

# Format as chat
def format_instruction(example):
    instruction = example[config["instruction_column"]]
    response = example[config["response_column"]]
    return {"text": f"### Instruction:\\n{instruction}\\n\\n### Response:\\n{response}"}

dataset = dataset.map(format_instruction)
dataset = dataset.train_test_split(test_size=0.1, seed=42)

# Training arguments
training_args = TrainingArguments(
    output_dir=config["output_dir"],
    learning_rate=config["learning_rate"],
    per_device_train_batch_size=config["batch_size"],
    per_device_eval_batch_size=config["batch_size"],
    num_train_epochs=config["num_epochs"],
    warmup_ratio=config["warmup_ratio"],
    weight_decay=config["weight_decay"],
    evaluation_strategy=config["evaluation_strategy"],
    save_strategy=config["save_strategy"],
    gradient_accumulation_steps=config["gradient_accumulation_steps"],
    fp16=True,
    logging_steps=10,
    report_to="none",
)

# Train with SFTTrainer
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["test"],
    tokenizer=tokenizer,
    dataset_text_field="text",
    max_seq_length=config["max_length"],
)

print("\\nStarting training...")
trainer.train()

# Save LoRA adapter
trainer.save_model(config["output_dir"])
print(f"\\nLoRA adapter saved to {config['output_dir']}")

# Merge and save full model (optional)
# merged_model = model.merge_and_unload()
# merged_model.save_pretrained(config["output_dir"] + "_merged")
'''


# ============================================================================
# TEMPLATE: eval.py
# ============================================================================

EVAL_SCRIPT = '''"""
TuneKit Generated Evaluation Script
====================================
Run: python eval.py
"""

import json
from transformers import pipeline

# Load config
with open("config.json") as f:
    config = json.load(f)

print(f"Loading model from {config['output_dir']}...")

# Create pipeline based on task
if config["task_type"] == "classification":
    pipe = pipeline("text-classification", model=config["output_dir"])
    
    # Test examples
    test_texts = [
        "This is amazing! I love it!",
        "Terrible product, waste of money.",
        "It's okay, nothing special.",
    ]
    
    print("\\nTest Predictions:")
    for text in test_texts:
        result = pipe(text)
        print(f"  '{text[:50]}...' -> {result[0]['label']} ({result[0]['score']:.2%})")

elif config["task_type"] == "ner":
    pipe = pipeline("ner", model=config["output_dir"], aggregation_strategy="simple")
    
    test_texts = [
        "John Smith works at Google in New York.",
        "Apple CEO Tim Cook announced the new iPhone.",
    ]
    
    print("\\nTest Predictions:")
    for text in test_texts:
        result = pipe(text)
        print(f"  '{text}'")
        for entity in result:
            print(f"    -> {entity['word']}: {entity['entity_group']}")

elif config["task_type"] == "instruction_tuning":
    from transformers import AutoTokenizer, AutoModelForCausalLM
    
    tokenizer = AutoTokenizer.from_pretrained(config["output_dir"])
    model = AutoModelForCausalLM.from_pretrained(config["output_dir"])
    
    test_prompts = [
        "What is machine learning?",
        "Explain gradient descent in simple terms.",
    ]
    
    print("\\nTest Generations:")
    for prompt in test_prompts:
        formatted = f"### Instruction:\\n{prompt}\\n\\n### Response:\\n"
        inputs = tokenizer(formatted, return_tensors="pt")
        outputs = model.generate(**inputs, max_new_tokens=100, temperature=0.7)
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"  Q: {prompt}")
        print(f"  A: {response.split('### Response:')[-1].strip()[:200]}...")
        print()

print("\\nEvaluation complete!")
'''


# ============================================================================
# TEMPLATE: requirements.txt
# ============================================================================

def _generate_requirements(task_type: str) -> str:
    """Generate requirements.txt based on task type."""
    base = [
        "torch>=2.0.0",
        "transformers>=4.36.0",
        "datasets>=2.14.0",
        "accelerate>=0.24.0",
        "scikit-learn>=1.3.0",
    ]
    
    if task_type == "ner":
        base.append("seqeval>=1.2.2")
    
    if task_type == "instruction_tuning":
        base.extend([
            "peft>=0.7.0",
            "trl>=0.7.0",
            "bitsandbytes>=0.41.0",
        ])
    
    return "\n".join(base)


# ============================================================================
# TEMPLATE: README.md
# ============================================================================

def _generate_readme(state: "TuneKitState") -> str:
    """Generate README with instructions and context."""
    config = state.get("training_config", {})
    task_type = state.get("final_task_type", "unknown")
    model = state.get("base_model", "unknown")
    reasoning = state.get("planning_reasoning", "")
    
    readme = f'''# TuneKit Training Package

## Overview
This training package was automatically generated by **TuneKit** based on your dataset and requirements.

- **Task Type:** {task_type}
- **Base Model:** {model}
- **Generated:** {datetime.now().strftime("%Y-%m-%d %H:%M")}

## Agent's Reasoning
> {reasoning}

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Training
```bash
python train.py
```

### 3. Evaluate Model
```bash
python eval.py
```

## Files Included

| File | Description |
|------|-------------|
| `config.json` | Training configuration (hyperparameters, columns, etc.) |
| `train.py` | Complete training script |
| `eval.py` | Evaluation and inference script |
| `requirements.txt` | Python dependencies |
| `README.md` | This file |

## Configuration

```json
{json.dumps(config, indent=2)}
```

## Training Details

- **Learning Rate:** {config.get("learning_rate", "N/A")}
- **Batch Size:** {config.get("batch_size", "N/A")}
- **Epochs:** {config.get("num_epochs", "N/A")}
- **Max Length:** {config.get("max_length", "N/A")}
'''
    
    if task_type == "instruction_tuning":
        readme += f'''
## QLoRA Settings
- **LoRA Rank (r):** {config.get("lora_r", 8)}
- **LoRA Alpha:** {config.get("lora_alpha", 16)}
- **LoRA Dropout:** {config.get("lora_dropout", 0.1)}
'''
    
    readme += '''
## Need Help?
- Check the [HuggingFace Transformers docs](https://huggingface.co/docs/transformers)
- For QLoRA issues, see [PEFT documentation](https://huggingface.co/docs/peft)

---
*Generated by TuneKit - Automated Fine-Tuning Pipeline*
'''
    
    return readme


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def generate_package(state: "TuneKitState") -> dict:
    """
    Generate a complete training package with 5 files.
    
    Inputs (from state):
        - final_task_type: str
        - base_model: str
        - training_config: dict
        - planning_reasoning: str
        - file_path: str (original data path)
    
    Outputs (to state):
        - package_path: str (path to generated package folder)
    
    Generated files:
        - config.json
        - train.py
        - eval.py
        - requirements.txt
        - README.md
    """
    task_type = state["final_task_type"]
    config = state["training_config"].copy()
    
    # Add data path to config
    config["data_path"] = os.path.abspath(state["file_path"])
    
    # Create output folder
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    package_name = f"{task_type}_package_{timestamp}"
    package_path = os.path.join("output", package_name)
    os.makedirs(package_path, exist_ok=True)
    
    # 1. Generate config.json
    config_path = os.path.join(package_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    
    # 2. Generate train.py (task-specific)
    if task_type == "classification":
        train_script = TRAIN_CLASSIFICATION
    elif task_type == "ner":
        train_script = TRAIN_NER
    elif task_type == "instruction_tuning":
        train_script = TRAIN_INSTRUCTION
    else:
        raise ValueError(f"Unknown task type: {task_type}")
    
    train_path = os.path.join(package_path, "train.py")
    with open(train_path, "w") as f:
        f.write(train_script)
    
    # 3. Generate eval.py
    eval_path = os.path.join(package_path, "eval.py")
    with open(eval_path, "w") as f:
        f.write(EVAL_SCRIPT)
    
    # 4. Generate requirements.txt
    req_path = os.path.join(package_path, "requirements.txt")
    with open(req_path, "w") as f:
        f.write(_generate_requirements(task_type))
    
    # 5. Generate README.md
    readme_path = os.path.join(package_path, "README.md")
    with open(readme_path, "w") as f:
        f.write(_generate_readme(state))
    
    print(f"\n{'='*60}")
    print("TRAINING PACKAGE GENERATED")
    print(f"{'='*60}")
    print(f"Location: {package_path}/")
    print(f"\nFiles created:")
    print(f"  - config.json      (training configuration)")
    print(f"  - train.py         ({task_type} training script)")
    print(f"  - eval.py          (evaluation script)")
    print(f"  - requirements.txt (dependencies)")
    print(f"  - README.md        (instructions)")
    print(f"\nTo train locally:")
    print(f"  cd {package_path}")
    print(f"  pip install -r requirements.txt")
    print(f"  python train.py")
    print(f"{'='*60}")
    
    return {
        "package_path": package_path,
    }

