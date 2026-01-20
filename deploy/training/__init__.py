"""
Deploy Training Module
=======================
Colab notebook generation for SLM fine-tuning.
"""

from .colab_generator import (
    generate_training_notebook,
    save_notebook,
)

__all__ = [
    "generate_training_notebook",
    "save_notebook",
]
