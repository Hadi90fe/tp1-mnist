# src/__init__.py

"""
src package for the MNIST TP1 project.

Contains:
- model: model architecture definition
- utils: helper functions (plots, logging, etc.)
- train: training script
- predict: prediction script
"""

from .model import create_model
from .utils import save_training_curves, log_training
