"""
Mini-Infer Model Training and Export Package

This package provides utilities for training and exporting models
to ONNX format for use with Mini-Infer inference engine.
"""

from .lenet5_model import LeNet5, print_model_summary

__version__ = "0.1.0"
__all__ = ["LeNet5", "print_model_summary"]
