"""
Weight Atlas - LLM Fingerprint Visualization Tool
==================================================

Visualize LLMs like brain scans. Each model gets a unique 'fingerprint'
extracted from its weight distributions, spectral properties, and
attention patterns.

Usage:
    python -m src.model_atlas --model path/to/model --output ./output
    python -m src.model_atlas --demo  # Run with synthetic models
    python -m src.model_atlas --compare model_a model_b
"""

__version__ = "0.1.0"
