"""
OpenGE: A crop trait prediction library with G×E interaction modeling.
Supports multi-crop adaptation, interpretable models, and sparse transformers.
"""

__version__ = "0.1.0"
__author__ = "OpenGE Contributors"

from .core.config import Config
from .core.registry import encoder_registry, fusion_registry, model_registry
from .core.engine import Trainer

__all__ = [
    "Config",
    "encoder_registry",
    "fusion_registry",
    "model_registry",
    "Trainer",
]
