"""Data loading and preprocessing module."""

# Dataset classes
from .dataloaders import GxEDataset, GxEDataLoader

# Individual loaders
from .loaders import GeneticLoader, EnvironmentLoader, PhenotypeLoader

# Preprocessing utilities
from .preprocess import (
    Preprocessor,
    check_and_handle_missing,
    aggregate_temporal_to_static,
    aggregate_temporal_features,
    get_growth_stages,
    create_fixed_windows,
    add_custom_growth_stages,
    CROP_GROWTH_STAGES
)

__all__ = [
    # Dataset classes
    "GxEDataset",
    "GxEDataLoader",
    # Individual loaders
    "GeneticLoader",
    "EnvironmentLoader", 
    "PhenotypeLoader",
    # Preprocessing
    "Preprocessor",
    "check_and_handle_missing",
    "aggregate_temporal_to_static",
    "aggregate_temporal_features",
    "get_growth_stages",
    "create_fixed_windows",
    "add_custom_growth_stages",
    "CROP_GROWTH_STAGES"
]
