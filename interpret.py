#!/usr/bin/env python
"""
Model interpretability and explanation script.

Usage:
    python interpret.py --model-path results/best_model.pt --config configs/maize_2024.yaml --data-dir ./data
"""

import argparse
import logging
from pathlib import Path

import torch
import numpy as np

from openge.core import Config
from openge.data import CropDataset
from openge.interpretability import (
    AttentionAnalyzer,
    FeatureImportance,
    GradientExplainer,
    SparsityAnalyzer,
)


def setup_logging(log_level: str = "INFO") -> logging.Logger:
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    return logging.getLogger(__name__)


def analyze_attention(model, data, output_dir: Path):
    """Analyze attention mechanisms."""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing attention mechanisms...")
    
    analyzer = AttentionAnalyzer(model)
    attention_heads = analyzer.extract_attention_heads()
    head_importance = analyzer.analyze_head_importance()
    
    logger.info(f"Found {len(attention_heads)} attention layers")
    logger.info(f"Head importance scores: {head_importance}")
    
    return attention_heads, head_importance


def analyze_feature_importance(model, data_loader, output_dir: Path):
    """Analyze feature importance."""
    logger = logging.getLogger(__name__)
    logger.info("Computing feature importance...")
    
    importance_analyzer = FeatureImportance(model)
    
    # Collect data for importance analysis
    X_genetic = []
    X_env = []
    y = []
    
    for genetic_data, env_data, targets in data_loader:
        X_genetic.append(genetic_data.numpy())
        X_env.append(env_data.numpy())
        y.append(targets.numpy())
    
    X_genetic = np.concatenate(X_genetic)
    X_env = np.concatenate(X_env)
    y = np.concatenate(y)
    
    # Compute G vs E contribution
    genetic_importance, env_importance = importance_analyzer.genetic_vs_environment_contribution(
        X_genetic, X_env
    )
    
    logger.info(f"Genetic importance: {genetic_importance:.4f}")
    logger.info(f"Environment importance: {env_importance:.4f}")
    
    return {"genetic": genetic_importance, "environment": env_importance}


def analyze_gradients(model, data, output_dir: Path):
    """Analyze gradient-based explanations."""
    logger = logging.getLogger(__name__)
    logger.info("Computing gradient-based explanations...")
    
    explainer = GradientExplainer(model)
    
    # Get sample data
    genetic_data, env_data, _ = next(iter(data))
    
    # Compute integrated gradients
    genetic_data = genetic_data.unsqueeze(0)  # Add batch dimension
    attr = explainer.integrated_gradients(genetic_data, target=0)
    
    logger.info(f"Attribution shape: {attr.shape}")
    logger.info(f"Top important features: {torch.topk(attr.abs(), k=5).indices.tolist()}")
    
    return attr


def analyze_sparsity(model, output_dir: Path):
    """Analyze model sparsity patterns."""
    logger = logging.getLogger(__name__)
    logger.info("Analyzing sparsity patterns...")
    
    analyzer = SparsityAnalyzer(model)
    sparsity_levels = analyzer.compute_sparsity_levels()
    
    logger.info("Sparsity levels by layer:")
    for layer_name, sparsity in sparsity_levels.items():
        logger.info(f"  {layer_name}: {sparsity:.2%}")
    
    return sparsity_levels


def main(args):
    """Main interpretability analysis function."""
    logger = setup_logging()
    logger.info("Starting model interpretability analysis...")
    
    # Load configuration
    config = Config(args.config)
    logger.info(f"Loaded configuration from {args.config}")
    
    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    # Model loading logic would go here
    model = checkpoint
    model.to(device)
    
    # Load data
    logger.info(f"Loading data from {args.data_dir}")
    dataset = CropDataset(args.data_dir, config.get("crop", "maize"))
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run different interpretability analyses
    if config.get("interpretability.enable_attention_analysis", True):
        attention_heads, head_importance = analyze_attention(model, dataset, output_dir)
    
    if config.get("interpretability.enable_feature_importance", True):
        # Create dummy data loader for feature importance
        # importance = analyze_feature_importance(model, data_loader, output_dir)
        logger.info("Feature importance analysis requires actual data loader")
    
    if config.get("interpretability.enable_shap", False):
        # importance = analyze_shap(model, data, output_dir)
        logger.info("SHAP analysis not enabled in config")
    
    if config.get("interpretability.enable_sparsity_analysis", False):
        sparsity_levels = analyze_sparsity(model, output_dir)
    
    # Gradient-based analysis
    # attr = analyze_gradients(model, data, output_dir)
    
    logger.info(f"Analysis complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze model interpretability and explainability"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/base.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data",
        help="Path to data directory"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./results/interpretability",
        help="Path to output directory"
    )
    
    args = parser.parse_args()
    main(args)
