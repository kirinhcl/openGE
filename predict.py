#!/usr/bin/env python
"""
Prediction script for crop trait prediction with G×E interactions.

Usage:
    python predict.py --model-path outputs/run_xxx/best_model.pt --data-dir ./Testing_data
    python predict.py --model-path outputs/run_xxx/best_model.pt --data-dir ./Testing_data --output-dir ./predictions
"""

import argparse
import logging
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from openge.core import Config
from openge.data import GxEDataLoader, Preprocessor, GxEDataset
from openge.models import (
    TransformerEncoder, MLPEncoder, CNNEncoder, TemporalEncoder,
    AttentionFusion, ConcatFusion,
    RegressionHead, GxEModel
)
from openge.utils.metrics import calculate_rmse, calculate_r2, calculate_mae, evaluate_model


def setup_logging(log_level: str = "INFO", log_file: str = None) -> logging.Logger:
    """Setup logging configuration."""
    handlers = [logging.StreamHandler()]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=handlers
    )
    return logging.getLogger(__name__)


def load_model(model_path: str, device: str, logger: logging.Logger):
    """
    Load trained model from checkpoint.
    
    Args:
        model_path: Path to model checkpoint (.pt file)
        device: Device to load model on
        logger: Logger instance
    
    Returns:
        model: Loaded PyTorch model
        checkpoint: Full checkpoint dict with config and data_info
    """
    logger.info(f"Loading model from {model_path}")
    
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract saved information
    data_info = checkpoint.get('data_info', {})
    config = checkpoint.get('config', {})
    
    logger.info(f"Model trained at epoch {checkpoint.get('epoch', 'unknown')}")
    logger.info(f"Validation R²: {checkpoint.get('val_r2', 'unknown'):.4f}")
    
    # Rebuild model architecture
    model = build_model_from_config(config, data_info, device, logger)
    
    # Load weights
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    logger.info("Model loaded successfully")
    
    return model, checkpoint


def build_model_from_config(config: dict, data_info: dict, device: str, logger: logging.Logger) -> nn.Module:
    """Rebuild model from saved configuration."""
    n_markers = data_info['n_markers']
    n_env_features = data_info['n_env_features']
    n_traits = data_info['n_traits']
    
    model_config = config.get('model', {})
    
    # Hidden dimensions
    genetic_hidden_dim = model_config.get('genetic_hidden_dim', 256)
    env_hidden_dim = model_config.get('env_hidden_dim', 64)
    fusion_dim = model_config.get('fusion_dim', 128)
    
    # Build genetic encoder
    genetic_encoder_type = model_config.get('genetic_encoder', 'mlp')
    logger.info(f"Building genetic encoder: {genetic_encoder_type}")
    
    if genetic_encoder_type == 'transformer':
        genetic_encoder = TransformerEncoder(
            input_dim=n_markers,
            output_dim=genetic_hidden_dim,
            n_heads=model_config.get('genetic_n_heads', 8),
            n_layers=model_config.get('genetic_n_layers', 2),
            dropout=model_config.get('dropout', 0.1)
        )
    elif genetic_encoder_type == 'cnn':
        genetic_encoder = CNNEncoder(
            input_dim=n_markers,
            output_dim=genetic_hidden_dim,
            hidden_channels=model_config.get('genetic_hidden_channels', [64, 128, 256]),
            kernel_sizes=model_config.get('genetic_kernel_sizes', [7, 5, 3])
        )
    elif genetic_encoder_type == 'cnn_transformer':
        # Hybrid CNN + Transformer encoder
        # CNN output dimension (intermediate dimension)
        cnn_output_dim = model_config.get('cnn_output_dim', genetic_hidden_dim)
        
        cnn_encoder = CNNEncoder(
            input_dim=n_markers,
            output_dim=cnn_output_dim,
            hidden_channels=model_config.get('genetic_hidden_channels', [64, 128, 256]),
            kernel_sizes=model_config.get('genetic_kernel_sizes', [7, 5, 3])
        )
        transformer_encoder = TransformerEncoder(
            input_dim=cnn_output_dim,
            output_dim=genetic_hidden_dim,
            n_heads=model_config.get('genetic_n_heads', 8),
            n_layers=model_config.get('genetic_n_layers', 2),
            dropout=model_config.get('dropout', 0.1)
        )
        genetic_encoder = nn.Sequential(cnn_encoder, transformer_encoder)
    else:  # MLP (default)
        genetic_encoder = MLPEncoder(
            input_dim=n_markers,
            hidden_dims=model_config.get('genetic_hidden_dims', [512, 256]),
            output_dim=genetic_hidden_dim,
            dropout=model_config.get('dropout', 0.1)
        )
    
    # Build environment encoder
    env_is_3d = data_info.get('env_is_3d', False)
    n_timesteps = data_info.get('n_timesteps', None)
    env_encoder_type = model_config.get('env_encoder', 'mlp')
    
    if env_is_3d and n_timesteps is not None:
        # Use TemporalEncoder for 3D weather time series
        logger.info(f"Using TemporalEncoder for 3D environment data ({n_timesteps} timesteps)")
        env_encoder = TemporalEncoder(
            n_features=n_env_features,
            n_timesteps=n_timesteps,
            output_dim=env_hidden_dim,
            hidden_dim=model_config.get('env_temporal_hidden_dim', 128),
            n_heads=model_config.get('env_n_heads', 4),
            n_layers=model_config.get('env_n_layers', 2),
            dropout=model_config.get('dropout', 0.1)
        )
    elif env_encoder_type == 'transformer':
        env_encoder = TransformerEncoder(
            input_dim=n_env_features,
            output_dim=env_hidden_dim,
            n_heads=model_config.get('env_n_heads', 4),
            n_layers=model_config.get('env_n_layers', 2),
            dropout=model_config.get('dropout', 0.1)
        )
    else:  # MLP (default for 2D)
        env_encoder = MLPEncoder(
            input_dim=n_env_features,
            hidden_dims=model_config.get('env_hidden_dims', [128, 64]),
            output_dim=env_hidden_dim,
            dropout=model_config.get('dropout', 0.1)
        )
    
    # Build fusion layer
    fusion_type = model_config.get('fusion', 'attention')
    if fusion_type == 'attention':
        fusion_layer = AttentionFusion(
            dim1=genetic_hidden_dim,
            dim2=env_hidden_dim
        )
        head_input_dim = genetic_hidden_dim
    else:
        fusion_layer = ConcatFusion(
            dim1=genetic_hidden_dim,
            dim2=env_hidden_dim,
            output_dim=fusion_dim
        )
        head_input_dim = fusion_dim
    
    # Build prediction head
    head = RegressionHead(
        input_dim=head_input_dim,
        n_traits=n_traits,
        hidden_dims=model_config.get('head_hidden_dims', [64])
    )
    
    # Residual connections
    use_residual = model_config.get('use_residual', False)
    
    model = GxEModel(
        genetic_encoder=genetic_encoder,
        env_encoder=env_encoder,
        fusion_layer=fusion_layer,
        prediction_head=head,
        use_residual=use_residual,
        genetic_dim=genetic_hidden_dim,
        env_dim=env_hidden_dim,
        fused_dim=head_input_dim,
    ).to(device)
    
    return model


def load_test_data(data_dir: str, checkpoint: dict, logger: logging.Logger):
    """
    Load and preprocess test data from Testing_data directory.
    
    Expected files:
    - 1_Submission_Template_2024.csv or 7_Testing_Observed_Values.csv (samples to predict)
    - 6_Testing_EC_Data_2024.csv (environmental covariates)
    - 3_Testing_Soil_Data_2024.csv (soil data)
    
    Note: Genotype data is loaded from Training_data (same hybrids)
    """
    logger.info(f"Loading test data from {data_dir}")
    
    data_path = Path(data_dir)
    data_info = checkpoint.get('data_info', {})
    
    # Define file paths for test data
    submission_file = data_path / "1_Submission_Template_2024.csv"
    observed_file = data_path / "7_Testing_Observed_Values.csv"
    ec_file = data_path / "6_Testing_EC_Data_2024.csv"
    soil_file = data_path / "3_Testing_Soil_Data_2024.csv"
    
    # Use observed values if available (for evaluation), otherwise submission template
    if observed_file.exists():
        test_pheno_file = observed_file
        has_targets = True
        logger.info("Using observed values file for evaluation")
    else:
        test_pheno_file = submission_file
        has_targets = False
        logger.info("Using submission template (no observed values)")
    
    # Genotype file is in Training_data (hybrids are the same)
    training_data_path = data_path.parent / "Training_data"
    genotype_file = training_data_path / "5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt"
    
    if not genotype_file.exists():
        raise FileNotFoundError(f"Genotype file not found: {genotype_file}")
    
    # Initialize data loader
    data_loader = GxEDataLoader()
    
    # 1. Load genetic data (same as training)
    logger.info("Loading genetic data...")
    genetic_data, genetic_ids, marker_names = data_loader.load_genetic(
        filepath=str(genotype_file),
        sample_col='<Marker>',
        handle_missing='mean',
        missing_threshold=0.5,
        encoding='keep'
    )
    logger.info(f"Genetic data shape: {genetic_data.shape}")
    
    # 2. Load test environment data
    logger.info("Loading test environment data...")
    try:
        env_data, env_ids, env_features = data_loader.load_environment(
            ec_file=str(ec_file),
            soil_file=str(soil_file) if soil_file.exists() else None,
            handle_missing='drop',
            use_3d=False
        )
    except Exception as e:
        logger.warning(f"Failed to load soil data, using EC only: {e}")
        env_data, env_ids, env_features = data_loader.load_environment(
            ec_file=str(ec_file),
            handle_missing='drop',
            use_3d=False
        )
    logger.info(f"Environment data shape: {env_data.shape}")
    
    # 3. Load test phenotype data (or submission template)
    logger.info("Loading test sample IDs...")
    pheno_data, pheno_ids, trait_names = data_loader.load_phenotype(
        filepath=str(test_pheno_file),
        traits=['Yield_Mg_ha'],
        sample_id_col='Hybrid',
        env_col='Env',
        handle_missing='keep' if not has_targets else 'drop',
        handle_outliers=False
    )
    logger.info(f"Test samples: {pheno_data.shape[0]}")
    
    # 4. Align samples
    logger.info("Aligning test samples...")
    aligned_genetic, aligned_env, aligned_pheno, aligned_ids = data_loader.align_samples(
        strategy='inner'
    )
    logger.info(f"Aligned test samples: {len(aligned_ids)}")
    
    # 5. Preprocess data (use same normalization as training)
    logger.info("Preprocessing data...")
    
    # Normalize genetic data
    genetic_preprocessor = Preprocessor(method='standard')
    aligned_genetic = genetic_preprocessor.normalize(aligned_genetic, fit=True)
    
    # Normalize environment data
    env_preprocessor = Preprocessor(method='standard')
    aligned_env = env_preprocessor.normalize(aligned_env, fit=True)
    
    # Store original phenotype values for de-normalization
    original_pheno = aligned_pheno.copy()
    
    # Normalize phenotype (if has targets)
    pheno_preprocessor = Preprocessor(method='standard')
    if has_targets:
        aligned_pheno_normalized = pheno_preprocessor.normalize(aligned_pheno, fit=True)
    else:
        aligned_pheno_normalized = aligned_pheno
    
    # Create dataset
    dataset = GxEDataset(
        genetic_data=aligned_genetic,
        environment_data=aligned_env,
        phenotype_data=aligned_pheno_normalized,
        sample_ids=aligned_ids,
        return_dict=False
    )
    
    # Print summary
    data_loader.print_summary()
    
    test_info = {
        'n_samples': len(aligned_ids),
        'sample_ids': aligned_ids,
        'has_targets': has_targets,
        'original_pheno': original_pheno,
        'pheno_preprocessor': pheno_preprocessor if has_targets else None,
        'trait_names': trait_names
    }
    
    return dataset, test_info


def predict(model, data_loader, device: str, logger: logging.Logger):
    """
    Make predictions on test data.
    
    Args:
        model: Trained PyTorch model
        data_loader: DataLoader for test data
        device: Device to run inference on
        logger: Logger instance
    
    Returns:
        predictions: numpy array of predictions
        targets: numpy array of targets (if available)
    """
    logger.info("Running inference...")
    model.eval()
    
    all_predictions = []
    all_targets = []
    
    # Create progress bar
    pbar = tqdm(
        data_loader,
        desc="Predicting",
        leave=True,
        ncols=100,
        bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]'
    )
    
    with torch.no_grad():
        for genetic, env, target in pbar:
            genetic = genetic.to(device)
            env = env.to(device)
            
            output = model(genetic, env)
            
            # Handle tuple output
            if isinstance(output, tuple):
                output = output[0]
            
            all_predictions.append(output.cpu().numpy())
            all_targets.append(target.numpy())
    
    predictions = np.vstack(all_predictions)
    targets = np.vstack(all_targets)
    
    logger.info(f"Inference complete: {predictions.shape[0]} predictions")
    
    return predictions, targets


def evaluate_predictions(predictions: np.ndarray, targets: np.ndarray, logger: logging.Logger):
    """Evaluate predictions against targets."""
    logger.info("\n" + "=" * 50)
    logger.info("Evaluation Results")
    logger.info("=" * 50)
    
    # Flatten for metrics
    pred_flat = predictions.flatten()
    target_flat = targets.flatten()
    
    # Calculate metrics
    rmse = calculate_rmse(target_flat, pred_flat)
    r2 = calculate_r2(target_flat, pred_flat)
    mae = calculate_mae(target_flat, pred_flat)
    
    # Pearson correlation
    correlation = np.corrcoef(target_flat, pred_flat)[0, 1]
    
    logger.info(f"  RMSE:        {rmse:.4f}")
    logger.info(f"  R²:          {r2:.4f}")
    logger.info(f"  MAE:         {mae:.4f}")
    logger.info(f"  Correlation: {correlation:.4f}")
    logger.info("=" * 50)
    
    return {
        "rmse": float(rmse),
        "r2": float(r2),
        "mae": float(mae),
        "correlation": float(correlation),
        "n_samples": len(pred_flat)
    }


def save_predictions(
    predictions: np.ndarray,
    sample_ids: list,
    trait_names: list,
    output_dir: Path,
    metrics: dict = None,
    logger: logging.Logger = None
):
    """Save predictions to files."""
    
    # Create submission format DataFrame
    results = []
    for i, sample_id in enumerate(sample_ids):
        # Parse Env_Hybrid format
        parts = sample_id.rsplit('_', 1)
        if len(parts) == 2:
            env, hybrid = parts
        else:
            env = sample_id
            hybrid = sample_id
        
        row = {'Env': env, 'Hybrid': hybrid}
        for j, trait in enumerate(trait_names):
            row[trait] = predictions[i, j] if predictions.ndim > 1 else predictions[i]
        results.append(row)
    
    df = pd.DataFrame(results)
    
    # Save predictions CSV
    pred_file = output_dir / "predictions.csv"
    df.to_csv(pred_file, index=False)
    if logger:
        logger.info(f"Predictions saved to {pred_file}")
    
    # Save submission format (Env, Hybrid, Yield_Mg_ha)
    submission_file = output_dir / "submission.csv"
    submission_cols = ['Env', 'Hybrid'] + trait_names
    df[submission_cols].to_csv(submission_file, index=False)
    if logger:
        logger.info(f"Submission file saved to {submission_file}")
    
    # Save metrics if available
    if metrics:
        metrics_file = output_dir / "evaluation_metrics.json"
        with open(metrics_file, 'w') as f:
            json.dump(metrics, f, indent=2)
        if logger:
            logger.info(f"Metrics saved to {metrics_file}")
    
    # Save raw predictions as numpy
    np.savez(
        output_dir / "predictions.npz",
        predictions=predictions,
        sample_ids=sample_ids
    )
    
    return df


def main(args):
    """Main prediction function."""
    # Setup output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_output_dir = Path(args.output_dir)
    output_dir = base_output_dir / f"pred_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup logging
    log_file = output_dir / "prediction.log"
    logger = setup_logging(args.log_level, str(log_file))
    
    logger.info("=" * 70)
    logger.info("G×E Crop Trait Prediction - Inference")
    logger.info("=" * 70)
    
    # Set device
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info(f"Using device: {device}")
    
    # Load model
    model, checkpoint = load_model(args.model_path, device, logger)
    
    # Load test data
    try:
        test_dataset, test_info = load_test_data(args.data_dir, checkpoint, logger)
    except Exception as e:
        logger.error(f"Failed to load test data: {e}")
        raise
    
    # Create data loader
    batch_size = args.batch_size
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Make predictions
    predictions, targets = predict(model, test_loader, device, logger)
    
    # De-normalize predictions if needed
    if test_info['pheno_preprocessor'] is not None:
        predictions = test_info['pheno_preprocessor'].inverse_normalize(predictions)
        targets = test_info['original_pheno']
    
    # Evaluate if targets available
    metrics = None
    if test_info['has_targets']:
        metrics = evaluate_predictions(predictions, targets, logger)
    
    # Save predictions
    save_predictions(
        predictions=predictions,
        sample_ids=test_info['sample_ids'],
        trait_names=test_info['trait_names'],
        output_dir=output_dir,
        metrics=metrics,
        logger=logger
    )
    
    logger.info(f"\nPrediction complete! Results saved to {output_dir}")
    logger.info("=" * 70)
    
    return predictions, metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Make predictions with trained G×E crop trait model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Basic prediction
    python predict.py --model-path outputs/run_xxx/best_model.pt --data-dir ./Testing_data
    
    # Specify output directory
    python predict.py --model-path outputs/run_xxx/best_model.pt --data-dir ./Testing_data --output-dir ./predictions
    
    # Use GPU
    python predict.py --model-path outputs/run_xxx/best_model.pt --data-dir ./Testing_data --device cuda
        """
    )
    
    parser.add_argument(
        "--model-path", "-m",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)"
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        default="./Testing_data",
        help="Path to test data directory (default: ./Testing_data)"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./predictions",
        help="Directory to save predictions (default: ./predictions)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for inference (default: 64)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device to use for inference"
    )
    
    parser.add_argument(
        "--log-level",
        type=str,
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging level (default: INFO)"
    )
    
    args = parser.parse_args()
    main(args)
