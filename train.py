#!/usr/bin/env python
"""
Main training script for crop trait prediction with G×E interactions.

Usage:
    python train.py --config configs/maize_2024.yaml --data-dir ./Training_data
    python train.py --data-dir ./Training_data --epochs 50 --batch-size 32
"""

import argparse
import logging
import os
import json
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from openge.core import Config, Trainer
from openge.data import GxEDataLoader, Preprocessor, GxEDataset
from openge.models import (
    TransformerEncoder, MLPEncoder, CNNEncoder,
    AttentionFusion, ConcatFusion,
    RegressionHead, ClassificationHead,
    GxEModel
)
from openge.utils.metrics import calculate_r2, calculate_rmse, evaluate_model


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


def load_data(config: dict, data_dir: str, logger: logging.Logger):
    """
    Load and preprocess training data from Training_data directory.
    
    Expected files:
    - 1_Training_Trait_Data_2014_2023.csv (phenotype/target)
    - 5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt (genetic markers)
    - 6_Training_EC_Data_2014_2023.csv (environmental covariates)
    
    Optional files:
    - 3_Training_Soil_Data_2015_2023.csv (soil data)
    - 4_Training_Weather_Data_2014_2023_seasons_only.csv (weather data)
    """
    logger.info(f"Loading data from {data_dir}")
    
    data_path = Path(data_dir)
    
    # Define file paths
    trait_file = data_path / "1_Training_Trait_Data_2014_2023.csv"
    genotype_file = data_path / "5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt"
    ec_file = data_path / "6_Training_EC_Data_2014_2023.csv"
    soil_file = data_path / "3_Training_Soil_Data_2015_2023.csv"
    weather_file = data_path / "4_Training_Weather_Data_2014_2023_seasons_only.csv"
    
    # Validate required files exist
    required_files = [trait_file, genotype_file, ec_file]
    for f in required_files:
        if not f.exists():
            raise FileNotFoundError(f"Required file not found: {f}")
    
    # Initialize data loader
    data_loader = GxEDataLoader()
    
    # 1. Load genetic data
    logger.info("Loading genetic data...")
    genetic_data, genetic_ids, marker_names = data_loader.load_genetic(
        filepath=str(genotype_file),
        sample_col='<Marker>',
        handle_missing='mean',
        missing_threshold=config.get('data', {}).get('genetic_missing_threshold', 0.5),
        encoding='keep'
    )
    logger.info(f"Genetic data shape: {genetic_data.shape}")
    
    # 2. Load environment data (EC data is simpler and more reliable)
    logger.info("Loading environment data...")
    try:
        env_data, env_ids, env_features = data_loader.load_environment(
            ec_file=str(ec_file),
            soil_file=str(soil_file) if soil_file.exists() else None,
            handle_missing='drop',
            use_3d=False  # Use 2D static features for simplicity
        )
    except Exception as e:
        logger.warning(f"Failed to load soil data, using EC only: {e}")
        env_data, env_ids, env_features = data_loader.load_environment(
            ec_file=str(ec_file),
            handle_missing='drop',
            use_3d=False
        )
    logger.info(f"Environment data shape: {env_data.shape}")
    
    # 3. Load phenotype data (target variable)
    logger.info("Loading phenotype data...")
    target_traits = config.get('data', {}).get('target_traits', ['Yield_Mg_ha'])
    
    pheno_data, pheno_ids, trait_names = data_loader.load_phenotype(
        filepath=str(trait_file),
        traits=target_traits,
        sample_id_col='Hybrid',
        env_col='Env',
        handle_missing='drop',
        handle_outliers=True,
        outlier_method='iqr'
    )
    logger.info(f"Phenotype data shape: {pheno_data.shape}")
    logger.info(f"Target traits: {trait_names}")
    
    # 4. Align samples across datasets
    logger.info("Aligning samples...")
    aligned_genetic, aligned_env, aligned_pheno, aligned_ids = data_loader.align_samples(
        strategy='inner'
    )
    logger.info(f"Aligned samples: {len(aligned_ids)}")
    
    # 5. Preprocess data
    logger.info("Preprocessing data...")
    
    # Normalize genetic data
    genetic_preprocessor = Preprocessor(method='standard')
    aligned_genetic = genetic_preprocessor.normalize(aligned_genetic, fit=True)
    
    # Normalize environment data
    env_preprocessor = Preprocessor(method='standard')
    aligned_env = env_preprocessor.normalize(aligned_env, fit=True)
    
    # Normalize phenotype data (targets) - optional but can help training
    pheno_preprocessor = Preprocessor(method='standard')
    aligned_pheno_normalized = pheno_preprocessor.normalize(aligned_pheno, fit=True)
    
    # Store preprocessors for inference
    preprocessors = {
        'genetic': genetic_preprocessor,
        'environment': env_preprocessor,
        'phenotype': pheno_preprocessor
    }
    
    # Create dataset
    dataset = GxEDataset(
        genetic_data=aligned_genetic,
        environment_data=aligned_env,
        phenotype_data=aligned_pheno_normalized,
        sample_ids=aligned_ids,
        return_dict=False
    )
    
    # Print data summary
    data_loader.print_summary()
    
    # Data info for model building
    data_info = {
        'n_samples': len(aligned_ids),
        'n_markers': aligned_genetic.shape[1],
        'n_env_features': aligned_env.shape[1],
        'n_traits': aligned_pheno.shape[1],
        'trait_names': trait_names,
        'marker_names': marker_names,
        'env_feature_names': env_features
    }
    
    return dataset, preprocessors, data_info


def build_model(config: dict, data_info: dict, device: str, logger: logging.Logger) -> nn.Module:
    """Build G×E model from configuration and data info."""
    logger.info("Building G×E model...")
    
    n_markers = data_info['n_markers']
    n_env_features = data_info['n_env_features']
    n_traits = data_info['n_traits']
    
    # Model hyperparameters
    model_config = config.get('model', {})
    
    # Hidden dimensions
    genetic_hidden_dim = model_config.get('genetic_hidden_dim', 256)
    env_hidden_dim = model_config.get('env_hidden_dim', 64)
    fusion_dim = model_config.get('fusion_dim', 128)
    
    # Build genetic encoder
    genetic_encoder_type = model_config.get('genetic_encoder', 'mlp')
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
            n_filters=model_config.get('genetic_n_filters', 64),
            kernel_sizes=model_config.get('genetic_kernel_sizes', [7, 5, 3])
        )
    else:  # MLP
        genetic_encoder = MLPEncoder(
            input_dim=n_markers,
            hidden_dims=model_config.get('genetic_hidden_dims', [512, 256]),
            output_dim=genetic_hidden_dim,
            dropout=model_config.get('dropout', 0.1)
        )
    
    # Build environment encoder
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
        head_input_dim = genetic_hidden_dim  # AttentionFusion outputs dim1
    else:  # concat
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
    
    # Combine into G×E model
    model = GxEModel(
        genetic_encoder=genetic_encoder,
        env_encoder=env_encoder,
        fusion_layer=fusion_layer,
        prediction_head=head,
    ).to(device)
    
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model built: {n_params:,} total parameters, {n_trainable:,} trainable")
    
    return model


def train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    n_batches = 0
    
    for batch_idx, (genetic, env, target) in enumerate(train_loader):
        genetic = genetic.to(device)
        env = env.to(device)
        target = target.to(device)
        
        optimizer.zero_grad()
        
        # Forward pass
        output = model(genetic, env)
        
        # Handle tuple output (some models return (output, attention_weights))
        if isinstance(output, tuple):
            output = output[0]
        
        loss = criterion(output, target)
        
        # Backward pass
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        if (batch_idx + 1) % 50 == 0:
            logger.debug(f"Epoch {epoch}, Batch {batch_idx + 1}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    return total_loss / n_batches


def validate(model, val_loader, criterion, device):
    """Validate model."""
    model.eval()
    total_loss = 0
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for genetic, env, target in val_loader:
            genetic = genetic.to(device)
            env = env.to(device)
            target = target.to(device)
            
            output = model(genetic, env)
            
            if isinstance(output, tuple):
                output = output[0]
            
            loss = criterion(output, target)
            total_loss += loss.item()
            
            all_preds.append(output.cpu().numpy())
            all_targets.append(target.cpu().numpy())
    
    all_preds = np.vstack(all_preds)
    all_targets = np.vstack(all_targets)
    
    # Calculate metrics
    r2 = calculate_r2(all_targets.flatten(), all_preds.flatten())
    rmse = calculate_rmse(all_targets.flatten(), all_preds.flatten())
    
    return total_loss / len(val_loader), r2, rmse, all_preds, all_targets


def main(args):
    """Main training function."""
    # Setup
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(args.output_dir) / f"run_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    log_file = output_dir / "training.log"
    logger = setup_logging(args.log_level, str(log_file))
    logger.info("=" * 70)
    logger.info("Starting G×E Crop Trait Prediction Training")
    logger.info("=" * 70)
    
    # Load configuration
    if args.config and Path(args.config).exists():
        config = Config(args.config).config
        logger.info(f"Loaded configuration from {args.config}")
    else:
        config = {}
        logger.info("Using default configuration")
    
    # Override config with command line arguments
    if args.batch_size:
        config.setdefault('training', {})['batch_size'] = args.batch_size
    if args.epochs:
        config.setdefault('training', {})['epochs'] = args.epochs
    if args.lr:
        config.setdefault('training', {})['learning_rate'] = args.lr
    
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
    
    # Load data
    try:
        dataset, preprocessors, data_info = load_data(config, args.data_dir, logger)
    except Exception as e:
        logger.error(f"Failed to load data: {e}")
        raise
    
    logger.info(f"Dataset: {data_info['n_samples']} samples, {data_info['n_markers']} markers, "
                f"{data_info['n_env_features']} env features, {data_info['n_traits']} traits")
    
    # Split dataset
    n_samples = len(dataset)
    indices = np.arange(n_samples)
    
    train_ratio = config.get('training', {}).get('train_ratio', 0.7)
    val_ratio = config.get('training', {}).get('val_ratio', 0.15)
    
    train_indices, temp_indices = train_test_split(
        indices, train_size=train_ratio, random_state=42
    )
    val_size = val_ratio / (1 - train_ratio)
    val_indices, test_indices = train_test_split(
        temp_indices, train_size=val_size, random_state=42
    )
    
    logger.info(f"Data split: train={len(train_indices)}, val={len(val_indices)}, test={len(test_indices)}")
    
    # Create subset datasets
    def create_subset(indices):
        return GxEDataset(
            genetic_data=dataset.genetic_data[indices],
            environment_data=dataset.environment_data[indices],
            phenotype_data=dataset.phenotype_data[indices],
            sample_ids=[dataset.sample_ids[i] for i in indices] if dataset.sample_ids else None,
            return_dict=False
        )
    
    train_dataset = create_subset(train_indices)
    val_dataset = create_subset(val_indices)
    test_dataset = create_subset(test_indices)
    
    # Create data loaders
    batch_size = config.get('training', {}).get('batch_size', 32)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    # Build model
    model = build_model(config, data_info, device, logger)
    
    # Setup training
    criterion = nn.MSELoss()
    
    learning_rate = config.get('training', {}).get('learning_rate', 0.001)
    weight_decay = config.get('training', {}).get('weight_decay', 0.0001)
    
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay
    )
    
    # Learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5
    )
    
    # Training loop
    n_epochs = config.get('training', {}).get('epochs', 100)
    early_stopping_patience = config.get('training', {}).get('early_stopping_patience', 15)
    
    best_val_loss = float('inf')
    best_val_r2 = -float('inf')
    patience_counter = 0
    history = {'train_loss': [], 'val_loss': [], 'val_r2': [], 'val_rmse': []}
    
    logger.info(f"\nStarting training for {n_epochs} epochs...")
    logger.info(f"Batch size: {batch_size}, Learning rate: {learning_rate}")
    logger.info("-" * 70)
    
    for epoch in range(1, n_epochs + 1):
        # Train
        train_loss = train_epoch(model, train_loader, criterion, optimizer, device, epoch, logger)
        
        # Validate
        val_loss, val_r2, val_rmse, _, _ = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        scheduler.step(val_loss)
        
        # Record history
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)
        history['val_rmse'].append(val_rmse)
        
        # Log progress
        logger.info(f"Epoch {epoch:3d}/{n_epochs} | "
                   f"Train Loss: {train_loss:.4f} | "
                   f"Val Loss: {val_loss:.4f} | "
                   f"Val R²: {val_r2:.4f} | "
                   f"Val RMSE: {val_rmse:.4f}")
        
        # Early stopping check
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_r2 = val_r2
            patience_counter = 0
            
            # Save best model
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'val_r2': val_r2,
                'data_info': data_info,
                'config': config
            }, output_dir / "best_model.pt")
            logger.info(f"  ✓ Saved best model (R²: {val_r2:.4f})")
        else:
            patience_counter += 1
            if patience_counter >= early_stopping_patience:
                logger.info(f"\nEarly stopping triggered at epoch {epoch}")
                break
    
    logger.info("-" * 70)
    logger.info(f"Training completed! Best validation R²: {best_val_r2:.4f}")
    
    # Load best model and evaluate on test set
    checkpoint = torch.load(output_dir / "best_model.pt")
    model.load_state_dict(checkpoint['model_state_dict'])
    
    test_loss, test_r2, test_rmse, test_preds, test_targets = validate(
        model, test_loader, criterion, device
    )
    
    logger.info(f"\nTest Set Performance:")
    logger.info(f"  - Loss: {test_loss:.4f}")
    logger.info(f"  - R²: {test_r2:.4f}")
    logger.info(f"  - RMSE: {test_rmse:.4f}")
    
    # Save final results
    results = {
        'best_epoch': checkpoint['epoch'],
        'best_val_loss': float(best_val_loss),
        'best_val_r2': float(best_val_r2),
        'test_loss': float(test_loss),
        'test_r2': float(test_r2),
        'test_rmse': float(test_rmse),
        'data_info': {k: v for k, v in data_info.items() if k not in ['marker_names', 'env_feature_names']},
        'config': config,
        'history': {k: [float(v) for v in vals] for k, vals in history.items()}
    }
    
    with open(output_dir / "results.json", 'w') as f:
        json.dump(results, f, indent=2)
    
    # Save test predictions
    np.savez(
        output_dir / "test_predictions.npz",
        predictions=test_preds,
        targets=test_targets,
        sample_ids=[test_dataset.sample_ids[i] for i in range(len(test_dataset))] if test_dataset.sample_ids else []
    )
    
    logger.info(f"\nResults saved to {output_dir}")
    logger.info("=" * 70)
    
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train G×E model for crop trait prediction",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Train with default settings
    python train.py --data-dir ./Training_data
    
    # Train with a specific configuration file
    python train.py --config configs/maize_2024.yaml --data-dir ./Training_data
    
    # Train with custom hyperparameters
    python train.py --data-dir ./Training_data --epochs 50 --batch-size 64 --lr 0.0005
    
    # Train on GPU
    python train.py --data-dir ./Training_data --device cuda
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        default=None,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--data-dir", "-d",
        type=str,
        required=True,
        help="Path to directory containing training data files"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        default="./outputs",
        help="Directory to save outputs (default: ./outputs)"
    )
    
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Batch size (overrides config)"
    )
    
    parser.add_argument(
        "--lr",
        type=float,
        default=None,
        help="Learning rate (overrides config)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda", "mps"],
        default=None,
        help="Device to use for training"
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
