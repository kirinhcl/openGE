"""Training engine and trainer for model optimization."""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Tuple, Callable
import numpy as np


class Trainer:
    """
    Training engine for crop trait prediction models.
    
    Handles training loops, validation, and metric computation.
    """
    
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        early_stopping_patience: int = 10,
    ):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model
            criterion: Loss function
            optimizer: Optimizer
            device: Device to train on ('cpu' or 'cuda')
            early_stopping_patience: Early stopping patience
        """
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.device = device
        self.early_stopping_patience = early_stopping_patience
        self.best_val_loss = float("inf")
        self.patience_counter = 0
        self.history = {"train_loss": [], "val_loss": []}
    
    def train_epoch(self, train_loader: DataLoader) -> float:
        """
        Train for one epoch.
        
        Args:
            train_loader: Training data loader
            
        Returns:
            Average training loss
        """
        self.model.train()
        total_loss = 0.0
        
        for batch_idx, (genetic_data, env_data, targets) in enumerate(train_loader):
            genetic_data = genetic_data.to(self.device)
            env_data = env_data.to(self.device)
            targets = targets.to(self.device)
            
            self.optimizer.zero_grad()
            outputs, _ = self.model(genetic_data, env_data)
            loss = self.criterion(outputs, targets)
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(train_loader)
    
    def validate(self, val_loader: DataLoader) -> float:
        """
        Validate model.
        
        Args:
            val_loader: Validation data loader
            
        Returns:
            Average validation loss
        """
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for genetic_data, env_data, targets in val_loader:
                genetic_data = genetic_data.to(self.device)
                env_data = env_data.to(self.device)
                targets = targets.to(self.device)
                
                outputs, _ = self.model(genetic_data, env_data)
                loss = self.criterion(outputs, targets)
                total_loss += loss.item()
        
        return total_loss / len(val_loader)
    
    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        epochs: int = 100,
    ) -> Dict:
        """
        Train model.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of epochs
            
        Returns:
            Training history
        """
        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)
            
            self.history["train_loss"].append(train_loss)
            self.history["val_loss"].append(val_loss)
            
            if epoch % 10 == 0:
                print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}")
            
            # Early stopping
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.early_stopping_patience:
                    print(f"Early stopping at epoch {epoch}")
                    break
        
        return self.history
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make predictions on test data.
        
        Args:
            data_loader: Test data loader
            
        Returns:
            Predictions and targets
        """
        self.model.eval()
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for genetic_data, env_data, targets in data_loader:
                genetic_data = genetic_data.to(self.device)
                env_data = env_data.to(self.device)
                
                outputs, _ = self.model(genetic_data, env_data)
                all_predictions.append(outputs.cpu().numpy())
                all_targets.append(targets.numpy())
        
        predictions = np.concatenate(all_predictions)
        targets = np.concatenate(all_targets)
        
        return predictions, targets
