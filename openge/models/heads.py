"""Prediction heads for regression and classification tasks."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple


class RegressionHead(nn.Module):
    """
    Regression head for continuous trait prediction.
    
    Predicts continuous values like yield, plant height, etc.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        n_traits: int = 1, 
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        use_residual: bool = False
    ):
        """
        Initialize regression head.
        
        Args:
            input_dim: Input dimension from fusion layer
            n_traits: Number of traits to predict
            hidden_dims: Optional hidden layer dimensions
            dropout: Dropout rate
            use_residual: Use residual connections
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_traits = n_traits
        self.use_residual = use_residual
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Final prediction layer
        self.output_layer = nn.Linear(prev_dim, n_traits)
        
        # Residual connection if dimensions match
        if use_residual and input_dim == n_traits:
            self.residual_weight = nn.Parameter(torch.tensor(0.1))
        else:
            self.use_residual = False
        
        # Initialize output layer with small weights
        nn.init.xavier_uniform_(self.output_layer.weight, gain=0.1)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict trait values.
        
        Args:
            x: Fused representation [batch_size, input_dim]
            
        Returns:
            Predicted trait values [batch_size, n_traits]
        """
        hidden = self.hidden_layers(x)
        output = self.output_layer(hidden)
        
        if self.use_residual:
            output = output + self.residual_weight * x[:, :self.n_traits]
        
        return output


class MultiTaskRegressionHead(nn.Module):
    """
    Multi-task regression head with shared and task-specific layers.
    
    For predicting multiple correlated traits simultaneously.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_traits: int,
        shared_dims: List[int] = None,
        task_specific_dim: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize multi-task regression head.
        
        Args:
            input_dim: Input dimension
            n_traits: Number of traits
            shared_dims: Shared hidden layer dimensions
            task_specific_dim: Task-specific layer dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_traits = n_traits
        
        if shared_dims is None:
            shared_dims = [input_dim // 2]
        
        # Shared layers
        layers = []
        prev_dim = input_dim
        for dim in shared_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.shared_layers = nn.Sequential(*layers)
        
        # Task-specific towers
        self.task_towers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(prev_dim, task_specific_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(task_specific_dim, 1)
            )
            for _ in range(n_traits)
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict multiple traits.
        
        Args:
            x: Fused representation [batch_size, input_dim]
            
        Returns:
            Predicted traits [batch_size, n_traits]
        """
        # Shared representation
        shared = self.shared_layers(x)
        
        # Task-specific predictions
        outputs = [tower(shared) for tower in self.task_towers]
        
        return torch.cat(outputs, dim=-1)


class ClassificationHead(nn.Module):
    """
    Classification head for categorical trait prediction.
    
    For predicting discrete categories like disease resistance levels.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        n_classes: int, 
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        use_softmax: bool = False
    ):
        """
        Initialize classification head.
        
        Args:
            input_dim: Input dimension from fusion layer
            n_classes: Number of classes
            hidden_dims: Optional hidden layer dimensions
            dropout: Dropout rate
            use_softmax: Apply softmax to output (for inference)
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_classes = n_classes
        self.use_softmax = use_softmax
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.hidden_layers = nn.Sequential(*layers)
        
        # Final classification layer
        self.output_layer = nn.Linear(prev_dim, n_classes)
        
        # Initialize
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict class probabilities.
        
        Args:
            x: Fused representation [batch_size, input_dim]
            
        Returns:
            Class logits/probabilities [batch_size, n_classes]
        """
        hidden = self.hidden_layers(x)
        logits = self.output_layer(hidden)
        
        if self.use_softmax:
            return F.softmax(logits, dim=-1)
        
        return logits


class UncertaintyHead(nn.Module):
    """
    Prediction head with uncertainty estimation.
    
    Outputs both mean prediction and uncertainty (variance).
    """
    
    def __init__(
        self,
        input_dim: int,
        n_traits: int = 1,
        hidden_dims: List[int] = None,
        dropout: float = 0.1,
        min_variance: float = 1e-6
    ):
        """
        Initialize uncertainty head.
        
        Args:
            input_dim: Input dimension
            n_traits: Number of traits
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
            min_variance: Minimum variance for numerical stability
        """
        super().__init__()
        
        self.n_traits = n_traits
        self.min_variance = min_variance
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
        
        # Shared layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.shared = nn.Sequential(*layers)
        
        # Mean prediction
        self.mean_head = nn.Linear(prev_dim, n_traits)
        
        # Log-variance prediction (for numerical stability)
        self.log_var_head = nn.Linear(prev_dim, n_traits)
        
        # Initialize
        nn.init.xavier_uniform_(self.mean_head.weight, gain=0.1)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.constant_(self.log_var_head.weight, 0.01)
        nn.init.constant_(self.log_var_head.bias, -2.0)  # Start with small variance
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Predict mean and variance.
        
        Args:
            x: Fused representation [batch_size, input_dim]
            
        Returns:
            Tuple of (mean, variance) each [batch_size, n_traits]
        """
        hidden = self.shared(x)
        
        mean = self.mean_head(hidden)
        log_var = self.log_var_head(hidden)
        
        # Convert log-variance to variance
        variance = F.softplus(log_var) + self.min_variance
        
        return mean, variance
    
    def nll_loss(
        self, 
        predictions: Tuple[torch.Tensor, torch.Tensor], 
        targets: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute negative log-likelihood loss.
        
        Args:
            predictions: Tuple of (mean, variance)
            targets: Ground truth values
            
        Returns:
            NLL loss value
        """
        mean, variance = predictions
        
        # Gaussian NLL
        nll = 0.5 * (torch.log(variance) + (targets - mean) ** 2 / variance)
        
        return nll.mean()


class QuantileHead(nn.Module):
    """
    Quantile regression head for prediction intervals.
    
    Predicts multiple quantiles to estimate prediction intervals.
    """
    
    def __init__(
        self,
        input_dim: int,
        n_traits: int = 1,
        quantiles: List[float] = None,
        hidden_dims: List[int] = None,
        dropout: float = 0.1
    ):
        """
        Initialize quantile head.
        
        Args:
            input_dim: Input dimension
            n_traits: Number of traits
            quantiles: Quantiles to predict (default: [0.1, 0.5, 0.9])
            hidden_dims: Hidden layer dimensions
            dropout: Dropout rate
        """
        super().__init__()
        
        if quantiles is None:
            quantiles = [0.1, 0.5, 0.9]
        
        self.quantiles = quantiles
        self.n_traits = n_traits
        self.n_quantiles = len(quantiles)
        
        if hidden_dims is None:
            hidden_dims = [input_dim // 2]
        
        # Shared layers
        layers = []
        prev_dim = input_dim
        for dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, dim),
                nn.LayerNorm(dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            ])
            prev_dim = dim
        self.shared = nn.Sequential(*layers)
        
        # Quantile-specific heads
        self.quantile_heads = nn.ModuleList([
            nn.Linear(prev_dim, n_traits) for _ in quantiles
        ])
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Predict quantiles.
        
        Args:
            x: Fused representation [batch_size, input_dim]
            
        Returns:
            Quantile predictions [batch_size, n_traits, n_quantiles]
        """
        hidden = self.shared(x)
        
        quantile_preds = [head(hidden) for head in self.quantile_heads]
        
        # Stack: [batch, n_traits, n_quantiles]
        output = torch.stack(quantile_preds, dim=-1)
        
        return output
    
    def quantile_loss(self, predictions: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Compute pinball (quantile) loss.
        
        Args:
            predictions: Quantile predictions [batch, n_traits, n_quantiles]
            targets: Ground truth [batch, n_traits]
            
        Returns:
            Quantile loss value
        """
        targets = targets.unsqueeze(-1)  # [batch, n_traits, 1]
        
        errors = targets - predictions
        
        losses = []
        for i, q in enumerate(self.quantiles):
            error = errors[..., i]
            loss = torch.max(q * error, (q - 1) * error)
            losses.append(loss.mean())
        
        return sum(losses) / len(losses)
