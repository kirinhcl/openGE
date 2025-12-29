"""Sparse transformer models for interpretable prediction."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple, List

from .sparse_attention import SparseMultiHeadAttention, TopKAttention


class SparseAttention(nn.Module):
    """
    Sparse attention mechanism with learnable sparsity patterns.
    
    Supports multiple sparsity strategies for efficient and
    interpretable attention computation.
    """
    
    def __init__(
        self, 
        dim: int, 
        n_heads: int = 8, 
        sparsity: float = 0.8,
        sparsity_type: str = 'topk',
        dropout: float = 0.1
    ):
        """
        Initialize sparse attention.
        
        Args:
            dim: Embedding dimension
            n_heads: Number of heads
            sparsity: Sparsity level (0-1)
            sparsity_type: Type of sparsity ('topk', 'threshold', 'learned')
            dropout: Dropout rate
        """
        super().__init__()
        
        assert dim % n_heads == 0
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.sparsity = sparsity
        self.sparsity_type = sparsity_type
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        # Learnable sparsity parameters
        if sparsity_type == 'learned':
            self.sparsity_threshold = nn.Parameter(torch.zeros(n_heads))
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention weights for interpretability
        self.attention_weights = None
        self.sparsity_mask = None
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with attention weights.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor and attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Transpose for attention
        q = q.transpose(1, 2)  # [batch, heads, seq, head_dim]
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Apply sparsity
        if self.sparsity_type == 'topk':
            scores, sparsity_mask = self._apply_topk_sparsity(scores)
        elif self.sparsity_type == 'threshold':
            scores, sparsity_mask = self._apply_threshold_sparsity(scores)
        elif self.sparsity_type == 'learned':
            scores, sparsity_mask = self._apply_learned_sparsity(scores)
        else:
            sparsity_mask = None
        
        self.sparsity_mask = sparsity_mask
        
        # Softmax and dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        self.attention_weights = attn_weights
        
        # Apply attention to values
        output = torch.matmul(attn_weights, v)
        
        # Reshape and project
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        return output, attn_weights
    
    def _apply_topk_sparsity(
        self, 
        scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply top-k sparsity."""
        seq_len = scores.size(-1)
        k = max(1, int(seq_len * (1 - self.sparsity)))
        
        topk_vals, topk_indices = torch.topk(scores, k, dim=-1)
        
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_indices, topk_vals)
        
        mask = sparse_scores > float('-inf')
        
        return sparse_scores, mask
    
    def _apply_threshold_sparsity(
        self, 
        scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply threshold-based sparsity."""
        threshold = torch.quantile(
            scores.flatten(-2), 
            self.sparsity, 
            dim=-1, 
            keepdim=True
        ).unsqueeze(-1)
        
        mask = scores >= threshold
        sparse_scores = torch.where(mask, scores, torch.tensor(float('-inf'), device=scores.device))
        
        return sparse_scores, mask
    
    def _apply_learned_sparsity(
        self, 
        scores: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply learned sparsity thresholds."""
        # Each head has its own threshold
        thresholds = torch.sigmoid(self.sparsity_threshold) * 2 - 1  # Range [-1, 1]
        thresholds = thresholds.view(1, -1, 1, 1)
        
        # Normalize scores to [-1, 1] range for comparison
        normalized_scores = torch.tanh(scores)
        mask = normalized_scores >= thresholds
        
        sparse_scores = torch.where(mask, scores, torch.tensor(float('-inf'), device=scores.device))
        
        return sparse_scores, mask
    
    def get_sparsity_stats(self) -> Dict[str, float]:
        """Get sparsity statistics from last forward pass."""
        if self.sparsity_mask is None:
            return {}
        
        actual_sparsity = 1 - self.sparsity_mask.float().mean().item()
        
        return {
            'target_sparsity': self.sparsity,
            'actual_sparsity': actual_sparsity,
            'n_active_connections': self.sparsity_mask.sum().item()
        }


class SparseTransformerBlock(nn.Module):
    """
    Transformer block with sparse attention.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        mlp_ratio: float = 4.0,
        sparsity: float = 0.8,
        dropout: float = 0.1
    ):
        """
        Initialize sparse transformer block.
        
        Args:
            dim: Embedding dimension
            n_heads: Number of attention heads
            mlp_ratio: MLP hidden dimension ratio
            sparsity: Attention sparsity level
            dropout: Dropout rate
        """
        super().__init__()
        
        self.norm1 = nn.LayerNorm(dim)
        self.attn = SparseAttention(dim, n_heads, sparsity, dropout=dropout)
        
        self.norm2 = nn.LayerNorm(dim)
        mlp_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq, dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor and attention weights
        """
        # Self-attention with residual
        normed = self.norm1(x)
        attn_out, attn_weights = self.attn(normed, mask)
        x = x + attn_out
        
        # MLP with residual
        x = x + self.mlp(self.norm2(x))
        
        return x, attn_weights


class WeightSparseTransformer(nn.Module):
    """
    Transformer with learned weight sparsity patterns.
    
    Enforces sparse connections in attention and feedforward layers
    for improved interpretability and efficiency.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        hidden_dim: int, 
        output_dim: int,
        n_heads: int = 8,
        n_layers: int = 4,
        sparsity_level: float = 0.8,
        dropout: float = 0.1,
        max_seq_len: int = 5000
    ):
        """
        Initialize weight-sparse transformer.
        
        Args:
            input_dim: Input feature dimension
            hidden_dim: Hidden dimension
            output_dim: Output dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            sparsity_level: Target sparsity level (0-1)
            dropout: Dropout rate
            max_seq_len: Maximum sequence length
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.sparsity_level = sparsity_level
        self.n_layers = n_layers
        
        # Input projection
        self.input_proj = nn.Linear(1, hidden_dim)
        
        # Positional encoding
        self.pos_encoding = self._create_positional_encoding(max_seq_len, hidden_dim)
        self.register_buffer('pe', self.pos_encoding)
        
        # Sparse transformer blocks
        self.blocks = nn.ModuleList([
            SparseTransformerBlock(
                dim=hidden_dim,
                n_heads=n_heads,
                sparsity=sparsity_level,
                dropout=dropout
            )
            for _ in range(n_layers)
        ])
        
        # Output layers
        self.norm = nn.LayerNorm(hidden_dim)
        self.output_proj = nn.Linear(hidden_dim, output_dim)
        
        # Weight sparsity masks (learned)
        self.weight_masks = nn.ParameterDict()
        self._init_weight_masks()
    
    def _create_positional_encoding(self, max_len: int, d_model: int) -> torch.Tensor:
        """Create sinusoidal positional encoding."""
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        return pe.unsqueeze(0)
    
    def _init_weight_masks(self):
        """Initialize learnable weight masks."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear) and 'proj' in name:
                # Create mask initialized to keep most weights
                mask = nn.Parameter(
                    torch.ones(module.weight.shape) * 3.0  # Sigmoid(3) ≈ 0.95
                )
                safe_name = name.replace('.', '_')
                self.weight_masks[safe_name] = mask
    
    def _apply_weight_masks(self):
        """Apply learned masks to weights during forward pass."""
        for name, module in self.named_modules():
            if isinstance(module, nn.Linear):
                safe_name = name.replace('.', '_')
                if safe_name in self.weight_masks:
                    mask = torch.sigmoid(self.weight_masks[safe_name])
                    # Apply mask with straight-through estimator
                    hard_mask = (mask > 0.5).float()
                    mask = hard_mask - mask.detach() + mask
                    module.weight.data *= mask
    
    def forward(
        self, 
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, List[torch.Tensor]]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len] or [batch, seq_len, 1]
            mask: Optional attention mask
            
        Returns:
            Output tensor and list of attention weights per layer
        """
        # Handle input shape
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        batch_size, seq_len, _ = x.shape
        
        # Input projection
        x = self.input_proj(x)
        
        # Add positional encoding
        x = x + self.pe[:, :seq_len, :]
        
        # Apply transformer blocks
        attention_weights = []
        for block in self.blocks:
            x, attn = block(x, mask)
            attention_weights.append(attn)
        
        # Output
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.output_proj(x)
        
        return x, attention_weights
    
    def get_sparsity_pattern(self) -> Dict[str, float]:
        """Get current sparsity patterns in model."""
        patterns = {}
        
        # Attention sparsity
        for i, block in enumerate(self.blocks):
            stats = block.attn.get_sparsity_stats()
            patterns[f'block_{i}_attention'] = stats
        
        # Weight sparsity
        for name, mask in self.weight_masks.items():
            sigmoid_mask = torch.sigmoid(mask)
            sparsity = (sigmoid_mask < 0.5).float().mean().item()
            patterns[f'weight_{name}'] = sparsity
        
        return patterns
    
    def get_attention_weights(self) -> List[torch.Tensor]:
        """Get attention weights from all layers."""
        weights = []
        for block in self.blocks:
            if block.attn.attention_weights is not None:
                weights.append(block.attn.attention_weights)
        return weights
    
    def compute_sparsity_loss(self, target_sparsity: Optional[float] = None) -> torch.Tensor:
        """
        Compute sparsity regularization loss.
        
        Args:
            target_sparsity: Target sparsity level (default: self.sparsity_level)
            
        Returns:
            Sparsity loss tensor
        """
        if target_sparsity is None:
            target_sparsity = self.sparsity_level
        
        total_loss = 0.0
        
        for name, mask in self.weight_masks.items():
            sigmoid_mask = torch.sigmoid(mask)
            actual_sparsity = (sigmoid_mask < 0.5).float().mean()
            
            # L1 loss towards target sparsity
            total_loss += (actual_sparsity - target_sparsity).abs()
            
            # Entropy regularization to encourage binary masks
            entropy = -sigmoid_mask * torch.log(sigmoid_mask + 1e-8) - \
                      (1 - sigmoid_mask) * torch.log(1 - sigmoid_mask + 1e-8)
            total_loss += 0.01 * entropy.mean()
        
        return total_loss


class InterpretableSparseTransformer(WeightSparseTransformer):
    """
    Sparse transformer with enhanced interpretability features.
    
    Adds feature importance tracking and attention analysis.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Feature importance tracking
        self.feature_importance = None
        self.input_gradients = None
    
    def forward_with_interpretation(
        self,
        x: torch.Tensor,
        compute_importance: bool = True
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass with interpretation data.
        
        Args:
            x: Input tensor
            compute_importance: Whether to compute feature importance
            
        Returns:
            Output and interpretation dictionary
        """
        if compute_importance:
            x.requires_grad_(True)
        
        output, attention_weights = self.forward(x)
        
        interpretation = {
            'attention_weights': attention_weights,
            'sparsity_pattern': self.get_sparsity_pattern()
        }
        
        if compute_importance and x.grad is not None:
            self.input_gradients = x.grad
            self.feature_importance = x.grad.abs().mean(dim=0)
            interpretation['feature_importance'] = self.feature_importance
        
        return output, interpretation
    
    def get_important_features(self, top_k: int = 10) -> List[int]:
        """
        Get indices of most important features.
        
        Args:
            top_k: Number of top features to return
            
        Returns:
            List of feature indices
        """
        if self.feature_importance is None:
            return []
        
        importance = self.feature_importance.squeeze()
        _, indices = torch.topk(importance, min(top_k, len(importance)))
        
        return indices.tolist()
