"""Sparse attention mechanisms for efficient and interpretable models."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class SparseAttentionHead(nn.Module):
    """
    Single attention head with sparse connectivity.
    
    Uses top-k selection or threshold-based sparsity to reduce
    computation and improve interpretability.
    """
    
    def __init__(
        self, 
        dim_per_head: int, 
        sparsity: float = 0.8,
        sparsity_type: str = 'topk'
    ):
        """
        Initialize sparse attention head.
        
        Args:
            dim_per_head: Dimension per head
            sparsity: Sparsity level (0-1, fraction of weights to zero out)
            sparsity_type: Type of sparsity ('topk', 'threshold', 'local')
        """
        super().__init__()
        
        self.dim_per_head = dim_per_head
        self.sparsity = sparsity
        self.sparsity_type = sparsity_type
        self.scale = dim_per_head ** -0.5
        
        # Learnable temperature for attention sharpening
        self.temperature = nn.Parameter(torch.ones(1))
    
    def forward(
        self, 
        query: torch.Tensor, 
        key: torch.Tensor, 
        value: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with sparse attention.
        
        Args:
            query: Query tensor [batch, seq_len, dim]
            key: Key tensor [batch, seq_len, dim]
            value: Value tensor [batch, seq_len, dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor and attention weights
        """
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        scores = scores / self.temperature.clamp(min=0.1)
        
        # Apply mask if provided
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        # Apply sparsity
        if self.sparsity_type == 'topk':
            scores = self._apply_topk_sparsity(scores)
        elif self.sparsity_type == 'threshold':
            scores = self._apply_threshold_sparsity(scores)
        elif self.sparsity_type == 'local':
            scores = self._apply_local_sparsity(scores)
        
        # Softmax
        attn_weights = F.softmax(scores, dim=-1)
        
        # Apply attention to values
        output = torch.matmul(attn_weights, value)
        
        return output, attn_weights
    
    def _apply_topk_sparsity(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply top-k sparsity to attention scores."""
        seq_len = scores.size(-1)
        k = max(1, int(seq_len * (1 - self.sparsity)))
        
        # Get top-k values and indices
        topk_vals, topk_indices = torch.topk(scores, k, dim=-1)
        
        # Create sparse scores
        sparse_scores = torch.full_like(scores, float('-inf'))
        sparse_scores.scatter_(-1, topk_indices, topk_vals)
        
        return sparse_scores
    
    def _apply_threshold_sparsity(self, scores: torch.Tensor) -> torch.Tensor:
        """Apply threshold-based sparsity."""
        # Compute threshold based on quantile
        threshold = torch.quantile(scores, self.sparsity, dim=-1, keepdim=True)
        
        # Zero out values below threshold
        mask = scores >= threshold
        sparse_scores = torch.where(mask, scores, torch.tensor(float('-inf'), device=scores.device))
        
        return sparse_scores
    
    def _apply_local_sparsity(self, scores: torch.Tensor, window_size: int = 32) -> torch.Tensor:
        """Apply local attention sparsity (sliding window)."""
        seq_len = scores.size(-1)
        
        # Create local attention mask
        indices = torch.arange(seq_len, device=scores.device)
        row_indices = indices.unsqueeze(1)
        col_indices = indices.unsqueeze(0)
        
        # Only attend within window
        local_mask = (row_indices - col_indices).abs() <= window_size // 2
        
        sparse_scores = torch.where(
            local_mask.unsqueeze(0).unsqueeze(0), 
            scores, 
            torch.tensor(float('-inf'), device=scores.device)
        )
        
        return sparse_scores


class TopKAttention(nn.Module):
    """
    Top-K attention mechanism for efficient sparse attention.
    
    Only attends to the K most relevant positions, reducing
    computational complexity from O(n²) to O(nk).
    """
    
    def __init__(
        self, 
        dim: int, 
        k: int, 
        n_heads: int = 8,
        dropout: float = 0.1,
        use_bias: bool = True
    ):
        """
        Initialize top-K attention.
        
        Args:
            dim: Embedding dimension
            k: Number of top elements to attend to
            n_heads: Number of heads
            dropout: Dropout rate
            use_bias: Use bias in projections
        """
        super().__init__()
        
        assert dim % n_heads == 0, "dim must be divisible by n_heads"
        
        self.dim = dim
        self.k = k
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        # Projections
        self.q_proj = nn.Linear(dim, dim, bias=use_bias)
        self.k_proj = nn.Linear(dim, dim, bias=use_bias)
        self.v_proj = nn.Linear(dim, dim, bias=use_bias)
        self.out_proj = nn.Linear(dim, dim, bias=use_bias)
        
        self.dropout = nn.Dropout(dropout)
        
        # Store attention patterns for interpretability
        self.attention_patterns = None
    
    def forward(
        self, 
        x: torch.Tensor,
        key_value: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass with top-K selection.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            key_value: Optional key-value tensor (for cross-attention)
            mask: Optional attention mask
            
        Returns:
            Output tensor and attention indices
        """
        batch_size, seq_len, _ = x.shape
        
        # Project queries, keys, values
        q = self.q_proj(x)
        
        kv_input = key_value if key_value is not None else x
        k = self.k_proj(kv_input)
        v = self.v_proj(kv_input)
        
        # Reshape for multi-head attention
        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # Compute attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        
        # Apply mask
        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(1).unsqueeze(2) == 0, float('-inf'))
        
        # Top-K selection
        effective_k = min(self.k, scores.size(-1))
        topk_scores, topk_indices = torch.topk(scores, effective_k, dim=-1)
        
        # Store attention patterns
        self.attention_patterns = topk_indices
        
        # Softmax over top-k only
        attn_weights = F.softmax(topk_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Gather top-k values
        # Expand indices for gathering from v
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, -1, -1, self.head_dim)
        v_expanded = v.unsqueeze(2).expand(-1, -1, seq_len, -1, -1)
        topk_values = torch.gather(v_expanded, 3, expanded_indices)
        
        # Weighted sum
        output = torch.einsum('bhqk,bhqkd->bhqd', attn_weights, topk_values)
        
        # Reshape and project output
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        return output, topk_indices
    
    def get_attention_patterns(self) -> Optional[torch.Tensor]:
        """Return the last computed attention patterns."""
        return self.attention_patterns


class LocalSparseAttention(nn.Module):
    """
    Local sparse attention with sliding window.
    
    Efficient for long sequences by limiting attention to local windows.
    """
    
    def __init__(
        self,
        dim: int,
        n_heads: int = 8,
        window_size: int = 64,
        dropout: float = 0.1
    ):
        """
        Initialize local sparse attention.
        
        Args:
            dim: Embedding dimension
            n_heads: Number of attention heads
            window_size: Size of local attention window
            dropout: Dropout rate
        """
        super().__init__()
        
        assert dim % n_heads == 0
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.window_size = window_size
        self.scale = self.head_dim ** -0.5
        
        self.qkv = nn.Linear(dim, 3 * dim)
        self.out_proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with local attention.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            
        Returns:
            Output tensor [batch, seq_len, dim]
        """
        batch_size, seq_len, _ = x.shape
        
        # Compute Q, K, V
        qkv = self.qkv(x)
        qkv = qkv.view(batch_size, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, batch, heads, seq, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        # Create local attention mask
        mask = self._create_local_mask(seq_len, x.device)
        
        # Compute attention
        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale
        scores = scores + mask
        
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention
        output = torch.matmul(attn, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        return output
    
    def _create_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Create local attention mask."""
        indices = torch.arange(seq_len, device=device)
        row = indices.unsqueeze(1)
        col = indices.unsqueeze(0)
        
        # Allow attention within window
        mask = (row - col).abs() <= self.window_size // 2
        
        # Convert to attention bias
        mask = torch.where(mask, torch.zeros(1, device=device), torch.full((1,), float('-inf'), device=device))
        
        return mask.unsqueeze(0).unsqueeze(0)  # [1, 1, seq, seq]


class SparseMultiHeadAttention(nn.Module):
    """
    Multi-head attention with configurable sparsity pattern.
    
    Supports various sparsity patterns for different use cases.
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
        Initialize sparse multi-head attention.
        
        Args:
            dim: Embedding dimension
            n_heads: Number of attention heads
            sparsity: Sparsity level
            sparsity_type: Type of sparsity pattern
            dropout: Dropout rate
        """
        super().__init__()
        
        assert dim % n_heads == 0
        
        self.dim = dim
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        
        # Create sparse attention heads
        self.heads = nn.ModuleList([
            SparseAttentionHead(self.head_dim, sparsity, sparsity_type)
            for _ in range(n_heads)
        ])
        
        # Projections
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(
        self, 
        x: torch.Tensor, 
        mask: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch, seq_len, dim]
            mask: Optional attention mask
            
        Returns:
            Output tensor and average attention weights
        """
        batch_size, seq_len, _ = x.shape
        
        # Project
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.head_dim)
        
        # Apply each head
        outputs = []
        attention_weights = []
        
        for i, head in enumerate(self.heads):
            head_q = q[:, :, i, :]
            head_k = k[:, :, i, :]
            head_v = v[:, :, i, :]
            
            head_out, head_attn = head(head_q, head_k, head_v, mask)
            outputs.append(head_out)
            attention_weights.append(head_attn)
        
        # Concatenate heads
        output = torch.stack(outputs, dim=2)  # [batch, seq, heads, head_dim]
        output = output.view(batch_size, seq_len, self.dim)
        output = self.out_proj(output)
        
        # Average attention weights
        avg_attention = torch.stack(attention_weights, dim=1).mean(dim=1)
        
        return output, avg_attention
