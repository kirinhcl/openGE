"""Fusion layers for combining genetic and environmental representations."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class ConcatFusion(nn.Module):
    """
    Simple concatenation fusion of two representations.
    
    Concatenates genetic and environmental embeddings and optionally
    projects to a lower dimension.
    """
    
    def __init__(self, dim1: int, dim2: int, output_dim: Optional[int] = None, dropout: float = 0.1):
        """
        Initialize concatenation fusion.
        
        Args:
            dim1: Dimension of first input (genetic)
            dim2: Dimension of second input (environment)
            output_dim: Optional output projection dimension
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim if output_dim else dim1 + dim2
        
        if output_dim is not None:
            self.projection = nn.Sequential(
                nn.Linear(dim1 + dim2, output_dim * 2),
                nn.LayerNorm(output_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim),
                nn.LayerNorm(output_dim)
            )
        else:
            self.projection = None
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Concatenate two inputs.
        
        Args:
            x1: First input representation [batch_size, dim1]
            x2: Second input representation [batch_size, dim2]
            
        Returns:
            Concatenated representation [batch_size, output_dim]
        """
        # Concatenate along feature dimension
        fused = torch.cat([x1, x2], dim=-1)
        
        # Project if specified
        if self.projection is not None:
            fused = self.projection(fused)
        
        return fused


class AttentionFusion(nn.Module):
    """
    Attention-based fusion of genetic and environmental data.
    
    Uses cross-attention to model G×E interactions where one modality
    attends to the other.
    """
    
    def __init__(
        self, 
        dim1: int, 
        dim2: int, 
        hidden_dim: int = 256,
        n_heads: int = 4,
        dropout: float = 0.1
    ):
        """
        Initialize attention fusion.
        
        Args:
            dim1: Dimension of first input (e.g., genetic)
            dim2: Dimension of second input (e.g., environment)
            hidden_dim: Hidden dimension for attention computation
            n_heads: Number of attention heads
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.hidden_dim = hidden_dim
        self.n_heads = n_heads
        
        # Project both inputs to common dimension
        self.proj1 = nn.Linear(dim1, hidden_dim)
        self.proj2 = nn.Linear(dim2, hidden_dim)
        
        # Cross-attention: genetic attends to environment
        self.cross_attn_g2e = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Cross-attention: environment attends to genetic
        self.cross_attn_e2g = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
        
        # Layer norms
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.norm2 = nn.LayerNorm(hidden_dim)
        
        # Feedforward fusion
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim)
        )
        
        # Store attention weights for interpretability
        self.attention_weights_g2e = None
        self.attention_weights_e2g = None
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Fuse inputs using attention mechanism.
        
        Args:
            x1: First input representation (genetic) [batch_size, dim1]
            x2: Second input representation (environment) [batch_size, dim2]
            
        Returns:
            Fused representation [batch_size, hidden_dim] and attention weights
        """
        # Project to common dimension
        h1 = self.proj1(x1).unsqueeze(1)  # [batch, 1, hidden_dim]
        h2 = self.proj2(x2).unsqueeze(1)  # [batch, 1, hidden_dim]
        
        # Cross-attention: genetic attends to environment
        h1_cross, attn_g2e = self.cross_attn_g2e(h1, h2, h2)
        h1_cross = self.norm1(h1 + h1_cross)
        
        # Cross-attention: environment attends to genetic
        h2_cross, attn_e2g = self.cross_attn_e2g(h2, h1, h1)
        h2_cross = self.norm2(h2 + h2_cross)
        
        # Store attention weights
        self.attention_weights_g2e = attn_g2e
        self.attention_weights_e2g = attn_e2g
        
        # Combine and fuse
        combined = torch.cat([h1_cross.squeeze(1), h2_cross.squeeze(1)], dim=-1)
        fused = self.ffn(combined)
        
        # Combine attention weights
        attention_weights = (attn_g2e + attn_e2g) / 2
        
        return fused, attention_weights.squeeze()


class GatingFusion(nn.Module):
    """
    Gating-based fusion with learned weights for each modality.
    
    Learns to dynamically weight the contribution of genetic vs 
    environmental information based on the input.
    """
    
    def __init__(self, dim1: int, dim2: int, output_dim: Optional[int] = None, dropout: float = 0.1):
        """
        Initialize gating fusion.
        
        Args:
            dim1: Dimension of first input (genetic)
            dim2: Dimension of second input (environment)
            output_dim: Output dimension (default: max(dim1, dim2))
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim if output_dim else max(dim1, dim2)
        
        # Project both to same dimension
        self.proj1 = nn.Linear(dim1, self.output_dim)
        self.proj2 = nn.Linear(dim2, self.output_dim)
        
        # Gating network - determines contribution of each modality
        self.gate = nn.Sequential(
            nn.Linear(dim1 + dim2, self.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(self.output_dim, 2),  # 2 gates for 2 modalities
            nn.Softmax(dim=-1)
        )
        
        # Output transformation
        self.output_transform = nn.Sequential(
            nn.Linear(self.output_dim, self.output_dim),
            nn.LayerNorm(self.output_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Store gate values for interpretability
        self.gate_values = None
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Fuse inputs using gating mechanism.
        
        Args:
            x1: First input representation (genetic) [batch_size, dim1]
            x2: Second input representation (environment) [batch_size, dim2]
            
        Returns:
            Fused representation with learned modality weights [batch_size, output_dim]
        """
        # Project both to same dimension
        h1 = self.proj1(x1)  # [batch, output_dim]
        h2 = self.proj2(x2)  # [batch, output_dim]
        
        # Compute gates
        concat = torch.cat([x1, x2], dim=-1)
        gates = self.gate(concat)  # [batch, 2]
        
        # Store gate values
        self.gate_values = gates
        
        # Weight and combine
        g1 = gates[:, 0:1]  # [batch, 1]
        g2 = gates[:, 1:2]  # [batch, 1]
        
        fused = g1 * h1 + g2 * h2  # [batch, output_dim]
        
        # Transform output
        fused = self.output_transform(fused)
        
        return fused
    
    def get_gate_values(self) -> Optional[torch.Tensor]:
        """Get the last computed gate values for interpretability."""
        return self.gate_values


class BilinearFusion(nn.Module):
    """
    Bilinear fusion for explicit G×E interaction modeling.
    
    Computes explicit outer product between genetic and environmental
    representations to model their interactions.
    """
    
    def __init__(
        self, 
        dim1: int, 
        dim2: int, 
        output_dim: int,
        rank: int = 32,
        dropout: float = 0.1
    ):
        """
        Initialize bilinear fusion.
        
        Args:
            dim1: Dimension of first input (genetic)
            dim2: Dimension of second input (environment)
            output_dim: Output dimension
            rank: Rank for low-rank approximation
            dropout: Dropout rate
        """
        super().__init__()
        
        self.dim1 = dim1
        self.dim2 = dim2
        self.output_dim = output_dim
        self.rank = rank
        
        # Low-rank factorization of bilinear weight tensor
        # W ≈ U @ V^T where U: [output_dim, rank, dim1], V: [output_dim, rank, dim2]
        self.U = nn.Parameter(torch.randn(output_dim, rank, dim1) * 0.01)
        self.V = nn.Parameter(torch.randn(output_dim, rank, dim2) * 0.01)
        
        self.bias = nn.Parameter(torch.zeros(output_dim))
        
        # Additional linear terms
        self.linear1 = nn.Linear(dim1, output_dim)
        self.linear2 = nn.Linear(dim2, output_dim)
        
        # Output normalization
        self.norm = nn.LayerNorm(output_dim)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """
        Fuse using bilinear interaction.
        
        Args:
            x1: First input (genetic) [batch_size, dim1]
            x2: Second input (environment) [batch_size, dim2]
            
        Returns:
            Fused representation [batch_size, output_dim]
        """
        batch_size = x1.size(0)
        
        # Bilinear term: sum over rank
        # x1 @ U^T: [batch, output_dim, rank]
        Ux1 = torch.einsum('bi,ori->bor', x1, self.U)
        # x2 @ V^T: [batch, output_dim, rank]
        Vx2 = torch.einsum('bi,ori->bor', x2, self.V)
        # Element-wise product and sum over rank
        bilinear = (Ux1 * Vx2).sum(dim=-1)  # [batch, output_dim]
        
        # Linear terms
        linear = self.linear1(x1) + self.linear2(x2)
        
        # Combine
        output = bilinear + linear + self.bias
        output = self.norm(output)
        output = self.dropout(output)
        
        return output


class MultiModalFusion(nn.Module):
    """
    Multi-modal fusion combining multiple fusion strategies.
    
    Can handle genetic, environmental (temporal), and static features.
    """
    
    def __init__(
        self,
        genetic_dim: int,
        env_dim: int,
        static_dim: Optional[int] = None,
        output_dim: int = 256,
        fusion_type: str = 'attention',
        dropout: float = 0.1
    ):
        """
        Initialize multi-modal fusion.
        
        Args:
            genetic_dim: Genetic embedding dimension
            env_dim: Environment embedding dimension
            static_dim: Static feature dimension (soil, EC) - optional
            output_dim: Output dimension
            fusion_type: Type of fusion ('concat', 'attention', 'gating', 'bilinear')
            dropout: Dropout rate
        """
        super().__init__()
        
        self.genetic_dim = genetic_dim
        self.env_dim = env_dim
        self.static_dim = static_dim
        self.output_dim = output_dim
        self.fusion_type = fusion_type
        
        # Primary G×E fusion
        if fusion_type == 'concat':
            self.gxe_fusion = ConcatFusion(genetic_dim, env_dim, output_dim, dropout)
        elif fusion_type == 'attention':
            self.gxe_fusion = AttentionFusion(genetic_dim, env_dim, output_dim, dropout=dropout)
        elif fusion_type == 'gating':
            self.gxe_fusion = GatingFusion(genetic_dim, env_dim, output_dim, dropout)
        elif fusion_type == 'bilinear':
            self.gxe_fusion = BilinearFusion(genetic_dim, env_dim, output_dim, dropout=dropout)
        else:
            raise ValueError(f"Unknown fusion type: {fusion_type}")
        
        # Static feature integration (if provided)
        if static_dim is not None:
            self.static_projection = nn.Sequential(
                nn.Linear(static_dim, output_dim // 2),
                nn.LayerNorm(output_dim // 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
            
            self.final_fusion = nn.Sequential(
                nn.Linear(output_dim + output_dim // 2, output_dim),
                nn.LayerNorm(output_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout)
            )
        else:
            self.static_projection = None
            self.final_fusion = None
    
    def forward(
        self, 
        genetic: torch.Tensor, 
        environment: torch.Tensor,
        static_features: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Fuse all modalities.
        
        Args:
            genetic: Genetic embedding [batch_size, genetic_dim]
            environment: Environment embedding [batch_size, env_dim]
            static_features: Optional static features [batch_size, static_dim]
            
        Returns:
            Fused representation [batch_size, output_dim]
        """
        # Primary G×E fusion
        if self.fusion_type == 'attention':
            gxe_fused, _ = self.gxe_fusion(genetic, environment)
        else:
            gxe_fused = self.gxe_fusion(genetic, environment)
        
        # Integrate static features if provided
        if static_features is not None and self.static_projection is not None:
            static_emb = self.static_projection(static_features)
            combined = torch.cat([gxe_fused, static_emb], dim=-1)
            output = self.final_fusion(combined)
        else:
            output = gxe_fused
        
        return output
