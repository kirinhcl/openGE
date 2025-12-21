"""G×E (Genotype × Environment) interaction model for trait prediction."""

import torch
import torch.nn as nn
from typing import Tuple, Optional, Dict


class GxEModel(nn.Module):
    """
    G×E interaction model combining genetic and environmental data.
    
    This model captures the interaction between genotype (G) and environment (E)
    to predict quantitative traits in crops.
    """
    
    def __init__(
        self,
        genetic_encoder: nn.Module,
        env_encoder: nn.Module,
        fusion_layer: nn.Module,
        prediction_head: nn.Module,
        use_gxe_interaction: bool = True,
    ):
        """
        Initialize G×E model.
        
        Args:
            genetic_encoder: Encoder for genetic data
            env_encoder: Encoder for environmental data
            fusion_layer: Fusion layer combining both modalities
            prediction_head: Head for final prediction
            use_gxe_interaction: Whether to explicitly model G×E interactions
        """
        super().__init__()
        self.genetic_encoder = genetic_encoder
        self.env_encoder = env_encoder
        self.fusion_layer = fusion_layer
        self.prediction_head = prediction_head
        self.use_gxe_interaction = use_gxe_interaction
    
    def forward(
        self,
        genetic_data: torch.Tensor,
        env_data: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Forward pass for G×E model.
        
        Args:
            genetic_data: Genetic/SNP data [batch_size, n_markers]
            env_data: Environmental data [batch_size, n_env_features]
            
        Returns:
            Predictions and intermediate representations
        """
        # Encode genetic data
        genetic_repr = self.genetic_encoder(genetic_data)
        
        # Encode environmental data
        env_repr = self.env_encoder(env_data)
        
        # Fuse representations
        fused_repr = self.fusion_layer(genetic_repr, env_repr)
        
        # Make predictions
        predictions = self.prediction_head(fused_repr)
        
        # Return predictions and intermediate features for interpretability
        intermediates = {
            "genetic_repr": genetic_repr,
            "env_repr": env_repr,
            "fused_repr": fused_repr,
        }
        
        return predictions, intermediates
    
    def get_gxe_weights(self) -> Optional[torch.Tensor]:
        """
        Extract G×E interaction weights if using explicit interaction term.
        
        Returns:
            G×E interaction weight matrix or None
        """
        pass
