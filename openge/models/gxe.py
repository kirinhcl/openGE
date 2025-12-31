"""
G×E (Genotype × Environment) 交互模型

用于作物性状预测的基因型-环境交互模型，支持：
    - 多种编码器组合 (CNN, Transformer, MLP 等)
    - 多种融合策略 (Attention, Gating, Bilinear 等)
    - 深度学习和线性模型的统一接口
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional, Dict, List, Union, Any

from .encoders import CNNEncoder, TransformerEncoder, MLPEncoder, TemporalEncoder, GeneticEncoder
from .fusion import ConcatFusion, AttentionFusion, GatingFusion, BilinearFusion, MultiModalFusion
from .heads import RegressionHead, ClassificationHead, UncertaintyHead, QuantileHead


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
        static_encoder: Optional[nn.Module] = None,
        use_residual: bool = False,
        genetic_dim: Optional[int] = None,
        env_dim: Optional[int] = None,
        fused_dim: Optional[int] = None,
    ):
        """
        Initialize G×E model.
        
        Args:
            genetic_encoder: Encoder for genetic data
            env_encoder: Encoder for environmental data
            fusion_layer: Fusion layer combining both modalities
            prediction_head: Head for final prediction
            use_gxe_interaction: Whether to explicitly model G×E interactions
            static_encoder: Optional encoder for static features (soil, EC)
            use_residual: Whether to add G and E as residual connections to GxE
            genetic_dim: Dimension of genetic representation (for residual projection)
            env_dim: Dimension of environment representation (for residual projection)
            fused_dim: Dimension of fused representation (for residual projection)
        """
        super().__init__()
        self.genetic_encoder = genetic_encoder
        self.env_encoder = env_encoder
        self.fusion_layer = fusion_layer
        self.prediction_head = prediction_head
        self.use_gxe_interaction = use_gxe_interaction
        self.static_encoder = static_encoder
        self.use_residual = use_residual
        
        # Residual projection layers for G and E
        # Projects G and E to same dimension as fused representation
        if use_residual and genetic_dim is not None and fused_dim is not None:
            self.genetic_residual_proj = nn.Linear(genetic_dim, fused_dim) if genetic_dim != fused_dim else nn.Identity()
            self.env_residual_proj = nn.Linear(env_dim, fused_dim) if env_dim != fused_dim else nn.Identity()
            # Learnable weights for residual contributions
            self.residual_gate = nn.Parameter(torch.ones(3) / 3)  # [gxe_weight, g_weight, e_weight]
        else:
            self.genetic_residual_proj = None
            self.env_residual_proj = None
            self.residual_gate = None
        
        # G×E interaction term if enabled
        if use_gxe_interaction:
            # Get output dimensions from encoders
            self.gxe_interaction = None  # Will be initialized on first forward
    
    def forward(
        self,
        genetic_data: torch.Tensor,
        env_data: torch.Tensor,
        static_data: Optional[torch.Tensor] = None,
        return_intermediates: bool = True
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict]]:
        """
        Forward pass for G×E model.
        
        Args:
            genetic_data: Genetic/SNP data [batch_size, n_markers]
            env_data: Environmental data [batch_size, n_timesteps, n_features] or [batch_size, n_features]
            static_data: Optional static features [batch_size, n_static_features]
            return_intermediates: Whether to return intermediate representations
            
        Returns:
            Predictions and optionally intermediate representations
        """
        # Encode genetic data
        genetic_repr = self.genetic_encoder(genetic_data)
        
        # Encode environmental data
        env_repr = self.env_encoder(env_data)
        
        # Encode static features if provided
        static_repr = None
        if static_data is not None and self.static_encoder is not None:
            static_repr = self.static_encoder(static_data)
        
        # Fuse representations
        if hasattr(self.fusion_layer, 'forward') and static_repr is not None:
            # Check if fusion layer supports static features
            if isinstance(self.fusion_layer, MultiModalFusion):
                fused_repr = self.fusion_layer(genetic_repr, env_repr, static_repr)
            else:
                # Concatenate static features with env representation
                env_repr = torch.cat([env_repr, static_repr], dim=-1)
                fused_result = self.fusion_layer(genetic_repr, env_repr)
                if isinstance(fused_result, tuple):
                    fused_repr = fused_result[0]
                else:
                    fused_repr = fused_result
        else:
            fused_result = self.fusion_layer(genetic_repr, env_repr)
            if isinstance(fused_result, tuple):
                fused_repr = fused_result[0]
            else:
                fused_repr = fused_result
        
        # Apply residual connections: fused = w1*GxE + w2*G + w3*E
        if self.use_residual and self.genetic_residual_proj is not None:
            # Normalize gate weights with softmax
            gate_weights = F.softmax(self.residual_gate, dim=0)
            
            # Project G and E to fused dimension
            genetic_proj = self.genetic_residual_proj(genetic_repr)
            env_proj = self.env_residual_proj(env_repr)
            
            # Combine with learnable weights
            fused_repr = (gate_weights[0] * fused_repr + 
                         gate_weights[1] * genetic_proj + 
                         gate_weights[2] * env_proj)
        
        # Make predictions
        predictions = self.prediction_head(fused_repr)
        
        if return_intermediates:
            # Return predictions and intermediate features for interpretability
            intermediates = {
                "genetic_repr": genetic_repr,
                "env_repr": env_repr,
                "fused_repr": fused_repr,
            }
            if static_repr is not None:
                intermediates["static_repr"] = static_repr
            
            # Add residual gate weights for interpretability
            if self.use_residual and self.residual_gate is not None:
                gate_weights = F.softmax(self.residual_gate, dim=0)
                intermediates["residual_weights"] = {
                    "gxe": gate_weights[0].item(),
                    "genetic": gate_weights[1].item(),
                    "env": gate_weights[2].item(),
                }
            
            return predictions, intermediates
        
        return predictions
    
    def get_gxe_weights(self) -> Optional[torch.Tensor]:
        """
        Extract G×E interaction weights if using explicit interaction term.
        
        Returns:
            G×E interaction weight matrix or None
        """
        if hasattr(self.fusion_layer, 'U') and hasattr(self.fusion_layer, 'V'):
            # BilinearFusion has explicit interaction weights
            return torch.einsum('ori,orj->oij', self.fusion_layer.U, self.fusion_layer.V)
        
        if hasattr(self.fusion_layer, 'gate_values'):
            return self.fusion_layer.gate_values
        
        return None
    
    def get_attention_weights(self) -> Optional[torch.Tensor]:
        """
        Get attention weights from fusion layer.
        
        Returns:
            Attention weights or None
        """
        if hasattr(self.fusion_layer, 'attention_weights_g2e'):
            return self.fusion_layer.attention_weights_g2e
        return None
    
    def get_residual_weights(self) -> Optional[Dict[str, float]]:
        """
        Get residual connection weights (G, E, GxE contributions).
        
        Returns:
            Dictionary with 'gxe', 'genetic', 'env' weights or None
        """
        if self.use_residual and self.residual_gate is not None:
            gate_weights = F.softmax(self.residual_gate, dim=0)
            return {
                "gxe": gate_weights[0].item(),
                "genetic": gate_weights[1].item(),
                "env": gate_weights[2].item(),
            }
        return None


class GxEModelBuilder:
    """
    Builder class for constructing GxEModel with various configurations.
    """
    
    def __init__(self):
        """Initialize builder with default settings."""
        self.config = {
            'genetic_encoder_type': 'cnn',
            'env_encoder_type': 'temporal',
            'fusion_type': 'attention',
            'head_type': 'regression',
            'embedding_dim': 256,
            'n_heads': 8,
            'n_layers': 4,
            'dropout': 0.1,
            'use_residual': False,  # Add G and E as residual to GxE
        }
    
    def set_genetic_encoder(
        self, 
        encoder_type: str, 
        n_markers: int,
        **kwargs
    ) -> 'GxEModelBuilder':
        """Set genetic encoder configuration."""
        self.config['genetic_encoder_type'] = encoder_type
        self.config['n_markers'] = n_markers
        self.config.update(kwargs)
        return self
    
    def set_env_encoder(
        self, 
        encoder_type: str,
        n_features: int,
        n_timesteps: Optional[int] = None,
        **kwargs
    ) -> 'GxEModelBuilder':
        """Set environment encoder configuration."""
        self.config['env_encoder_type'] = encoder_type
        self.config['n_env_features'] = n_features
        self.config['n_timesteps'] = n_timesteps
        self.config.update(kwargs)
        return self
    
    def set_fusion(self, fusion_type: str, **kwargs) -> 'GxEModelBuilder':
        """Set fusion layer configuration."""
        self.config['fusion_type'] = fusion_type
        self.config.update(kwargs)
        return self
    
    def set_head(
        self, 
        head_type: str, 
        n_outputs: int,
        **kwargs
    ) -> 'GxEModelBuilder':
        """Set prediction head configuration."""
        self.config['head_type'] = head_type
        self.config['n_outputs'] = n_outputs
        self.config.update(kwargs)
        return self
    
    def build(self) -> GxEModel:
        """Build and return the GxEModel."""
        config = self.config
        embedding_dim = config.get('embedding_dim', 256)
        dropout = config.get('dropout', 0.1)
        
        # Build genetic encoder
        n_markers = config.get('n_markers', 2000)
        if config['genetic_encoder_type'] == 'cnn':
            genetic_encoder = CNNEncoder(
                input_dim=n_markers,
                output_dim=embedding_dim,
                dropout=dropout
            )
        elif config['genetic_encoder_type'] == 'transformer':
            genetic_encoder = TransformerEncoder(
                input_dim=n_markers,
                output_dim=embedding_dim,
                n_heads=config.get('n_heads', 8),
                n_layers=config.get('n_layers', 4),
                dropout=dropout
            )
        elif config['genetic_encoder_type'] == 'mlp':
            genetic_encoder = MLPEncoder(
                input_dim=n_markers,
                output_dim=embedding_dim,
                dropout=dropout
            )
        elif config['genetic_encoder_type'] == 'genetic':
            genetic_encoder = GeneticEncoder(
                n_markers=n_markers,
                output_dim=embedding_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown genetic encoder type: {config['genetic_encoder_type']}")
        
        # Build environment encoder
        n_features = config.get('n_env_features', 16)
        n_timesteps = config.get('n_timesteps', 366)
        
        if config['env_encoder_type'] == 'temporal':
            env_encoder = TemporalEncoder(
                n_features=n_features,
                n_timesteps=n_timesteps,
                output_dim=embedding_dim,
                dropout=dropout
            )
        elif config['env_encoder_type'] == 'mlp':
            env_encoder = MLPEncoder(
                input_dim=n_features * (n_timesteps if n_timesteps else 1),
                output_dim=embedding_dim,
                dropout=dropout
            )
        elif config['env_encoder_type'] == 'transformer':
            env_encoder = TransformerEncoder(
                input_dim=n_features,
                output_dim=embedding_dim,
                n_heads=config.get('n_heads', 8),
                n_layers=config.get('n_layers', 4),
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown env encoder type: {config['env_encoder_type']}")
        
        # Build fusion layer
        if config['fusion_type'] == 'concat':
            fusion_layer = ConcatFusion(
                dim1=embedding_dim,
                dim2=embedding_dim,
                output_dim=embedding_dim,
                dropout=dropout
            )
        elif config['fusion_type'] == 'attention':
            fusion_layer = AttentionFusion(
                dim1=embedding_dim,
                dim2=embedding_dim,
                hidden_dim=embedding_dim,
                dropout=dropout
            )
        elif config['fusion_type'] == 'gating':
            fusion_layer = GatingFusion(
                dim1=embedding_dim,
                dim2=embedding_dim,
                output_dim=embedding_dim,
                dropout=dropout
            )
        elif config['fusion_type'] == 'bilinear':
            fusion_layer = BilinearFusion(
                dim1=embedding_dim,
                dim2=embedding_dim,
                output_dim=embedding_dim,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown fusion type: {config['fusion_type']}")
        
        # Build prediction head
        n_outputs = config.get('n_outputs', 1)
        if config['head_type'] == 'regression':
            prediction_head = RegressionHead(
                input_dim=embedding_dim,
                n_traits=n_outputs,
                dropout=dropout
            )
        elif config['head_type'] == 'classification':
            prediction_head = ClassificationHead(
                input_dim=embedding_dim,
                n_classes=n_outputs,
                dropout=dropout
            )
        elif config['head_type'] == 'uncertainty':
            prediction_head = UncertaintyHead(
                input_dim=embedding_dim,
                n_traits=n_outputs,
                dropout=dropout
            )
        else:
            raise ValueError(f"Unknown head type: {config['head_type']}")
        
        return GxEModel(
            genetic_encoder=genetic_encoder,
            env_encoder=env_encoder,
            fusion_layer=fusion_layer,
            prediction_head=prediction_head,
            use_gxe_interaction=config.get('use_gxe_interaction', True),
            use_residual=config.get('use_residual', False),
            genetic_dim=embedding_dim,
            env_dim=embedding_dim,
            fused_dim=embedding_dim,
        )


def create_gxe_model(
    n_markers: int,
    n_env_features: int,
    n_traits: int = 1,
    n_timesteps: Optional[int] = None,
    genetic_encoder: str = 'cnn',
    env_encoder: str = 'mlp',
    fusion: str = 'attention',
    embedding_dim: int = 256,
    dropout: float = 0.1,
    use_residual: bool = False,
) -> GxEModel:
    """
    Convenience function to create a GxEModel with common configurations.
    
    Args:
        n_markers: Number of SNP markers
        n_env_features: Number of environment features
        n_traits: Number of traits to predict
        n_timesteps: Number of timesteps (for temporal data)
        genetic_encoder: Type of genetic encoder ('cnn', 'mlp', 'transformer')
        env_encoder: Type of environment encoder ('mlp', 'temporal', 'transformer')
        fusion: Type of fusion ('concat', 'attention', 'gating', 'bilinear')
        embedding_dim: Embedding dimension
        dropout: Dropout rate
        use_residual: Whether to add G and E as residual connections to GxE
        
    Returns:
        Configured GxEModel
    """
    builder = GxEModelBuilder()
    builder.set_genetic_encoder(genetic_encoder, n_markers)
    builder.set_env_encoder(
        env_encoder, 
        n_env_features, 
        n_timesteps=n_timesteps
    )
    builder.set_fusion(fusion)
    builder.set_head('regression', n_traits)
    builder.config['embedding_dim'] = embedding_dim
    builder.config['dropout'] = dropout
    builder.config['use_residual'] = use_residual
    
    return builder.build()


class ModelFactory:
    """
    模型工厂类
    
    提供统一接口创建深度学习和线性模型。
    支持快速切换不同模型进行对比实验。
    """
    
    # 注册的模型类型
    _deep_learning_models = {
        'gxe': 'create_gxe_model',
        'gxe_cnn': 'cnn',
        'gxe_transformer': 'transformer',
        'gxe_mlp': 'mlp',
    }
    
    _linear_models = {
        'ridge': 'RidgeRegression',
        'lasso': 'LassoRegression',
        'elastic_net': 'ElasticNet',
        'bayesian_ridge': 'BayesianRidge',
        'gblup': 'GBLUP',
        'rkhs': 'RKHS',
        'mixed': 'LinearMixedModel',
    }
    
    @classmethod
    def list_models(cls) -> Dict[str, List[str]]:
        """列出所有可用的模型类型。"""
        return {
            'deep_learning': list(cls._deep_learning_models.keys()),
            'linear': list(cls._linear_models.keys()),
        }
    
    @classmethod
    def create(
        cls,
        model_type: str,
        input_dim: int,
        output_dim: int = 1,
        **kwargs
    ) -> nn.Module:
        """
        创建模型实例。
        
        Args:
            model_type: 模型类型 (见 list_models())
            input_dim: 输入维度 (标记数量)
            output_dim: 输出维度 (性状数量)
            **kwargs: 模型特定参数
            
        Returns:
            模型实例
        """
        # 深度学习模型
        if model_type in cls._deep_learning_models:
            return cls._create_deep_model(
                model_type, input_dim, output_dim, **kwargs
            )
        
        # 线性模型
        if model_type in cls._linear_models:
            return cls._create_linear_model(
                model_type, input_dim, output_dim, **kwargs
            )
        
        raise ValueError(
            f"未知模型类型: {model_type}。"
            f"可用模型: {cls.list_models()}"
        )
    
    @classmethod
    def _create_deep_model(
        cls,
        model_type: str,
        n_markers: int,
        n_traits: int,
        n_env_features: int = 16,
        n_timesteps: Optional[int] = 366,  # 默认一年的天数
        embedding_dim: int = 256,
        **kwargs
    ) -> nn.Module:
        """创建深度学习模型。"""
        
        # 确保 n_timesteps 有值
        if n_timesteps is None:
            n_timesteps = 366
        
        if model_type == 'gxe':
            genetic_encoder = kwargs.get('genetic_encoder', 'cnn')
            env_encoder = kwargs.get('env_encoder', 'mlp')
            fusion = kwargs.get('fusion', 'attention')
        elif model_type == 'gxe_cnn':
            genetic_encoder = 'cnn'
            env_encoder = kwargs.get('env_encoder', 'temporal')
            fusion = kwargs.get('fusion', 'attention')
        elif model_type == 'gxe_transformer':
            genetic_encoder = 'transformer'
            env_encoder = kwargs.get('env_encoder', 'transformer')
            fusion = kwargs.get('fusion', 'attention')
        elif model_type == 'gxe_mlp':
            genetic_encoder = 'mlp'
            env_encoder = 'mlp'
            fusion = kwargs.get('fusion', 'concat')
        else:
            genetic_encoder = 'cnn'
            env_encoder = 'mlp'
            fusion = 'attention'
        
        return create_gxe_model(
            n_markers=n_markers,
            n_env_features=n_env_features,
            n_traits=n_traits,
            n_timesteps=n_timesteps,
            genetic_encoder=genetic_encoder,
            env_encoder=env_encoder,
            fusion=fusion,
            embedding_dim=embedding_dim,
            dropout=kwargs.get('dropout', 0.1)
        )
    
    @classmethod
    def _create_linear_model(
        cls,
        model_type: str,
        input_dim: int,
        output_dim: int,
        **kwargs
    ) -> nn.Module:
        """创建线性模型。"""
        from .linear import (
            RidgeRegression, LassoRegression, ElasticNet,
            BayesianRidge, GBLUP, RKHS, LinearMixedModel
        )
        
        if model_type == 'ridge':
            return RidgeRegression(
                input_dim=input_dim,
                output_dim=output_dim,
                alpha=kwargs.get('alpha', 1.0)
            )
        
        elif model_type == 'lasso':
            return LassoRegression(
                input_dim=input_dim,
                output_dim=output_dim,
                alpha=kwargs.get('alpha', 0.01)
            )
        
        elif model_type == 'elastic_net':
            return ElasticNet(
                input_dim=input_dim,
                output_dim=output_dim,
                alpha=kwargs.get('alpha', 1.0),
                l1_ratio=kwargs.get('l1_ratio', 0.5)
            )
        
        elif model_type == 'bayesian_ridge':
            return BayesianRidge(
                input_dim=input_dim,
                output_dim=output_dim
            )
        
        elif model_type == 'gblup':
            n_samples = kwargs.get('n_samples', input_dim)
            return GBLUP(
                n_samples=n_samples,
                heritability=kwargs.get('heritability', 0.5)
            )
        
        elif model_type == 'rkhs':
            return RKHS(
                kernel=kwargs.get('kernel', 'rbf'),
                alpha=kwargs.get('alpha', 1.0)
            )
        
        elif model_type == 'mixed':
            return LinearMixedModel(
                n_fixed=kwargs.get('n_fixed', 10),
                n_random=kwargs.get('n_random', input_dim),
                heritability=kwargs.get('heritability', 0.5)
            )
        
        raise ValueError(f"未知线性模型类型: {model_type}")


class ModelEnsemble(nn.Module):
    """
    模型集成
    
    组合多个模型的预测结果。
    
    支持的集成策略:
        - mean: 简单平均
        - weighted: 加权平均
        - stacking: 堆叠学习
    """
    
    def __init__(
        self,
        models: List[nn.Module],
        strategy: str = 'mean',
        weights: Optional[List[float]] = None
    ):
        """
        Args:
            models: 模型列表
            strategy: 集成策略 ('mean', 'weighted', 'stacking')
            weights: 加权平均的权重
        """
        super().__init__()
        
        self.models = nn.ModuleList(models)
        self.strategy = strategy
        self.n_models = len(models)
        
        if strategy == 'weighted':
            if weights is None:
                weights = [1.0 / self.n_models] * self.n_models
            self.register_buffer('weights', torch.tensor(weights))
        
        elif strategy == 'stacking':
            # 元学习器：学习如何组合模型输出
            self.meta_learner = nn.Linear(self.n_models, 1)
    
    def forward(
        self, 
        *args, 
        **kwargs
    ) -> torch.Tensor:
        """
        前向传播。
        
        将相同输入传递给所有模型，并组合输出。
        """
        # 获取所有模型的预测
        predictions = []
        for model in self.models:
            pred = model(*args, **kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]  # 取第一个输出 (预测值)
            predictions.append(pred)
        
        # 堆叠预测 [batch, n_models, ...]
        stacked = torch.stack(predictions, dim=1)
        
        # 集成
        if self.strategy == 'mean':
            return stacked.mean(dim=1)
        
        elif self.strategy == 'weighted':
            weights = self.weights.view(1, -1, *([1] * (stacked.dim() - 2)))
            return (stacked * weights).sum(dim=1)
        
        elif self.strategy == 'stacking':
            # 对最后一个维度应用元学习器
            # [batch, n_models] -> [batch, 1]
            if stacked.dim() == 2:
                return self.meta_learner(stacked)
            else:
                # 更高维度：展平后处理
                shape = stacked.shape
                flat = stacked.view(shape[0], shape[1], -1)
                # 对每个位置独立应用
                outputs = []
                for i in range(flat.size(-1)):
                    outputs.append(self.meta_learner(flat[..., i]))
                return torch.stack(outputs, dim=-1).view(shape[0], *shape[2:])
        
        return stacked.mean(dim=1)
    
    def get_individual_predictions(
        self, 
        *args, 
        **kwargs
    ) -> List[torch.Tensor]:
        """获取每个模型的单独预测。"""
        predictions = []
        for model in self.models:
            pred = model(*args, **kwargs)
            if isinstance(pred, tuple):
                pred = pred[0]
            predictions.append(pred)
        return predictions
