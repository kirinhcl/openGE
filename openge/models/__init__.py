"""
模型模块 (Models Module)

包含深度学习和传统线性模型，用于基因组预测和 G×E 交互建模。

子模块:
    - encoders: 特征编码器 (CNN, Transformer, MLP 等)
    - fusion: 多模态融合层
    - heads: 预测头 (回归、分类、不确定性估计等)
    - gxe: G×E 交互模型
    - linear: 传统线性模型 (Ridge, LASSO, GBLUP 等)
    - sparse: 稀疏注意力和模型剪枝
"""

from .encoders import (
    # 基础编码器
    CNNEncoder, 
    TransformerEncoder, 
    MLPEncoder,
    TemporalEncoder,
    GeneticEncoder,
    # 位置编码
    PositionalEncoding,  # 向后兼容别名
    SinusoidalPositionalEncoding,
    LearnablePositionalEncoding,
    RotaryPositionalEncoding,
    ALiBiPositionalEncoding,
    RelativePositionalEncoding,
    create_positional_encoding,
    # 组合编码器
    StackedEncoder,
    ParallelEncoder,
    HierarchicalEncoder,
    MultiInputEncoder,
)
from .fusion import (
    ConcatFusion, 
    AttentionFusion, 
    GatingFusion,
    BilinearFusion,
    MultiModalFusion
)
from .heads import (
    RegressionHead, 
    ClassificationHead,
    MultiTaskRegressionHead,
    UncertaintyHead,
    QuantileHead
)
from .gxe import GxEModel, GxEModelBuilder, create_gxe_model, ModelFactory, ModelEnsemble

# 线性模型
from .linear import (
    RidgeRegression,
    LassoRegression,
    ElasticNet,
    BayesianRidge,
    GBLUP,
    RKHS,
    LinearMixedModel,
    MultiTraitLinear,
)

__all__ = [
    # === 深度学习编码器 ===
    "CNNEncoder",
    "TransformerEncoder",
    "MLPEncoder",
    "TemporalEncoder",
    "GeneticEncoder",
    
    # === 位置编码 ===
    "PositionalEncoding",
    "SinusoidalPositionalEncoding",
    "LearnablePositionalEncoding",
    "RotaryPositionalEncoding",
    "ALiBiPositionalEncoding",
    "RelativePositionalEncoding",
    "create_positional_encoding",
    
    # === 组合编码器 ===
    "StackedEncoder",
    "ParallelEncoder",
    "HierarchicalEncoder",
    "MultiInputEncoder",
    
    # === 融合层 ===
    "ConcatFusion",
    "AttentionFusion",
    "GatingFusion",
    "BilinearFusion",
    "MultiModalFusion",
    
    # === 预测头 ===
    "RegressionHead",
    "ClassificationHead",
    "MultiTaskRegressionHead",
    "UncertaintyHead",
    "QuantileHead",
    
    # === G×E 模型 ===
    "GxEModel",
    "GxEModelBuilder",
    "create_gxe_model",
    "ModelFactory",
    "ModelEnsemble",
    
    # === 线性模型 ===
    "RidgeRegression",
    "LassoRegression",
    "ElasticNet",
    "BayesianRidge",
    "GBLUP",
    "RKHS",
    "LinearMixedModel",
    "MultiTraitLinear",
]
