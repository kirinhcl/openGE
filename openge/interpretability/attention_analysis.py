"""
注意力机制分析 (Attention Mechanism Analysis)

用于分析和可视化模型中的注意力权重，理解模型关注点。

功能:
    - 提取多头注意力权重
    - 分析注意力头的重要性
    - 可视化注意力流
    - 识别关键位置/特征
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, Optional, List, Union


class AttentionAnalyzer:
    """
    注意力权重分析器。
    
    用于提取、分析和可视化模型中的注意力机制。
    """
    
    def __init__(self, model: nn.Module):
        """
        初始化注意力分析器。
        
        Args:
            model: 包含注意力机制的模型
        """
        self.model = model
        self.attention_weights: Dict[str, torch.Tensor] = {}
        self.hooks = []
        self._register_hooks()
    
    def _register_hooks(self):
        """注册前向钩子以捕获注意力权重。"""
        def get_attention_hook(name):
            def hook(module, input, output):
                # 处理不同类型的注意力模块
                if isinstance(output, tuple) and len(output) >= 2:
                    # (output, attention_weights)
                    self.attention_weights[name] = output[1].detach()
                elif hasattr(module, 'attention_weights'):
                    self.attention_weights[name] = module.attention_weights.detach()
            return hook
        
        for name, module in self.model.named_modules():
            # 检测注意力模块
            if any(attn_name in name.lower() for attn_name in ['attention', 'attn', 'cross_attn']):
                if isinstance(module, nn.MultiheadAttention):
                    hook = module.register_forward_hook(get_attention_hook(name))
                    self.hooks.append(hook)
                elif hasattr(module, 'forward'):
                    hook = module.register_forward_hook(get_attention_hook(name))
                    self.hooks.append(hook)
    
    def remove_hooks(self):
        """移除所有注册的钩子。"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_attention_heads(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        layer_names: Optional[List[str]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        提取所有注意力头的权重。
        
        Args:
            inputs: 模型输入
            layer_names: 指定要提取的层名称 (None 表示全部)
            
        Returns:
            字典映射层名称到注意力权重
        """
        self.attention_weights = {}
        
        # 前向传播
        self.model.eval()
        with torch.no_grad():
            if isinstance(inputs, tuple):
                _ = self.model(*inputs)
            else:
                _ = self.model(inputs)
        
        # 过滤指定层
        if layer_names is not None:
            return {k: v for k, v in self.attention_weights.items() if k in layer_names}
        
        return self.attention_weights
    
    def analyze_head_importance(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        method: str = 'entropy'
    ) -> Dict[str, np.ndarray]:
        """
        分析不同注意力头的重要性。
        
        Args:
            inputs: 模型输入
            method: 重要性计算方法 ('entropy', 'max', 'variance')
            
        Returns:
            每层每个头的重要性分数
        """
        attention_weights = self.extract_attention_heads(inputs)
        importance_scores = {}
        
        for layer_name, weights in attention_weights.items():
            # weights: [batch, n_heads, query_len, key_len] 或 [batch, query_len, key_len]
            if weights.dim() == 3:
                weights = weights.unsqueeze(1)  # 添加头维度
            
            weights = weights.cpu().numpy()
            n_heads = weights.shape[1]
            
            if method == 'entropy':
                # 熵越高，注意力越分散
                eps = 1e-10
                entropy = -np.sum(weights * np.log(weights + eps), axis=-1)
                head_importance = entropy.mean(axis=(0, 2))  # 平均 batch 和 query
            
            elif method == 'max':
                # 最大注意力值
                head_importance = weights.max(axis=-1).mean(axis=(0, 2))
            
            elif method == 'variance':
                # 方差越高，注意力越集中
                head_importance = weights.var(axis=-1).mean(axis=(0, 2))
            
            else:
                raise ValueError(f"未知方法: {method}")
            
            importance_scores[layer_name] = head_importance
        
        return importance_scores
    
    def get_attention_to_position(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        position: int,
        layer_name: Optional[str] = None
    ) -> Dict[str, np.ndarray]:
        """
        获取指向特定位置的注意力权重。
        
        Args:
            inputs: 模型输入
            position: 目标位置索引
            layer_name: 指定层名称
            
        Returns:
            从所有位置指向目标位置的注意力权重
        """
        attention_weights = self.extract_attention_heads(inputs)
        
        position_attention = {}
        for name, weights in attention_weights.items():
            if layer_name is not None and name != layer_name:
                continue
            
            weights = weights.cpu().numpy()
            
            # 提取指向该位置的注意力
            if weights.ndim == 4:  # [batch, heads, query, key]
                attn_to_pos = weights[:, :, :, position]  # [batch, heads, query]
            else:  # [batch, query, key]
                attn_to_pos = weights[:, :, position]  # [batch, query]
            
            position_attention[name] = attn_to_pos
        
        return position_attention
    
    def visualize_attention_flow(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        aggregate_heads: bool = True
    ) -> np.ndarray:
        """
        可视化注意力在层间的流动。
        
        使用注意力滚动 (Attention Rollout) 方法。
        
        Args:
            inputs: 模型输入
            aggregate_heads: 是否聚合多个头
            
        Returns:
            注意力流矩阵
        """
        attention_weights = self.extract_attention_heads(inputs)
        
        # 按层顺序排列
        layer_names = sorted(attention_weights.keys())
        
        if not layer_names:
            raise ValueError("未找到注意力权重")
        
        # 初始化为单位矩阵
        first_weight = attention_weights[layer_names[0]]
        seq_len = first_weight.shape[-1]
        rollout = np.eye(seq_len)
        
        for layer_name in layer_names:
            weights = attention_weights[layer_name].cpu().numpy()
            
            # 聚合多个头
            if aggregate_heads and weights.ndim == 4:
                weights = weights.mean(axis=1)  # [batch, query, key]
            
            # 取 batch 平均
            weights = weights.mean(axis=0)  # [query, key]
            
            # 添加残差连接
            weights = 0.5 * weights + 0.5 * np.eye(seq_len)
            
            # 归一化
            weights = weights / weights.sum(axis=-1, keepdims=True)
            
            # 滚动乘法
            rollout = rollout @ weights
        
        return rollout
    
    def identify_key_positions(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]],
        top_k: int = 10,
        method: str = 'received'
    ) -> Dict[str, np.ndarray]:
        """
        识别接收/发送最多注意力的关键位置。
        
        Args:
            inputs: 模型输入
            top_k: 返回前 k 个位置
            method: 'received' (被关注) 或 'sent' (发出关注)
            
        Returns:
            每层的关键位置索引和分数
        """
        attention_weights = self.extract_attention_heads(inputs)
        key_positions = {}
        
        for layer_name, weights in attention_weights.items():
            weights = weights.cpu().numpy()
            
            # 聚合头和 batch
            if weights.ndim == 4:
                weights = weights.mean(axis=(0, 1))  # [query, key]
            else:
                weights = weights.mean(axis=0)
            
            if method == 'received':
                # 每个位置接收的总注意力
                scores = weights.sum(axis=0)
            elif method == 'sent':
                # 每个位置发送的总注意力 (通常归一化为1)
                # 使用方差代替
                scores = weights.var(axis=1)
            else:
                raise ValueError(f"未知方法: {method}")
            
            top_indices = np.argsort(scores)[::-1][:top_k]
            key_positions[layer_name] = {
                'indices': top_indices,
                'scores': scores[top_indices]
            }
        
        return key_positions
    
    def compute_attention_distance(
        self,
        inputs: Union[torch.Tensor, Tuple[torch.Tensor, ...]]
    ) -> Dict[str, float]:
        """
        计算注意力的平均距离 (衡量局部 vs 全局注意力)。
        
        Args:
            inputs: 模型输入
            
        Returns:
            每层的平均注意力距离
        """
        attention_weights = self.extract_attention_heads(inputs)
        distances = {}
        
        for layer_name, weights in attention_weights.items():
            weights = weights.cpu().numpy()
            
            if weights.ndim == 4:
                weights = weights.mean(axis=(0, 1))
            else:
                weights = weights.mean(axis=0)
            
            seq_len = weights.shape[0]
            
            # 位置距离矩阵
            pos_i, pos_j = np.meshgrid(np.arange(seq_len), np.arange(seq_len), indexing='ij')
            distance_matrix = np.abs(pos_i - pos_j)
            
            # 加权平均距离
            avg_distance = (weights * distance_matrix).sum() / weights.sum()
            distances[layer_name] = float(avg_distance)
        
        return distances
