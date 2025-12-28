"""
模型稀疏性分析 (Sparsity Analysis)

分析稀疏模型的权重分布和稀疏模式:
    - 层级稀疏度统计
    - 权重分布分析
    - 瓶颈层识别
    - 稀疏模式可视化
    - 剪枝前后模型比较
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional, Any


class SparsityAnalyzer:
    """
    稀疏模型分析器。
    
    分析稀疏模型的权重分布、稀疏模式，并提供可视化支持。
    """
    
    def __init__(self, model: nn.Module, threshold: float = 1e-6):
        """
        初始化稀疏性分析器。
        
        Args:
            model: 要分析的稀疏模型
            threshold: 判断为零的阈值
        """
        self.model = model
        self.threshold = threshold
        
        # 缓存层信息
        self._layer_info = self._collect_layer_info()
    
    def _collect_layer_info(self) -> Dict[str, Dict[str, Any]]:
        """收集所有可训练层的信息。"""
        layer_info = {}
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'weight') and module.weight is not None:
                weight = module.weight.detach()
                layer_info[name] = {
                    'module': module,
                    'weight_shape': tuple(weight.shape),
                    'n_params': weight.numel(),
                    'has_bias': hasattr(module, 'bias') and module.bias is not None
                }
        
        return layer_info
    
    def compute_sparsity_levels(self) -> Dict[str, float]:
        """
        计算每层的稀疏度。
        
        Returns:
            层名到稀疏度百分比的映射
        """
        sparsity_levels = {}
        
        for name, info in self._layer_info.items():
            weight = info['module'].weight.detach()
            
            # 计算零值的比例
            n_zeros = torch.sum(torch.abs(weight) < self.threshold).item()
            n_total = weight.numel()
            
            sparsity = n_zeros / n_total * 100
            sparsity_levels[name] = sparsity
        
        # 添加全局统计
        total_zeros = 0
        total_params = 0
        for name, info in self._layer_info.items():
            weight = info['module'].weight.detach()
            total_zeros += torch.sum(torch.abs(weight) < self.threshold).item()
            total_params += weight.numel()
        
        if total_params > 0:
            sparsity_levels['__global__'] = total_zeros / total_params * 100
        
        return sparsity_levels
    
    def analyze_weight_distribution(self) -> Dict:
        """
        分析权重分布 (包括零值)。
        
        Returns:
            权重分布统计信息
        """
        all_weights = []
        layer_stats = {}
        
        for name, info in self._layer_info.items():
            weight = info['module'].weight.detach().cpu().numpy().flatten()
            all_weights.extend(weight)
            
            # 非零权重
            nonzero_weight = weight[np.abs(weight) > self.threshold]
            
            layer_stats[name] = {
                'mean': float(np.mean(weight)),
                'std': float(np.std(weight)),
                'min': float(np.min(weight)),
                'max': float(np.max(weight)),
                'median': float(np.median(weight)),
                'nonzero_mean': float(np.mean(nonzero_weight)) if len(nonzero_weight) > 0 else 0.0,
                'nonzero_std': float(np.std(nonzero_weight)) if len(nonzero_weight) > 0 else 0.0,
                'percentile_5': float(np.percentile(weight, 5)),
                'percentile_95': float(np.percentile(weight, 95)),
                'n_positive': int(np.sum(weight > self.threshold)),
                'n_negative': int(np.sum(weight < -self.threshold)),
                'n_zero': int(np.sum(np.abs(weight) <= self.threshold))
            }
        
        all_weights = np.array(all_weights)
        
        # 全局统计
        global_stats = {
            'mean': float(np.mean(all_weights)),
            'std': float(np.std(all_weights)),
            'min': float(np.min(all_weights)),
            'max': float(np.max(all_weights)),
            'median': float(np.median(all_weights)),
            'total_params': len(all_weights),
            'histogram': np.histogram(all_weights, bins=50)
        }
        
        return {
            'layer_stats': layer_stats,
            'global_stats': global_stats
        }
    
    def find_bottleneck_layers(
        self,
        top_k: int = 5,
        min_params: int = 100
    ) -> List[Tuple[str, float]]:
        """
        识别稀疏度最高的层 (潜在瓶颈)。
        
        Args:
            top_k: 返回前 k 个最稀疏的层
            min_params: 最小参数量阈值 (忽略太小的层)
            
        Returns:
            瓶颈层列表 [(层名, 稀疏度), ...]
        """
        sparsity_levels = self.compute_sparsity_levels()
        
        # 过滤并排序
        filtered_layers = []
        for name, sparsity in sparsity_levels.items():
            if name == '__global__':
                continue
            if self._layer_info[name]['n_params'] >= min_params:
                filtered_layers.append((name, sparsity))
        
        # 按稀疏度降序排序
        filtered_layers.sort(key=lambda x: x[1], reverse=True)
        
        return filtered_layers[:top_k]
    
    def visualize_sparsity_pattern(
        self,
        layer_name: str,
        mode: str = 'binary'
    ) -> np.ndarray:
        """
        可视化某层的稀疏模式。
        
        Args:
            layer_name: 层名
            mode: 可视化模式
                - 'binary': 0/1 掩码
                - 'magnitude': 权重绝对值
                - 'signed': 保留符号
            
        Returns:
            用于可视化的 2D 数组
        """
        if layer_name not in self._layer_info:
            raise ValueError(f"Layer '{layer_name}' not found")
        
        weight = self._layer_info[layer_name]['module'].weight.detach().cpu().numpy()
        
        # 展平为 2D 矩阵以便可视化
        original_shape = weight.shape
        if len(original_shape) == 1:
            weight = weight.reshape(1, -1)
        elif len(original_shape) > 2:
            # 将高维权重展平为 2D
            weight = weight.reshape(original_shape[0], -1)
        
        if mode == 'binary':
            pattern = (np.abs(weight) > self.threshold).astype(float)
        elif mode == 'magnitude':
            pattern = np.abs(weight)
        elif mode == 'signed':
            pattern = weight
        else:
            raise ValueError(f"Unknown mode: {mode}")
        
        return pattern
    
    def compare_pruned_vs_dense(
        self,
        dense_model: nn.Module
    ) -> Dict:
        """
        比较剪枝后模型与原始密集模型。
        
        Args:
            dense_model: 原始未剪枝模型
            
        Returns:
            比较指标
        """
        dense_analyzer = SparsityAnalyzer(dense_model, self.threshold)
        
        sparse_sparsity = self.compute_sparsity_levels()
        dense_sparsity = dense_analyzer.compute_sparsity_levels()
        
        sparse_distribution = self.analyze_weight_distribution()
        dense_distribution = dense_analyzer.analyze_weight_distribution()
        
        # 参数数量比较
        sparse_nonzero = 0
        sparse_total = 0
        dense_nonzero = 0
        dense_total = 0
        
        for name in self._layer_info:
            weight = self._layer_info[name]['module'].weight.detach()
            sparse_nonzero += torch.sum(torch.abs(weight) > self.threshold).item()
            sparse_total += weight.numel()
        
        for name in dense_analyzer._layer_info:
            weight = dense_analyzer._layer_info[name]['module'].weight.detach()
            dense_nonzero += torch.sum(torch.abs(weight) > self.threshold).item()
            dense_total += weight.numel()
        
        compression_ratio = dense_total / max(sparse_nonzero, 1)
        
        return {
            'sparse_global_sparsity': sparse_sparsity.get('__global__', 0),
            'dense_global_sparsity': dense_sparsity.get('__global__', 0),
            'sparsity_increase': sparse_sparsity.get('__global__', 0) - dense_sparsity.get('__global__', 0),
            'sparse_nonzero_params': sparse_nonzero,
            'dense_nonzero_params': dense_nonzero,
            'compression_ratio': compression_ratio,
            'memory_reduction_pct': (1 - sparse_nonzero / max(dense_total, 1)) * 100,
            'layer_comparison': {
                name: {
                    'sparse_sparsity': sparse_sparsity.get(name, 0),
                    'dense_sparsity': dense_sparsity.get(name, 0),
                    'difference': sparse_sparsity.get(name, 0) - dense_sparsity.get(name, 0)
                }
                for name in self._layer_info
            }
        }
    
    def get_pruning_mask(self, layer_name: str) -> torch.Tensor:
        """
        获取某层的剪枝掩码。
        
        Args:
            layer_name: 层名
            
        Returns:
            布尔掩码张量
        """
        if layer_name not in self._layer_info:
            raise ValueError(f"Layer '{layer_name}' not found")
        
        weight = self._layer_info[layer_name]['module'].weight.detach()
        mask = torch.abs(weight) > self.threshold
        
        return mask
    
    def layer_wise_summary(self) -> Dict:
        """
        生成层级汇总报告。
        
        Returns:
            每层的详细汇总
        """
        sparsity_levels = self.compute_sparsity_levels()
        distribution = self.analyze_weight_distribution()
        
        summary = {}
        
        for name, info in self._layer_info.items():
            layer_dist = distribution['layer_stats'].get(name, {})
            
            summary[name] = {
                'shape': info['weight_shape'],
                'n_params': info['n_params'],
                'sparsity_pct': sparsity_levels.get(name, 0),
                'nonzero_params': info['n_params'] - layer_dist.get('n_zero', 0),
                'mean': layer_dist.get('mean', 0),
                'std': layer_dist.get('std', 0),
                'has_bias': info['has_bias']
            }
        
        return summary
    
    def structured_sparsity_analysis(self) -> Dict:
        """
        分析结构化稀疏性 (行/列/通道级别)。
        
        Returns:
            结构化稀疏性指标
        """
        results = {}
        
        for name, info in self._layer_info.items():
            weight = info['module'].weight.detach()
            
            if weight.dim() < 2:
                continue
            
            # 对于 2D 权重 (Linear 层)
            weight_2d = weight.view(weight.shape[0], -1)
            
            # 行稀疏性 (输出神经元)
            row_norms = torch.norm(weight_2d, dim=1)
            zero_rows = torch.sum(row_norms < self.threshold).item()
            row_sparsity = zero_rows / weight_2d.shape[0] * 100
            
            # 列稀疏性 (输入特征)
            col_norms = torch.norm(weight_2d, dim=0)
            zero_cols = torch.sum(col_norms < self.threshold).item()
            col_sparsity = zero_cols / weight_2d.shape[1] * 100
            
            results[name] = {
                'row_sparsity': row_sparsity,
                'col_sparsity': col_sparsity,
                'zero_rows': zero_rows,
                'zero_cols': zero_cols,
                'total_rows': weight_2d.shape[0],
                'total_cols': weight_2d.shape[1]
            }
            
            # 对于 4D 权重 (Conv 层)
            if weight.dim() == 4:
                # 通道/滤波器稀疏性
                filter_norms = torch.norm(weight.view(weight.shape[0], -1), dim=1)
                zero_filters = torch.sum(filter_norms < self.threshold).item()
                filter_sparsity = zero_filters / weight.shape[0] * 100
                
                results[name]['filter_sparsity'] = filter_sparsity
                results[name]['zero_filters'] = zero_filters
                results[name]['total_filters'] = weight.shape[0]
        
        return results
    
    def importance_weighted_sparsity(
        self,
        importance_scores: Optional[Dict[str, np.ndarray]] = None
    ) -> Dict:
        """
        计算重要性加权的稀疏度。
        
        Args:
            importance_scores: 层名到重要性分数的映射
                              如果为 None，使用权重大小作为重要性
        
        Returns:
            加权稀疏度指标
        """
        results = {}
        
        for name, info in self._layer_info.items():
            weight = info['module'].weight.detach().cpu().numpy()
            
            if importance_scores is not None and name in importance_scores:
                importance = importance_scores[name]
                # 确保形状匹配
                if importance.shape != weight.shape:
                    importance = np.broadcast_to(importance, weight.shape)
            else:
                # 使用权重大小作为重要性
                importance = np.abs(weight)
            
            # 归一化重要性
            importance = importance / (np.sum(importance) + 1e-10)
            
            # 计算被剪枝权重的重要性损失
            pruned_mask = np.abs(weight) <= self.threshold
            importance_lost = np.sum(importance[pruned_mask])
            
            results[name] = {
                'sparsity_pct': np.mean(pruned_mask) * 100,
                'importance_lost': float(importance_lost),
                'retained_importance': 1.0 - float(importance_lost)
            }
        
        return results
