"""
SHAP (SHapley Additive exPlanations) 模型解释

基于博弈论的 Shapley 值来解释模型预测。
提供:
    - 单实例解释
    - 全局特征重要性
    - 特征交互效应分析
"""

import math
import numpy as np
from typing import Dict, Optional, List, Tuple, Union
import torch
import torch.nn as nn


class SHAPExplainer:
    """
    基于 SHAP 的模型解释和分析。
    
    实现多种 SHAP 解释方法，包括:
    - Kernel SHAP (模型无关)
    - 梯度 SHAP (基于梯度)
    - 采样 SHAP (采样近似)
    """
    
    def __init__(
        self,
        model: nn.Module,
        background_data: np.ndarray,
        device: str = 'cpu'
    ):
        """
        初始化 SHAP 解释器。
        
        Args:
            model: 训练好的模型
            background_data: 背景/参考数据，用于计算期望值
            device: 计算设备
        """
        self.model = model
        self.background_data = background_data
        self.device = device
        self.model.to(device)
        self.model.eval()
        
        # 预计算背景数据的预测
        self._background_preds = self._predict(background_data)
        self._expected_value = np.mean(self._background_preds)
    
    def _predict(self, X: np.ndarray) -> np.ndarray:
        """使用模型进行预测。"""
        with torch.no_grad():
            X_tensor = torch.tensor(X, dtype=torch.float32).to(self.device)
            output = self.model(X_tensor)
            if isinstance(output, tuple):
                output = output[0]
            return output.cpu().numpy().flatten()
    
    def explain_instance(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None,
        n_samples: int = 100
    ) -> Dict:
        """
        计算单个实例的 SHAP 值。
        
        使用 Kernel SHAP 近似方法计算 Shapley 值。
        
        Args:
            instance: 单个数据实例 [n_features]
            feature_names: 特征名称
            n_samples: 采样数量
            
        Returns:
            包含 SHAP 值和特征名称的字典
        """
        if instance.ndim == 1:
            instance = instance.reshape(1, -1)
        
        n_features = instance.shape[1]
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # 使用采样方法近似 Shapley 值
        shap_values = self._kernel_shap(instance[0], n_samples)
        
        # 预测值分解: f(x) ≈ E[f(x)] + sum(SHAP_i)
        prediction = self._predict(instance)[0]
        
        return {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'expected_value': self._expected_value,
            'prediction': prediction,
            'attribution_sum': np.sum(shap_values),
            # 重要性排序
            'feature_importance_order': np.argsort(np.abs(shap_values))[::-1]
        }
    
    def _kernel_shap(
        self,
        instance: np.ndarray,
        n_samples: int
    ) -> np.ndarray:
        """
        使用 Kernel SHAP 计算 Shapley 值。
        
        Kernel SHAP 使用加权线性回归来近似 Shapley 值。
        """
        n_features = len(instance)
        
        # 生成联盟样本
        coalitions = np.random.randint(0, 2, size=(n_samples, n_features))
        
        # 确保包含空集和全集
        coalitions[0] = 0
        coalitions[1] = 1
        
        # 计算每个联盟的预测
        predictions = np.zeros(n_samples)
        
        for i, coalition in enumerate(coalitions):
            # 创建混合实例
            mixed_instance = np.zeros((len(self.background_data), n_features))
            
            for j in range(len(self.background_data)):
                for k in range(n_features):
                    if coalition[k] == 1:
                        mixed_instance[j, k] = instance[k]
                    else:
                        mixed_instance[j, k] = self.background_data[j, k]
            
            # 平均预测
            predictions[i] = np.mean(self._predict(mixed_instance))
        
        # 计算 Shapley 权重
        weights = self._shapley_kernel_weights(coalitions)
        
        # 加权线性回归
        shap_values = self._weighted_linear_regression(
            coalitions, predictions, weights
        )
        
        return shap_values
    
    def _shapley_kernel_weights(self, coalitions: np.ndarray) -> np.ndarray:
        """计算 Kernel SHAP 权重。"""
        n_samples, n_features = coalitions.shape
        weights = np.zeros(n_samples)
        
        for i, coalition in enumerate(coalitions):
            n_present = np.sum(coalition)
            if n_present == 0 or n_present == n_features:
                weights[i] = 1e-10  # 避免除以零
            else:
                # SHAP 核权重
                weights[i] = (n_features - 1) / (
                    math.comb(n_features, int(n_present)) * 
                    n_present * (n_features - n_present)
                )
        
        return weights
    
    def _weighted_linear_regression(
        self,
        X: np.ndarray,
        y: np.ndarray,
        weights: np.ndarray
    ) -> np.ndarray:
        """加权线性回归求解 SHAP 值。"""
        # 添加截距项
        X_with_intercept = np.column_stack([np.ones(len(X)), X])
        
        # 加权最小二乘
        W = np.diag(weights)
        try:
            XtWX = X_with_intercept.T @ W @ X_with_intercept
            XtWy = X_with_intercept.T @ W @ y
            
            # 正则化以避免奇异矩阵
            reg = 1e-6 * np.eye(XtWX.shape[0])
            coefficients = np.linalg.solve(XtWX + reg, XtWy)
        except np.linalg.LinAlgError:
            # 使用伪逆
            coefficients = np.linalg.lstsq(
                X_with_intercept, y, rcond=None
            )[0]
        
        # 返回特征系数（不包括截距）
        return coefficients[1:]
    
    def summary_plot_data(
        self,
        X: np.ndarray,
        feature_names: Optional[List[str]] = None,
        max_samples: int = 100
    ) -> Dict:
        """
        准备 SHAP 摘要图数据。
        
        Args:
            X: 输入数据
            feature_names: 特征名称
            max_samples: 最大样本数
            
        Returns:
            用于绘制摘要图的 SHAP 值和元数据
        """
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
        
        n_samples, n_features = X.shape
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # 计算所有样本的 SHAP 值
        all_shap_values = np.zeros((n_samples, n_features))
        
        for i in range(n_samples):
            result = self.explain_instance(X[i], feature_names, n_samples=50)
            all_shap_values[i] = result['shap_values']
        
        # 计算全局特征重要性
        global_importance = np.abs(all_shap_values).mean(axis=0)
        importance_order = np.argsort(global_importance)[::-1]
        
        return {
            'shap_values': all_shap_values,
            'feature_values': X,
            'feature_names': feature_names,
            'global_importance': global_importance,
            'importance_order': importance_order,
            'expected_value': self._expected_value
        }
    
    def interaction_effects(
        self,
        X: np.ndarray,
        max_samples: int = 50
    ) -> np.ndarray:
        """
        分析特征交互效应。
        
        使用 SHAP 交互值来量化特征对之间的交互。
        
        Args:
            X: 输入数据
            max_samples: 最大样本数
            
        Returns:
            交互矩阵 [n_features, n_features]
        """
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X = X[indices]
        
        n_samples, n_features = X.shape
        interaction_matrix = np.zeros((n_features, n_features))
        
        for sample in X:
            interactions = self._compute_pairwise_interactions(sample)
            interaction_matrix += np.abs(interactions)
        
        interaction_matrix /= n_samples
        
        return interaction_matrix
    
    def _compute_pairwise_interactions(
        self,
        instance: np.ndarray
    ) -> np.ndarray:
        """计算单个实例的成对交互。"""
        n_features = len(instance)
        interactions = np.zeros((n_features, n_features))
        
        # 计算单特征 SHAP 值
        single_shap = self._kernel_shap(instance, n_samples=30)
        
        # 估计交互效应
        for i in range(n_features):
            for j in range(i + 1, n_features):
                # 计算联合效应
                coalition = np.zeros(n_features)
                coalition[i] = 1
                coalition[j] = 1
                
                joint_effect = self._coalition_value(instance, coalition)
                individual_sum = single_shap[i] + single_shap[j]
                
                # 交互 = 联合效应 - 个体效应之和
                interaction = joint_effect - individual_sum
                interactions[i, j] = interaction
                interactions[j, i] = interaction
        
        return interactions
    
    def _coalition_value(
        self,
        instance: np.ndarray,
        coalition: np.ndarray
    ) -> float:
        """计算联盟的边际贡献。"""
        n_features = len(instance)
        
        # 创建混合实例
        predictions = []
        for bg in self.background_data:
            mixed = np.where(coalition == 1, instance, bg)
            pred = self._predict(mixed.reshape(1, -1))[0]
            predictions.append(pred)
        
        coalition_pred = np.mean(predictions)
        return coalition_pred - self._expected_value
    
    def force_plot_data(
        self,
        instance: np.ndarray,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        准备 SHAP 力图数据。
        
        Args:
            instance: 单个实例
            feature_names: 特征名称
            
        Returns:
            力图所需的数据
        """
        result = self.explain_instance(instance, feature_names)
        
        shap_values = result['shap_values']
        
        # 排序特征
        sorted_indices = np.argsort(np.abs(shap_values))[::-1]
        
        # 正/负贡献
        positive_mask = shap_values > 0
        negative_mask = shap_values < 0
        
        return {
            'shap_values': shap_values,
            'feature_names': result['feature_names'],
            'sorted_indices': sorted_indices,
            'positive_contributions': shap_values[positive_mask],
            'negative_contributions': shap_values[negative_mask],
            'positive_features': np.array(result['feature_names'])[positive_mask].tolist(),
            'negative_features': np.array(result['feature_names'])[negative_mask].tolist(),
            'expected_value': self._expected_value,
            'prediction': result['prediction']
        }
    
    def dependence_plot_data(
        self,
        X: np.ndarray,
        feature_idx: int,
        interaction_idx: Optional[int] = None,
        feature_names: Optional[List[str]] = None
    ) -> Dict:
        """
        准备 SHAP 依赖图数据。
        
        Args:
            X: 输入数据
            feature_idx: 要分析的特征索引
            interaction_idx: 交互特征索引
            feature_names: 特征名称
            
        Returns:
            依赖图数据
        """
        n_samples = len(X)
        n_features = X.shape[1]
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # 计算 SHAP 值
        shap_values = np.zeros(n_samples)
        for i in range(n_samples):
            result = self.explain_instance(X[i], feature_names, n_samples=30)
            shap_values[i] = result['shap_values'][feature_idx]
        
        # 如果未指定交互特征，自动选择
        if interaction_idx is None:
            # 选择与目标特征相关性最高的特征
            correlations = []
            for j in range(n_features):
                if j != feature_idx:
                    corr = np.corrcoef(shap_values, X[:, j])[0, 1]
                    correlations.append((j, abs(corr) if not np.isnan(corr) else 0))
            
            if correlations:
                interaction_idx = max(correlations, key=lambda x: x[1])[0]
        
        return {
            'feature_values': X[:, feature_idx],
            'shap_values': shap_values,
            'feature_name': feature_names[feature_idx],
            'interaction_values': X[:, interaction_idx] if interaction_idx else None,
            'interaction_name': feature_names[interaction_idx] if interaction_idx else None
        }
