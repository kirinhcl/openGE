"""
特征重要性分析 (Feature Importance Analysis)

提供多种方法计算和分析特征对模型预测的重要性。

方法:
    - 置换重要性 (Permutation Importance)
    - 基于梯度的重要性
    - 遗传 vs 环境贡献分析
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, Tuple, Optional, List, Callable, Union
from sklearn.metrics import mean_squared_error, r2_score


class FeatureImportance:
    """
    特征重要性分析器。
    
    计算和可视化特征对模型预测的影响。
    """
    
    def __init__(self, model: nn.Module, device: str = 'cpu'):
        """
        初始化特征重要性分析器。
        
        Args:
            model: 训练好的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.model.to(device)
        self.model.eval()
    
    def permutation_importance(
        self,
        X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        y: np.ndarray,
        n_repeats: int = 10,
        metric: str = 'r2',
        feature_names: Optional[List[str]] = None,
        random_state: int = 42
    ) -> Dict:
        """
        计算置换重要性。
        
        通过置换每个特征并测量性能下降来评估重要性。
        
        Args:
            X: 输入特征 (单个数组或 (遗传, 环境) 元组)
            y: 目标值
            n_repeats: 重复次数
            metric: 评估指标 ('r2', 'mse', 'mae')
            feature_names: 特征名称
            random_state: 随机种子
            
        Returns:
            特征重要性结果
        """
        np.random.seed(random_state)
        
        # 处理输入格式
        if isinstance(X, tuple):
            X_genetic, X_env = X
            is_gxe = True
        else:
            X_genetic = X
            X_env = None
            is_gxe = False
        
        n_features = X_genetic.shape[1]
        
        if feature_names is None:
            feature_names = [f'Feature_{i}' for i in range(n_features)]
        
        # 基线性能
        baseline_score = self._evaluate(X_genetic, X_env, y, metric)
        
        # 计算每个特征的重要性
        importances = np.zeros((n_features, n_repeats))
        
        for feat_idx in range(n_features):
            for rep in range(n_repeats):
                # 复制并置换特征
                X_permuted = X_genetic.copy()
                np.random.shuffle(X_permuted[:, feat_idx])
                
                # 评估置换后的性能
                permuted_score = self._evaluate(X_permuted, X_env, y, metric)
                
                # 重要性 = 性能下降
                importances[feat_idx, rep] = baseline_score - permuted_score
        
        return {
            'importances_mean': importances.mean(axis=1),
            'importances_std': importances.std(axis=1),
            'feature_names': feature_names,
            'baseline_score': baseline_score,
            'metric': metric
        }
    
    def _evaluate(
        self,
        X_genetic: np.ndarray,
        X_env: Optional[np.ndarray],
        y: np.ndarray,
        metric: str
    ) -> float:
        """评估模型性能。"""
        with torch.no_grad():
            X_g = torch.tensor(X_genetic, dtype=torch.float32).to(self.device)
            
            if X_env is not None:
                X_e = torch.tensor(X_env, dtype=torch.float32).to(self.device)
                predictions, _ = self.model(X_g, X_e)
            else:
                predictions = self.model(X_g)
                if isinstance(predictions, tuple):
                    predictions = predictions[0]
            
            y_pred = predictions.cpu().numpy().flatten()
        
        y_true = y.flatten()
        
        if metric == 'r2':
            return r2_score(y_true, y_pred)
        elif metric == 'mse':
            return -mean_squared_error(y_true, y_pred)  # 负数，使得越大越好
        elif metric == 'mae':
            return -np.mean(np.abs(y_true - y_pred))
        else:
            raise ValueError(f"未知指标: {metric}")
    
    def gradient_importance(
        self,
        X: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        aggregate: str = 'mean'
    ) -> Dict:
        """
        基于梯度的特征重要性。
        
        Args:
            X: 输入特征
            aggregate: 聚合方法 ('mean', 'max', 'sum')
            
        Returns:
            梯度重要性分数
        """
        if isinstance(X, tuple):
            X_genetic, X_env = X
            is_gxe = True
        else:
            X_genetic = X
            X_env = None
            is_gxe = False
        
        X_g = torch.tensor(X_genetic, dtype=torch.float32, requires_grad=True).to(self.device)
        
        if is_gxe and X_env is not None:
            X_e = torch.tensor(X_env, dtype=torch.float32).to(self.device)
            outputs, _ = self.model(X_g, X_e)
        else:
            outputs = self.model(X_g)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
        
        # 计算梯度
        outputs.sum().backward()
        gradients = X_g.grad.cpu().numpy()
        
        # 聚合
        if aggregate == 'mean':
            importance = np.abs(gradients).mean(axis=0)
        elif aggregate == 'max':
            importance = np.abs(gradients).max(axis=0)
        elif aggregate == 'sum':
            importance = np.abs(gradients).sum(axis=0)
        else:
            raise ValueError(f"未知聚合方法: {aggregate}")
        
        return {
            'importance': importance,
            'raw_gradients': gradients,
            'method': 'gradient',
            'aggregate': aggregate
        }
    
    def genetic_vs_environment_contribution(
        self,
        genetic_data: np.ndarray,
        env_data: np.ndarray,
        y: np.ndarray,
        n_permutations: int = 20
    ) -> Dict:
        """
        量化遗传因素 vs 环境因素的相对贡献。
        
        通过置换分析估计每个模态的贡献。
        
        Args:
            genetic_data: 遗传特征
            env_data: 环境特征
            y: 目标值
            n_permutations: 置换次数
            
        Returns:
            遗传和环境贡献估计
        """
        # 基线性能
        baseline = self._evaluate(genetic_data, env_data, y, 'r2')
        
        # 置换遗传数据
        genetic_importance = []
        for _ in range(n_permutations):
            perm_idx = np.random.permutation(len(genetic_data))
            genetic_perm = genetic_data[perm_idx]
            score = self._evaluate(genetic_perm, env_data, y, 'r2')
            genetic_importance.append(baseline - score)
        
        # 置换环境数据
        env_importance = []
        for _ in range(n_permutations):
            perm_idx = np.random.permutation(len(env_data))
            env_perm = env_data[perm_idx]
            score = self._evaluate(genetic_data, env_perm, y, 'r2')
            env_importance.append(baseline - score)
        
        genetic_contrib = np.mean(genetic_importance)
        env_contrib = np.mean(env_importance)
        total = genetic_contrib + env_contrib
        
        # 归一化为比例
        if total > 0:
            genetic_ratio = genetic_contrib / total
            env_ratio = env_contrib / total
        else:
            genetic_ratio = 0.5
            env_ratio = 0.5
        
        return {
            'genetic_contribution': genetic_contrib,
            'environment_contribution': env_contrib,
            'genetic_ratio': genetic_ratio,
            'environment_ratio': env_ratio,
            'genetic_std': np.std(genetic_importance),
            'environment_std': np.std(env_importance),
            'baseline_r2': baseline
        }
    
    def marker_group_importance(
        self,
        genetic_data: np.ndarray,
        env_data: Optional[np.ndarray],
        y: np.ndarray,
        groups: Dict[str, List[int]],
        n_repeats: int = 5
    ) -> Dict:
        """
        计算标记组 (如染色体、基因区域) 的重要性。
        
        Args:
            genetic_data: 遗传特征
            env_data: 环境特征
            y: 目标值
            groups: 组名到特征索引的映射
            n_repeats: 重复次数
            
        Returns:
            每个组的重要性
        """
        baseline = self._evaluate(genetic_data, env_data, y, 'r2')
        
        group_importance = {}
        for group_name, indices in groups.items():
            scores = []
            for _ in range(n_repeats):
                X_perm = genetic_data.copy()
                # 置换组内所有特征
                for idx in indices:
                    np.random.shuffle(X_perm[:, idx])
                score = self._evaluate(X_perm, env_data, y, 'r2')
                scores.append(baseline - score)
            
            group_importance[group_name] = {
                'mean': np.mean(scores),
                'std': np.std(scores),
                'n_features': len(indices)
            }
        
        return {
            'group_importance': group_importance,
            'baseline_r2': baseline
        }
    
    def feature_interaction_importance(
        self,
        genetic_data: np.ndarray,
        env_data: Optional[np.ndarray],
        y: np.ndarray,
        feature_pairs: List[Tuple[int, int]],
        n_repeats: int = 5
    ) -> Dict:
        """
        计算特征对之间的交互重要性。
        
        Args:
            genetic_data: 遗传特征
            env_data: 环境特征
            y: 目标值
            feature_pairs: 要分析的特征对
            n_repeats: 重复次数
            
        Returns:
            特征交互重要性
        """
        baseline = self._evaluate(genetic_data, env_data, y, 'r2')
        
        # 计算单个特征重要性
        single_importance = {}
        for i, j in feature_pairs:
            for feat in [i, j]:
                if feat not in single_importance:
                    scores = []
                    for _ in range(n_repeats):
                        X_perm = genetic_data.copy()
                        np.random.shuffle(X_perm[:, feat])
                        score = self._evaluate(X_perm, env_data, y, 'r2')
                        scores.append(baseline - score)
                    single_importance[feat] = np.mean(scores)
        
        # 计算成对重要性
        pair_importance = {}
        for i, j in feature_pairs:
            scores = []
            for _ in range(n_repeats):
                X_perm = genetic_data.copy()
                np.random.shuffle(X_perm[:, i])
                np.random.shuffle(X_perm[:, j])
                score = self._evaluate(X_perm, env_data, y, 'r2')
                scores.append(baseline - score)
            
            pair_imp = np.mean(scores)
            # 交互效应 = 成对重要性 - 各自重要性之和
            interaction = pair_imp - single_importance[i] - single_importance[j]
            
            pair_importance[(i, j)] = {
                'pair_importance': pair_imp,
                'interaction_effect': interaction,
                'feature_i_importance': single_importance[i],
                'feature_j_importance': single_importance[j]
            }
        
        return {
            'pair_importance': pair_importance,
            'single_importance': single_importance,
            'baseline_r2': baseline
        }
