"""
线性模型 (Linear Models)

包含基因组预测中常用的线性模型，作为深度学习模型的基线对比。
这些模型在遗传学研究中广泛使用，具有良好的可解释性。

模型列表:
    - RidgeRegression: 岭回归 (L2 正则化)
    - LassoRegression: Lasso 回归 (L1 正则化)
    - ElasticNet: 弹性网络 (L1 + L2 正则化)
    - BayesianRidge: 贝叶斯岭回归
    - GBLUP: 基因组最佳线性无偏预测
    - RKHS: 再生核希尔伯特空间回归
    - LinearMixedModel: 线性混合模型 (简化版)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, List, Tuple, Dict, Union
import math


class RidgeRegression(nn.Module):
    """
    岭回归 (Ridge Regression)
    
    带 L2 正则化的线性回归，等价于 BLUP。
    在遗传学中也称为 RR-BLUP (Ridge Regression BLUP)。
    
    损失函数:
        L = ||y - Xβ||² + λ||β||²
    
    优点:
        - 处理多重共线性
        - 防止过拟合
        - 所有标记都有非零效应
    
    适用场景:
        - 多基因性状 (polygenic traits)
        - 标记数量 >> 样本数量
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        alpha: float = 1.0,
        bias: bool = True,
        normalize_input: bool = True
    ):
        """
        Args:
            input_dim: 输入特征维度 (标记数量)
            output_dim: 输出维度 (性状数量)
            alpha: L2 正则化系数 (λ)
            bias: 是否包含偏置项
            normalize_input: 是否标准化输入
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.normalize_input = normalize_input
        
        # 线性层
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
        # 输入标准化参数
        if normalize_input:
            self.register_buffer('input_mean', torch.zeros(input_dim))
            self.register_buffer('input_std', torch.ones(input_dim))
        
        # 初始化权重
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        if bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            
        Returns:
            预测值 [batch_size, output_dim]
        """
        if self.normalize_input:
            x = (x - self.input_mean) / (self.input_std + 1e-8)
        
        return self.linear(x)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """获取 L2 正则化损失。"""
        return self.alpha * torch.sum(self.linear.weight ** 2)
    
    def get_marker_effects(self) -> torch.Tensor:
        """获取标记效应 (权重)。"""
        return self.linear.weight.data.clone()
    
    def fit_closed_form(
        self, 
        X: torch.Tensor, 
        y: torch.Tensor
    ) -> 'RidgeRegression':
        """
        使用闭式解拟合模型 (比梯度下降更快)。
        
        β = (X'X + λI)^(-1) X'y
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            y: 目标值 [n_samples, n_outputs]
        """
        n_samples, n_features = X.shape
        
        # 标准化
        if self.normalize_input:
            self.input_mean = X.mean(dim=0)
            self.input_std = X.std(dim=0)
            X = (X - self.input_mean) / (self.input_std + 1e-8)
        
        # 添加偏置列
        if self.linear.bias is not None:
            X_with_bias = torch.cat([X, torch.ones(n_samples, 1, device=X.device)], dim=1)
        else:
            X_with_bias = X
        
        # 闭式解
        XtX = X_with_bias.T @ X_with_bias
        reg_matrix = self.alpha * torch.eye(XtX.size(0), device=X.device)
        if self.linear.bias is not None:
            reg_matrix[-1, -1] = 0  # 不正则化偏置
        
        Xty = X_with_bias.T @ y
        
        # 求解
        beta = torch.linalg.solve(XtX + reg_matrix, Xty)
        
        # 设置权重
        if self.linear.bias is not None:
            self.linear.weight.data = beta[:-1].T
            self.linear.bias.data = beta[-1]
        else:
            self.linear.weight.data = beta.T
        
        return self


class LassoRegression(nn.Module):
    """
    Lasso 回归 (Least Absolute Shrinkage and Selection Operator)
    
    带 L1 正则化的线性回归，可以产生稀疏解。
    
    损失函数:
        L = ||y - Xβ||² + λ||β||₁
    
    优点:
        - 自动特征选择 (部分权重变为 0)
        - 识别重要标记
    
    适用场景:
        - 寡基因性状 (oligogenic traits)
        - 需要识别关键 QTL
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        alpha: float = 0.01,
        bias: bool = True
    ):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出维度
            alpha: L1 正则化系数
            bias: 是否包含偏置项
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
        # 初始化
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        if bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return self.linear(x)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """获取 L1 正则化损失。"""
        return self.alpha * torch.sum(torch.abs(self.linear.weight))
    
    def get_nonzero_markers(self, threshold: float = 1e-4) -> torch.Tensor:
        """获取非零效应的标记索引。"""
        effects = self.linear.weight.abs().sum(dim=0)
        return torch.where(effects > threshold)[0]
    
    def get_sparsity(self, threshold: float = 1e-4) -> float:
        """计算稀疏度 (零权重比例)。"""
        total = self.linear.weight.numel()
        zeros = (self.linear.weight.abs() < threshold).sum().item()
        return zeros / total


class ElasticNet(nn.Module):
    """
    弹性网络 (Elastic Net)
    
    结合 L1 和 L2 正则化的线性回归。
    
    损失函数:
        L = ||y - Xβ||² + α(ρ||β||₁ + (1-ρ)||β||²/2)
    
    优点:
        - 兼具 Lasso 的特征选择和 Ridge 的稳定性
        - 处理高度相关的特征
    
    适用场景:
        - 混合遗传架构
        - 存在连锁不平衡的标记
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        alpha: float = 1.0,
        l1_ratio: float = 0.5,
        bias: bool = True
    ):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出维度
            alpha: 总正则化强度
            l1_ratio: L1 正则化比例 (ρ), 0~1 之间
            bias: 是否包含偏置项
        """
        super().__init__()
        
        assert 0 <= l1_ratio <= 1, "l1_ratio 必须在 0 到 1 之间"
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        
        self.linear = nn.Linear(input_dim, output_dim, bias=bias)
        
        nn.init.normal_(self.linear.weight, mean=0, std=0.01)
        if bias:
            nn.init.zeros_(self.linear.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播。"""
        return self.linear(x)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """获取弹性网络正则化损失。"""
        l1_loss = torch.sum(torch.abs(self.linear.weight))
        l2_loss = torch.sum(self.linear.weight ** 2)
        return self.alpha * (
            self.l1_ratio * l1_loss + 
            (1 - self.l1_ratio) * l2_loss / 2
        )


class BayesianRidge(nn.Module):
    """
    贝叶斯岭回归 (Bayesian Ridge Regression)
    
    使用贝叶斯推断估计权重的后验分布。
    实现为变分近似版本。
    
    先验:
        β ~ N(0, σ²_β I)
        y ~ N(Xβ, σ²_y I)
    
    优点:
        - 提供预测不确定性
        - 自动确定正则化强度
        - 更好的泛化能力
    
    适用场景:
        - 需要不确定性量化
        - 小样本情况
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int = 1,
        alpha_init: float = 1.0,
        lambda_init: float = 1.0,
        bias: bool = True
    ):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出维度
            alpha_init: 噪声精度的初始值
            lambda_init: 权重精度的初始值
            bias: 是否包含偏置项
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # 权重均值
        self.weight_mean = nn.Parameter(torch.zeros(output_dim, input_dim))
        
        # 权重对数方差 (用于重参数化)
        self.weight_log_var = nn.Parameter(torch.full((output_dim, input_dim), -5.0))
        
        # 偏置
        if bias:
            self.bias = nn.Parameter(torch.zeros(output_dim))
        else:
            self.register_parameter('bias', None)
        
        # 精度参数 (log scale)
        self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)))
        self.log_lambda = nn.Parameter(torch.tensor(math.log(lambda_init)))
    
    def forward(
        self, 
        x: torch.Tensor, 
        return_std: bool = False,
        n_samples: int = 1
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        前向传播。
        
        Args:
            x: 输入特征 [batch_size, input_dim]
            return_std: 是否返回预测标准差
            n_samples: Monte Carlo 采样次数
            
        Returns:
            预测均值，可选地返回标准差
        """
        if self.training or n_samples > 1:
            # 从权重后验采样
            weight_std = torch.exp(0.5 * self.weight_log_var)
            eps = torch.randn(n_samples, *self.weight_mean.shape, device=x.device)
            weights = self.weight_mean + weight_std * eps  # [n_samples, out, in]
            
            # 预测
            outputs = torch.einsum('bj,sij->sbi', x, weights)
            if self.bias is not None:
                outputs = outputs + self.bias
            
            mean = outputs.mean(dim=0)
            
            if return_std:
                if n_samples > 1:
                    std = outputs.std(dim=0)
                else:
                    std = torch.zeros_like(mean)
                return mean, std
            return mean
        else:
            # 推理时使用均值
            output = F.linear(x, self.weight_mean, self.bias)
            
            if return_std:
                # 解析计算方差
                weight_var = torch.exp(self.weight_log_var)
                pred_var = (x.unsqueeze(1) ** 2) @ weight_var.T
                pred_std = torch.sqrt(pred_var.squeeze(1) + 1e-8)
                return output, pred_std
            return output
    
    def get_kl_divergence(self) -> torch.Tensor:
        """计算 KL 散度损失 (权重后验 vs 先验)。"""
        lambda_val = torch.exp(self.log_lambda)
        
        weight_var = torch.exp(self.weight_log_var)
        
        # KL(q(w) || p(w)) where p(w) = N(0, 1/λ)
        kl = 0.5 * lambda_val * (
            torch.sum(self.weight_mean ** 2) + 
            torch.sum(weight_var)
        ) - 0.5 * torch.sum(self.weight_log_var) - 0.5 * self.weight_mean.numel()
        
        return kl


class GBLUP(nn.Module):
    """
    基因组最佳线性无偏预测 (Genomic Best Linear Unbiased Prediction)
    
    使用基因组关系矩阵 (G-matrix) 进行预测。
    这是一个简化的 PyTorch 实现版本。
    
    模型:
        y = μ + Zu + ε
        u ~ N(0, Gσ²_u)
        ε ~ N(0, Iσ²_ε)
    
    其中 G 是基因组关系矩阵。
    
    优点:
        - 考虑个体间的遗传相似性
        - 对多基因性状效果好
    
    注意:
        - 需要预先计算 G 矩阵
        - 计算复杂度 O(n³)
    """
    
    def __init__(
        self,
        n_samples: int,
        heritability: float = 0.5,
        use_vr: bool = True
    ):
        """
        Args:
            n_samples: 训练样本数量
            heritability: 遗传力估计值 (h²)
            use_vr: 是否使用方差比例参数化
        """
        super().__init__()
        
        self.n_samples = n_samples
        self.use_vr = use_vr
        
        # 固定效应 (截距)
        self.intercept = nn.Parameter(torch.zeros(1))
        
        # 方差组分
        if use_vr:
            # 使用方差比例: λ = σ²_ε / σ²_u = (1-h²) / h²
            init_lambda = (1 - heritability) / (heritability + 1e-8)
            self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda + 1e-8)))
        else:
            self.log_var_u = nn.Parameter(torch.zeros(1))
            self.log_var_e = nn.Parameter(torch.zeros(1))
        
        # 存储训练数据
        self.register_buffer('G_matrix', None)
        self.register_buffer('train_y', None)
        self.register_buffer('breeding_values', None)
    
    def set_G_matrix(self, markers: torch.Tensor):
        """
        计算并设置基因组关系矩阵。
        
        G = ZZ' / p, 其中 Z 是标准化的标记矩阵
        
        Args:
            markers: 标记矩阵 [n_samples, n_markers]
        """
        n, p = markers.shape
        
        # 标准化标记
        freq = markers.mean(dim=0)
        Z = (markers - 2 * freq) / torch.sqrt(2 * freq * (1 - freq) + 1e-8)
        
        # 计算 G 矩阵
        G = Z @ Z.T / p
        
        # 确保正定
        G = G + 0.01 * torch.eye(n, device=markers.device)
        
        self.G_matrix = G
    
    def fit(self, y: torch.Tensor):
        """
        使用 BLUP 方程求解。
        
        Args:
            y: 表型值 [n_samples]
        """
        if self.G_matrix is None:
            raise ValueError("请先调用 set_G_matrix() 设置 G 矩阵")
        
        n = self.G_matrix.size(0)
        
        # 获取方差比
        if self.use_vr:
            lambda_val = torch.exp(self.log_lambda)
        else:
            var_u = torch.exp(self.log_var_u)
            var_e = torch.exp(self.log_var_e)
            lambda_val = var_e / (var_u + 1e-8)
        
        # 混合模型方程
        # [u] = (G + λI)^(-1) (y - μ)
        y_centered = y - y.mean()
        
        G_reg = self.G_matrix + lambda_val * torch.eye(n, device=y.device)
        u = torch.linalg.solve(G_reg, y_centered)
        
        self.intercept.data = y.mean().unsqueeze(0)
        self.breeding_values = u
        self.train_y = y
    
    def forward(
        self, 
        markers: torch.Tensor,
        train_markers: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        预测新个体的育种值。
        
        Args:
            markers: 新个体的标记 [n_new, n_markers]
            train_markers: 训练个体的标记 (用于计算关系)
            
        Returns:
            预测的育种值
        """
        if self.breeding_values is None:
            raise ValueError("请先调用 fit() 拟合模型")
        
        if train_markers is None:
            # 假设预测训练集本身
            return self.intercept + self.breeding_values
        
        # 计算新个体与训练个体的关系
        n_train, p = train_markers.shape
        freq = train_markers.mean(dim=0)
        
        Z_train = (train_markers - 2 * freq) / torch.sqrt(2 * freq * (1 - freq) + 1e-8)
        Z_new = (markers - 2 * freq) / torch.sqrt(2 * freq * (1 - freq) + 1e-8)
        
        # G_new,train
        G_cross = Z_new @ Z_train.T / p
        
        # 预测
        return self.intercept + G_cross @ self.breeding_values


class RKHS(nn.Module):
    """
    再生核希尔伯特空间回归 (Reproducing Kernel Hilbert Space Regression)
    
    使用核方法捕获非线性遗传效应。
    
    模型:
        y = f(X) + ε, f ∈ H_K
    
    常用核函数:
        - 高斯核 (RBF): K(x,x') = exp(-||x-x'||² / 2σ²)
        - 多项式核: K(x,x') = (x·x' + c)^d
    
    优点:
        - 捕获上位性效应
        - 处理非加性遗传效应
    
    适用场景:
        - 存在显著上位性
        - 非线性性状
    """
    
    def __init__(
        self,
        kernel: str = 'rbf',
        gamma: Optional[float] = None,
        degree: int = 3,
        coef0: float = 1.0,
        alpha: float = 1.0
    ):
        """
        Args:
            kernel: 核函数类型 ('rbf', 'polynomial', 'linear')
            gamma: RBF 核的参数 (默认 1/n_features)
            degree: 多项式核的度数
            coef0: 多项式核的常数项
            alpha: 正则化系数
        """
        super().__init__()
        
        self.kernel = kernel
        self.gamma = gamma
        self.degree = degree
        self.coef0 = coef0
        self.alpha = alpha
        
        # 训练数据
        self.register_buffer('X_train', None)
        self.register_buffer('alpha_coef', None)
        self.intercept = nn.Parameter(torch.zeros(1))
    
    def _compute_kernel(
        self, 
        X1: torch.Tensor, 
        X2: torch.Tensor
    ) -> torch.Tensor:
        """计算核矩阵。"""
        if self.kernel == 'linear':
            return X1 @ X2.T
        
        elif self.kernel == 'rbf':
            gamma = self.gamma or (1.0 / X1.size(1))
            
            # 计算距离矩阵
            X1_sq = (X1 ** 2).sum(dim=1, keepdim=True)
            X2_sq = (X2 ** 2).sum(dim=1, keepdim=True)
            dist = X1_sq + X2_sq.T - 2 * X1 @ X2.T
            
            return torch.exp(-gamma * dist)
        
        elif self.kernel == 'polynomial':
            return (X1 @ X2.T + self.coef0) ** self.degree
        
        else:
            raise ValueError(f"未知核函数: {self.kernel}")
    
    def fit(self, X: torch.Tensor, y: torch.Tensor):
        """
        拟合 RKHS 回归模型。
        
        Args:
            X: 特征矩阵 [n_samples, n_features]
            y: 目标值 [n_samples]
        """
        n = X.size(0)
        
        # 计算核矩阵
        K = self._compute_kernel(X, X)
        
        # 求解 (K + αI)α = y
        K_reg = K + self.alpha * torch.eye(n, device=X.device)
        
        y_centered = y - y.mean()
        alpha_coef = torch.linalg.solve(K_reg, y_centered)
        
        self.X_train = X
        self.alpha_coef = alpha_coef
        self.intercept.data = y.mean().unsqueeze(0)
    
    def forward(self, X: torch.Tensor) -> torch.Tensor:
        """
        预测新样本。
        
        Args:
            X: 新样本特征 [n_samples, n_features]
            
        Returns:
            预测值 [n_samples]
        """
        if self.X_train is None or self.alpha_coef is None:
            raise ValueError("请先调用 fit() 拟合模型")
        
        K_new = self._compute_kernel(X, self.X_train)
        return self.intercept + K_new @ self.alpha_coef


class LinearMixedModel(nn.Module):
    """
    线性混合模型 (Linear Mixed Model) - 简化版
    
    包含固定效应和随机效应的模型。
    
    模型:
        y = Xβ + Zu + ε
        u ~ N(0, Gσ²_u)
        ε ~ N(0, Iσ²_ε)
    
    其中:
        - β: 固定效应 (如环境效应)
        - u: 随机遗传效应
        - G: 遗传关系矩阵
    
    适用场景:
        - 需要同时估计固定和随机效应
        - G×E 交互建模
    """
    
    def __init__(
        self,
        n_fixed: int,
        n_random: int,
        heritability: float = 0.5
    ):
        """
        Args:
            n_fixed: 固定效应数量
            n_random: 随机效应数量 (样本数)
            heritability: 遗传力估计
        """
        super().__init__()
        
        self.n_fixed = n_fixed
        self.n_random = n_random
        
        # 固定效应
        self.fixed_effects = nn.Linear(n_fixed, 1, bias=True)
        
        # 随机效应 (育种值)
        self.random_effects = nn.Parameter(torch.zeros(n_random))
        
        # 方差组分
        init_lambda = (1 - heritability) / (heritability + 1e-8)
        self.log_lambda = nn.Parameter(torch.tensor(math.log(init_lambda + 1e-8)))
        
        # G 矩阵
        self.register_buffer('G_matrix', None)
    
    def set_G_matrix(self, G: torch.Tensor):
        """设置遗传关系矩阵。"""
        self.G_matrix = G
    
    def forward(
        self, 
        X_fixed: torch.Tensor,
        sample_indices: torch.Tensor
    ) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            X_fixed: 固定效应设计矩阵 [batch, n_fixed]
            sample_indices: 样本索引 (用于索引随机效应)
            
        Returns:
            预测值 [batch]
        """
        # 固定效应
        fixed = self.fixed_effects(X_fixed).squeeze(-1)
        
        # 随机效应
        random = self.random_effects[sample_indices]
        
        return fixed + random
    
    def get_regularization_loss(self) -> torch.Tensor:
        """获取随机效应的正则化损失。"""
        lambda_val = torch.exp(self.log_lambda)
        
        if self.G_matrix is not None:
            # u' G^(-1) u
            G_inv = torch.linalg.inv(self.G_matrix + 0.01 * torch.eye(
                self.G_matrix.size(0), device=self.G_matrix.device
            ))
            return lambda_val * self.random_effects @ G_inv @ self.random_effects
        else:
            return lambda_val * torch.sum(self.random_effects ** 2)


class MultiTraitLinear(nn.Module):
    """
    多性状线性模型
    
    同时预测多个相关性状，共享部分信息。
    
    模型:
        Y = XB + E
        vec(E) ~ N(0, Σ ⊗ I)
    
    其中 Σ 是性状间的协方差矩阵。
    """
    
    def __init__(
        self,
        input_dim: int,
        n_traits: int,
        shared_effects: bool = True,
        alpha: float = 1.0
    ):
        """
        Args:
            input_dim: 输入维度
            n_traits: 性状数量
            shared_effects: 是否共享效应
            alpha: 正则化系数
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.n_traits = n_traits
        self.alpha = alpha
        
        if shared_effects:
            # 共享表示 + 性状特异性输出
            self.shared_layer = nn.Linear(input_dim, input_dim // 2)
            self.trait_layers = nn.ModuleList([
                nn.Linear(input_dim // 2, 1) for _ in range(n_traits)
            ])
        else:
            # 独立的每个性状
            self.linear = nn.Linear(input_dim, n_traits)
        
        self.shared_effects = shared_effects
        
        # 性状相关矩阵 (对角为1)
        self.register_buffer(
            'trait_corr', 
            torch.eye(n_traits)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。
        
        Args:
            x: 输入特征 [batch, input_dim]
            
        Returns:
            多性状预测 [batch, n_traits]
        """
        if self.shared_effects:
            shared = F.relu(self.shared_layer(x))
            outputs = [layer(shared) for layer in self.trait_layers]
            return torch.cat(outputs, dim=-1)
        else:
            return self.linear(x)
    
    def get_regularization_loss(self) -> torch.Tensor:
        """获取正则化损失。"""
        if self.shared_effects:
            reg = self.alpha * torch.sum(self.shared_layer.weight ** 2)
            for layer in self.trait_layers:
                reg = reg + self.alpha * torch.sum(layer.weight ** 2)
            return reg
        else:
            return self.alpha * torch.sum(self.linear.weight ** 2)
