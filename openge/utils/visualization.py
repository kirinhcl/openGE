"""
可视化工具 (Visualization Utilities)

提供模型分析和结果展示的可视化函数。

包含:
    - 预测 vs 实际值散点图
    - 注意力权重热图
    - 训练历史曲线
    - 特征重要性图
    - 遗传效应分布图
    - G×E 交互热图
"""

import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List, Dict, Union
import warnings

# 设置中文字体支持
try:
    plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    pass


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    title: str = "Predictions vs Actual",
    xlabel: str = "Actual Values",
    ylabel: str = "Predicted Values",
    figsize: Tuple[int, int] = (8, 6),
    alpha: float = 0.6,
    show_metrics: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    绘制预测值 vs 实际值散点图。
    
    Args:
        y_true: 真实值
        y_pred: 预测值
        title: 图标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        figsize: 图像尺寸
        alpha: 点透明度
        show_metrics: 是否显示评估指标
        save_path: 保存路径
        show: 是否显示图像
        
    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 散点图
    ax.scatter(y_true, y_pred, alpha=alpha, edgecolors='none', s=50)
    
    # 对角线 (完美预测)
    min_val = min(y_true.min(), y_pred.min())
    max_val = max(y_true.max(), y_pred.max())
    margin = (max_val - min_val) * 0.05
    ax.plot([min_val - margin, max_val + margin], 
            [min_val - margin, max_val + margin], 
            'r--', lw=2, label='Perfect Prediction')
    
    # 计算并显示指标
    if show_metrics:
        from sklearn.metrics import r2_score, mean_squared_error
        r2 = r2_score(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        corr = np.corrcoef(y_true.flatten(), y_pred.flatten())[0, 1]
        
        metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nCorr = {corr:.4f}'
        ax.text(0.05, 0.95, metrics_text, transform=ax.transAxes, 
                verticalalignment='top', fontsize=10,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    ax.set_xlim(min_val - margin, max_val + margin)
    ax.set_ylim(min_val - margin, max_val + margin)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_attention_weights(
    attention_weights: np.ndarray,
    title: str = "Attention Weights",
    xlabel: str = "Key Position",
    ylabel: str = "Query Position",
    figsize: Tuple[int, int] = (10, 8),
    cmap: str = 'viridis',
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    可视化注意力权重热图。
    
    Args:
        attention_weights: 注意力权重矩阵 [query_len, key_len] 或 [n_heads, q, k]
        title: 图标题
        xlabel: X 轴标签
        ylabel: Y 轴标签
        figsize: 图像尺寸
        cmap: 颜色映射
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        matplotlib Figure 对象
    """
    if attention_weights.ndim == 3:
        # 多头注意力：显示每个头
        n_heads = attention_weights.shape[0]
        n_cols = min(4, n_heads)
        n_rows = (n_heads + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3))
        axes = np.atleast_2d(axes)
        
        for i in range(n_heads):
            row, col = i // n_cols, i % n_cols
            im = axes[row, col].imshow(attention_weights[i], cmap=cmap, aspect='auto')
            axes[row, col].set_title(f'Head {i + 1}')
            axes[row, col].set_xlabel(xlabel)
            axes[row, col].set_ylabel(ylabel)
            plt.colorbar(im, ax=axes[row, col])
        
        # 隐藏多余的子图
        for i in range(n_heads, n_rows * n_cols):
            row, col = i // n_cols, i % n_cols
            axes[row, col].axis('off')
        
        fig.suptitle(title, fontsize=14)
    else:
        # 单个注意力矩阵
        fig, ax = plt.subplots(figsize=figsize)
        im = ax.imshow(attention_weights, cmap=cmap, aspect='auto')
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_training_history(
    train_loss: List[float],
    val_loss: List[float],
    train_metric: Optional[List[float]] = None,
    val_metric: Optional[List[float]] = None,
    metric_name: str = "R²",
    title: str = "Training History",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    绘制训练和验证损失/指标曲线。
    
    Args:
        train_loss: 训练损失历史
        val_loss: 验证损失历史
        train_metric: 训练指标历史 (可选)
        val_metric: 验证指标历史 (可选)
        metric_name: 指标名称
        title: 图标题
        figsize: 图像尺寸
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        matplotlib Figure 对象
    """
    n_plots = 2 if train_metric is not None else 1
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    
    if n_plots == 1:
        axes = [axes]
    
    epochs = range(1, len(train_loss) + 1)
    
    # 损失曲线
    axes[0].plot(epochs, train_loss, 'b-', label='Training Loss', linewidth=2)
    axes[0].plot(epochs, val_loss, 'r-', label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curves')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # 标记最佳验证损失
    best_epoch = np.argmin(val_loss)
    axes[0].axvline(x=best_epoch + 1, color='g', linestyle='--', alpha=0.7, 
                    label=f'Best: Epoch {best_epoch + 1}')
    
    # 指标曲线
    if train_metric is not None and n_plots > 1:
        axes[1].plot(epochs, train_metric, 'b-', label=f'Training {metric_name}', linewidth=2)
        axes[1].plot(epochs, val_metric, 'r-', label=f'Validation {metric_name}', linewidth=2)
        axes[1].set_xlabel('Epoch')
        axes[1].set_ylabel(metric_name)
        axes[1].set_title(f'{metric_name} Curves')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_feature_importance(
    importance_scores: np.ndarray,
    feature_names: Optional[List[str]] = None,
    top_k: int = 20,
    title: str = "Feature Importance",
    figsize: Tuple[int, int] = (10, 8),
    horizontal: bool = True,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    绘制特征重要性条形图。
    
    Args:
        importance_scores: 特征重要性分数
        feature_names: 特征名称
        top_k: 显示前 k 个重要特征
        title: 图标题
        figsize: 图像尺寸
        horizontal: 是否水平显示
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        matplotlib Figure 对象
    """
    if feature_names is None:
        feature_names = [f'Feature {i}' for i in range(len(importance_scores))]
    
    # 排序并取 top_k
    sorted_idx = np.argsort(importance_scores)[::-1][:top_k]
    sorted_scores = importance_scores[sorted_idx]
    sorted_names = [feature_names[i] for i in sorted_idx]
    
    fig, ax = plt.subplots(figsize=figsize)
    
    if horizontal:
        y_pos = np.arange(len(sorted_scores))[::-1]
        ax.barh(y_pos, sorted_scores, align='center', color='steelblue', alpha=0.8)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(sorted_names)
        ax.set_xlabel('Importance Score')
        ax.set_ylabel('Feature')
    else:
        x_pos = np.arange(len(sorted_scores))
        ax.bar(x_pos, sorted_scores, align='center', color='steelblue', alpha=0.8)
        ax.set_xticks(x_pos)
        ax.set_xticklabels(sorted_names, rotation=45, ha='right')
        ax.set_ylabel('Importance Score')
        ax.set_xlabel('Feature')
    
    ax.set_title(title)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_marker_effects(
    effects: np.ndarray,
    marker_positions: Optional[np.ndarray] = None,
    chromosome: Optional[np.ndarray] = None,
    title: str = "Marker Effects (Manhattan Plot)",
    threshold: Optional[float] = None,
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    绘制标记效应曼哈顿图。
    
    Args:
        effects: 标记效应大小 (绝对值)
        marker_positions: 标记位置 (bp)
        chromosome: 染色体编号
        title: 图标题
        threshold: 显著性阈值线
        figsize: 图像尺寸
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    n_markers = len(effects)
    
    if chromosome is not None:
        # 按染色体着色
        unique_chr = np.unique(chromosome)
        colors = plt.cm.tab10(np.linspace(0, 1, len(unique_chr)))
        
        cumulative_pos = 0
        chr_centers = []
        
        for i, chr_num in enumerate(unique_chr):
            mask = chromosome == chr_num
            chr_effects = np.abs(effects[mask])
            
            if marker_positions is not None:
                chr_pos = marker_positions[mask]
                x = chr_pos - chr_pos.min() + cumulative_pos
            else:
                x = np.arange(np.sum(mask)) + cumulative_pos
            
            ax.scatter(x, chr_effects, c=[colors[i]], alpha=0.6, s=10, label=f'Chr {chr_num}')
            chr_centers.append((x.min() + x.max()) / 2)
            cumulative_pos = x.max() + 1000000  # Gap between chromosomes
        
        ax.set_xticks(chr_centers)
        ax.set_xticklabels(unique_chr)
        ax.set_xlabel('Chromosome')
    else:
        # 简单的索引
        x = np.arange(n_markers)
        ax.scatter(x, np.abs(effects), alpha=0.6, s=10, c='steelblue')
        ax.set_xlabel('Marker Index')
    
    ax.set_ylabel('|Effect|')
    ax.set_title(title)
    
    if threshold is not None:
        ax.axhline(y=threshold, color='red', linestyle='--', label=f'Threshold = {threshold}')
        ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_gxe_heatmap(
    gxe_matrix: np.ndarray,
    genotype_names: Optional[List[str]] = None,
    environment_names: Optional[List[str]] = None,
    title: str = "G×E Interaction Heatmap",
    figsize: Tuple[int, int] = (12, 10),
    cmap: str = 'RdBu_r',
    center: float = 0,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    绘制 G×E 交互效应热图。
    
    Args:
        gxe_matrix: G×E 交互矩阵 [n_genotypes, n_environments]
        genotype_names: 基因型名称
        environment_names: 环境名称
        title: 图标题
        figsize: 图像尺寸
        cmap: 颜色映射
        center: 颜色中心值
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 居中颜色映射
    vmax = max(abs(gxe_matrix.min() - center), abs(gxe_matrix.max() - center))
    
    im = ax.imshow(gxe_matrix, cmap=cmap, aspect='auto', 
                   vmin=center - vmax, vmax=center + vmax)
    
    # 标签
    if genotype_names is not None:
        if len(genotype_names) <= 50:
            ax.set_yticks(np.arange(len(genotype_names)))
            ax.set_yticklabels(genotype_names, fontsize=8)
        else:
            ax.set_ylabel(f'Genotypes (n={len(genotype_names)})')
    else:
        ax.set_ylabel('Genotype Index')
    
    if environment_names is not None:
        ax.set_xticks(np.arange(len(environment_names)))
        ax.set_xticklabels(environment_names, rotation=45, ha='right', fontsize=8)
    else:
        ax.set_xlabel('Environment Index')
    
    ax.set_title(title)
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('G×E Effect')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_model_comparison(
    results: Dict[str, Dict[str, float]],
    metrics: List[str] = ['r2', 'rmse', 'mae'],
    title: str = "Model Comparison",
    figsize: Tuple[int, int] = (12, 5),
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    绘制多模型比较图。
    
    Args:
        results: 模型结果字典 {model_name: {metric_name: value}}
        metrics: 要显示的指标列表
        title: 图标题
        figsize: 图像尺寸
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        matplotlib Figure 对象
    """
    model_names = list(results.keys())
    n_models = len(model_names)
    n_metrics = len(metrics)
    
    fig, axes = plt.subplots(1, n_metrics, figsize=figsize)
    if n_metrics == 1:
        axes = [axes]
    
    x = np.arange(n_models)
    colors = plt.cm.Set2(np.linspace(0, 1, n_models))
    
    for i, metric in enumerate(metrics):
        values = [results[model].get(metric, 0) for model in model_names]
        bars = axes[i].bar(x, values, color=colors, alpha=0.8)
        axes[i].set_xticks(x)
        axes[i].set_xticklabels(model_names, rotation=45, ha='right')
        axes[i].set_ylabel(metric.upper())
        axes[i].set_title(f'{metric.upper()} Comparison')
        
        # 在柱子上方显示数值
        for bar, val in zip(bars, values):
            axes[i].text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                        f'{val:.4f}', ha='center', va='bottom', fontsize=9)
    
    fig.suptitle(title, fontsize=14)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig


def plot_uncertainty(
    y_true: np.ndarray,
    y_pred_mean: np.ndarray,
    y_pred_std: np.ndarray,
    title: str = "Predictions with Uncertainty",
    figsize: Tuple[int, int] = (10, 6),
    n_samples: int = 100,
    save_path: Optional[str] = None,
    show: bool = True
) -> plt.Figure:
    """
    绘制带不确定性的预测图。
    
    Args:
        y_true: 真实值
        y_pred_mean: 预测均值
        y_pred_std: 预测标准差
        title: 图标题
        figsize: 图像尺寸
        n_samples: 显示的样本数
        save_path: 保存路径
        show: 是否显示
        
    Returns:
        matplotlib Figure 对象
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # 限制显示的样本数
    if len(y_true) > n_samples:
        idx = np.random.choice(len(y_true), n_samples, replace=False)
        idx = np.sort(idx)
    else:
        idx = np.arange(len(y_true))
    
    x = np.arange(len(idx))
    
    # 绘制预测区间
    ax.fill_between(x, 
                    y_pred_mean[idx] - 2 * y_pred_std[idx],
                    y_pred_mean[idx] + 2 * y_pred_std[idx],
                    alpha=0.2, color='blue', label='95% CI')
    ax.fill_between(x,
                    y_pred_mean[idx] - y_pred_std[idx],
                    y_pred_mean[idx] + y_pred_std[idx],
                    alpha=0.4, color='blue', label='68% CI')
    
    # 绘制预测均值和真实值
    ax.plot(x, y_pred_mean[idx], 'b-', label='Prediction', linewidth=2)
    ax.scatter(x, y_true[idx], c='red', s=20, alpha=0.7, label='Actual', zorder=5)
    
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.set_title(title)
    ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    if show:
        plt.show()
    
    return fig
