"""Data preprocessing utilities for normalization, imputation, and feature engineering."""

import numpy as np
import pandas as pd
from typing import Union, Tuple, Optional, List, Dict
from sklearn.preprocessing import StandardScaler, MinMaxScaler


# 作物生育期定义
CROP_GROWTH_STAGES = {
    'maize': {
        'VE': (0, 10),
        'V_stage': (10, 40),
        'VT': (40, 60),
        'R_stage': (60, 120),
    },
    'wheat': {
        'Emergence': (0, 15),
        'Tillering': (15, 45),
        'Booting': (45, 60),
        'Flowering': (60, 75),
        'Grain_filling': (75, 120),
    },
    'rice': {
        'Emergence': (0, 15),
        'Tillering': (15, 60),
        'Booting': (60, 80),
        'Flowering': (80, 100),
        'Maturity': (100, 150),
    }
}


class Preprocessor:
    """Universal preprocessor for genetic and environmental data."""
    
    def __init__(self, method: str = "standard"):
        """
        Initialize preprocessor.
        
        Args:
            method: Normalization method ('standard' or 'minmax')
        """
        self.method = method
        self.scaler = StandardScaler() if method == "standard" else MinMaxScaler()
        self.is_fitted = False
        
    def normalize(self, data: np.ndarray, fit: bool = True) -> np.ndarray:
        """
        Normalize data.
        
        Args:
            data: Input data array (2D)
            fit: Whether to fit the scaler (True for training, False for inference)
            
        Returns:
            Normalized data
        """
        if data.ndim != 2:
            raise ValueError(f"期望 2D 数组，但得到 {data.ndim}D")
        
        if fit:
            normalized = self.scaler.fit_transform(data)
            self.is_fitted = True
        else:
            if not self.is_fitted:
                raise ValueError("Scaler 未拟合，请先调用 normalize(data, fit=True)")
            normalized = self.scaler.transform(data)
        
        return normalized.astype(np.float32)
    
    def inverse_normalize(self, data: np.ndarray) -> np.ndarray:
        """
        反归一化数据
        
        Args:
            data: 归一化后的数据
            
        Returns:
            原始尺度的数据
        """
        if not self.is_fitted:
            raise ValueError("Scaler 未拟合")
        return self.scaler.inverse_transform(data).astype(np.float32)
    
    def handle_missing_values(self, data: np.ndarray, strategy: str = "mean") -> np.ndarray:
        """
        Handle missing values in NumPy array.
        
        Args:
            data: Input data with potential NaN values (2D)
            strategy: Imputation strategy ('mean', 'median', 'zero', 'forward_fill')
            
        Returns:
            Data with missing values filled
        """
        if data.ndim != 2:
            raise ValueError(f"期望 2D 数组，但得到 {data.ndim}D")
        
        result = data.copy()
        
        if strategy == 'mean':
            col_means = np.nanmean(result, axis=0)
            for j in range(result.shape[1]):
                mask = np.isnan(result[:, j])
                result[mask, j] = col_means[j]
        
        elif strategy == 'median':
            col_medians = np.nanmedian(result, axis=0)
            for j in range(result.shape[1]):
                mask = np.isnan(result[:, j])
                result[mask, j] = col_medians[j]
        
        elif strategy == 'zero':
            result = np.nan_to_num(result, nan=0.0)
        
        elif strategy == 'forward_fill':
            for j in range(result.shape[1]):
                for i in range(1, result.shape[0]):
                    if np.isnan(result[i, j]):
                        result[i, j] = result[i-1, j]
        
        else:
            raise ValueError(f"未知的填充策略: {strategy}")
        
        return result.astype(np.float32)
    
    def feature_engineering(self, genetic_data: np.ndarray, env_data: np.ndarray) -> np.ndarray:
        """
        Create interaction features between genetic and environmental data.
        
        Args:
            genetic_data: Genetic data array (n_samples, n_genetic_features)
            env_data: Environmental data array (n_samples, n_env_features)
            
        Returns:
            Feature matrix with G×E interactions (n_samples, n_genetic * n_env)
        """
        if genetic_data.shape[0] != env_data.shape[0]:
            raise ValueError(f"样本数不匹配: genetic={genetic_data.shape[0]}, env={env_data.shape[0]}")
        
        n_samples = genetic_data.shape[0]
        n_genetic = genetic_data.shape[1]
        n_env = env_data.shape[1]
        
        # 创建 G×E 交互特征
        interactions = np.zeros((n_samples, n_genetic * n_env), dtype=np.float32)
        
        for i in range(n_genetic):
            for j in range(n_env):
                interactions[:, i * n_env + j] = genetic_data[:, i] * env_data[:, j]
        
        return interactions


def check_and_handle_missing(df: pd.DataFrame, 
                             method: str = 'drop',
                             threshold: float = 0.5,
                             name: str = 'Data') -> pd.DataFrame:
    """
    检查和处理 DataFrame 中的缺失值
    
    Parameters:
        df: 输入 DataFrame
        method: 处理方法 ('drop', 'forward_fill', 'backward_fill', 'mean', 'interpolate', 'none')
        threshold: 缺失率阈值，超过此阈值的列将被删除
        name: 数据名称（用于打印信息）
    
    Returns:
        处理后的 DataFrame
    """
    n_rows, n_cols = df.shape
    n_missing_total = df.isna().sum().sum()
    
    if n_missing_total == 0:
        print(f"✓ {name}: 无缺失值")
        return df
    
    print(f"\n📊 {name} 缺失值分析:")
    print(f"   总缺失数：{n_missing_total} / {n_rows * n_cols} ({100 * n_missing_total / (n_rows * n_cols):.2f}%)")
    
    missing_per_col = df.isna().sum()
    cols_with_missing = missing_per_col[missing_per_col > 0]
    
    print(f"   有缺失值的列：{len(cols_with_missing)}")
    for col, count in cols_with_missing.items():
        missing_rate = 100 * count / n_rows
        print(f"      • {col}: {count} ({missing_rate:.1f}%)")
    
    missing_per_row = df.isna().sum(axis=1)
    rows_with_missing = (missing_per_row > 0).sum()
    print(f"   有缺失值的行：{rows_with_missing}")
    
    # 删除缺失率过高的列
    high_missing_cols = missing_per_col[missing_per_col / n_rows > threshold]
    if len(high_missing_cols) > 0:
        print(f"\n   🗑️ 删除缺失率超过 {100*threshold:.0f}% 的列：")
        for col in high_missing_cols.index:
            print(f"      • {col} ({100 * high_missing_cols[col] / n_rows:.1f}%)")
        df = df.drop(columns=high_missing_cols.index)
    
    if method == 'none':
        print(f"\n   ⏭️ 跳过缺失值处理")
        return df
    
    elif method == 'drop':
        n_before = len(df)
        df = df.dropna()
        n_after = len(df)
        print(f"\n   🗑️ 删除有缺失值的行：{n_before - n_after} 行被删除")
        
    elif method == 'forward_fill':
        df = df.fillna(method='ffill')
        print(f"   ✓ 使用向前填充处理缺失值")
        
    elif method == 'backward_fill':
        df = df.fillna(method='bfill')
        print(f"   ✓ 使用向后填充处理缺失值")
        
    elif method == 'mean':
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if df[col].isna().sum() > 0:
                mean_val = df[col].mean()
                df[col].fillna(mean_val, inplace=True)
        print(f"   ✓ 使用均值填充缺失值")
        
    elif method == 'interpolate':
        df = df.interpolate(method='linear', limit_direction='both')
        print(f"   ✓ 使用插值填充缺失值")
        
    else:
        raise ValueError(f"未知的处理方法: {method}")
    
    n_missing_after = df.isna().sum().sum()
    if n_missing_after == 0:
        print(f"   ✅ 处理完成：无剩余缺失值")
    else:
        print(f"   ⚠️ 处理后仍有 {n_missing_after} 个缺失值")
    
    return df


def aggregate_temporal_to_static(data: np.ndarray) -> np.ndarray:
    """
    将 3D 时间序列数据聚合为 2D 静态数据
    
    Parameters:
        data: 3D 数组，形状 (n_samples, n_timesteps, n_features)
    
    Returns:
        2D 数组，形状 (n_samples, n_features * 4)
        每个原始特征产生 4 个聚合特征：mean, max, min, std
    """
    if data.ndim != 3:
        raise ValueError(f"期望 3D 数组，但得到 {data.ndim}D，形状: {data.shape}")
    
    n_samples, n_timesteps, n_features = data.shape
    
    n_nan = np.isnan(data).sum()
    if n_nan > 0:
        nan_rate = 100 * n_nan / (n_samples * n_timesteps * n_features)
        print(f"⚠️ 注意：聚合数据中有 {n_nan} 个 NaN ({nan_rate:.2f}%)")
    
    mean_val = np.nanmean(data, axis=1)
    max_val = np.nanmax(data, axis=1)
    min_val = np.nanmin(data, axis=1)
    std_val = np.nanstd(data, axis=1)
    
    # 检查是否有完全为 NaN 的样本
    invalid_samples = np.where(np.isnan(mean_val).any(axis=1))[0]
    if len(invalid_samples) > 0:
        print(f"❌ 错误：发现 {len(invalid_samples)} 个完全为 NaN 的样本")
        raise ValueError("存在完全为 NaN 的样本，无法聚合")
    
    aggregated = np.concatenate([mean_val, max_val, min_val, std_val], axis=1)
    return aggregated.astype(np.float32)


def aggregate_temporal_features(data: np.ndarray, 
                               temporal_windows: Union[list, str, dict] = None,
                               crop_name: str = 'maize',
                               return_feature_names: bool = False) -> Union[np.ndarray, Tuple]:
    """
    按时间窗口聚合时间特征
    
    Parameters:
        data: 3D 数组，形状 (n_samples, n_timesteps, n_features)
        temporal_windows: 时间窗口定义
            - str: 作物名称 ('maize', 'wheat', 'rice')，使用预定义生育期
            - list: 窗口大小列表 [30, 60, 90]
            - dict: 自定义窗口 {'stage1': (0, 30), 'stage2': (30, 60)}
        crop_name: 作物名称（当 temporal_windows 为 str 时使用）
        return_feature_names: 是否返回特征名列表
    
    Returns:
        如果 return_feature_names=False:
            np.ndarray: 聚合特征 (n_samples, n_aggregated_features)
        如果 return_feature_names=True:
            Tuple[np.ndarray, List[str]]: (聚合特征, 特征名列表)
    """
    if data.ndim != 3:
        raise ValueError(f"期望 3D 数组，但得到 {data.ndim}D，形状: {data.shape}")
    
    n_samples, n_timesteps, n_features = data.shape
    aggregated_list = []
    
    # 确定时间窗口
    if isinstance(temporal_windows, str):
        window_info = get_growth_stages(temporal_windows)
    elif isinstance(temporal_windows, dict):
        window_info = temporal_windows
    elif isinstance(temporal_windows, list):
        window_info = create_fixed_windows(temporal_windows, n_timesteps)
    else:
        raise ValueError("temporal_windows 必须是 list、str 或 dict")
    
    # 按窗口聚合
    for stage_name, (start_idx, end_idx) in window_info.items():
        original_start, original_end = start_idx, end_idx
        start_idx = max(0, start_idx)
        end_idx = min(n_timesteps, end_idx)
        
        if start_idx >= end_idx:
            continue
        
        window_data = data[:, start_idx:end_idx, :]
        
        mean_val = np.nanmean(window_data, axis=1)
        max_val = np.nanmax(window_data, axis=1)
        min_val = np.nanmin(window_data, axis=1)
        std_val = np.nanstd(window_data, axis=1)
        
        aggregated_list.append((f"{stage_name}_mean", mean_val))
        aggregated_list.append((f"{stage_name}_max", max_val))
        aggregated_list.append((f"{stage_name}_min", min_val))
        aggregated_list.append((f"{stage_name}_std", std_val))
    
    if not aggregated_list:
        raise ValueError("没有有效的时间窗口！")
    
    feature_names = [name for name, _ in aggregated_list]
    feature_arrays = [arr for _, arr in aggregated_list]
    aggregated_features = np.concatenate(feature_arrays, axis=1)
    
    if return_feature_names:
        return aggregated_features.astype(np.float32), feature_names
    
    return aggregated_features.astype(np.float32)


def get_growth_stages(crop_name: str) -> Dict[str, tuple]:
    """
    获取作物生育期定义
    
    Parameters:
        crop_name: 作物名称 ('maize', 'wheat', 'rice')
    
    Returns:
        Dict: 生育期名称到 (开始日, 结束日) 的映射
    """
    crop_name = crop_name.lower()
    if crop_name not in CROP_GROWTH_STAGES:
        raise ValueError(f"不支持的作物: {crop_name}。支持的作物: {list(CROP_GROWTH_STAGES.keys())}")
    return CROP_GROWTH_STAGES[crop_name]


def create_fixed_windows(window_sizes: list, n_timesteps: int) -> Dict[str, tuple]:
    """
    创建固定大小的时间窗口
    
    Parameters:
        window_sizes: 窗口大小列表，如 [30, 60, 90]
        n_timesteps: 总时间步数
    
    Returns:
        Dict: 窗口名称到 (开始, 结束) 的映射
    """
    windows_dict = {}
    for window_size in sorted(window_sizes):
        n_windows = n_timesteps // window_size
        
        for i in range(n_windows):
            start = i * window_size
            end = (i + 1) * window_size
            stage_name = f"window_{window_size}d_seg{i+1}"
            windows_dict[stage_name] = (start, end)
        
        # 处理余数
        remainder = n_timesteps % window_size
        if remainder > 0:
            stage_name = f"window_{window_size}d_remainder"
            windows_dict[stage_name] = (n_windows * window_size, n_timesteps)
    
    return windows_dict


def add_custom_growth_stages(crop_name: str, stages: Dict[str, tuple]) -> None:
    """
    添加自定义作物生育期
    
    Parameters:
        crop_name: 作物名称
        stages: 生育期定义 {'stage_name': (start_day, end_day), ...}
    """
    CROP_GROWTH_STAGES[crop_name.lower()] = stages
    print(f"✓ 已添加作物 {crop_name} 的生育期定义")
