"""Loader for phenotype/trait data."""

import numpy as np
import pandas as pd
from typing import Union, Dict, List, Tuple, Optional
from pathlib import Path
from datetime import datetime


class PhenotypeLoader:
    """Loader for phenotype/trait data (target variables)."""
    
    # 定义可用的表型性状及其元数据
    TRAIT_METADATA = {
        'Yield_Mg_ha': {
            'name': 'Grain Yield',
            'unit': 'Mg/ha',
            'description': '籽粒产量',
            'type': 'continuous',
            'range': (0, 30),
            'primary': True
        },
        'Plant_Height_cm': {
            'name': 'Plant Height',
            'unit': 'cm',
            'description': '株高',
            'type': 'continuous',
            'range': (50, 400)
        },
        'Ear_Height_cm': {
            'name': 'Ear Height',
            'unit': 'cm',
            'description': '穗高',
            'type': 'continuous',
            'range': (30, 250)
        },
        'Pollen_DAP_days': {
            'name': 'Days to Pollen',
            'unit': 'days',
            'description': '开花期',
            'type': 'continuous',
            'range': (40, 100)
        },
        'Silk_DAP_days': {
            'name': 'Days to Silk',
            'unit': 'days',
            'description': '吐丝期',
            'type': 'continuous',
            'range': (40, 100)
        },
        'Grain_Moisture': {
            'name': 'Grain Moisture',
            'unit': '%',
            'description': '籽粒水分',
            'type': 'continuous',
            'range': (10, 40)
        },
        'Twt_kg_m3': {
            'name': 'Test Weight',
            'unit': 'kg/m³',
            'description': '容重',
            'type': 'continuous',
            'range': (500, 850)
        },
        'Stand_Count_plants': {
            'name': 'Stand Count',
            'unit': 'plants',
            'description': '成株数',
            'type': 'count',
            'range': (0, 100)
        },
        'Root_Lodging_plants': {
            'name': 'Root Lodging',
            'unit': 'plants',
            'description': '根倒伏株数',
            'type': 'count',
            'range': (0, 100)
        },
        'Stalk_Lodging_plants': {
            'name': 'Stalk Lodging',
            'unit': 'plants',
            'description': '茎倒伏株数',
            'type': 'count',
            'range': (0, 100)
        }
    }
    
    def __init__(self):
        """Initialize phenotype data loader."""
        self.data: Optional[pd.DataFrame] = None
        self.trait_names: Optional[List[str]] = None
        self.sample_ids: Optional[List[str]] = None
        self.metadata: Optional[Dict] = None
    
    def load_trait_data(self, 
                       filepath: str,
                       traits: Optional[List[str]] = None,
                       sample_id_col: str = 'Hybrid',
                       env_col: str = 'Env',
                       handle_missing: str = 'drop',
                       filter_by_env: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load trait/phenotype data from CSV file.
        
        Args:
            filepath: Path to trait data file
            traits: List of trait names to load. If None, loads all available traits
            sample_id_col: Column name for sample IDs (default: 'Hybrid')
            env_col: Column name for environment IDs (default: 'Env')
            handle_missing: How to handle missing values
                           'drop': Drop samples with any missing traits
                           'mean': Fill with mean value
                           'median': Fill with median value
                           'keep': Keep NaN values
            filter_by_env: List of environments to include (None = all)
            
        Returns:
            tuple: (trait_matrix, sample_ids, trait_names)
                   trait_matrix shape: (n_samples, n_traits)
        """
        print(f"\n{'=' * 70}")
        print(f"加载表型数据: {Path(filepath).name}")
        print(f"{'=' * 70}")
        
        # 读取数据
        df = pd.read_csv(filepath)
        print(f"✓ 文件加载成功: {df.shape}")
        
        # 过滤环境
        if filter_by_env is not None:
            df = df[df[env_col].isin(filter_by_env)]
            print(f"✓ 过滤环境: {len(filter_by_env)} 个环境, {len(df)} 行数据")
        
        # 确定要加载的性状
        if traits is None:
            traits = [t for t in self.TRAIT_METADATA.keys() if t in df.columns]
        else:
            # 验证性状是否存在
            missing_traits = [t for t in traits if t not in df.columns]
            if missing_traits:
                raise ValueError(f"未找到性状: {missing_traits}")
        
        print(f"✓ 目标性状: {len(traits)} 个")
        for trait in traits:
            meta = self.TRAIT_METADATA.get(trait, {})
            print(f"  - {trait}: {meta.get('description', 'N/A')} ({meta.get('unit', 'N/A')})")
        
        # 提取样本ID
        if sample_id_col not in df.columns:
            raise ValueError(f"未找到样本ID列: {sample_id_col}")
        
        # 构建样本ID（环境+杂交组合）
        sample_ids = (df[env_col].astype(str) + '_' + df[sample_id_col].astype(str)).tolist()
        print(f"✓ 样本数: {len(sample_ids)} (环境_杂交组合)")
        
        # 提取性状数据
        trait_data = df[traits].values
        
        # 统计缺失值
        n_missing = np.isnan(trait_data.astype(float)).sum()
        missing_rate = 100 * n_missing / trait_data.size
        print(f"\n📊 缺失值分析:")
        print(f"   - 总缺失值: {n_missing} / {trait_data.size} ({missing_rate:.2f}%)")
        
        # 按性状统计缺失率
        for i, trait in enumerate(traits):
            trait_missing = np.isnan(trait_data[:, i].astype(float)).sum()
            trait_missing_rate = 100 * trait_missing / len(trait_data)
            print(f"   - {trait}: {trait_missing} / {len(trait_data)} ({trait_missing_rate:.2f}%)")
        
        # 处理缺失值
        original_size = len(trait_data)
        
        if handle_missing == 'drop':
            # 删除有缺失值的样本
            valid_mask = ~np.isnan(trait_data.astype(float)).any(axis=1)
            trait_data = trait_data[valid_mask]
            sample_ids = [sid for sid, valid in zip(sample_ids, valid_mask) if valid]
            print(f"   - 删除缺失样本: {original_size - len(trait_data)} 个")
            
        elif handle_missing == 'mean':
            # 用均值填充
            trait_data = trait_data.astype(float)
            for i in range(trait_data.shape[1]):
                col_mean = np.nanmean(trait_data[:, i])
                trait_data[np.isnan(trait_data[:, i]), i] = col_mean
            print(f"   - 用均值填充缺失值")
            
        elif handle_missing == 'median':
            # 用中位数填充
            trait_data = trait_data.astype(float)
            for i in range(trait_data.shape[1]):
                col_median = np.nanmedian(trait_data[:, i])
                trait_data[np.isnan(trait_data[:, i]), i] = col_median
            print(f"   - 用中位数填充缺失值")
            
        elif handle_missing == 'keep':
            trait_data = trait_data.astype(float)
            print(f"   - 保留缺失值")
        
        else:
            raise ValueError(f"未知的缺失值处理方法: {handle_missing}")
        
        # 统计每个性状
        print(f"\n✓ 性状统计:")
        trait_data_float = trait_data.astype(float)
        for i, trait in enumerate(traits):
            valid_data = trait_data_float[:, i][~np.isnan(trait_data_float[:, i])]
            if len(valid_data) > 0:
                print(f"   - {trait}:")
                print(f"     均值={valid_data.mean():.2f}, "
                      f"标准差={valid_data.std():.2f}, "
                      f"范围=[{valid_data.min():.2f}, {valid_data.max():.2f}]")
        
        # 保存元数据
        self.data = df
        self.trait_names = traits
        self.sample_ids = sample_ids
        
        print(f"\n✓ 最终数据形状: {trait_data.shape}")
        print(f"{'=' * 70}")
        print(f"表型数据加载完成")
        print(f"{'=' * 70}\n")
        
        return trait_data.astype(np.float32), sample_ids, traits
    
    def handle_outliers(self, 
                       data: np.ndarray,
                       trait_names: Optional[List[str]] = None,
                       method: str = "iqr",
                       iqr_factor: float = 1.5,
                       zscore_threshold: float = 3.0,
                       replace_with: str = 'median') -> Tuple[np.ndarray, Dict]:
        """
        Handle outliers in phenotype data.
        
        Args:
            data: Phenotype data (n_samples, n_traits)
            trait_names: Names of traits (for logging)
            method: Outlier detection method
                   'iqr': Interquartile range method
                   'zscore': Z-score method
                   'range': Use predefined valid ranges from TRAIT_METADATA
            iqr_factor: IQR multiplier for outlier detection (default: 1.5)
            zscore_threshold: Z-score threshold (default: 3.0)
            replace_with: How to replace outliers ('median', 'mean', 'clip', 'nan')
            
        Returns:
            tuple: (cleaned_data, outlier_info)
        """
        print(f"\n{'=' * 70}")
        print(f"异常值检测与处理 (方法: {method})")
        print(f"{'=' * 70}")
        
        cleaned_data = data.copy().astype(float)
        outlier_info = {'method': method, 'traits': {}}
        
        for i in range(data.shape[1]):
            trait_name = trait_names[i] if trait_names and i < len(trait_names) else f"Trait_{i}"
            col_data = data[:, i].astype(float)
            valid_mask = ~np.isnan(col_data)
            valid_data = col_data[valid_mask]
            
            if len(valid_data) == 0:
                continue
            
            outlier_mask = np.zeros(len(col_data), dtype=bool)
            
            if method == 'iqr':
                # IQR方法
                q1 = np.percentile(valid_data, 25)
                q3 = np.percentile(valid_data, 75)
                iqr = q3 - q1
                lower_bound = q1 - iqr_factor * iqr
                upper_bound = q3 + iqr_factor * iqr
                
                outlier_mask = valid_mask & ((col_data < lower_bound) | (col_data > upper_bound))
                bounds = (lower_bound, upper_bound)
                
            elif method == 'zscore':
                # Z-score方法
                mean = valid_data.mean()
                std = valid_data.std()
                if std > 0:
                    z_scores = np.abs((col_data - mean) / std)
                    outlier_mask = valid_mask & (z_scores > zscore_threshold)
                bounds = (mean - zscore_threshold * std, mean + zscore_threshold * std)
                
            elif method == 'range':
                # 使用预定义范围
                if trait_name in self.TRAIT_METADATA:
                    valid_range = self.TRAIT_METADATA[trait_name]['range']
                    outlier_mask = valid_mask & ((col_data < valid_range[0]) | (col_data > valid_range[1]))
                    bounds = valid_range
                else:
                    # 如果没有预定义范围，使用IQR
                    q1 = np.percentile(valid_data, 25)
                    q3 = np.percentile(valid_data, 75)
                    iqr = q3 - q1
                    bounds = (q1 - 1.5 * iqr, q3 + 1.5 * iqr)
                    outlier_mask = valid_mask & ((col_data < bounds[0]) | (col_data > bounds[1]))
            
            else:
                raise ValueError(f"未知的异常值检测方法: {method}")
            
            n_outliers = outlier_mask.sum()
            outlier_info['traits'][trait_name] = {
                'n_outliers': int(n_outliers),
                'outlier_rate': float(n_outliers / valid_mask.sum() * 100),
                'bounds': bounds
            }
            
            # 替换异常值
            if n_outliers > 0:
                if replace_with == 'median':
                    replacement = np.median(valid_data[~outlier_mask[valid_mask]])
                elif replace_with == 'mean':
                    replacement = np.mean(valid_data[~outlier_mask[valid_mask]])
                elif replace_with == 'clip':
                    cleaned_data[outlier_mask & (col_data < bounds[0]), i] = bounds[0]
                    cleaned_data[outlier_mask & (col_data > bounds[1]), i] = bounds[1]
                    replacement = None
                elif replace_with == 'nan':
                    cleaned_data[outlier_mask, i] = np.nan
                    replacement = None
                else:
                    raise ValueError(f"未知的替换方法: {replace_with}")
                
                if replacement is not None and replace_with not in ['clip', 'nan']:
                    cleaned_data[outlier_mask, i] = replacement
                
                print(f"✓ {trait_name}: {n_outliers} 个异常值 "
                      f"({n_outliers / valid_mask.sum() * 100:.2f}%), "
                      f"范围: [{bounds[0]:.2f}, {bounds[1]:.2f}]")
        
        print(f"{'=' * 70}\n")
        return cleaned_data.astype(np.float32), outlier_info
    
    def get_trait_info(self, trait_names: Optional[List[str]] = None) -> Dict:
        """
        Get metadata about traits.
        
        Args:
            trait_names: List of trait names. If None, returns all available traits.
            
        Returns:
            Dictionary with trait information (name, unit, description, etc.)
        """
        if trait_names is None:
            return self.TRAIT_METADATA.copy()
        
        return {name: self.TRAIT_METADATA[name] for name in trait_names 
                if name in self.TRAIT_METADATA}
    
    def save_to_numpy(self,
                     trait_data: np.ndarray,
                     sample_ids: List[str],
                     trait_names: List[str],
                     output_path: str):
        """
        Save phenotype data to .npz file.
        
        Args:
            trait_data: Trait/phenotype data
            sample_ids: Sample IDs
            trait_names: Trait names
            output_path: Output file path
        """
        np.savez_compressed(
            output_path,
            traits=trait_data,
            sample_ids=sample_ids,
            trait_names=trait_names
        )
        
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        print(f"✓ 保存表型数据: {output_path}")
        print(f"  - 文件大小: {file_size:.2f} MB")
    
    def compute_correlations(self, 
                            trait_data: np.ndarray,
                            trait_names: List[str]) -> Tuple[np.ndarray, pd.DataFrame]:
        """
        Compute correlation matrix between traits.
        
        Args:
            trait_data: Trait data (n_samples, n_traits)
            trait_names: List of trait names
            
        Returns:
            tuple: (correlation_matrix, correlation_dataframe)
        """
        # 计算相关系数矩阵
        corr_matrix = np.corrcoef(trait_data.T)
        
        # 创建DataFrame便于查看
        corr_df = pd.DataFrame(corr_matrix, 
                              index=trait_names,
                              columns=trait_names)
        
        return corr_matrix, corr_df
