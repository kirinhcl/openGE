"""DataLoader for GxE deep learning with PyTorch compatibility."""

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from typing import Dict, Tuple, Optional, List, Union
from pathlib import Path
from .loaders.genetic import GeneticLoader
from .loaders.environment import EnvironmentLoader
from .loaders.phenotype import PhenotypeLoader


class GxEDataset(Dataset):
    """
    PyTorch Dataset for Genotype × Environment (GxE) interaction modeling.
    
    Loads and aligns genetic, environmental, and phenotypic data.
    Phenotype is treated as the dependent variable (target).
    """
    
    def __init__(self,
                 genetic_data: Optional[np.ndarray] = None,
                 environment_data: Optional[np.ndarray] = None,
                 phenotype_data: Optional[np.ndarray] = None,
                 sample_ids: Optional[List[str]] = None,
                 return_dict: bool = False):
        """
        Initialize GxE dataset.
        
        Args:
            genetic_data: Genotype data (n_samples, n_markers)
            environment_data: Environment data (n_samples, n_env_features) or 
                            (n_samples, n_timesteps, n_env_features) for temporal
            phenotype_data: Phenotype/trait data (n_samples, n_traits) - TARGET
            sample_ids: Sample identifiers
            return_dict: If True, __getitem__ returns dict; if False, returns tuple
        """
        self.genetic_data = genetic_data
        self.environment_data = environment_data
        self.phenotype_data = phenotype_data
        self.sample_ids = sample_ids
        self.return_dict = return_dict
        
        # Validate data
        self._validate_data()
        
    def _validate_data(self):
        """Validate that all data arrays have consistent sample counts."""
        n_samples = None
        
        if self.genetic_data is not None:
            n_samples = len(self.genetic_data)
            
        if self.environment_data is not None:
            if n_samples is None:
                n_samples = len(self.environment_data)
            elif len(self.environment_data) != n_samples:
                raise ValueError(f"Environment data has {len(self.environment_data)} samples, "
                               f"but expected {n_samples}")
        
        if self.phenotype_data is not None:
            if n_samples is None:
                n_samples = len(self.phenotype_data)
            elif len(self.phenotype_data) != n_samples:
                raise ValueError(f"Phenotype data has {len(self.phenotype_data)} samples, "
                               f"but expected {n_samples}")
        
        if self.sample_ids is not None:
            if len(self.sample_ids) != n_samples:
                raise ValueError(f"Sample IDs has {len(self.sample_ids)} entries, "
                               f"but expected {n_samples}")
        
        self.n_samples = n_samples if n_samples is not None else 0
        
    def __len__(self) -> int:
        """Return number of samples."""
        return self.n_samples
    
    def __getitem__(self, idx: int) -> Union[Dict, Tuple]:
        """
        Get a single sample.
        
        Args:
            idx: Sample index
            
        Returns:
            If return_dict=True: Dictionary with keys 'genetic', 'environment', 'phenotype', 'sample_id'
            If return_dict=False: Tuple (genetic, environment, phenotype)
        """
        genetic = self.genetic_data[idx] if self.genetic_data is not None else None
        environment = self.environment_data[idx] if self.environment_data is not None else None
        phenotype = self.phenotype_data[idx] if self.phenotype_data is not None else None
        sample_id = self.sample_ids[idx] if self.sample_ids is not None else None
        
        # Convert to tensors
        if genetic is not None and not isinstance(genetic, torch.Tensor):
            genetic = torch.from_numpy(genetic.astype(np.float32))
        if environment is not None and not isinstance(environment, torch.Tensor):
            environment = torch.from_numpy(environment.astype(np.float32))
        if phenotype is not None and not isinstance(phenotype, torch.Tensor):
            phenotype = torch.from_numpy(phenotype.astype(np.float32))
        
        if self.return_dict:
            return {
                'genetic': genetic,
                'environment': environment,
                'phenotype': phenotype,  # TARGET
                'sample_id': sample_id
            }
        else:
            return genetic, environment, phenotype
    
    def get_shapes(self) -> Dict[str, tuple]:
        """Get shapes of data arrays."""
        return {
            'genetic': self.genetic_data.shape if self.genetic_data is not None else None,
            'environment': self.environment_data.shape if self.environment_data is not None else None,
            'phenotype': self.phenotype_data.shape if self.phenotype_data is not None else None,
            'n_samples': self.n_samples
        }


class GxEDataLoader:
    """
    High-level data loader that integrates genetic, environmental, and phenotypic data.
    Handles data loading, alignment, preprocessing, and train/val/test splitting.
    """
    
    def __init__(self):
        """Initialize data loaders."""
        self.genetic_loader = GeneticLoader()
        self.environment_loader = EnvironmentLoader()
        self.phenotype_loader = PhenotypeLoader()
        
        # Store loaded data
        self.genetic_data = None
        self.environment_data = None
        self.phenotype_data = None
        
        self.genetic_sample_ids = None
        self.environment_sample_ids = None
        self.phenotype_sample_ids = None
        
        self.genetic_marker_names = None
        self.environment_feature_names = None
        self.phenotype_trait_names = None
    
    def load_genetic(self, 
                    filepath: str,
                    sample_col: str = '<Marker>',
                    handle_missing: str = 'mean',
                    missing_threshold: float = 0.5,
                    maf_threshold: Optional[float] = None,
                    encoding: str = 'keep') -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load genetic data.
        
        Args:
            filepath: Path to genotype file
            sample_col: Sample ID column name
            handle_missing: Missing value handling ('mean', 'drop', 'zero')
            missing_threshold: Threshold for removing high-missing markers
            maf_threshold: Minor allele frequency threshold (None = no filtering)
            encoding: Genotype encoding method ('keep', 'additive', 'standardized', etc.)
            
        Returns:
            tuple: (genotype_matrix, sample_ids, marker_names)
        """
        genotypes, sample_ids, marker_names = self.genetic_loader.load_from_numerical_file(
            filepath=filepath,
            sample_col=sample_col,
            handle_missing=handle_missing,
            missing_threshold=missing_threshold
        )
        
        # Apply MAF filtering if requested
        if maf_threshold is not None:
            genotypes, marker_names = self.genetic_loader.filter_markers(
                genotypes, marker_names, maf_threshold=maf_threshold
            )
        
        # Apply encoding
        if encoding != 'keep':
            result = self.genetic_loader.encode_genotypes(genotypes, encoding=encoding)
            if isinstance(result, tuple):
                genotypes, _ = result  # Ignore stats for now
            else:
                genotypes = result
        
        self.genetic_data = genotypes
        self.genetic_sample_ids = sample_ids
        self.genetic_marker_names = marker_names
        
        return genotypes, sample_ids, marker_names
    
    def load_environment(self,
                        weather_file: str,
                        soil_file: Optional[str] = None,
                        ec_file: Optional[str] = None,
                        use_3d: bool = True,
                        handle_missing: str = 'drop') -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load environmental data.
        
        Args:
            weather_file: Path to weather data CSV
            soil_file: Path to soil data CSV (optional)
            ec_file: Path to EC data CSV (optional)
            use_3d: If True, load as 3D temporal data; if False, aggregate to static
            handle_missing: How to handle missing values
            
        Returns:
            tuple: (environment_matrix, sample_ids, feature_names)
        """
        if use_3d:
            # Load 3D temporal weather data
            weather_3d, sample_ids, feature_names = self.environment_loader.load_weather_data_3d(
                weather_file,
                handle_missing=handle_missing
            )
            env_data = weather_3d
            
        else:
            # Load static weather data
            weather_static = self.environment_loader.load_weather_data(weather_file)
            sample_ids = list(weather_static.keys())
            env_data = np.array([weather_static[sid] for sid in sample_ids])
            feature_names = list(weather_static[sample_ids[0]].keys()) if sample_ids else []
        
        # TODO: Integrate soil and EC data if provided
        # For now, we focus on weather data
        
        self.environment_data = env_data
        self.environment_sample_ids = sample_ids
        self.environment_feature_names = feature_names
        
        return env_data, sample_ids, feature_names
    
    def load_phenotype(self,
                      filepath: str,
                      traits: Optional[List[str]] = None,
                      sample_id_col: str = 'Hybrid',
                      env_col: str = 'Env',
                      handle_missing: str = 'drop',
                      filter_by_env: Optional[List[str]] = None,
                      handle_outliers: bool = False,
                      outlier_method: str = 'iqr') -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load phenotype/trait data (dependent variable).
        
        Args:
            filepath: Path to trait data CSV
            traits: List of trait names to load (None = all available)
            sample_id_col: Column name for sample IDs
            env_col: Column name for environment IDs
            handle_missing: Missing value handling
            filter_by_env: List of environments to include
            handle_outliers: Whether to detect and handle outliers
            outlier_method: Outlier detection method ('iqr', 'zscore', 'range')
            
        Returns:
            tuple: (trait_matrix, sample_ids, trait_names)
        """
        trait_data, sample_ids, trait_names = self.phenotype_loader.load_trait_data(
            filepath=filepath,
            traits=traits,
            sample_id_col=sample_id_col,
            env_col=env_col,
            handle_missing=handle_missing,
            filter_by_env=filter_by_env
        )
        
        # Handle outliers if requested
        if handle_outliers:
            trait_data, _ = self.phenotype_loader.handle_outliers(
                trait_data,
                trait_names=trait_names,
                method=outlier_method,
                replace_with='median'
            )
        
        self.phenotype_data = trait_data
        self.phenotype_sample_ids = sample_ids
        self.phenotype_trait_names = trait_names
        
        return trait_data, sample_ids, trait_names
    
    def align_samples(self, 
                     strategy: str = 'inner') -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str]]:
        """
        Align samples across genetic, environmental, and phenotypic data.
        通过Env列关联三种数据：
        - 表型样本ID格式: "Env_Hybrid" (如 "DEH1_2014_M0088/LH185")
        - 环境样本ID: "Env" (如 "DEH1_2014")
        - 基因型样本ID: "Hybrid" (如 "M0088/LH185")
        
        Args:
            strategy: Alignment strategy
                     'inner': Keep only samples present in all datasets (recommended)
                     'left': Use phenotype samples as reference
        
        Returns:
            tuple: (aligned_genetic, aligned_environment, aligned_phenotype, aligned_sample_ids)
        """
        print(f"\n{'=' * 70}")
        print(f"样本对齐 (策略: {strategy}) - 通过Env列关联")
        print(f"{'=' * 70}")
        
        def parse_phenotype_id(sample_id):
            """解析表型样本ID，提取Env和Hybrid"""
            # 格式: "Env_Hybrid" 如 "DEH1_2014_M0088/LH185"
            # Env通常是前两部分，如 "DEH1_2014"
            parts = sample_id.split('_')
            if len(parts) >= 3:
                # 假设Env格式为: Location_Year (如 DEH1_2014)
                env = '_'.join(parts[:2])
                hybrid = '_'.join(parts[2:])
                return env, hybrid
            return None, sample_id
        
        # 构建映射表
        print(f"\n原始数据:")
        print(f"  - 表型样本: {len(self.phenotype_sample_ids)}")
        print(f"  - 基因型样本: {len(self.genetic_sample_ids)}")
        print(f"  - 环境样本: {len(self.environment_sample_ids)}")
        
        # 解析表型样本ID
        phenotype_mapping = {}  # {(env, hybrid): index}
        for i, sid in enumerate(self.phenotype_sample_ids):
            env, hybrid = parse_phenotype_id(sid)
            if env and hybrid:
                phenotype_mapping[(env, hybrid)] = i
        
        # 构建基因型和环境的映射
        genetic_mapping = {sid: i for i, sid in enumerate(self.genetic_sample_ids)}
        environment_mapping = {sid: i for i, sid in enumerate(self.environment_sample_ids)}
        
        print(f"\n解析后:")
        print(f"  - 表型 (Env, Hybrid) 组合: {len(phenotype_mapping)}")
        print(f"  - 基因型 Hybrid: {len(genetic_mapping)}")
        print(f"  - 环境 Env: {len(environment_mapping)}")
        
        # 打印示例
        if phenotype_mapping:
            sample_keys = list(phenotype_mapping.keys())[:3]
            print(f"\n表型样本示例:")
            for env, hybrid in sample_keys:
                print(f"  - Env={env}, Hybrid={hybrid}")
        
        if self.genetic_sample_ids:
            print(f"\n基因型样本示例: {self.genetic_sample_ids[:3]}")
        
        if self.environment_sample_ids:
            print(f"\n环境样本示例: {self.environment_sample_ids[:3]}")
        
        # 找到可以对齐的样本
        aligned_genetic = []
        aligned_environment = []
        aligned_phenotype = []
        aligned_sample_ids = []
        
        n_missing_genetic = 0
        n_missing_environment = 0
        
        for (env, hybrid), pheno_idx in phenotype_mapping.items():
            # 检查基因型是否存在
            if hybrid not in genetic_mapping:
                n_missing_genetic += 1
                if strategy == 'inner':
                    continue
            
            # 检查环境是否存在
            if env not in environment_mapping:
                n_missing_environment += 1
                if strategy == 'inner':
                    continue
            
            # 如果两者都存在，添加到对齐数组
            if hybrid in genetic_mapping and env in environment_mapping:
                geno_idx = genetic_mapping[hybrid]
                env_idx = environment_mapping[env]
                
                aligned_genetic.append(self.genetic_data[geno_idx])
                aligned_environment.append(self.environment_data[env_idx])
                aligned_phenotype.append(self.phenotype_data[pheno_idx])
                aligned_sample_ids.append(self.phenotype_sample_ids[pheno_idx])
        
        if n_missing_genetic > 0:
            print(f"\n⚠️ 警告: {n_missing_genetic} 个表型样本缺少对应的基因型数据")
        if n_missing_environment > 0:
            print(f"⚠️ 警告: {n_missing_environment} 个表型样本缺少对应的环境数据")
        
        if len(aligned_genetic) == 0:
            raise ValueError(
                "没有找到可以对齐的样本！\n"
                "请检查：\n"
                "1. 表型数据样本ID格式是否为 'Env_Hybrid' (如 'DEH1_2014_M0088/LH185')\n"
                "2. 环境数据样本ID是否包含对应的Env (如 'DEH1_2014')\n"
                "3. 基因型数据样本ID是否包含对应的Hybrid (如 'M0088/LH185')"
            )
        
        # 转换为数组
        aligned_genetic = np.array(aligned_genetic)
        aligned_environment = np.array(aligned_environment)
        aligned_phenotype = np.array(aligned_phenotype)
        
        print(f"\n✓ 对齐成功:")
        print(f"  - 对齐样本数: {len(aligned_sample_ids)}")
        print(f"  - 基因型形状: {aligned_genetic.shape}")
        print(f"  - 环境形状: {aligned_environment.shape}")
        print(f"  - 表型形状: {aligned_phenotype.shape}")
        
        # 打印一些对齐的样本示例
        print(f"\n对齐样本示例:")
        for i, sid in enumerate(aligned_sample_ids[:5]):
            env, hybrid = parse_phenotype_id(sid)
            print(f"  {i+1}. {sid}")
            print(f"     └─ Env: {env}, Hybrid: {hybrid}")
        
        print(f"{'=' * 70}\n")
        
        return aligned_genetic, aligned_environment, aligned_phenotype, aligned_sample_ids
    
    def create_dataset(self,
                      genetic_data: np.ndarray,
                      environment_data: np.ndarray,
                      phenotype_data: np.ndarray,
                      sample_ids: List[str],
                      return_dict: bool = False) -> GxEDataset:
        """
        Create a GxEDataset from aligned data.
        
        Args:
            genetic_data: Aligned genotype data
            environment_data: Aligned environment data
            phenotype_data: Aligned phenotype data (TARGET)
            sample_ids: Aligned sample IDs
            return_dict: Return dict format in __getitem__
            
        Returns:
            GxEDataset instance
        """
        return GxEDataset(
            genetic_data=genetic_data,
            environment_data=environment_data,
            phenotype_data=phenotype_data,
            sample_ids=sample_ids,
            return_dict=return_dict
        )
    
    def split_dataset(self,
                     dataset: GxEDataset,
                     train_ratio: float = 0.7,
                     val_ratio: float = 0.15,
                     test_ratio: float = 0.15,
                     random_seed: int = 42,
                     stratify_by: Optional[str] = None) -> Tuple[GxEDataset, GxEDataset, GxEDataset]:
        """
        Split dataset into train/validation/test sets.
        
        Args:
            dataset: GxEDataset to split
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
            stratify_by: Strategy for stratification (None, 'environment', 'trait_bins')
            
        Returns:
            tuple: (train_dataset, val_dataset, test_dataset)
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "train_ratio + val_ratio + test_ratio must equal 1.0"
        
        n_samples = len(dataset)
        indices = np.arange(n_samples)
        
        # Shuffle indices
        np.random.seed(random_seed)
        np.random.shuffle(indices)
        
        # Calculate split points
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        print(f"\n数据集划分:")
        print(f"  - 训练集: {len(train_indices)} 样本 ({len(train_indices)/n_samples*100:.1f}%)")
        print(f"  - 验证集: {len(val_indices)} 样本 ({len(val_indices)/n_samples*100:.1f}%)")
        print(f"  - 测试集: {len(test_indices)} 样本 ({len(test_indices)/n_samples*100:.1f}%)\n")
        
        # Create subset datasets
        def create_subset(indices):
            return GxEDataset(
                genetic_data=dataset.genetic_data[indices] if dataset.genetic_data is not None else None,
                environment_data=dataset.environment_data[indices] if dataset.environment_data is not None else None,
                phenotype_data=dataset.phenotype_data[indices] if dataset.phenotype_data is not None else None,
                sample_ids=[dataset.sample_ids[i] for i in indices] if dataset.sample_ids else None,
                return_dict=dataset.return_dict
            )
        
        train_dataset = create_subset(train_indices)
        val_dataset = create_subset(val_indices)
        test_dataset = create_subset(test_indices)
        
        return train_dataset, val_dataset, test_dataset
    
    def create_dataloader(self,
                         dataset: GxEDataset,
                         batch_size: int = 32,
                         shuffle: bool = True,
                         num_workers: int = 0,
                         pin_memory: bool = True) -> DataLoader:
        """
        Create PyTorch DataLoader from GxEDataset.
        
        Args:
            dataset: GxEDataset instance
            batch_size: Batch size
            shuffle: Whether to shuffle data
            num_workers: Number of worker processes
            pin_memory: Whether to pin memory for faster GPU transfer
            
        Returns:
            PyTorch DataLoader
        """
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def get_summary(self) -> Dict:
        """
        Get summary of loaded data.
        
        Returns:
            Dictionary with data statistics
        """
        summary = {
            'genetic': {
                'loaded': self.genetic_data is not None,
                'shape': self.genetic_data.shape if self.genetic_data is not None else None,
                'n_samples': len(self.genetic_sample_ids) if self.genetic_sample_ids else 0,
                'n_markers': len(self.genetic_marker_names) if self.genetic_marker_names else 0
            },
            'environment': {
                'loaded': self.environment_data is not None,
                'shape': self.environment_data.shape if self.environment_data is not None else None,
                'n_samples': len(self.environment_sample_ids) if self.environment_sample_ids else 0,
                'n_features': len(self.environment_feature_names) if self.environment_feature_names else 0
            },
            'phenotype': {
                'loaded': self.phenotype_data is not None,
                'shape': self.phenotype_data.shape if self.phenotype_data is not None else None,
                'n_samples': len(self.phenotype_sample_ids) if self.phenotype_sample_ids else 0,
                'n_traits': len(self.phenotype_trait_names) if self.phenotype_trait_names else 0,
                'traits': self.phenotype_trait_names
            }
        }
        
        return summary
    
    def print_summary(self):
        """Print data loading summary."""
        summary = self.get_summary()
        
        print("\n" + "=" * 70)
        print("数据加载摘要")
        print("=" * 70)
        
        print("\n【基因型数据】")
        if summary['genetic']['loaded']:
            print(f"  ✓ 已加载: {summary['genetic']['shape']}")
            print(f"  - 样本数: {summary['genetic']['n_samples']}")
            print(f"  - 标记数: {summary['genetic']['n_markers']}")
        else:
            print("  ✗ 未加载")
        
        print("\n【环境数据】")
        if summary['environment']['loaded']:
            print(f"  ✓ 已加载: {summary['environment']['shape']}")
            print(f"  - 样本数: {summary['environment']['n_samples']}")
            print(f"  - 特征数: {summary['environment']['n_features']}")
        else:
            print("  ✗ 未加载")
        
        print("\n【表型数据】 (目标变量)")
        if summary['phenotype']['loaded']:
            print(f"  ✓ 已加载: {summary['phenotype']['shape']}")
            print(f"  - 样本数: {summary['phenotype']['n_samples']}")
            print(f"  - 性状数: {summary['phenotype']['n_traits']}")
            print(f"  - 性状: {summary['phenotype']['traits']}")
        else:
            print("  ✗ 未加载")
        
        print("\n" + "=" * 70 + "\n")
