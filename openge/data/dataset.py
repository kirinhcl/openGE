"""Universal dataset loader for crop trait prediction."""

import pandas as pd
import numpy as np
from typing import Dict, Tuple, Optional, List
from pathlib import Path
from .loaders import GeneticLoader, EnvironmentLoader, PhenotypeLoader


class CropDataset:
    """Generic crop dataset loader supporting multiple data types."""
    
    def __init__(self, data_dir: str, crop_name: str):
        """
        Initialize dataset loader.
        
        Args:
            data_dir: Directory containing data files
            crop_name: Name of crop (e.g., 'maize', 'wheat')
        """
        self.data_dir = Path(data_dir)
        self.crop_name = crop_name
        self.data = {}
        
        # 初始化各类型的 loader
        self.genetic_loader = GeneticLoader()
        self.environment_loader = EnvironmentLoader()
        self.phenotype_loader = PhenotypeLoader()
    
    def load_genetic_data(self, filepath: str) -> np.ndarray:
        """
        Load genetic/SNP data using GeneticLoader.
        
        Args:
            filepath: Path to genetic data file
            
        Returns:
            Genetic data array [n_samples, n_markers]
        """
        # ✅ 使用 GeneticLoader 来加载
        genetic_data = self.genetic_loader.load_snp_data(filepath)
        self.data['genetic'] = genetic_data
        print(f"✓ 加载基因数据: {genetic_data.shape}")
        return genetic_data
    
    def load_environment_data(self, weather_file: str, 
                             soil_file: str, 
                             ec_file: str,
                             temporal_file: str = None) -> Dict[str, np.ndarray]:
        """
        Load environmental data using EnvironmentLoader.
        
        Args:
            weather_file: Path to weather data
            soil_file: Path to soil data
            ec_file: Path to EC (environmental covariate) data
            temporal_file: Optional path to temporal weather data
            
        Returns:
            Dictionary with different environment types
        """
        # ✅ 使用 EnvironmentLoader 来加载
        env_data = self.environment_loader.load_all_environment_data(
            weather_file=weather_file,
            soil_file=soil_file,
            ec_file=ec_file,
            temporal_weather_file=temporal_file,
            temporal_windows=[90, 180, 365]
        )
        
        self.data['environment'] = env_data
        print(f"✓ 加载环境数据: {env_data.shape}")
        return {'all': env_data}
    
    def load_phenotype_data(self, filepath: str) -> np.ndarray:
        """
        Load phenotype/trait data using PhenotypeLoader.
        
        Args:
            filepath: Path to phenotype data file
            
        Returns:
            Phenotype array [n_samples, n_traits]
        """
        # ✅ 使用 PhenotypeLoader 来加载
        phenotype_data = self.phenotype_loader.load_traits(filepath)
        self.data['phenotype'] = phenotype_data
        print(f"✓ 加载表型数据: {phenotype_data.shape}")
        return phenotype_data
    
    def load_all(self, 
                 genetic_file: str,
                 weather_file: str,
                 soil_file: str,
                 ec_file: str,
                 phenotype_file: str,
                 temporal_file: str = None) -> None:
        """
        Load all data at once.
        
        Args:
            genetic_file: Path to genetic data
            weather_file: Path to weather data
            soil_file: Path to soil data
            ec_file: Path to EC data
            phenotype_file: Path to phenotype data
            temporal_file: Optional temporal weather data
        """
        print(f"\n正在加载 {self.crop_name} 数据集...")
        
        self.load_genetic_data(genetic_file)
        self.load_environment_data(weather_file, soil_file, ec_file, temporal_file)
        self.load_phenotype_data(phenotype_file)
        
        # 验证数据一致性
        n_samples = self.data['genetic'].shape[0]
        assert self.data['environment'].shape[0] == n_samples
        assert self.data['phenotype'].shape[0] == n_samples
        print(f"\n✓ 所有数据加载完成！样本数: {n_samples}\n")
    
    def __len__(self) -> int:
        """Return dataset size."""
        if 'genetic' in self.data:
            return self.data['genetic'].shape[0]
        return 0
    
    def __getitem__(self, idx: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get single sample (genetic, environment, phenotype).
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (genetic_data, environment_data, phenotype_data)
        """
        genetic = self.data['genetic'][idx]        # [n_markers]
        environment = self.data['environment'][idx] # [n_env_features]
        phenotype = self.data['phenotype'][idx]     # [n_traits]
        
        return genetic, environment, phenotype
    
    def get_train_val_test_split(self, 
                                 train_ratio: float = 0.7,
                                 val_ratio: float = 0.15,
                                 test_ratio: float = 0.15,
                                 random_seed: int = 42) -> Dict:
        """
        Split dataset into train, validation, and test sets.
        
        Args:
            train_ratio: Proportion for training
            val_ratio: Proportion for validation
            test_ratio: Proportion for testing
            random_seed: Random seed for reproducibility
            
        Returns:
            Dictionary with split indices
        """
        np.random.seed(random_seed)
        n_samples = len(self)
        indices = np.random.permutation(n_samples)
        
        train_end = int(n_samples * train_ratio)
        val_end = train_end + int(n_samples * val_ratio)
        
        return {
            'train': indices[:train_end],
            'val': indices[train_end:val_end],
            'test': indices[val_end:]
        }
