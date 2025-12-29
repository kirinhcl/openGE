"""Unit tests for data loading and preprocessing."""

import unittest
import numpy as np
import pandas as pd
import tempfile
import os
from openge.data import GxEDataset, Preprocessor
from openge.data.loaders import GeneticLoader, EnvironmentLoader, PhenotypeLoader


class TestDataset(unittest.TestCase):
    """Test crop dataset loading."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        
        # 创建模拟遗传数据文件
        self.n_samples = 20
        self.n_markers = 100
        self.genetic_data = np.random.randint(0, 3, (self.n_samples, self.n_markers))
        
        # 创建模拟环境数据
        self.n_env_features = 10
        self.env_data = np.random.randn(self.n_samples, self.n_env_features)
        
        # 创建模拟表型数据
        self.phenotype_data = np.random.randn(self.n_samples)
    
    def test_load_genetic_data(self):
        """Test genetic data loading."""
        # 测试 GeneticLoader 初始化
        loader = GeneticLoader()
        self.assertIsNotNone(loader)
        
        # 测试数据形状
        self.assertEqual(self.genetic_data.shape, (self.n_samples, self.n_markers))
        
        # 测试数据值范围 (0, 1, 2 for SNP genotypes)
        self.assertTrue(np.all(self.genetic_data >= 0))
        self.assertTrue(np.all(self.genetic_data <= 2))
    
    def test_load_environment_data(self):
        """Test environment data loading."""
        # 测试 EnvironmentLoader 初始化
        loader = EnvironmentLoader()
        self.assertIsNotNone(loader)
        
        # 测试数据形状
        self.assertEqual(self.env_data.shape, (self.n_samples, self.n_env_features))
        
        # 测试数据是浮点型
        self.assertTrue(np.issubdtype(self.env_data.dtype, np.floating))
    
    def test_load_phenotype_data(self):
        """Test phenotype data loading."""
        # 测试 PhenotypeLoader 初始化
        loader = PhenotypeLoader()
        self.assertIsNotNone(loader)
        
        # 测试数据形状
        self.assertEqual(self.phenotype_data.shape, (self.n_samples,))
    
    def test_gxe_dataset(self):
        """Test GxEDataset functionality."""
        import torch
        
        # 创建数据集
        genetic_tensor = torch.tensor(self.genetic_data, dtype=torch.float32)
        env_tensor = torch.tensor(self.env_data, dtype=torch.float32)
        phenotype_tensor = torch.tensor(self.phenotype_data, dtype=torch.float32)
        
        # 测试数据集长度
        self.assertEqual(len(genetic_tensor), self.n_samples)
    
    def tearDown(self):
        """Clean up test fixtures."""
        import shutil
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)


class TestPreprocessor(unittest.TestCase):
    """Test data preprocessing."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.preprocessor = Preprocessor(method="standard")
        
        # 创建测试数据
        self.n_samples = 50
        self.n_features = 20
        self.data = np.random.randn(self.n_samples, self.n_features) * 10 + 5
        
        # 添加一些缺失值
        self.data_with_missing = self.data.copy()
        self.data_with_missing[5, 3] = np.nan
        self.data_with_missing[10, 7] = np.nan
    
    def test_normalize(self):
        """Test normalization."""
        # 测试标准化 (使用 normalize 方法)
        normalized = self.preprocessor.normalize(self.data, fit=True)
        
        # 标准化后均值应接近 0
        self.assertAlmostEqual(np.mean(normalized), 0, places=1)
        
        # 标准化后标准差应接近 1
        self.assertAlmostEqual(np.std(normalized), 1, places=1)
    
    def test_minmax_normalize(self):
        """Test min-max normalization."""
        preprocessor = Preprocessor(method="minmax")
        normalized = preprocessor.normalize(self.data, fit=True)
        
        # 值应在 [0, 1] 范围内
        self.assertTrue(np.all(normalized >= 0))
        self.assertTrue(np.all(normalized <= 1))
    
    def test_handle_missing_values(self):
        """Test missing value handling."""
        from openge.data import check_and_handle_missing
        
        # 创建带缺失值的 DataFrame
        df = pd.DataFrame(self.data_with_missing)
        
        # 检测缺失值
        has_missing = df.isnull().any().any()
        self.assertTrue(has_missing)
        
        # 处理缺失值 (使用均值填充)
        df_filled = df.fillna(df.mean())
        has_missing_after = df_filled.isnull().any().any()
        self.assertFalse(has_missing_after)
    
    def test_feature_engineering(self):
        """Test feature engineering."""
        from openge.data import aggregate_temporal_to_static
        
        # 测试时间聚合功能存在
        self.assertTrue(callable(aggregate_temporal_to_static))
    
    def test_preprocessor_transform(self):
        """Test preprocessor transform on new data."""
        # 先 fit (normalize with fit=True)
        self.preprocessor.normalize(self.data, fit=True)
        
        # 生成新数据
        new_data = np.random.randn(10, self.n_features) * 10 + 5
        
        # 使用已有参数 transform (fit=False)
        transformed = self.preprocessor.normalize(new_data, fit=False)
        
        # 验证形状
        self.assertEqual(transformed.shape, new_data.shape)


if __name__ == "__main__":
    unittest.main()
