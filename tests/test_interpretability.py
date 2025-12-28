"""Unit tests for interpretability modules."""

import unittest
import torch
import numpy as np
from openge.interpretability import (
    AttentionAnalyzer,
    FeatureImportance,
    GradientExplainer,
    SparsityAnalyzer,
    SHAPExplainer
)


class TestAttentionAnalyzer(unittest.TestCase):
    """Test attention analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        # 创建带注意力的简单模型
        self.model = torch.nn.Sequential(
            torch.nn.Linear(64, 32),
            torch.nn.ReLU(),
            torch.nn.Linear(32, 1)
        )
        self.analyzer = AttentionAnalyzer(self.model)
        
    def test_initialization(self):
        """Test analyzer initialization."""
        self.assertIsNotNone(self.analyzer)
        self.assertEqual(self.analyzer.model, self.model)
    
    def test_extract_attention_heads(self):
        """Test attention head extraction."""
        # 对于没有注意力层的模型，应返回空
        x = torch.randn(2, 64)
        attention = self.analyzer.extract_attention_heads(x)
        self.assertIsInstance(attention, dict)
    
    def test_head_importance(self):
        """Test attention head importance analysis."""
        # 测试分析方法存在
        self.assertTrue(hasattr(self.analyzer, 'analyze_head_importance'))


class TestFeatureImportance(unittest.TestCase):
    """Test feature importance computation."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 50
        self.model = torch.nn.Linear(self.input_dim, 1)
        self.fi = FeatureImportance(self.model)
        
        # 生成测试数据
        self.n_samples = 30
        self.X = np.random.randn(self.n_samples, self.input_dim).astype(np.float32)
        self.y = self.X @ np.random.randn(self.input_dim) + np.random.randn(self.n_samples) * 0.1
    
    def test_gradient_importance(self):
        """Test gradient-based importance."""
        result = self.fi.gradient_importance(self.X)
        
        self.assertIn('importance', result)
        self.assertEqual(result['importance'].shape, (self.input_dim,))
        
        # 重要性应为非负
        self.assertTrue(np.all(result['importance'] >= 0))
    
    def test_permutation_importance(self):
        """Test permutation importance."""
        result = self.fi.permutation_importance(
            self.X, self.y.astype(np.float32), 
            n_repeats=3, metric='r2'
        )
        
        self.assertIn('importances_mean', result)
        self.assertIn('importances_std', result)
        self.assertEqual(len(result['importances_mean']), self.input_dim)
    
    def test_genetic_vs_env_contribution(self):
        """Test G vs E contribution analysis."""
        # 创建双输入模型
        class DualInputModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.genetic_layer = torch.nn.Linear(30, 16)
                self.env_layer = torch.nn.Linear(20, 16)
                self.output = torch.nn.Linear(32, 1)
            
            def forward(self, g, e):
                g_feat = torch.relu(self.genetic_layer(g))
                e_feat = torch.relu(self.env_layer(e))
                combined = torch.cat([g_feat, e_feat], dim=-1)
                return self.output(combined), None
        
        model = DualInputModel()
        fi = FeatureImportance(model)
        
        genetic_data = np.random.randn(20, 30).astype(np.float32)
        env_data = np.random.randn(20, 20).astype(np.float32)
        y = np.random.randn(20).astype(np.float32)
        
        result = fi.genetic_vs_environment_contribution(
            genetic_data, env_data, y, n_permutations=5
        )
        
        self.assertIn('genetic_ratio', result)
        self.assertIn('environment_ratio', result)
        
        # 比例之和应接近 1
        total = result['genetic_ratio'] + result['environment_ratio']
        self.assertAlmostEqual(total, 1.0, places=5)


class TestGradientExplainer(unittest.TestCase):
    """Test gradient-based explanation methods."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(40, 20),
            torch.nn.ReLU(),
            torch.nn.Linear(20, 1)
        )
        self.explainer = GradientExplainer(self.model)
        self.x = np.random.randn(1, 40).astype(np.float32)
    
    def test_saliency_map(self):
        """Test saliency map generation."""
        saliency = self.explainer.saliency_map(self.x)
        
        self.assertEqual(saliency.shape, self.x.shape)
        # 默认取绝对值，应为非负
        self.assertTrue(np.all(saliency >= 0))
    
    def test_integrated_gradients(self):
        """Test integrated gradients."""
        ig = self.explainer.integrated_gradients(self.x, steps=20)
        
        self.assertEqual(ig.shape, torch.Size(self.x.shape))
    
    def test_smoothgrad(self):
        """Test SmoothGrad."""
        smoothgrad = self.explainer.smoothgrad(self.x, n_samples=10, noise_level=0.1)
        
        self.assertEqual(smoothgrad.shape, self.x.shape)
    
    def test_gradient_times_input(self):
        """Test Gradient × Input."""
        gxi = self.explainer.gradient_times_input(self.x)
        
        self.assertEqual(gxi.shape, self.x.shape)
    
    def test_compare_methods(self):
        """Test method comparison."""
        result = self.explainer.compare_methods(
            self.x, methods=['saliency', 'smoothgrad']
        )
        
        self.assertIn('saliency', result)
        self.assertIn('smoothgrad', result)
        self.assertIn('correlation_matrix', result)


class TestSparsityAnalyzer(unittest.TestCase):
    """Test sparsity analysis."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = torch.nn.Sequential(
            torch.nn.Linear(100, 50),
            torch.nn.ReLU(),
            torch.nn.Linear(50, 10)
        )
        
        # 手动设置一些权重为零以创建稀疏性
        with torch.no_grad():
            mask = torch.rand_like(self.model[0].weight) > 0.7
            self.model[0].weight *= mask.float()
        
        self.analyzer = SparsityAnalyzer(self.model)
    
    def test_sparsity_levels(self):
        """Test sparsity level computation."""
        sparsity = self.analyzer.compute_sparsity_levels()
        
        self.assertIn('__global__', sparsity)
        self.assertTrue(0 <= sparsity['__global__'] <= 100)
        
        # 第一层应该有约 70% 稀疏度
        self.assertTrue(sparsity['0'] > 50)
    
    def test_weight_distribution(self):
        """Test weight distribution analysis."""
        distribution = self.analyzer.analyze_weight_distribution()
        
        self.assertIn('layer_stats', distribution)
        self.assertIn('global_stats', distribution)
        
        global_stats = distribution['global_stats']
        self.assertIn('mean', global_stats)
        self.assertIn('std', global_stats)
    
    def test_find_bottleneck_layers(self):
        """Test bottleneck layer identification."""
        bottlenecks = self.analyzer.find_bottleneck_layers(top_k=2)
        
        self.assertIsInstance(bottlenecks, list)
        self.assertTrue(len(bottlenecks) <= 2)
        
        # 每个元素应该是 (层名, 稀疏度) 元组
        if len(bottlenecks) > 0:
            self.assertEqual(len(bottlenecks[0]), 2)
    
    def test_visualize_sparsity_pattern(self):
        """Test sparsity pattern visualization."""
        pattern = self.analyzer.visualize_sparsity_pattern('0', mode='binary')
        
        self.assertIsInstance(pattern, np.ndarray)
        # Binary 模式下值应为 0 或 1
        self.assertTrue(np.all((pattern == 0) | (pattern == 1)))
    
    def test_layer_wise_summary(self):
        """Test layer-wise summary."""
        summary = self.analyzer.layer_wise_summary()
        
        self.assertIn('0', summary)
        self.assertIn('sparsity_pct', summary['0'])
        self.assertIn('n_params', summary['0'])


class TestSHAPExplainer(unittest.TestCase):
    """Test SHAP explainer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.input_dim = 15
        self.model = torch.nn.Linear(self.input_dim, 1)
        self.background = np.random.randn(10, self.input_dim).astype(np.float32)
        self.explainer = SHAPExplainer(self.model, self.background)
    
    def test_explain_instance(self):
        """Test single instance explanation."""
        instance = np.random.randn(self.input_dim).astype(np.float32)
        result = self.explainer.explain_instance(instance, n_samples=20)
        
        self.assertIn('shap_values', result)
        self.assertIn('expected_value', result)
        self.assertEqual(result['shap_values'].shape, (self.input_dim,))
    
    def test_force_plot_data(self):
        """Test force plot data preparation."""
        instance = np.random.randn(self.input_dim).astype(np.float32)
        result = self.explainer.force_plot_data(instance)
        
        self.assertIn('shap_values', result)
        self.assertIn('positive_contributions', result)
        self.assertIn('negative_contributions', result)


if __name__ == "__main__":
    unittest.main()
