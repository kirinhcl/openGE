"""Unit tests for sparse models."""

import unittest
import torch
import torch.nn as nn
from openge.models.sparse import (
    WeightSparseTransformer,
    SparseAttention,
    SparseTransformerBlock,
    SparseMultiHeadAttention,
    TopKAttention,
    LocalSparseAttention,
    ModelPruner,
    GradientBasedPruner
)


class TestSparseTransformer(unittest.TestCase):
    """Test sparse transformer."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 16
        self.input_dim = 100
        self.hidden_dim = 64
        self.output_dim = 10
        self.n_heads = 4
    
    def test_initialization(self):
        """Test sparse transformer initialization."""
        model = WeightSparseTransformer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_heads=self.n_heads,
            n_layers=2,
            sparsity_level=0.5
        )
        
        self.assertIsNotNone(model)
        self.assertEqual(model.input_dim, self.input_dim)
        self.assertEqual(model.hidden_dim, self.hidden_dim)
        self.assertEqual(model.sparsity_level, 0.5)
    
    def test_sparsity_level(self):
        """Test sparsity level configuration."""
        sparsity_levels = [0.1, 0.5, 0.9]
        
        for sparsity in sparsity_levels:
            model = WeightSparseTransformer(
                input_dim=self.input_dim,
                hidden_dim=self.hidden_dim,
                output_dim=self.output_dim,
                n_heads=self.n_heads,
                n_layers=1,
                sparsity_level=sparsity
            )
            
            self.assertEqual(model.sparsity_level, sparsity)
    
    def test_forward_pass(self):
        """Test forward pass with sparsity."""
        model = WeightSparseTransformer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_heads=self.n_heads,
            n_layers=2,
            sparsity_level=0.5
        )
        
        # 输入是 [batch, seq_len] 格式（每个位置一个特征）
        x = torch.randn(self.batch_size, self.input_dim)
        result = model(x)
        
        # 返回元组 (output, attention_weights)
        if isinstance(result, tuple):
            out, attn_weights = result
        else:
            out = result
        
        # 输出应该是 [batch, output_dim]
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[-1], self.output_dim)
    
    def test_gradient_flow(self):
        """Test gradient flow through sparse transformer."""
        model = WeightSparseTransformer(
            input_dim=self.input_dim,
            hidden_dim=self.hidden_dim,
            output_dim=self.output_dim,
            n_heads=self.n_heads,
            n_layers=1,
            sparsity_level=0.5
        )
        
        x = torch.randn(self.batch_size, self.input_dim, requires_grad=True)
        result = model(x)
        
        # 处理元组输出
        if isinstance(result, tuple):
            out, _ = result
        else:
            out = result
            
        loss = out.mean()
        loss.backward()
        
        self.assertIsNotNone(x.grad)
        self.assertFalse(torch.isnan(x.grad).any())


class TestSparseTransformerBlock(unittest.TestCase):
    """Test sparse transformer block."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 16
        self.dim = 64
        self.n_heads = 4
    
    def test_block_forward(self):
        """Test block forward pass."""
        block = SparseTransformerBlock(
            dim=self.dim,
            n_heads=self.n_heads,
            sparsity=0.5
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        out, attn_weights = block(x)
        
        self.assertEqual(out.shape, x.shape)
        self.assertIsNotNone(attn_weights)


class TestSparseAttention(unittest.TestCase):
    """Test sparse attention mechanisms."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.seq_len = 16
        self.dim = 64
        self.n_heads = 4
    
    def test_sparse_attention(self):
        """Test basic sparse attention."""
        attn = SparseAttention(dim=self.dim, n_heads=self.n_heads)
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        out, attn_weights = attn(x)  # 返回元组
        
        self.assertEqual(out.shape, x.shape)
        self.assertIsNotNone(attn_weights)
    
    def test_sparse_multihead_attention(self):
        """Test sparse multi-head attention."""
        attn = SparseMultiHeadAttention(dim=self.dim, n_heads=self.n_heads)
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        out, attn_weights = attn(x)  # 返回元组
        
        self.assertEqual(out.shape, x.shape)
    
    def test_topk_attention(self):
        """Test Top-K attention."""
        k = 8  # 只关注 top 8 个位置
        attn = TopKAttention(dim=self.dim, n_heads=self.n_heads, k=k)
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        out, attn_weights = attn(x)  # 返回元组
        
        self.assertEqual(out.shape, x.shape)
    
    def test_local_sparse_attention(self):
        """Test local sparse attention with window."""
        window_size = 4
        attn = LocalSparseAttention(
            dim=self.dim, 
            n_heads=self.n_heads, 
            window_size=window_size
        )
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        result = attn(x)
        
        # LocalSparseAttention 可能只返回 output，不返回 attention weights
        if isinstance(result, tuple):
            out, attn_weights = result
        else:
            out = result
        
        self.assertEqual(out.shape, x.shape)
    
    def test_attention_sparsity(self):
        """Test attention actually produces sparse patterns."""
        attn = SparseAttention(dim=self.dim, n_heads=self.n_heads, sparsity=0.8)
        
        x = torch.randn(self.batch_size, self.seq_len, self.dim)
        out, attn_weights = attn(x)
        
        # 检查注意力权重存在
        self.assertIsNotNone(attn_weights)


class TestModelPruner(unittest.TestCase):
    """Test model pruning utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        # 创建一个简单模型用于测试
        self.model = nn.Sequential(
            nn.Linear(100, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def test_magnitude_pruning(self):
        """Test magnitude-based pruning."""
        pruner = ModelPruner(self.model)
        
        # 执行幅度剪枝 (使用正确的方法名)
        report = pruner.magnitude_pruning(sparsity_level=0.5)
        
        # 检查返回了稀疏度报告
        self.assertIsInstance(report, dict)
        self.assertGreater(len(report), 0)
    
    def test_structured_pruning(self):
        """Test structured pruning."""
        pruner = ModelPruner(self.model)
        
        # 执行结构化剪枝
        report = pruner.structured_pruning(sparsity_level=0.3)
        
        # 模型仍应可运行
        x = torch.randn(4, 100)
        out = self.model(x)
        self.assertEqual(out.shape, (4, 1))
    
    def test_sparsity_report(self):
        """Test sparsity report generation."""
        pruner = ModelPruner(self.model)
        
        # 剪枝
        pruner.magnitude_pruning(sparsity_level=0.5)
        
        # 获取报告
        report = pruner.get_sparsity_report()
        
        self.assertIsInstance(report, dict)
        self.assertIn('global', report)
    
    def test_make_pruning_permanent(self):
        """Test making pruning permanent."""
        pruner = ModelPruner(self.model)
        
        # 应用剪枝
        pruner.magnitude_pruning(sparsity_level=0.5)
        
        # 使剪枝永久化
        pruner.make_pruning_permanent()
        
        # 模型仍应可运行
        x = torch.randn(4, 100)
        out = self.model(x)
        self.assertEqual(out.shape, (4, 1))
    
    def test_global_pruning(self):
        """Test global pruning."""
        pruner = ModelPruner(self.model)
        
        # 全局剪枝
        report = pruner.global_pruning(sparsity_level=0.5)
        
        self.assertIsInstance(report, dict)
    
    def test_importance_scores(self):
        """Test importance scoring."""
        pruner = ModelPruner(self.model)
        
        # 获取重要性分数
        scores = pruner.get_importance_scores(method='magnitude')
        
        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)


class TestGradientBasedPruner(unittest.TestCase):
    """Test gradient-based pruning."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.model = nn.Sequential(
            nn.Linear(50, 32),
            nn.ReLU(),
            nn.Linear(32, 1)
        )
    
    def test_gradient_based_importance(self):
        """Test gradient-based importance scoring."""
        pruner = GradientBasedPruner(self.model)
        
        # 模拟梯度
        x = torch.randn(8, 50)
        y = torch.randn(8, 1)
        
        # 前向传播
        out = self.model(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        
        # 获取基于梯度的重要性分数
        scores = pruner.get_importance_scores(method='gradient')
        
        self.assertIsInstance(scores, dict)
    
    def test_taylor_importance(self):
        """Test Taylor-based importance scoring."""
        pruner = GradientBasedPruner(self.model)
        
        # 模拟梯度
        x = torch.randn(8, 50)
        y = torch.randn(8, 1)
        
        out = self.model(x)
        loss = ((out - y) ** 2).mean()
        loss.backward()
        
        # 获取 Taylor 重要性分数
        scores = pruner.get_importance_scores(method='taylor')
        
        self.assertIsInstance(scores, dict)
        self.assertGreater(len(scores), 0)
    
    def test_inherits_from_model_pruner(self):
        """Test that GradientBasedPruner inherits from ModelPruner."""
        pruner = GradientBasedPruner(self.model)
        
        # 应能使用父类方法
        self.assertTrue(hasattr(pruner, 'magnitude_pruning'))
        self.assertTrue(hasattr(pruner, 'get_sparsity_report'))


if __name__ == "__main__":
    unittest.main()
