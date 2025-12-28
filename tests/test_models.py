"""Unit tests for model architectures."""

import unittest
import torch
from openge.models import (
    CNNEncoder, TransformerEncoder, MLPEncoder,
    ConcatFusion, AttentionFusion, GatingFusion,
    RegressionHead, ClassificationHead,
    GxEModel
)


class TestEncoders(unittest.TestCase):
    """Test encoder architectures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.input_dim = 100
        self.output_dim = 64
    
    def test_cnn_encoder(self):
        """Test CNN encoder."""
        encoder = CNNEncoder(
            input_dim=self.input_dim, 
            output_dim=self.output_dim,
            hidden_channels=[32, 64],
            kernel_sizes=[5, 3]
        )
        
        x = torch.randn(self.batch_size, self.input_dim)
        out = encoder(x)
        
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[1], self.output_dim)
    
    def test_transformer_encoder(self):
        """Test Transformer encoder."""
        encoder = TransformerEncoder(
            input_dim=self.input_dim,
            d_model=64,
            n_heads=4,
            n_layers=2,
            output_dim=self.output_dim
        )
        
        x = torch.randn(self.batch_size, self.input_dim)
        out = encoder(x)
        
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[1], self.output_dim)
    
    def test_mlp_encoder(self):
        """Test MLP encoder."""
        encoder = MLPEncoder(
            input_dim=self.input_dim,
            hidden_dims=[128, 64],
            output_dim=self.output_dim
        )
        
        x = torch.randn(self.batch_size, self.input_dim)
        out = encoder(x)
        
        self.assertEqual(out.shape, (self.batch_size, self.output_dim))
    
    def test_encoder_training_mode(self):
        """Test encoder in training vs eval mode."""
        encoder = MLPEncoder(
            input_dim=self.input_dim,
            hidden_dims=[64],
            output_dim=self.output_dim,
            dropout=0.5
        )
        
        x = torch.randn(self.batch_size, self.input_dim)
        
        # Training mode
        encoder.train()
        out1 = encoder(x)
        out2 = encoder(x)
        # 由于 dropout，两次输出可能不同
        
        # Eval mode
        encoder.eval()
        out3 = encoder(x)
        out4 = encoder(x)
        # 无 dropout，输出应相同
        self.assertTrue(torch.allclose(out3, out4))


class TestFusionLayers(unittest.TestCase):
    """Test fusion layer architectures."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.dim1 = 64
        self.dim2 = 32
    
    def test_concat_fusion(self):
        """Test concatenation fusion."""
        fusion = ConcatFusion(dim1=self.dim1, dim2=self.dim2)
        
        x1 = torch.randn(self.batch_size, self.dim1)
        x2 = torch.randn(self.batch_size, self.dim2)
        
        out = fusion(x1, x2)
        
        # 默认输出维度为 dim1 + dim2
        self.assertEqual(out.shape, (self.batch_size, self.dim1 + self.dim2))
    
    def test_concat_fusion_with_projection(self):
        """Test concatenation fusion with projection."""
        output_dim = 48
        fusion = ConcatFusion(dim1=self.dim1, dim2=self.dim2, output_dim=output_dim)
        
        x1 = torch.randn(self.batch_size, self.dim1)
        x2 = torch.randn(self.batch_size, self.dim2)
        
        out = fusion(x1, x2)
        
        self.assertEqual(out.shape, (self.batch_size, output_dim))
    
    def test_attention_fusion(self):
        """Test attention fusion."""
        fusion = AttentionFusion(dim1=self.dim1, dim2=self.dim2)
        
        x1 = torch.randn(self.batch_size, self.dim1)
        x2 = torch.randn(self.batch_size, self.dim2)
        
        result = fusion(x1, x2)
        
        # AttentionFusion 返回元组 (out, attention_weights)
        if isinstance(result, tuple):
            out, attn_weights = result
        else:
            out = result
        
        self.assertEqual(out.shape[0], self.batch_size)
    
    def test_gating_fusion(self):
        """Test gating fusion."""
        fusion = GatingFusion(dim1=self.dim1, dim2=self.dim2)
        
        x1 = torch.randn(self.batch_size, self.dim1)
        x2 = torch.randn(self.batch_size, self.dim2)
        
        out = fusion(x1, x2)
        
        self.assertEqual(out.shape[0], self.batch_size)


class TestPredictionHeads(unittest.TestCase):
    """Test prediction heads."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.input_dim = 64
    
    def test_regression_head(self):
        """Test regression head."""
        n_traits = 3
        head = RegressionHead(input_dim=self.input_dim, n_traits=n_traits)
        
        x = torch.randn(self.batch_size, self.input_dim)
        out = head(x)
        
        self.assertEqual(out.shape, (self.batch_size, n_traits))
    
    def test_regression_head_single_trait(self):
        """Test regression head with single trait."""
        head = RegressionHead(input_dim=self.input_dim, n_traits=1)
        
        x = torch.randn(self.batch_size, self.input_dim)
        out = head(x)
        
        self.assertEqual(out.shape, (self.batch_size, 1))
    
    def test_classification_head(self):
        """Test classification head."""
        n_classes = 5
        head = ClassificationHead(input_dim=self.input_dim, n_classes=n_classes)
        
        x = torch.randn(self.batch_size, self.input_dim)
        out = head(x)
        
        self.assertEqual(out.shape, (self.batch_size, n_classes))
        
        # 输出应为 logits，可以 softmax
        probs = torch.softmax(out, dim=-1)
        self.assertTrue(torch.allclose(probs.sum(dim=-1), torch.ones(self.batch_size)))


class TestGxEModel(unittest.TestCase):
    """Test full G×E model."""
    
    def setUp(self):
        """Set up test fixtures."""
        self.batch_size = 4
        self.n_markers = 100
        self.n_env_features = 20
        self.n_timesteps = 10
    
    def test_forward_pass(self):
        """Test model forward pass."""
        from openge.models import GxEModel
        from openge.models.encoders import MLPEncoder
        from openge.models.fusion import ConcatFusion
        from openge.models.heads import RegressionHead
        
        # 直接构建模型组件
        genetic_encoder = MLPEncoder(
            input_dim=self.n_markers,
            output_dim=64,
            dropout=0.1
        )
        
        env_encoder = MLPEncoder(
            input_dim=self.n_env_features,
            output_dim=64,
            dropout=0.1
        )
        
        fusion_layer = ConcatFusion(dim1=64, dim2=64, output_dim=64)
        prediction_head = RegressionHead(input_dim=64, n_traits=1)
        
        model = GxEModel(
            genetic_encoder=genetic_encoder,
            env_encoder=env_encoder,
            fusion_layer=fusion_layer,
            prediction_head=prediction_head
        )
        
        x_g = torch.randn(self.batch_size, self.n_markers)
        x_e = torch.randn(self.batch_size, self.n_env_features)
        
        result = model(x_g, x_e)
        
        # 处理可能的元组返回
        if isinstance(result, tuple):
            out = result[0]
        else:
            out = result
        
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[-1], 1)
    
    def test_output_shape(self):
        """Test output shape with multiple traits."""
        from openge.models import GxEModel
        from openge.models.encoders import MLPEncoder
        from openge.models.fusion import ConcatFusion
        from openge.models.heads import RegressionHead
        
        n_outputs = 3
        
        genetic_encoder = MLPEncoder(
            input_dim=self.n_markers,
            output_dim=64,
            dropout=0.1
        )
        
        env_encoder = MLPEncoder(
            input_dim=self.n_env_features,
            output_dim=64,
            dropout=0.1
        )
        
        fusion_layer = ConcatFusion(dim1=64, dim2=64, output_dim=64)
        prediction_head = RegressionHead(input_dim=64, n_traits=n_outputs)
        
        model = GxEModel(
            genetic_encoder=genetic_encoder,
            env_encoder=env_encoder,
            fusion_layer=fusion_layer,
            prediction_head=prediction_head
        )
        
        x_g = torch.randn(self.batch_size, self.n_markers)
        x_e = torch.randn(self.batch_size, self.n_env_features)
        
        result = model(x_g, x_e)
        
        # 处理可能的元组返回
        if isinstance(result, tuple):
            out = result[0]
        else:
            out = result
        
        self.assertEqual(out.shape[0], self.batch_size)
        self.assertEqual(out.shape[-1], n_outputs)
    
    def test_model_parameters(self):
        """Test model has trainable parameters."""
        from openge.models import GxEModelBuilder
        
        builder = GxEModelBuilder()
        builder.set_genetic_encoder('mlp', n_markers=self.n_markers)
        builder.set_env_encoder('mlp', n_features=self.n_env_features)
        builder.set_fusion('concat')
        builder.set_head('regression', n_outputs=1)
        
        model = builder.build()
        
        # 检查有可训练参数
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        
        self.assertGreater(total_params, 0)
        self.assertEqual(total_params, trainable_params)
    
    def test_cnn_encoder(self):
        """Test model with CNN encoder."""
        from openge.models import GxEModel
        from openge.models.encoders import CNNEncoder, MLPEncoder
        from openge.models.fusion import AttentionFusion
        from openge.models.heads import RegressionHead
        
        genetic_encoder = CNNEncoder(
            input_dim=self.n_markers,
            output_dim=64,
            hidden_channels=[32, 64],
            kernel_sizes=[5, 3]
        )
        
        env_encoder = MLPEncoder(
            input_dim=self.n_env_features,
            output_dim=64,
            dropout=0.1
        )
        
        fusion_layer = AttentionFusion(dim1=64, dim2=64, hidden_dim=64)
        prediction_head = RegressionHead(input_dim=64, n_traits=1)
        
        model = GxEModel(
            genetic_encoder=genetic_encoder,
            env_encoder=env_encoder,
            fusion_layer=fusion_layer,
            prediction_head=prediction_head
        )
        
        x_g = torch.randn(self.batch_size, self.n_markers)
        x_e = torch.randn(self.batch_size, self.n_env_features)
        
        result = model(x_g, x_e)
        
        # 处理可能的元组返回
        if isinstance(result, tuple):
            out = result[0]
        else:
            out = result
        
        self.assertEqual(out.shape[0], self.batch_size)


if __name__ == "__main__":
    unittest.main()
