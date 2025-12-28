"""Encoder architectures: CNN, Transformer, MLP for genetic and environmental encoding."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, List, Tuple, Union


# =============================================================================
# 位置编码方法 (Position Encoding Methods)
# =============================================================================

class SinusoidalPositionalEncoding(nn.Module):
    """
    正弦位置编码 (Sinusoidal Positional Encoding)
    
    来自 "Attention is All You Need" 论文的原始位置编码方法。
    使用不同频率的正弦和余弦函数为每个位置生成唯一的编码。
    
    优点:
        - 可以泛化到训练时未见过的序列长度
        - 相对位置信息可以通过线性变换表达
        - 不需要学习参数
    
    缺点:
        - 固定的编码模式，无法适应特定任务
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 模型的嵌入维度
            max_len: 支持的最大序列长度
            dropout: Dropout 率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        
        # 创建位置编码矩阵
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)  # 偶数维度用 sin
        pe[:, 1::2] = torch.cos(position * div_term)  # 奇数维度用 cos
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        
        self.register_buffer('pe', pe)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将位置编码加到输入上。
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)


class LearnablePositionalEncoding(nn.Module):
    """
    可学习位置编码 (Learnable Positional Encoding)
    
    为每个位置学习一个独立的嵌入向量，类似于 BERT 的位置编码。
    
    优点:
        - 可以学习任务特定的位置表示
        - 简单直观
    
    缺点:
        - 无法泛化到训练时未见过的序列长度
        - 增加了模型参数
    """
    
    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            d_model: 模型的嵌入维度
            max_len: 支持的最大序列长度
            dropout: Dropout 率
        """
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        self.d_model = d_model
        self.max_len = max_len
        
        # 可学习的位置嵌入
        self.pe = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        将可学习的位置编码加到输入上。
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            添加位置编码后的张量 [batch_size, seq_len, d_model]
        """
        seq_len = x.size(1)
        if seq_len > self.max_len:
            raise ValueError(f"序列长度 {seq_len} 超过最大长度 {self.max_len}")
        
        x = x + self.pe[:, :seq_len, :]
        return self.dropout(x)


class RotaryPositionalEncoding(nn.Module):
    """
    旋转位置编码 (Rotary Position Embedding, RoPE)
    
    来自 "RoFormer: Enhanced Transformer with Rotary Position Embedding" 论文。
    通过旋转操作将位置信息编码到查询和键向量中。
    
    核心思想:
        - 将位置编码实现为对查询/键向量的旋转
        - 内积自动包含相对位置信息: q_m · k_n 只依赖于 (m-n)
    
    优点:
        - 相对位置信息自然地编码在注意力中
        - 可以泛化到更长的序列
        - 与自注意力机制完美兼容
    
    缺点:
        - 实现略复杂
        - 需要修改注意力计算
    
    数学原理:
        对于位置 m 的向量 x = [x1, x2, x3, x4, ...]:
        RoPE(x, m) = [x1*cos(mθ1) - x2*sin(mθ1),
                      x1*sin(mθ1) + x2*cos(mθ1),
                      x3*cos(mθ2) - x4*sin(mθ2),
                      x3*sin(mθ2) + x4*cos(mθ2), ...]
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 5000, 
        base: float = 10000.0,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型的嵌入维度 (必须是偶数)
            max_len: 支持的最大序列长度
            base: 频率基数 (默认 10000)
            dropout: Dropout 率
        """
        super().__init__()
        
        assert d_model % 2 == 0, "d_model 必须是偶数"
        
        self.d_model = d_model
        self.max_len = max_len
        self.base = base
        self.dropout = nn.Dropout(p=dropout)
        
        # 预计算频率
        # θ_i = base^(-2i/d) for i = 0, 1, ..., d/2-1
        inv_freq = 1.0 / (base ** (torch.arange(0, d_model, 2).float() / d_model))
        self.register_buffer('inv_freq', inv_freq)
        
        # 预计算位置编码 (cos 和 sin)
        self._build_cache(max_len)
    
    def _build_cache(self, seq_len: int):
        """预计算 cos 和 sin 缓存。"""
        positions = torch.arange(seq_len, dtype=self.inv_freq.dtype, device=self.inv_freq.device)
        
        # [seq_len, d_model/2]
        freqs = torch.einsum('i,j->ij', positions, self.inv_freq)
        
        # [seq_len, d_model] - 每个频率复制两次
        emb = torch.cat([freqs, freqs], dim=-1)
        
        # [1, seq_len, d_model]
        cos_cached = emb.cos().unsqueeze(0)
        sin_cached = emb.sin().unsqueeze(0)
        
        self.register_buffer('cos_cached', cos_cached, persistent=False)
        self.register_buffer('sin_cached', sin_cached, persistent=False)
    
    def _rotate_half(self, x: torch.Tensor) -> torch.Tensor:
        """
        将张量的后半部分旋转到前面并取负。
        [x1, x2, x3, x4] -> [-x3, -x4, x1, x2]
        
        但实际上我们需要的是:
        [x1, x2, x3, x4] -> [-x2, x1, -x4, x3]
        """
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat([-x2, x1], dim=-1)
    
    def forward(self, x: torch.Tensor, seq_len: Optional[int] = None) -> torch.Tensor:
        """
        应用旋转位置编码。
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            seq_len: 序列长度 (可选，默认从 x 推断)
            
        Returns:
            应用 RoPE 后的张量 [batch_size, seq_len, d_model]
        """
        if seq_len is None:
            seq_len = x.size(1)
        
        # 如果需要，扩展缓存
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        
        cos = self.cos_cached[:, :seq_len, :]
        sin = self.sin_cached[:, :seq_len, :]
        
        # 应用旋转: x * cos + rotate_half(x) * sin
        x = (x * cos) + (self._rotate_half(x) * sin)
        
        return self.dropout(x)
    
    def apply_rotary_pos_emb(
        self, 
        q: torch.Tensor, 
        k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        将 RoPE 应用于查询和键张量。
        
        这是在注意力计算中使用 RoPE 的推荐方法。
        
        Args:
            q: 查询张量 [batch, n_heads, seq_len, head_dim]
            k: 键张量 [batch, n_heads, seq_len, head_dim]
            
        Returns:
            应用 RoPE 后的 (q, k) 元组
        """
        seq_len = q.size(2)
        
        if seq_len > self.cos_cached.size(1):
            self._build_cache(seq_len)
        
        # [1, 1, seq_len, d_model]
        cos = self.cos_cached[:, :seq_len, :].unsqueeze(1)
        sin = self.sin_cached[:, :seq_len, :].unsqueeze(1)
        
        q_embed = (q * cos) + (self._rotate_half(q) * sin)
        k_embed = (k * cos) + (self._rotate_half(k) * sin)
        
        return q_embed, k_embed


class ALiBiPositionalEncoding(nn.Module):
    """
    线性偏置注意力 (Attention with Linear Biases, ALiBi)
    
    来自 "Train Short, Test Long" 论文。
    不直接修改输入嵌入，而是在注意力分数上添加线性位置偏置。
    
    核心思想:
        - 对注意力分数添加偏置: attention_scores - m * |i - j|
        - m 是每个注意力头的斜率 (slope)
        - 更远的位置受到更大的惩罚
    
    优点:
        - 极好的长度泛化能力
        - 无需位置嵌入参数
        - 训练短序列，推理长序列
    
    缺点:
        - 只能用于注意力机制，不能加到嵌入上
        - 需要修改注意力计算
    """
    
    def __init__(self, n_heads: int, max_len: int = 5000, dropout: float = 0.1):
        """
        Args:
            n_heads: 注意力头数
            max_len: 支持的最大序列长度
            dropout: Dropout 率 (用于兼容接口，ALiBi 本身不使用)
        """
        super().__init__()
        
        self.n_heads = n_heads
        self.max_len = max_len
        self.dropout = nn.Dropout(p=dropout)
        
        # 计算每个头的斜率
        # 斜率按几何序列分布: 2^(-8/n_heads), 2^(-16/n_heads), ...
        slopes = self._get_slopes(n_heads)
        self.register_buffer('slopes', slopes)
        
        # 预计算位置偏置矩阵
        self._build_alibi_bias(max_len)
    
    def _get_slopes(self, n_heads: int) -> torch.Tensor:
        """
        计算 ALiBi 斜率。
        
        对于 n_heads 个头，斜率为:
        2^(-8/n), 2^(-8*2/n), ..., 2^(-8)
        """
        def get_slopes_power_of_2(n: int) -> List[float]:
            start = 2 ** (-(2 ** -(math.log2(n) - 3)))
            ratio = start
            return [start * (ratio ** i) for i in range(n)]
        
        if math.log2(n_heads).is_integer():
            slopes = get_slopes_power_of_2(n_heads)
        else:
            # 对于非 2 的幂次，使用插值
            closest_power_of_2 = 2 ** math.floor(math.log2(n_heads))
            slopes = get_slopes_power_of_2(closest_power_of_2)
            extra_slopes = get_slopes_power_of_2(2 * closest_power_of_2)
            slopes = slopes + extra_slopes[0::2][:n_heads - closest_power_of_2]
        
        return torch.tensor(slopes, dtype=torch.float32).view(1, n_heads, 1, 1)
    
    def _build_alibi_bias(self, seq_len: int):
        """预计算 ALiBi 偏置矩阵。"""
        # 创建相对位置矩阵 [seq_len, seq_len]
        # bias[i, j] = -|i - j|
        positions = torch.arange(seq_len)
        relative_positions = positions.unsqueeze(0) - positions.unsqueeze(1)
        relative_positions = -torch.abs(relative_positions).float()
        
        # [1, 1, seq_len, seq_len]
        self.register_buffer(
            'alibi_bias', 
            relative_positions.unsqueeze(0).unsqueeze(0),
            persistent=False
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        ALiBi 不修改输入嵌入，直接返回输入。
        位置信息通过 get_bias() 在注意力中添加。
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            原样返回输入张量
        """
        return self.dropout(x)
    
    def get_bias(self, seq_len: int) -> torch.Tensor:
        """
        获取 ALiBi 偏置，用于添加到注意力分数上。
        
        Args:
            seq_len: 序列长度
            
        Returns:
            偏置张量 [1, n_heads, seq_len, seq_len]
            
        使用方法:
            attention_scores = q @ k.T / sqrt(d)
            attention_scores = attention_scores + alibi.get_bias(seq_len)
        """
        if seq_len > self.alibi_bias.size(2):
            self._build_alibi_bias(seq_len)
        
        bias = self.alibi_bias[:, :, :seq_len, :seq_len]
        return bias * self.slopes


class RelativePositionalEncoding(nn.Module):
    """
    相对位置编码 (Relative Positional Encoding)
    
    来自 "Self-Attention with Relative Position Representations" 论文。
    学习相对位置的嵌入，而不是绝对位置。
    
    优点:
        - 直接建模相对位置关系
        - 更好的泛化能力
    
    缺点:
        - 需要修改注意力计算
        - 参数量与最大相对距离相关
    """
    
    def __init__(
        self, 
        d_model: int, 
        max_len: int = 5000, 
        max_relative_position: int = 128,
        dropout: float = 0.1
    ):
        """
        Args:
            d_model: 模型的嵌入维度
            max_len: 支持的最大序列长度 (用于兼容接口)
            max_relative_position: 最大相对位置距离
            dropout: Dropout 率
        """
        super().__init__()
        
        self.d_model = d_model
        self.max_relative_position = max_relative_position
        self.dropout = nn.Dropout(p=dropout)
        
        # 相对位置嵌入: [-max_rel, ..., -1, 0, 1, ..., max_rel]
        # 共 2 * max_relative_position + 1 个
        num_embeddings = 2 * max_relative_position + 1
        self.relative_embeddings = nn.Embedding(num_embeddings, d_model)
        
        nn.init.xavier_uniform_(self.relative_embeddings.weight)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        相对位置编码不直接加到输入上，返回原始输入。
        使用 get_relative_embeddings() 获取相对位置嵌入。
        
        Args:
            x: 输入张量 [batch_size, seq_len, d_model]
            
        Returns:
            原样返回输入张量
        """
        return self.dropout(x)
    
    def get_relative_embeddings(self, seq_len: int) -> torch.Tensor:
        """
        获取相对位置嵌入矩阵。
        
        Args:
            seq_len: 序列长度
            
        Returns:
            相对位置嵌入 [seq_len, seq_len, d_model]
        """
        # 创建相对位置索引
        range_vec = torch.arange(seq_len, device=self.relative_embeddings.weight.device)
        relative_positions = range_vec.unsqueeze(0) - range_vec.unsqueeze(1)
        
        # 裁剪到 [-max_relative_position, max_relative_position]
        relative_positions = torch.clamp(
            relative_positions, 
            -self.max_relative_position, 
            self.max_relative_position
        )
        
        # 转换为非负索引
        relative_positions = relative_positions + self.max_relative_position
        
        return self.relative_embeddings(relative_positions)


# =============================================================================
# 统一位置编码接口
# =============================================================================

def create_positional_encoding(
    encoding_type: str,
    d_model: int,
    max_len: int = 5000,
    dropout: float = 0.1,
    n_heads: int = 8,
    **kwargs
) -> nn.Module:
    """
    创建指定类型的位置编码。
    
    Args:
        encoding_type: 位置编码类型
            - 'sinusoidal': 正弦位置编码 (原始 Transformer)
            - 'learnable': 可学习位置编码 (BERT 风格)
            - 'rope': 旋转位置编码 (RoFormer)
            - 'alibi': 线性偏置注意力 (ALiBi)
            - 'relative': 相对位置编码
        d_model: 模型维度
        max_len: 最大序列长度
        dropout: Dropout 率
        n_heads: 注意力头数 (仅 ALiBi 需要)
        **kwargs: 传递给具体实现的额外参数
        
    Returns:
        位置编码模块
        
    Example:
        >>> pe = create_positional_encoding('rope', d_model=256, max_len=2048)
        >>> x = torch.randn(2, 100, 256)
        >>> x_with_pos = pe(x)
    """
    encoding_type = encoding_type.lower()
    
    if encoding_type in ['sinusoidal', 'sin', 'absolute']:
        return SinusoidalPositionalEncoding(d_model, max_len, dropout)
    
    elif encoding_type in ['learnable', 'learned', 'bert']:
        return LearnablePositionalEncoding(d_model, max_len, dropout)
    
    elif encoding_type in ['rope', 'rotary']:
        base = kwargs.get('base', 10000.0)
        return RotaryPositionalEncoding(d_model, max_len, base, dropout)
    
    elif encoding_type in ['alibi', 'linear_bias']:
        return ALiBiPositionalEncoding(n_heads, max_len, dropout)
    
    elif encoding_type in ['relative', 'rel']:
        max_relative_position = kwargs.get('max_relative_position', 128)
        return RelativePositionalEncoding(d_model, max_len, max_relative_position, dropout)
    
    else:
        raise ValueError(
            f"未知的位置编码类型: {encoding_type}. "
            f"支持: sinusoidal, learnable, rope, alibi, relative"
        )


# 向后兼容：保留原始名称
PositionalEncoding = SinusoidalPositionalEncoding


class CNNEncoder(nn.Module):
    """
    CNN encoder for sequential genetic data.
    
    Suitable for encoding SNP markers with local patterns and correlations
    (e.g., linkage disequilibrium between nearby SNPs).
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        hidden_channels: List[int] = None,
        kernel_sizes: List[int] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ):
        """
        Initialize CNN encoder.
        
        Args:
            input_dim: Number of input features (e.g., number of SNP markers)
            output_dim: Output embedding dimension
            hidden_channels: List of channel sizes for conv layers
            kernel_sizes: List of kernel sizes for conv layers
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
        """
        super().__init__()
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        if kernel_sizes is None:
            kernel_sizes = [7, 5, 3]
        
        assert len(hidden_channels) == len(kernel_sizes), \
            "hidden_channels and kernel_sizes must have same length"
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.use_batch_norm = use_batch_norm
        
        # Build convolutional layers
        layers = []
        in_channels = 1  # Start with 1 channel (SNP values as single channel)
        
        for i, (out_channels, kernel_size) in enumerate(zip(hidden_channels, kernel_sizes)):
            layers.append(
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size // 2)
            )
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            layers.append(nn.MaxPool1d(2))
            layers.append(nn.Dropout(dropout))
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Calculate output size after convolutions
        with torch.no_grad():
            dummy_input = torch.zeros(1, 1, input_dim)
            conv_output = self.conv_layers(dummy_input)
            self.conv_output_size = conv_output.view(1, -1).size(1)
        
        # Final projection to output_dim
        self.fc = nn.Sequential(
            nn.Linear(self.conv_output_size, output_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, n_markers]
            
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        # Add channel dimension: [batch, n_markers] -> [batch, 1, n_markers]
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Apply convolutions
        x = self.conv_layers(x)
        
        # Flatten and project
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x


class TransformerEncoder(nn.Module):
    """
    Transformer encoder for capturing long-range dependencies.
    
    Ideal for modeling complex interactions across the genome or 
    temporal patterns in environmental data.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int, 
        d_model: int = 256,
        n_heads: int = 8, 
        n_layers: int = 4,
        dim_feedforward: int = 1024,
        dropout: float = 0.1,
        max_seq_len: int = 5000,
        pool_method: str = 'mean'
    ):
        """
        Initialize Transformer encoder.
        
        Args:
            input_dim: Input feature dimension
            output_dim: Output dimension
            d_model: Model dimension (embedding size)
            n_heads: Number of attention heads
            n_layers: Number of transformer blocks
            dim_feedforward: Feedforward dimension
            dropout: Dropout rate
            max_seq_len: Maximum sequence length for positional encoding
            pool_method: Pooling method ('mean', 'cls', 'max')
        """
        super().__init__()
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.d_model = d_model
        self.pool_method = pool_method
        
        # Input projection
        self.input_projection = nn.Linear(1, d_model)
        
        # CLS token for classification-style pooling
        if pool_method == 'cls':
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_model))
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(d_model, max_seq_len, dropout)
        
        # Transformer encoder layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(d_model)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(d_model, output_dim * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim * 2, output_dim)
        )
    
    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, seq_len] or [batch_size, seq_len, input_dim]
            attention_mask: Optional mask for padding
            
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        # Handle different input shapes
        if x.dim() == 2:
            # [batch, seq_len] -> [batch, seq_len, 1]
            x = x.unsqueeze(-1)
        
        batch_size, seq_len, _ = x.shape
        
        # Project to model dimension
        x = self.input_projection(x)  # [batch, seq_len, d_model]
        
        # Add CLS token if using cls pooling
        if self.pool_method == 'cls':
            cls_tokens = self.cls_token.expand(batch_size, -1, -1)
            x = torch.cat([cls_tokens, x], dim=1)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply transformer
        x = self.transformer(x, src_key_padding_mask=attention_mask)
        x = self.layer_norm(x)
        
        # Pool sequence
        if self.pool_method == 'cls':
            x = x[:, 0, :]  # Use CLS token
        elif self.pool_method == 'mean':
            if attention_mask is not None:
                mask = ~attention_mask.unsqueeze(-1)
                x = (x * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                x = x.mean(dim=1)
        elif self.pool_method == 'max':
            x = x.max(dim=1)[0]
        
        # Output projection
        x = self.output_projection(x)
        
        return x
    
    def get_attention_weights(self, x: torch.Tensor) -> List[torch.Tensor]:
        """
        Get attention weights from all layers.
        
        Args:
            x: Input tensor
            
        Returns:
            List of attention weight tensors
        """
        # This requires modifying the forward pass to return attention weights
        # For now, returns empty list
        return []


class MLPEncoder(nn.Module):
    """
    Multi-layer perceptron encoder for tabular data.
    
    Suitable for static environmental covariates (EC) or soil data,
    where spatial/temporal structure is already encoded in features.
    """
    
    def __init__(
        self, 
        input_dim: int, 
        output_dim: int,
        hidden_dims: List[int] = None, 
        dropout: float = 0.1,
        use_batch_norm: bool = True,
        activation: str = 'relu'
    ):
        """
        Initialize MLP encoder.
        
        Args:
            input_dim: Input feature dimension
            hidden_dims: List of hidden layer dimensions
            output_dim: Output dimension
            dropout: Dropout rate
            use_batch_norm: Whether to use batch normalization
            activation: Activation function ('relu', 'gelu', 'leaky_relu')
        """
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [512, 256]
        
        self.input_dim = input_dim
        self.output_dim = output_dim
        
        # Activation function
        if activation == 'relu':
            act_fn = nn.ReLU(inplace=True)
        elif activation == 'gelu':
            act_fn = nn.GELU()
        elif activation == 'leaky_relu':
            act_fn = nn.LeakyReLU(0.1, inplace=True)
        else:
            act_fn = nn.ReLU(inplace=True)
        
        # Build MLP layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_dim))
            layers.append(act_fn)
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        # Final layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.mlp = nn.Sequential(*layers)
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, input_dim]
            
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        return self.mlp(x)


class TemporalEncoder(nn.Module):
    """
    Temporal encoder for 3D weather data.
    
    Combines CNN for local temporal patterns with attention for 
    long-range dependencies across the growing season.
    """
    
    def __init__(
        self,
        n_features: int,
        n_timesteps: int,
        output_dim: int,
        hidden_dim: int = 128,
        n_heads: int = 4,
        n_layers: int = 2,
        dropout: float = 0.1
    ):
        """
        Initialize temporal encoder.
        
        Args:
            n_features: Number of weather features per timestep
            n_timesteps: Number of timesteps (days)
            output_dim: Output dimension
            hidden_dim: Hidden dimension
            n_heads: Number of attention heads
            n_layers: Number of transformer layers
            dropout: Dropout rate
        """
        super().__init__()
        
        self.n_features = n_features
        self.n_timesteps = n_timesteps
        self.output_dim = output_dim
        
        # Temporal convolution to capture local patterns
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(n_features, hidden_dim, kernel_size=7, padding=3),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Conv1d(hidden_dim, hidden_dim, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(inplace=True),
        )
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_dim, max_len=n_timesteps + 100, dropout=dropout)
        
        # Temporal attention
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=n_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.temporal_attention = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # Output projection
        self.output_projection = nn.Sequential(
            nn.Linear(hidden_dim, output_dim),
            nn.LayerNorm(output_dim)
        )
        
        # Aggregation weights (learned)
        self.aggregation_weights = nn.Parameter(torch.ones(1, n_timesteps, 1) / n_timesteps)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor [batch_size, n_timesteps, n_features]
            
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        # Transpose for Conv1d: [batch, n_features, n_timesteps]
        x = x.transpose(1, 2)
        
        # Apply temporal convolution
        x = self.temporal_conv(x)
        
        # Transpose back: [batch, n_timesteps, hidden_dim]
        x = x.transpose(1, 2)
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply temporal attention
        x = self.temporal_attention(x)
        
        # Weighted aggregation across time
        weights = F.softmax(self.aggregation_weights, dim=1)
        x = (x * weights).sum(dim=1)  # [batch, hidden_dim]
        
        # Output projection
        x = self.output_projection(x)
        
        return x


class GeneticEncoder(nn.Module):
    """
    Specialized encoder for genetic (SNP) data.
    
    Combines embedding-based and CNN-based encoding for SNP markers.
    """
    
    def __init__(
        self,
        n_markers: int,
        output_dim: int,
        embedding_dim: int = 8,
        hidden_channels: List[int] = None,
        dropout: float = 0.1,
        encoding_type: str = 'cnn'
    ):
        """
        Initialize genetic encoder.
        
        Args:
            n_markers: Number of SNP markers
            output_dim: Output dimension
            embedding_dim: Embedding dimension for each marker
            hidden_channels: CNN hidden channels
            dropout: Dropout rate
            encoding_type: 'embedding', 'cnn', or 'hybrid'
        """
        super().__init__()
        
        self.n_markers = n_markers
        self.output_dim = output_dim
        self.encoding_type = encoding_type
        
        if hidden_channels is None:
            hidden_channels = [64, 128, 256]
        
        if encoding_type in ['embedding', 'hybrid']:
            # Learnable embedding for each marker position
            self.marker_embeddings = nn.Embedding(n_markers, embedding_dim)
            # Value projection (SNP value 0, 0.5, 1 -> embedding)
            self.value_projection = nn.Linear(1, embedding_dim)
            
        if encoding_type in ['cnn', 'hybrid']:
            self.cnn_encoder = CNNEncoder(
                input_dim=n_markers,
                output_dim=output_dim,
                hidden_channels=hidden_channels,
                dropout=dropout
            )
        
        if encoding_type == 'embedding':
            self.output_fc = nn.Sequential(
                nn.Linear(n_markers * embedding_dim * 2, output_dim * 2),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
                nn.Linear(output_dim * 2, output_dim)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: SNP values [batch_size, n_markers] with values in {0, 0.5, 1}
            
        Returns:
            Encoded representation [batch_size, output_dim]
        """
        batch_size = x.size(0)
        
        if self.encoding_type == 'cnn':
            return self.cnn_encoder(x)
        
        # Get marker position embeddings
        marker_indices = torch.arange(self.n_markers, device=x.device)
        marker_emb = self.marker_embeddings(marker_indices)  # [n_markers, emb_dim]
        marker_emb = marker_emb.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, n_markers, emb_dim]
        
        # Get value embeddings
        value_emb = self.value_projection(x.unsqueeze(-1))  # [batch, n_markers, emb_dim]
        
        # Combine position and value embeddings
        combined = torch.cat([marker_emb, value_emb], dim=-1)  # [batch, n_markers, emb_dim*2]
        
        if self.encoding_type == 'embedding':
            x = combined.view(batch_size, -1)
            return self.output_fc(x)
        
        # Hybrid: average of cnn and embedding
        cnn_out = self.cnn_encoder(x)
        emb_out = self.output_fc(combined.view(batch_size, -1))
        return (cnn_out + emb_out) / 2


# =============================================================================
# 组合编码器 (Composite Encoders)
# =============================================================================

class StackedEncoder(nn.Module):
    """
    串行叠加编码器 (Stacked Encoder)
    
    将多个编码器串行连接，前一个的输出作为后一个的输入。
    
    使用场景:
        - CNN 提取局部特征 → Transformer 捕获全局依赖
        - MLP 降维 → Transformer 建模交互
    
    Example:
        >>> # CNN + Transformer 串行叠加
        >>> encoder = StackedEncoder([
        ...     CNNEncoder(input_dim=2000, output_dim=512),
        ...     MLPEncoder(input_dim=512, output_dim=256),
        ... ])
        >>> x = torch.randn(32, 2000)
        >>> out = encoder(x)  # [32, 256]
    """
    
    def __init__(
        self, 
        encoders: List[nn.Module],
        residual: bool = False,
        dropout: float = 0.1
    ):
        """
        Args:
            encoders: 编码器列表，按顺序串行执行
            residual: 是否使用残差连接 (需要维度匹配)
            dropout: 层间 dropout 率
        """
        super().__init__()
        
        self.encoders = nn.ModuleList(encoders)
        self.residual = residual
        self.dropout = nn.Dropout(dropout)
        
        # 获取输入输出维度
        self.input_dim = getattr(encoders[0], 'input_dim', None)
        self.output_dim = getattr(encoders[-1], 'output_dim', None)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        串行执行所有编码器。
        
        Args:
            x: 输入张量
            
        Returns:
            最终编码结果
        """
        for i, encoder in enumerate(self.encoders):
            identity = x
            x = encoder(x)
            x = self.dropout(x)
            
            # 残差连接 (如果维度匹配)
            if self.residual and identity.shape == x.shape:
                x = x + identity
        
        return x


class ParallelEncoder(nn.Module):
    """
    并行编码器 (Parallel Encoder)
    
    将多个编码器并行执行，然后融合结果。
    
    使用场景:
        - 同时使用 CNN 和 Transformer 编码同一输入
        - 多视角特征提取
    
    融合方式:
        - 'concat': 拼接所有输出
        - 'mean': 平均所有输出 (需要相同维度)
        - 'sum': 求和所有输出 (需要相同维度)
        - 'attention': 注意力加权融合
    
    Example:
        >>> encoder = ParallelEncoder(
        ...     encoders=[
        ...         CNNEncoder(input_dim=2000, output_dim=256),
        ...         GeneticEncoder(n_markers=2000, output_dim=256, encoding_type='embedding'),
        ...     ],
        ...     fusion='attention'
        ... )
        >>> x = torch.randn(32, 2000)
        >>> out = encoder(x)  # [32, 256]
    """
    
    def __init__(
        self,
        encoders: List[nn.Module],
        fusion: str = 'concat',
        output_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            encoders: 并行执行的编码器列表
            fusion: 融合方式 ('concat', 'mean', 'sum', 'attention')
            output_dim: 输出维度 (仅 concat 模式需要)
            dropout: Dropout 率
        """
        super().__init__()
        
        self.encoders = nn.ModuleList(encoders)
        self.fusion = fusion
        self.dropout = nn.Dropout(dropout)
        
        # 计算各编码器输出维度
        self.encoder_output_dims = [
            getattr(enc, 'output_dim', 256) for enc in encoders
        ]
        
        if fusion == 'concat':
            total_dim = sum(self.encoder_output_dims)
            self.output_dim = output_dim or total_dim
            if output_dim and output_dim != total_dim:
                self.projection = nn.Linear(total_dim, output_dim)
            else:
                self.projection = None
                
        elif fusion == 'attention':
            # 注意力融合
            assert len(set(self.encoder_output_dims)) == 1, \
                "attention 融合需要所有编码器输出维度相同"
            self.output_dim = self.encoder_output_dims[0]
            self.attention_weights = nn.Parameter(
                torch.ones(len(encoders)) / len(encoders)
            )
            
        else:  # mean, sum
            assert len(set(self.encoder_output_dims)) == 1, \
                f"{fusion} 融合需要所有编码器输出维度相同"
            self.output_dim = self.encoder_output_dims[0]
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        并行执行所有编码器并融合结果。
        
        Args:
            x: 输入张量
            
        Returns:
            融合后的编码结果
        """
        # 并行编码
        outputs = [encoder(x) for encoder in self.encoders]
        
        # 融合
        if self.fusion == 'concat':
            out = torch.cat(outputs, dim=-1)
            if self.projection is not None:
                out = self.projection(out)
                
        elif self.fusion == 'mean':
            out = torch.stack(outputs, dim=0).mean(dim=0)
            
        elif self.fusion == 'sum':
            out = torch.stack(outputs, dim=0).sum(dim=0)
            
        elif self.fusion == 'attention':
            # 可学习的注意力权重
            weights = F.softmax(self.attention_weights, dim=0)
            stacked = torch.stack(outputs, dim=0)  # [n_encoders, batch, dim]
            out = (stacked * weights.view(-1, 1, 1)).sum(dim=0)
        
        return self.dropout(out)
    
    def get_attention_weights(self) -> torch.Tensor:
        """获取各编码器的注意力权重 (仅 attention 融合模式)。"""
        if self.fusion == 'attention':
            return F.softmax(self.attention_weights, dim=0)
        return None


class HierarchicalEncoder(nn.Module):
    """
    层次化编码器 (Hierarchical Encoder)
    
    先用局部编码器处理序列片段，再用全局编码器处理整体。
    
    适用场景:
        - 超长基因序列: 先分块处理，再整合
        - 多尺度特征提取
    
    Example:
        >>> # 处理 10000 个 SNP: 分成 10 个块，每块 1000
        >>> encoder = HierarchicalEncoder(
        ...     local_encoder=CNNEncoder(input_dim=1000, output_dim=128),
        ...     global_encoder=TransformerEncoder(input_dim=128, output_dim=256),
        ...     chunk_size=1000
        ... )
        >>> x = torch.randn(32, 10000)
        >>> out = encoder(x)  # [32, 256]
    """
    
    def __init__(
        self,
        local_encoder: nn.Module,
        global_encoder: nn.Module,
        chunk_size: int,
        overlap: int = 0,
        pool_chunks: str = 'none'
    ):
        """
        Args:
            local_encoder: 局部编码器，处理每个块
            global_encoder: 全局编码器，处理所有块的表示
            chunk_size: 每个块的大小
            overlap: 块之间的重叠大小
            pool_chunks: 块表示的池化方式 ('none', 'mean', 'max')
        """
        super().__init__()
        
        self.local_encoder = local_encoder
        self.global_encoder = global_encoder
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.pool_chunks = pool_chunks
        
        self.output_dim = getattr(global_encoder, 'output_dim', 256)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        层次化编码。
        
        Args:
            x: 输入张量 [batch_size, seq_len]
            
        Returns:
            编码结果 [batch_size, output_dim]
        """
        batch_size, seq_len = x.shape[:2]
        
        # 分块
        step = self.chunk_size - self.overlap
        chunks = []
        for start in range(0, seq_len, step):
            end = min(start + self.chunk_size, seq_len)
            chunk = x[:, start:end]
            
            # 如果块太小，填充
            if chunk.size(1) < self.chunk_size:
                padding = torch.zeros(
                    batch_size, self.chunk_size - chunk.size(1),
                    *chunk.shape[2:], device=x.device, dtype=x.dtype
                )
                chunk = torch.cat([chunk, padding], dim=1)
            
            chunks.append(chunk)
        
        # 局部编码
        chunk_embeddings = []
        for chunk in chunks:
            emb = self.local_encoder(chunk)
            chunk_embeddings.append(emb)
        
        # [batch, n_chunks, local_dim]
        chunk_embeddings = torch.stack(chunk_embeddings, dim=1)
        
        # 块级池化 (可选)
        if self.pool_chunks == 'mean':
            return chunk_embeddings.mean(dim=1)
        elif self.pool_chunks == 'max':
            return chunk_embeddings.max(dim=1)[0]
        
        # 全局编码
        # 需要调整 global_encoder 的输入格式
        if hasattr(self.global_encoder, 'input_projection'):
            # Transformer: 期望 [batch, seq, dim]
            out = self.global_encoder(chunk_embeddings)
        else:
            # MLP: 展平后处理
            out = self.global_encoder(chunk_embeddings.view(batch_size, -1))
        
        return out


class MultiInputEncoder(nn.Module):
    """
    多输入编码器 (Multi-Input Encoder)
    
    为不同类型的输入使用不同的编码器，然后融合。
    
    适用场景:
        - 基因型 + 环境数据
        - 多模态输入
    
    Example:
        >>> encoder = MultiInputEncoder(
        ...     encoders={
        ...         'genetic': CNNEncoder(input_dim=2000, output_dim=256),
        ...         'weather': TemporalEncoder(n_features=16, n_timesteps=366, output_dim=256),
        ...         'soil': MLPEncoder(input_dim=20, output_dim=256),
        ...     },
        ...     fusion='concat',
        ...     output_dim=512
        ... )
        >>> outputs = encoder({
        ...     'genetic': genetic_data,
        ...     'weather': weather_data,
        ...     'soil': soil_data
        ... })
    """
    
    def __init__(
        self,
        encoders: dict,
        fusion: str = 'concat',
        output_dim: Optional[int] = None,
        dropout: float = 0.1
    ):
        """
        Args:
            encoders: 字典，键为输入名称，值为对应的编码器
            fusion: 融合方式 ('concat', 'mean', 'sum', 'attention')
            output_dim: 输出维度
            dropout: Dropout 率
        """
        super().__init__()
        
        self.encoder_names = list(encoders.keys())
        self.encoders = nn.ModuleDict(encoders)
        self.fusion = fusion
        self.dropout = nn.Dropout(dropout)
        
        # 计算输出维度
        encoder_dims = [getattr(enc, 'output_dim', 256) for enc in encoders.values()]
        
        if fusion == 'concat':
            total_dim = sum(encoder_dims)
            self.output_dim = output_dim or total_dim
            if output_dim and output_dim != total_dim:
                self.projection = nn.Linear(total_dim, output_dim)
            else:
                self.projection = None
        else:
            assert len(set(encoder_dims)) == 1, \
                f"{fusion} 融合需要所有编码器输出维度相同"
            self.output_dim = encoder_dims[0]
            
            if fusion == 'attention':
                self.attention = nn.Sequential(
                    nn.Linear(encoder_dims[0], encoder_dims[0] // 4),
                    nn.ReLU(),
                    nn.Linear(encoder_dims[0] // 4, 1)
                )
    
    def forward(self, inputs: dict) -> torch.Tensor:
        """
        编码多个输入并融合。
        
        Args:
            inputs: 字典，键为输入名称，值为对应的数据张量
            
        Returns:
            融合后的编码结果
        """
        # 编码各输入
        outputs = {}
        for name in self.encoder_names:
            if name in inputs and inputs[name] is not None:
                outputs[name] = self.encoders[name](inputs[name])
        
        if len(outputs) == 0:
            raise ValueError("至少需要一个有效输入")
        
        output_list = list(outputs.values())
        
        # 融合
        if self.fusion == 'concat':
            out = torch.cat(output_list, dim=-1)
            if self.projection is not None:
                out = self.projection(out)
                
        elif self.fusion == 'mean':
            out = torch.stack(output_list, dim=0).mean(dim=0)
            
        elif self.fusion == 'sum':
            out = torch.stack(output_list, dim=0).sum(dim=0)
            
        elif self.fusion == 'attention':
            # 计算注意力权重
            stacked = torch.stack(output_list, dim=1)  # [batch, n_inputs, dim]
            attn_scores = self.attention(stacked).squeeze(-1)  # [batch, n_inputs]
            attn_weights = F.softmax(attn_scores, dim=-1)
            out = (stacked * attn_weights.unsqueeze(-1)).sum(dim=1)
        
        return self.dropout(out)

