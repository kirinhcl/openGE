"""测试不同的基因型编码方法"""

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, '/Users/lu/openGE')
from openge.data import GeneticLoader

def test_encoding_methods():
    """测试所有编码方法"""
    
    print("=" * 80)
    print("基因型编码方法对比测试")
    print("=" * 80)
    
    # 加载数据
    loader = GeneticLoader()
    genotypes, sample_ids, marker_names = loader.load_from_numerical_file(
        'Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt',
        sample_col='<Marker>',
        handle_missing='mean',
        missing_threshold=0.5
    )
    
    # 使用前100个样本和100个标记进行快速测试
    test_data = genotypes[:100, :100]
    
    print(f"\n测试数据: {test_data.shape}")
    print(f"原始数据统计: 均值={test_data.mean():.4f}, 标准差={test_data.std():.4f}")
    print(f"原始数据范围: [{test_data.min():.4f}, {test_data.max():.4f}]")
    
    # 测试所有编码方法
    encoding_methods = [
        'keep', 'additive', 'standardized', 'centered', 
        'minmax', 'binary', 'onehot', 'dominant', 'recessive'
    ]
    
    results = {}
    
    print("\n" + "=" * 80)
    print("测试各种编码方法")
    print("=" * 80)
    
    for method in encoding_methods:
        print(f"\n【{method.upper()}】")
        result = loader.encode_genotypes(test_data, encoding=method)
        
        # 处理返回tuple的情况
        if isinstance(result, tuple):
            encoded, stats = result
            results[method] = (encoded, stats)
            print(f"  返回统计信息: {list(stats.keys())}")
        else:
            encoded = result
            results[method] = encoded
        
        # 打印编码后的统计信息
        if method == 'onehot':
            print(f"  形状: {encoded.shape}")
            print(f"  各类别比例: 类0={encoded[:,:,0].sum()/(encoded.shape[0]*encoded.shape[1]):.2%}, "
                  f"类1={encoded[:,:,1].sum()/(encoded.shape[0]*encoded.shape[1]):.2%}, "
                  f"类2={encoded[:,:,2].sum()/(encoded.shape[0]*encoded.shape[1]):.2%}")
        else:
            print(f"  形状: {encoded.shape}")
            print(f"  均值: {encoded.mean():.4f}")
            print(f"  标准差: {encoded.std():.4f}")
            print(f"  范围: [{encoded.min():.4f}, {encoded.max():.4f}]")
            print(f"  独特值数量: {len(np.unique(encoded))}")
    
    # 可视化对比
    print("\n生成可视化对比图...")
    fig = plt.figure(figsize=(18, 12))
    
    plot_methods = [m for m in encoding_methods if m != 'onehot']  # onehot是3D，单独处理
    
    for idx, method in enumerate(plot_methods, 1):
        ax = plt.subplot(3, 3, idx)
        
        if isinstance(results[method], tuple):
            encoded, _ = results[method]
        else:
            encoded = results[method]
        
        # 绘制前20个样本的热图
        im = ax.imshow(encoded[:20, :50], aspect='auto', cmap='viridis')
        ax.set_title(f'{method.upper()}\nμ={encoded.mean():.3f}, σ={encoded.std():.3f}')
        ax.set_xlabel('Marker Index')
        ax.set_ylabel('Sample Index')
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    output_file = 'output/encoding_comparison.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 保存可视化: {output_file}")
    plt.close()
    
    # 分布对比
    fig2 = plt.figure(figsize=(18, 10))
    
    for idx, method in enumerate(plot_methods, 1):
        ax = plt.subplot(3, 3, idx)
        
        if isinstance(results[method], tuple):
            encoded, _ = results[method]
        else:
            encoded = results[method]
        
        ax.hist(encoded.flatten(), bins=50, edgecolor='black', alpha=0.7)
        ax.set_title(f'{method.upper()} Distribution')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file2 = 'output/encoding_distributions.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"✓ 保存分布图: {output_file2}")
    plt.close()
    
    # 使用建议
    print("\n" + "=" * 80)
    print("编码方法使用建议")
    print("=" * 80)
    
    recommendations = {
        'keep (dosage)': '【推荐用于深度学习】保持[0, 0.5, 1]，符合加性遗传模型，无需额外预处理',
        'additive': '强调等位基因计数[0, 1, 2]，适合线性模型和GBLUP',
        'standardized': '标准化到均值0、方差1，适合不同频率标记的公平比较，深度学习常用',
        'centered': '中心化但保持方差，适合需要解释原始变异的场景',
        'minmax': '归一化到[0, 1]，适合神经网络输入层',
        'binary': '二值化，丢失杂合信息，不推荐用于复杂模型',
        'onehot': '独热编码，捕捉非加性效应，模型参数量增加3倍',
        'dominant': '显性遗传模型，适合单基因显性性状',
        'recessive': '隐性遗传模型，适合单基因隐性性状'
    }
    
    for method, desc in recommendations.items():
        print(f"\n• {method:20s}: {desc}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)

if __name__ == "__main__":
    test_encoding_methods()
