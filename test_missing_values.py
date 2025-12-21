"""测试不同的缺失值处理方法对基因型分布的影响"""

import sys
sys.path.insert(0, '/Users/lu/openGE')

from openge.data.loaders.genetic import GeneticLoader
import numpy as np
import matplotlib.pyplot as plt


def test_missing_strategies():
    """比较不同缺失值处理策略"""
    
    print("\n" + "=" * 80)
    print("缺失值处理策略对比")
    print("=" * 80)
    
    strategies = {
        'mean': '均值填充 (产生连续值)',
        'zero': '零值填充 (保持离散)',
        'mode': '众数填充 (保持离散 - 推荐)',
    }
    
    results = {}
    
    for strategy, description in strategies.items():
        print(f"\n{'─' * 80}")
        print(f"策略: {strategy} - {description}")
        print(f"{'─' * 80}")
        
        loader = GeneticLoader()
        
        if strategy == 'mode':
            # 众数填充需要手动实现
            # 先用mean加载，然后手动替换
            genotypes, sample_ids, marker_names = loader.load_from_numerical_file(
                filepath='Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt',
                sample_col='<Marker>',
                handle_missing='mean',
                missing_threshold=0.5
            )
            
            print(f"\n🔄 转换为众数填充...")
            # 重新加载以获取原始缺失值位置
            import pandas as pd
            df = pd.read_csv('Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt', 
                           sep='\t', index_col=0)
            
            original_data = df.values.astype(np.float32)
            missing_mask = np.isnan(original_data)
            
            # 对每个标记，用众数填充
            for j in range(original_data.shape[1]):
                if missing_mask[:, j].any():
                    # 获取非缺失值
                    valid_values = original_data[~missing_mask[:, j], j]
                    # 找到众数（最常见的值）
                    unique, counts = np.unique(valid_values, return_counts=True)
                    mode_value = unique[np.argmax(counts)]
                    # 填充
                    original_data[missing_mask[:, j], j] = mode_value
            
            genotypes = original_data
            print(f"✓ 众数填充完成")
        else:
            genotypes, sample_ids, marker_names = loader.load_from_numerical_file(
                filepath='Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt',
                sample_col='<Marker>',
                handle_missing=strategy,
                missing_threshold=0.5
            )
        
        # 应用MAF过滤
        genotypes, marker_names = loader.filter_markers(
            genotype_matrix=genotypes,
            marker_names=marker_names,
            maf_threshold=0.05
        )
        
        # 统计基因型值分布
        unique, counts = np.unique(genotypes.flatten(), return_counts=True)
        
        print(f"\n基因型值分布:")
        # 按频率排序
        sorted_idx = np.argsort(counts)[::-1][:15]  # 前15个最常见值
        
        total = counts.sum()
        for idx in sorted_idx:
            percentage = 100 * counts[idx] / total
            print(f"  {unique[idx]:8.3f}: {counts[idx]:12,} 次 ({percentage:6.2f}%)")
        
        # 检查是否只有标准值
        standard_values = {0.0, 0.5, 1.0}
        non_standard = [v for v in unique if not any(abs(v - sv) < 0.001 for sv in standard_values)]
        
        print(f"\n是否只有标准值 (0, 0.5, 1): {'✓ 是' if len(non_standard) == 0 else f'✗ 否，有{len(non_standard)}个非标准值'}")
        
        if len(non_standard) > 0 and len(non_standard) <= 10:
            print(f"非标准值示例: {non_standard[:10]}")
        
        results[strategy] = {
            'genotypes': genotypes,
            'unique': unique,
            'counts': counts,
            'non_standard': non_standard
        }
    
    # 可视化对比
    print(f"\n{'=' * 80}")
    print("生成可视化对比")
    print(f"{'=' * 80}")
    
    fig, axes = plt.subplots(1, len(strategies), figsize=(15, 5))
    
    for idx, (strategy, description) in enumerate(strategies.items()):
        ax = axes[idx]
        
        unique = results[strategy]['unique']
        counts = results[strategy]['counts']
        
        # 只显示前10个最常见值
        sorted_idx = np.argsort(counts)[::-1][:10]
        
        ax.bar(range(len(sorted_idx)), counts[sorted_idx])
        ax.set_xticks(range(len(sorted_idx)))
        ax.set_xticklabels([f"{unique[i]:.3f}" for i in sorted_idx], rotation=45)
        ax.set_title(f"{strategy}\n{description}", fontsize=10)
        ax.set_ylabel('频次')
        ax.set_xlabel('基因型值')
        
        # 标记标准值
        for i, sort_i in enumerate(sorted_idx):
            val = unique[sort_i]
            if abs(val - 0.0) < 0.001 or abs(val - 0.5) < 0.001 or abs(val - 1.0) < 0.001:
                ax.get_xticklabels()[i].set_color('green')
                ax.get_xticklabels()[i].set_weight('bold')
            else:
                ax.get_xticklabels()[i].set_color('red')
    
    plt.tight_layout()
    plt.savefig('output/missing_value_strategies_comparison.png', dpi=150, bbox_inches='tight')
    print(f"✓ 保存对比图: output/missing_value_strategies_comparison.png")
    
    # 推荐
    print(f"\n{'=' * 80}")
    print("📋 推荐使用策略")
    print(f"{'=' * 80}")
    print(f"1. 如果需要严格的离散值 (0, 0.5, 1):")
    print(f"   👉 使用 'mode' (众数填充) 或 'zero' (零值填充)")
    print(f"   - mode: 用该标记最常见的值填充，更符合群体分布")
    print(f"   - zero: 统一用0填充，简单但可能引入偏差")
    print(f"\n2. 如果允许连续值 (适合深度学习):")
    print(f"   👉 使用 'mean' (均值填充) - 当前使用")
    print(f"   - 优点: 不丢失信息，保持群体统计特性")
    print(f"   - 缺点: 引入非标准值 (如0.737, 0.818等)")
    print(f"\n3. 如果数据质量要求极高:")
    print(f"   👉 使用 'drop' (删除有缺失的样本)")
    print(f"   - 但可能损失较多样本")
    print(f"{'=' * 80}\n")


if __name__ == '__main__':
    test_missing_strategies()
