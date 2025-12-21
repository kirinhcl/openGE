"""测试基因型数据加载器"""

import sys
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# 直接导入genetic模块，避免torch依赖
sys.path.insert(0, '/Users/lu/openGE')
from openge.data.loaders.genetic import GeneticLoader

def test_genetic_loader():
    """测试基因型数据加载"""
    
    # 初始化加载器
    loader = GeneticLoader()
    
    # 加载基因型数据
    print("\n测试基因型数据加载...")
    genotype_file = "Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt"
    
    # 加载数据
    genotypes, sample_ids, marker_names = loader.load_from_numerical_file(
        genotype_file,
        sample_col='<Marker>',
        handle_missing='mean',  # 用均值填充缺失值
        missing_threshold=0.5    # 删除缺失率>50%的标记
    )
    
    # 基本统计
    print("\n" + "=" * 70)
    print("基因型数据统计")
    print("=" * 70)
    print(f"数据形状: {genotypes.shape}")
    print(f"样本数: {len(sample_ids)}")
    print(f"标记数: {len(marker_names)}")
    print(f"\n样本ID示例:")
    for i, sid in enumerate(sample_ids[:5]):
        print(f"  {i+1}. {sid}")
    
    print(f"\n标记名称示例:")
    for i, marker in enumerate(marker_names[:10]):
        print(f"  {i+1}. {marker}")
    
    # 数据质量检查
    print(f"\n数据质量:")
    print(f"  - 最小值: {genotypes.min():.4f}")
    print(f"  - 最大值: {genotypes.max():.4f}")
    print(f"  - 均值: {genotypes.mean():.4f}")
    print(f"  - 标准差: {genotypes.std():.4f}")
    print(f"  - 缺失值: {np.isnan(genotypes).sum()}")
    
    # 等位基因频率分析
    allele_freq = np.mean(genotypes, axis=0)
    print(f"\n等位基因频率统计:")
    print(f"  - 均值: {allele_freq.mean():.4f}")
    print(f"  - 标准差: {allele_freq.std():.4f}")
    print(f"  - 范围: [{allele_freq.min():.4f}, {allele_freq.max():.4f}]")
    
    # MAF分析
    maf = np.minimum(allele_freq, 1 - allele_freq)
    print(f"\n次要等位基因频率(MAF)统计:")
    print(f"  - 均值: {maf.mean():.4f}")
    print(f"  - MAF < 0.05: {(maf < 0.05).sum()} ({100*(maf < 0.05).sum()/len(maf):.2f}%)")
    print(f"  - MAF < 0.01: {(maf < 0.01).sum()} ({100*(maf < 0.01).sum()/len(maf):.2f}%)")
    
    # 可视化
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    fig = plt.figure(figsize=(16, 10))
    
    # 1. 基因型热图 (前50个样本和标记)
    ax1 = plt.subplot(2, 3, 1)
    n_samples_show = min(50, genotypes.shape[0])
    n_markers_show = min(100, genotypes.shape[1])
    im = ax1.imshow(genotypes[:n_samples_show, :n_markers_show], 
                    aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax1.set_xlabel('标记索引')
    ax1.set_ylabel('样本索引')
    ax1.set_title(f'基因型热图 (前{n_samples_show}样本 × {n_markers_show}标记)')
    plt.colorbar(im, ax=ax1, label='基因型值')
    
    # 2. 基因型值分布
    ax2 = plt.subplot(2, 3, 2)
    ax2.hist(genotypes.flatten(), bins=50, edgecolor='black', alpha=0.7)
    ax2.set_xlabel('基因型值')
    ax2.set_ylabel('频次')
    ax2.set_title('基因型值分布')
    ax2.axvline(x=0, color='r', linestyle='--', alpha=0.5, label='0')
    ax2.axvline(x=0.5, color='g', linestyle='--', alpha=0.5, label='0.5')
    ax2.axvline(x=1, color='b', linestyle='--', alpha=0.5, label='1')
    ax2.legend()
    
    # 3. 等位基因频率分布
    ax3 = plt.subplot(2, 3, 3)
    ax3.hist(allele_freq, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax3.set_xlabel('等位基因频率')
    ax3.set_ylabel('标记数量')
    ax3.set_title('等位基因频率分布')
    ax3.axvline(x=0.5, color='r', linestyle='--', alpha=0.5, label='0.5')
    ax3.legend()
    
    # 4. MAF分布
    ax4 = plt.subplot(2, 3, 4)
    ax4.hist(maf, bins=50, edgecolor='black', alpha=0.7, color='green')
    ax4.set_xlabel('次要等位基因频率 (MAF)')
    ax4.set_ylabel('标记数量')
    ax4.set_title('MAF分布')
    ax4.axvline(x=0.05, color='r', linestyle='--', alpha=0.5, label='MAF=0.05')
    ax4.axvline(x=0.01, color='orange', linestyle='--', alpha=0.5, label='MAF=0.01')
    ax4.legend()
    
    # 5. 样本间遗传相似度
    ax5 = plt.subplot(2, 3, 5)
    # 计算前50个样本的相关系数矩阵
    n_samples_corr = min(50, genotypes.shape[0])
    similarity = np.corrcoef(genotypes[:n_samples_corr])
    im = ax5.imshow(similarity, cmap='coolwarm', vmin=-1, vmax=1)
    ax5.set_xlabel('样本索引')
    ax5.set_ylabel('样本索引')
    ax5.set_title(f'样本间遗传相似度 (前{n_samples_corr}样本)')
    plt.colorbar(im, ax=ax5, label='相关系数')
    
    # 6. 每个样本的杂合度
    ax6 = plt.subplot(2, 3, 6)
    # 杂合度 = 基因型值为0.5的比例
    heterozygosity = np.sum(np.abs(genotypes - 0.5) < 0.01, axis=1) / genotypes.shape[1]
    ax6.bar(range(len(heterozygosity)), heterozygosity, alpha=0.7, color='purple')
    ax6.set_xlabel('样本索引')
    ax6.set_ylabel('杂合度')
    ax6.set_title('每个样本的杂合度')
    ax6.axhline(y=heterozygosity.mean(), color='r', linestyle='--', 
                label=f'均值={heterozygosity.mean():.3f}')
    ax6.legend()
    
    plt.tight_layout()
    output_file = f'output/genetic_visualization_{timestamp}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n✓ 可视化保存: {output_file}")
    plt.close()
    
    # 保存NPZ文件
    npz_file = f'output/genotypes_{timestamp}.npz'
    loader.save_to_numpy(genotypes, sample_ids, marker_names, npz_file)
    
    # 测试MAF过滤
    print(f"\n测试 MAF 过滤...")
    filtered_genotypes, filtered_markers = loader.filter_markers(
        genotypes, marker_names, maf_threshold=0.05
    )
    
    print(f"\n过滤后的数据形状: {filtered_genotypes.shape}")
    
    # 按染色体统计标记数
    print(f"\n按染色体统计标记数:")
    chr_counts = {}
    for marker in marker_names:
        # 提取染色体编号 (S1, S2, ..., S10)
        chr_name = marker.split('_')[0]
        chr_counts[chr_name] = chr_counts.get(chr_name, 0) + 1
    
    for chr_name in sorted(chr_counts.keys(), key=lambda x: int(x[1:])):
        print(f"  {chr_name}: {chr_counts[chr_name]} 标记")
    
    print("\n" + "=" * 70)
    print("测试完成！")
    print("=" * 70)

if __name__ == "__main__":
    test_genetic_loader()
