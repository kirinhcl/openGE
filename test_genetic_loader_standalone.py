"""独立的基因型数据加载器测试脚本"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from pathlib import Path
from typing import Tuple, List, Optional

class GeneticLoader:
    """Loader for genetic data (SNP markers, VCF files, etc.)."""
    
    def __init__(self):
        """Initialize genetic data loader."""
        self.marker_names: Optional[List[str]] = None
        self.sample_ids: Optional[List[str]] = None
    
    def load_from_numerical_file(self, 
                                 filepath: str,
                                 sample_col: str = '<Marker>',
                                 handle_missing: str = 'mean',
                                 missing_threshold: float = 0.5) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load genetic data from numerical text file (space/tab separated).
        
        Expected format:
        - First column: Sample IDs
        - Remaining columns: SNP markers
        - Values: 0, 0.5, 1, NA (representing genotype dosages)
        
        Args:
            filepath: Path to numerical genotype file
            sample_col: Name of the sample ID column (default: '<Marker>')
            handle_missing: How to handle missing values ('mean', 'drop', 'zero')
            missing_threshold: Threshold for removing markers with too many missing values
            
        Returns:
            tuple: (genotype_matrix, sample_ids, marker_names)
                   genotype_matrix shape: (n_samples, n_markers)
        """
        print(f"\n{'=' * 70}")
        print(f"加载基因型数据: {Path(filepath).name}")
        print(f"{'=' * 70}")
        
        # 读取文件
        df = pd.read_csv(filepath, sep=r'\s+')
        print(f"✓ 文件加载成功: {df.shape}")
        
        # 提取样本ID
        if sample_col not in df.columns:
            raise ValueError(f"未找到样本ID列 '{sample_col}'。可用列: {df.columns[:5].tolist()}")
        
        sample_ids = df[sample_col].tolist()
        print(f"✓ 样本数: {len(sample_ids)}")
        
        # 提取标记数据
        marker_cols = [col for col in df.columns if col != sample_col]
        marker_names = marker_cols
        genotype_data = df[marker_cols].values
        
        print(f"✓ 标记数: {len(marker_names)}")
        print(f"  - 标记名称示例: {marker_names[:5]}")
        
        # 检查缺失值
        n_missing = np.isnan(genotype_data).sum()
        missing_rate = 100 * n_missing / genotype_data.size
        print(f"\n📊 缺失值分析:")
        print(f"   - 总缺失值: {n_missing} / {genotype_data.size} ({missing_rate:.2f}%)")
        
        # 删除缺失率过高的标记
        missing_per_marker = np.isnan(genotype_data).sum(axis=0)
        high_missing_markers = missing_per_marker > (len(sample_ids) * missing_threshold)
        
        if high_missing_markers.sum() > 0:
            print(f"   - 删除缺失率 > {100*missing_threshold:.0f}% 的标记: {high_missing_markers.sum()} 个")
            genotype_data = genotype_data[:, ~high_missing_markers]
            marker_names = [m for m, remove in zip(marker_names, high_missing_markers) if not remove]
        
        # 处理剩余缺失值
        n_missing_after = np.isnan(genotype_data).sum()
        if n_missing_after > 0:
            if handle_missing == 'mean':
                # 用每个标记的均值填充
                col_means = np.nanmean(genotype_data, axis=0)
                for j in range(genotype_data.shape[1]):
                    mask = np.isnan(genotype_data[:, j])
                    genotype_data[mask, j] = col_means[j]
                print(f"   - 用均值填充剩余缺失值")
                
            elif handle_missing == 'zero':
                genotype_data = np.nan_to_num(genotype_data, nan=0.0)
                print(f"   - 用0填充剩余缺失值")
                
            elif handle_missing == 'drop':
                # 删除有缺失值的样本
                has_missing = np.isnan(genotype_data).any(axis=1)
                genotype_data = genotype_data[~has_missing]
                sample_ids = [s for s, keep in zip(sample_ids, ~has_missing) if keep]
                print(f"   - 删除有缺失值的样本: {has_missing.sum()} 个")
            
            else:
                raise ValueError(f"未知的缺失值处理方法: {handle_missing}")
        
        # 验证数据范围
        unique_vals = np.unique(genotype_data[~np.isnan(genotype_data)])
        print(f"\n✓ 基因型数据统计:")
        print(f"   - 最终形状: {genotype_data.shape}")
        print(f"   - 数据类型: {genotype_data.dtype}")
        print(f"   - 数据范围: [{genotype_data.min():.2f}, {genotype_data.max():.2f}]")
        print(f"   - 独特值: {unique_vals[:10]}")  # 显示前10个
        
        # 保存元数据
        self.marker_names = marker_names
        self.sample_ids = sample_ids
        
        print(f"\n{'=' * 70}")
        print(f"基因型数据加载完成")
        print(f"{'=' * 70}\n")
        
        return genotype_data.astype(np.float32), sample_ids, marker_names
    
    def filter_markers(self,
                      genotype_matrix: np.ndarray,
                      marker_names: List[str],
                      maf_threshold: float = 0.05) -> Tuple[np.ndarray, List[str]]:
        """
        Filter markers by Minor Allele Frequency (MAF).
        
        Args:
            genotype_matrix: Genotype data (n_samples, n_markers)
            marker_names: List of marker names
            maf_threshold: Minimum MAF threshold (default: 0.05)
            
        Returns:
            tuple: (filtered_matrix, filtered_marker_names)
        """
        # 计算 MAF
        allele_freq = np.mean(genotype_matrix, axis=0)
        maf = np.minimum(allele_freq, 1 - allele_freq)
        
        # 过滤
        keep_mask = maf >= maf_threshold
        filtered_matrix = genotype_matrix[:, keep_mask]
        filtered_names = [name for name, keep in zip(marker_names, keep_mask) if keep]
        
        print(f"✓ MAF 过滤 (阈值={maf_threshold}):")
        print(f"  - 保留标记: {keep_mask.sum()} / {len(marker_names)}")
        print(f"  - 删除标记: {(~keep_mask).sum()}")
        
        return filtered_matrix, filtered_names
    
    def save_to_numpy(self,
                     genotype_matrix: np.ndarray,
                     sample_ids: List[str],
                     marker_names: List[str],
                     output_path: str):
        """
        Save genotype data to .npz file.
        
        Args:
            genotype_matrix: Genotype data
            sample_ids: Sample IDs
            marker_names: Marker names
            output_path: Output file path
        """
        np.savez_compressed(
            output_path,
            genotypes=genotype_matrix,
            sample_ids=sample_ids,
            marker_names=marker_names
        )
        
        file_size = Path(output_path).stat().st_size / (1024 * 1024)  # MB
        print(f"✓ 保存基因型数据: {output_path}")
        print(f"  - 文件大小: {file_size:.2f} MB")


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
    for i, sid in enumerate(sample_ids[:10]):
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
