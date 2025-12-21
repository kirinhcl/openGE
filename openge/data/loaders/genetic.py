"""Loader for genetic/SNP marker data."""

import numpy as np
import pandas as pd
from typing import Union, Tuple, List, Optional, Dict
from pathlib import Path


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
    
    def load_from_vcf(self, 
                     filepath: str,
                     encoding: str = 'dosage') -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load genetic data from VCF file.
        
        Args:
            filepath: Path to VCF file
            encoding: Encoding method ('dosage', 'additive')
                     'dosage': 0/0→0, 0/1→0.5, 1/1→1
                     'additive': 0/0→0, 0/1→1, 1/1→2
            
        Returns:
            tuple: (genotype_matrix, sample_ids, marker_names)
        """
        print(f"\n加载 VCF 文件: {filepath}")
        
        sample_ids = []
        marker_names = []
        genotypes = []
        
        with open(filepath, 'r') as f:
            for line in f:
                if line.startswith('##'):
                    continue  # 跳过元数据行
                    
                if line.startswith('#CHROM'):
                    # 提取样本ID
                    parts = line.strip().split('\t')
                    sample_ids = parts[9:]  # 从第10列开始是样本
                    print(f"✓ 找到 {len(sample_ids)} 个样本")
                    continue
                
                # 解析变异位点
                parts = line.strip().split('\t')
                chrom = parts[0]
                pos = parts[1]
                ref = parts[3]
                alt = parts[4]
                
                marker_name = f"{chrom}_{pos}_{ref}_{alt}"
                marker_names.append(marker_name)
                
                # 解析基因型
                format_field = parts[8]
                gt_idx = format_field.split(':').index('GT')
                
                sample_genotypes = []
                for sample_data in parts[9:]:
                    gt = sample_data.split(':')[gt_idx]
                    
                    # 编码基因型
                    if gt in ['./.', '.']:
                        sample_genotypes.append(np.nan)
                    elif encoding == 'dosage':
                        if gt in ['0/0', '0|0']:
                            sample_genotypes.append(0.0)
                        elif gt in ['0/1', '1/0', '0|1', '1|0']:
                            sample_genotypes.append(0.5)
                        elif gt in ['1/1', '1|1']:
                            sample_genotypes.append(1.0)
                        else:
                            sample_genotypes.append(np.nan)
                    elif encoding == 'additive':
                        if gt in ['0/0', '0|0']:
                            sample_genotypes.append(0.0)
                        elif gt in ['0/1', '1/0', '0|1', '1|0']:
                            sample_genotypes.append(1.0)
                        elif gt in ['1/1', '1|1']:
                            sample_genotypes.append(2.0)
                        else:
                            sample_genotypes.append(np.nan)
                
                genotypes.append(sample_genotypes)
        
        # 转换为数组 (markers × samples) 然后转置为 (samples × markers)
        genotype_matrix = np.array(genotypes, dtype=np.float32).T
        
        print(f"✓ 加载完成: {genotype_matrix.shape}")
        print(f"  - 样本数: {len(sample_ids)}")
        print(f"  - 标记数: {len(marker_names)}")
        
        self.marker_names = marker_names
        self.sample_ids = sample_ids
        
        return genotype_matrix, sample_ids, marker_names
    
    def load_from_csv(self, 
                     filepath: str,
                     sample_col: str = 'sample_id') -> Tuple[np.ndarray, List[str], List[str]]:
        """
        Load genetic data from CSV file.
        
        Args:
            filepath: Path to CSV file
            sample_col: Name of sample ID column
            
        Returns:
            tuple: (genotype_matrix, sample_ids, marker_names)
        """
        df = pd.read_csv(filepath)
        
        if sample_col not in df.columns:
            raise ValueError(f"Sample column '{sample_col}' not found in CSV")
        
        sample_ids = df[sample_col].tolist()
        marker_cols = [col for col in df.columns if col != sample_col]
        genotype_matrix = df[marker_cols].values.astype(np.float32)
        
        self.marker_names = marker_cols
        self.sample_ids = sample_ids
        
        print(f"✓ 从 CSV 加载完成: {genotype_matrix.shape}")
        
        return genotype_matrix, sample_ids, marker_cols
    
    def encode_genotypes(self, 
                        genotypes: np.ndarray,
                        encoding: str = 'keep') -> Union[np.ndarray, Tuple[np.ndarray, dict]]:
        """
        Encode genotypes to numerical values with various schemes.
        
        Args:
            genotypes: Raw genotype data (n_samples, n_markers)
            encoding: Encoding method
                     'keep': Keep original dosage values (0, 0.5, 1) [DEFAULT]
                     'additive': Scale to allele count (0, 1, 2)
                     'standardized': Z-score normalization per marker
                     'centered': Center to mean=0, keep original scale
                     'minmax': Scale to [0, 1] per marker
                     'binary': Threshold at 0.5 → {0, 1}
                     'onehot': One-hot encoding (n_samples, n_markers, 3)
                     'dominant': Dominant model (0→0, 0.5/1→1)
                     'recessive': Recessive model (0/0.5→0, 1→1)
            
        Returns:
            Encoded genetic matrix. For 'standardized' and 'centered', also returns
            a dict with 'mean' and 'std' for inverse transformation.
        """
        if encoding == 'keep':
            return genotypes.astype(np.float32)
        
        elif encoding == 'additive':
            # 0 → 0, 0.5 → 1, 1 → 2 (allele count)
            encoded = genotypes * 2
            print(f"✓ Additive编码: 范围 [{encoded.min():.2f}, {encoded.max():.2f}]")
            return encoded.astype(np.float32)
        
        elif encoding == 'standardized':
            # Z-score: (x - μ) / σ per marker
            means = np.mean(genotypes, axis=0, keepdims=True)
            stds = np.std(genotypes, axis=0, keepdims=True)
            stds = np.where(stds < 1e-8, 1.0, stds)  # 避免除0
            encoded = (genotypes - means) / stds
            print(f"✓ 标准化编码: 均值={encoded.mean():.4f}, 标准差={encoded.std():.4f}")
            return encoded.astype(np.float32), {'mean': means.squeeze(), 'std': stds.squeeze()}
        
        elif encoding == 'centered':
            # Center to zero mean, keep variance
            means = np.mean(genotypes, axis=0, keepdims=True)
            encoded = genotypes - means
            print(f"✓ 中心化编码: 均值={encoded.mean():.4f}, 范围 [{encoded.min():.2f}, {encoded.max():.2f}]")
            return encoded.astype(np.float32), {'mean': means.squeeze()}
        
        elif encoding == 'minmax':
            # Scale to [0, 1] per marker
            mins = np.min(genotypes, axis=0, keepdims=True)
            maxs = np.max(genotypes, axis=0, keepdims=True)
            ranges = maxs - mins
            ranges = np.where(ranges < 1e-8, 1.0, ranges)  # 避免除0
            encoded = (genotypes - mins) / ranges
            print(f"✓ MinMax编码: 范围 [{encoded.min():.2f}, {encoded.max():.2f}]")
            return encoded.astype(np.float32)
        
        elif encoding == 'binary':
            # Threshold at 0.5: 0, 0.5 → 0; 1 → 1
            encoded = (genotypes >= 0.75).astype(np.float32)
            print(f"✓ 二值编码: {(encoded==0).sum()} 个0, {(encoded==1).sum()} 个1")
            return encoded
        
        elif encoding == 'onehot':
            # One-hot encoding: (n_samples, n_markers, 3)
            n_samples, n_markers = genotypes.shape
            encoded = np.zeros((n_samples, n_markers, 3), dtype=np.float32)
            
            # Class 0: homozygous reference (genotype ≈ 0)
            encoded[:, :, 0] = (np.abs(genotypes - 0.0) < 0.25)
            # Class 1: heterozygous (genotype ≈ 0.5)
            encoded[:, :, 1] = (np.abs(genotypes - 0.5) < 0.25)
            # Class 2: homozygous alternate (genotype ≈ 1)
            encoded[:, :, 2] = (np.abs(genotypes - 1.0) < 0.25)
            
            print(f"✓ One-hot编码: 形状 {encoded.shape}")
            return encoded
        
        elif encoding == 'dominant':
            # Dominant model: at least one alternate allele
            # 0 → 0, 0.5 → 1, 1 → 1
            encoded = (genotypes > 0.25).astype(np.float32)
            print(f"✓ 显性编码: {(encoded==0).sum()} 个0, {(encoded==1).sum()} 个1")
            return encoded
        
        elif encoding == 'recessive':
            # Recessive model: two alternate alleles required
            # 0 → 0, 0.5 → 0, 1 → 1
            encoded = (genotypes >= 0.75).astype(np.float32)
            print(f"✓ 隐性编码: {(encoded==0).sum()} 个0, {(encoded==1).sum()} 个1")
            return encoded
        
        else:
            raise ValueError(f"Unknown encoding method: {encoding}. "
                           f"Available: keep, additive, standardized, centered, minmax, "
                           f"binary, onehot, dominant, recessive")
    
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
        allele_freq = np.nanmean(genotype_matrix, axis=0)
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
