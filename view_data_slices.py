"""查看加载后的GxE数据切片"""

import sys
sys.path.insert(0, '/Users/lu/openGE')

from openge.data import GxEDataLoader
import numpy as np
import pandas as pd


def view_data_slices():
    """查看数据切片"""
    
    print("\n" + "=" * 80)
    print("GxE 数据切片查看器")
    print("=" * 80)
    
    # 初始化数据加载器
    loader = GxEDataLoader()
    
    # 1. 加载基因型数据
    print("\n【加载基因型数据】")
    genetic_data, genetic_ids, marker_names = loader.load_genetic(
        filepath='Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt',
        sample_col='<Marker>',
        handle_missing='mean',
        maf_threshold=0.05,
        encoding='keep'
    )
    
    # 2. 加载环境数据
    print("\n【加载环境数据】")
    environment_data, env_ids, env_features = loader.load_environment(
        weather_file='Training_data/4_Training_Weather_Data_2014_2023_full_year.csv',
        use_3d=True,
        handle_missing='drop'
    )
    
    # 3. 加载表型数据
    print("\n【加载表型数据】")
    phenotype_data, pheno_ids, trait_names = loader.load_phenotype(
        filepath='Training_data/1_Training_Trait_Data_2014_2023.csv',
        traits=['Yield_Mg_ha'],
        sample_id_col='Hybrid',
        env_col='Env',
        handle_missing='drop',
        handle_outliers=True,
        outlier_method='iqr'
    )
    
    # 4. 对齐样本
    print("\n【对齐样本】")
    aligned_genetic, aligned_env, aligned_pheno, aligned_ids = loader.align_samples(strategy='inner')
    
    print("\n" + "=" * 80)
    print("数据切片展示")
    print("=" * 80)
    
    # 显示前5个样本的详细信息
    n_samples_to_show = 5
    
    for i in range(n_samples_to_show):
        print(f"\n{'─' * 80}")
        print(f"样本 {i+1}: {aligned_ids[i]}")
        print(f"{'─' * 80}")
        
        # 解析样本ID
        parts = aligned_ids[i].split('_')
        env = '_'.join(parts[:2])
        hybrid = '_'.join(parts[2:])
        print(f"  环境 (Env): {env}")
        print(f"  杂交种 (Hybrid): {hybrid}")
        
        # 基因型数据切片
        print(f"\n  【基因型数据】")
        print(f"    形状: {aligned_genetic[i].shape}")
        print(f"    前10个SNP标记:")
        for j in range(min(10, len(marker_names))):
            genotype_val = aligned_genetic[i, j]
            print(f"      {marker_names[j]:15s}: {genotype_val:.3f}", end="")
            if genotype_val == 0:
                print(" (纯合AA)")
            elif genotype_val == 0.5:
                print(" (杂合Aa)")
            elif genotype_val == 1:
                print(" (纯合aa)")
            else:
                print(f" (填充值)")
        
        # 环境数据切片 (显示前5天和最后5天)
        print(f"\n  【环境数据 (3D时间序列)】")
        print(f"    形状: {aligned_env[i].shape}")
        print(f"    气象特征: {env_features}")
        print(f"\n    前5天气象数据:")
        
        # 创建DataFrame显示
        env_df = pd.DataFrame(
            aligned_env[i, :5, :],
            columns=env_features,
            index=[f"第{d+1}天" for d in range(5)]
        )
        print(env_df.to_string())
        
        print(f"\n    最后5天气象数据:")
        last_days = aligned_env[i, -5:, :]
        env_df_last = pd.DataFrame(
            last_days,
            columns=env_features,
            index=[f"第{366-4+d}天" for d in range(5)]
        )
        print(env_df_last.to_string())
        
        # 选择几个关键时间点显示
        print(f"\n    关键时间点气象数据:")
        key_days = [0, 91, 182, 273, 365]  # 大约每季度一个点
        key_data = aligned_env[i, key_days, :]
        key_df = pd.DataFrame(
            key_data,
            columns=env_features,
            index=[f"第{d+1}天" for d in key_days]
        )
        print(key_df.to_string())
        
        # 表型数据
        print(f"\n  【表型数据 (目标变量)】")
        print(f"    {trait_names[0]}: {aligned_pheno[i, 0]:.3f} Mg/ha")
        
    # 统计摘要
    print("\n" + "=" * 80)
    print("数据统计摘要")
    print("=" * 80)
    
    print(f"\n【基因型统计】")
    print(f"  - 样本数: {aligned_genetic.shape[0]}")
    print(f"  - SNP标记数: {aligned_genetic.shape[1]}")
    print(f"  - 数据范围: [{aligned_genetic.min():.3f}, {aligned_genetic.max():.3f}]")
    print(f"  - 均值: {aligned_genetic.mean():.3f}")
    print(f"  - 标准差: {aligned_genetic.std():.3f}")
    
    # 基因型分布
    unique, counts = np.unique(aligned_genetic.flatten(), return_counts=True)
    print(f"\n  基因型值分布 (前10个最常见值):")
    sorted_idx = np.argsort(counts)[::-1][:10]
    for idx in sorted_idx:
        print(f"    {unique[idx]:.3f}: {counts[idx]:,} 次 ({100*counts[idx]/counts.sum():.2f}%)")
    
    print(f"\n【环境数据统计】")
    print(f"  - 样本数: {aligned_env.shape[0]}")
    print(f"  - 时间步数: {aligned_env.shape[1]} 天")
    print(f"  - 气象特征数: {aligned_env.shape[2]}")
    print(f"  - 数据范围: [{aligned_env.min():.3f}, {aligned_env.max():.3f}]")
    print(f"  - 均值: {aligned_env.mean():.3f}")
    print(f"  - 标准差: {aligned_env.std():.3f}")
    
    # 每个气象特征的统计
    print(f"\n  各气象特征统计:")
    for j, feature in enumerate(env_features):
        feature_data = aligned_env[:, :, j]
        print(f"    {feature:20s}: 均值={feature_data.mean():8.2f}, "
              f"标准差={feature_data.std():8.2f}, "
              f"范围=[{feature_data.min():8.2f}, {feature_data.max():8.2f}]")
    
    print(f"\n【表型数据统计】")
    print(f"  - 样本数: {aligned_pheno.shape[0]}")
    print(f"  - 性状数: {aligned_pheno.shape[1]}")
    print(f"  - {trait_names[0]}:")
    print(f"    均值: {aligned_pheno.mean():.3f} Mg/ha")
    print(f"    标准差: {aligned_pheno.std():.3f} Mg/ha")
    print(f"    范围: [{aligned_pheno.min():.3f}, {aligned_pheno.max():.3f}] Mg/ha")
    print(f"    中位数: {np.median(aligned_pheno):.3f} Mg/ha")
    
    # 分位数
    quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    print(f"\n  产量分位数:")
    for q in quantiles:
        val = np.quantile(aligned_pheno, q)
        print(f"    {int(q*100)}%: {val:.3f} Mg/ha")
    
    print("\n" + "=" * 80)
    print("数据切片查看完成")
    print("=" * 80)


if __name__ == '__main__':
    view_data_slices()
