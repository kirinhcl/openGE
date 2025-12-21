"""测试表型数据加载器"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import sys
sys.path.insert(0, '/Users/lu/openGE')
from openge.data.loaders.phenotype import PhenotypeLoader


def test_phenotype_loader():
    """测试表型数据加载"""
    
    print("=" * 80)
    print("表型数据加载器测试")
    print("=" * 80)
    
    # 初始化加载器
    loader = PhenotypeLoader()
    
    # 1. 查看可用的性状信息
    print("\n【可用性状信息】")
    trait_info = loader.get_trait_info()
    for trait_name, info in trait_info.items():
        primary = " ⭐" if info.get('primary', False) else ""
        print(f"  {trait_name}{primary}:")
        print(f"    - 名称: {info['name']}")
        print(f"    - 描述: {info['description']}")
        print(f"    - 单位: {info['unit']}")
        print(f"    - 类型: {info['type']}")
        print(f"    - 有效范围: {info['range']}")
    
    # 2. 加载主要性状
    print("\n【测试1: 加载主要性状】")
    main_traits = ['Yield_Mg_ha', 'Plant_Height_cm', 'Ear_Height_cm']
    
    trait_data, sample_ids, trait_names = loader.load_trait_data(
        filepath='Training_data/1_Training_Trait_Data_2014_2023.csv',
        traits=main_traits,
        sample_id_col='Hybrid',
        env_col='Env',
        handle_missing='drop'  # 删除有缺失值的样本
    )
    
    print(f"\n加载结果:")
    print(f"  - 数据形状: {trait_data.shape}")
    print(f"  - 样本数: {len(sample_ids)}")
    print(f"  - 性状数: {len(trait_names)}")
    print(f"\n前5个样本ID:")
    for i, sid in enumerate(sample_ids[:5], 1):
        print(f"  {i}. {sid}")
    
    # 3. 测试异常值检测
    print("\n【测试2: 异常值检测 - IQR方法】")
    cleaned_data_iqr, outlier_info_iqr = loader.handle_outliers(
        trait_data,
        trait_names=trait_names,
        method='iqr',
        iqr_factor=1.5,
        replace_with='median'
    )
    
    print("\n【测试3: 异常值检测 - Z-score方法】")
    cleaned_data_zscore, outlier_info_zscore = loader.handle_outliers(
        trait_data,
        trait_names=trait_names,
        method='zscore',
        zscore_threshold=3.0,
        replace_with='median'
    )
    
    print("\n【测试4: 异常值检测 - 预定义范围方法】")
    cleaned_data_range, outlier_info_range = loader.handle_outliers(
        trait_data,
        trait_names=trait_names,
        method='range',
        replace_with='clip'
    )
    
    # 4. 计算性状间相关性
    print("\n【测试5: 性状相关性分析】")
    corr_matrix, corr_df = loader.compute_correlations(trait_data, trait_names)
    print("\n相关系数矩阵:")
    print(corr_df.round(3))
    
    # 5. 可视化
    print("\n【生成可视化】")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 图1: 性状分布
    fig1 = plt.figure(figsize=(15, 10))
    
    for i, trait in enumerate(trait_names, 1):
        ax = plt.subplot(2, 3, i)
        data_col = trait_data[:, i-1]
        valid_data = data_col[~np.isnan(data_col)]
        
        ax.hist(valid_data, bins=50, edgecolor='black', alpha=0.7, color='steelblue')
        ax.set_xlabel(trait)
        ax.set_ylabel('Frequency')
        ax.set_title(f'{trait} Distribution\nμ={valid_data.mean():.2f}, σ={valid_data.std():.2f}')
        ax.axvline(valid_data.mean(), color='r', linestyle='--', label='Mean')
        ax.axvline(np.median(valid_data), color='g', linestyle='--', label='Median')
        ax.legend()
        ax.grid(alpha=0.3)
    
    # 异常值对比
    for i, trait in enumerate(trait_names, 1):
        ax = plt.subplot(2, 3, i+3)
        
        # 原始数据 vs 清洗后数据
        ax.boxplot([trait_data[:, i-1], 
                    cleaned_data_iqr[:, i-1],
                    cleaned_data_zscore[:, i-1]],
                   labels=['Original', 'IQR', 'Z-score'])
        ax.set_ylabel(trait)
        ax.set_title(f'{trait} - Outlier Handling Comparison')
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_file1 = f'output/phenotype_distributions_{timestamp}.png'
    plt.savefig(output_file1, dpi=150, bbox_inches='tight')
    print(f"✓ 保存分布图: {output_file1}")
    plt.close()
    
    # 图2: 相关性热图
    fig2 = plt.figure(figsize=(10, 8))
    sns.heatmap(corr_df, annot=True, fmt='.3f', cmap='coolwarm', 
                center=0, square=True, linewidths=1,
                cbar_kws={'label': 'Correlation Coefficient'})
    plt.title('Trait Correlation Matrix', fontsize=14, fontweight='bold')
    plt.tight_layout()
    output_file2 = f'output/phenotype_correlation_{timestamp}.png'
    plt.savefig(output_file2, dpi=150, bbox_inches='tight')
    print(f"✓ 保存相关性热图: {output_file2}")
    plt.close()
    
    # 图3: 散点图矩阵
    fig3 = plt.figure(figsize=(12, 12))
    
    for i in range(len(trait_names)):
        for j in range(len(trait_names)):
            ax = plt.subplot(len(trait_names), len(trait_names), 
                           i * len(trait_names) + j + 1)
            
            if i == j:
                # 对角线显示直方图
                data_col = trait_data[:, i]
                valid_data = data_col[~np.isnan(data_col)]
                ax.hist(valid_data, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
                ax.set_ylabel('Frequency')
            else:
                # 非对角线显示散点图
                x_data = trait_data[:, j]
                y_data = trait_data[:, i]
                valid_mask = ~(np.isnan(x_data) | np.isnan(y_data))
                ax.scatter(x_data[valid_mask], y_data[valid_mask], 
                          alpha=0.3, s=10, color='steelblue')
                
                # 添加相关系数
                corr = corr_matrix[i, j]
                ax.text(0.05, 0.95, f'r={corr:.3f}',
                       transform=ax.transAxes,
                       verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            # 设置标签
            if i == len(trait_names) - 1:
                ax.set_xlabel(trait_names[j], fontsize=8)
            else:
                ax.set_xticklabels([])
            
            if j == 0:
                ax.set_ylabel(trait_names[i], fontsize=8)
            else:
                ax.set_yticklabels([])
            
            ax.tick_params(labelsize=6)
    
    plt.tight_layout()
    output_file3 = f'output/phenotype_scatter_matrix_{timestamp}.png'
    plt.savefig(output_file3, dpi=150, bbox_inches='tight')
    print(f"✓ 保存散点图矩阵: {output_file3}")
    plt.close()
    
    # 6. 保存NPZ文件
    npz_file = f'output/phenotypes_{timestamp}.npz'
    loader.save_to_numpy(cleaned_data_iqr, sample_ids, trait_names, npz_file)
    
    # 7. 测试加载所有性状
    print("\n【测试6: 加载所有可用性状】")
    all_trait_data, all_sample_ids, all_trait_names = loader.load_trait_data(
        filepath='Training_data/1_Training_Trait_Data_2014_2023.csv',
        traits=None,  # 加载所有可用性状
        handle_missing='median'
    )
    
    print(f"\n所有性状加载结果:")
    print(f"  - 数据形状: {all_trait_data.shape}")
    print(f"  - 加载的性状: {all_trait_names}")
    
    # 8. 环境过滤测试
    print("\n【测试7: 按环境过滤】")
    env_filtered_data, env_sample_ids, env_trait_names = loader.load_trait_data(
        filepath='Training_data/1_Training_Trait_Data_2014_2023.csv',
        traits=['Yield_Mg_ha'],
        filter_by_env=['DEH1_2014', 'NEH1_2014'],  # 只加载这两个环境
        handle_missing='drop'
    )
    
    print(f"\n环境过滤结果:")
    print(f"  - 数据形状: {env_filtered_data.shape}")
    print(f"  - 包含的环境示例:")
    unique_envs = set([sid.split('_')[0] + '_' + sid.split('_')[1] for sid in env_sample_ids[:10]])
    for env in unique_envs:
        print(f"    - {env}")
    
    print("\n" + "=" * 80)
    print("测试完成！")
    print("=" * 80)
    
    # 返回测试结果供进一步分析
    return {
        'trait_data': trait_data,
        'cleaned_data_iqr': cleaned_data_iqr,
        'sample_ids': sample_ids,
        'trait_names': trait_names,
        'outlier_info': outlier_info_iqr,
        'correlation': corr_df
    }


if __name__ == "__main__":
    results = test_phenotype_loader()
