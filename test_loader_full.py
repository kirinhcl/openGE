"""完整的数据加载器测试 - 整合基因型、环境和表型数据"""

import sys
sys.path.insert(0, '/Users/lu/openGE')

from openge.data import GxEDataLoader, GxEDataset
import numpy as np
import matplotlib.pyplot as plt


def test_full_dataloader():
    """测试完整的GxE数据加载流程"""
    
    print("\n" + "=" * 80)
    print("完整 GxE 数据加载器测试")
    print("=" * 80)
    
    # 初始化数据加载器
    loader = GxEDataLoader()
    
    # 1. 加载基因型数据
    print("\n【步骤1: 加载基因型数据】")
    genetic_data, genetic_ids, marker_names = loader.load_genetic(
        filepath='Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt',
        sample_col='<Marker>',
        handle_missing='mean',
        missing_threshold=0.5,
        maf_threshold=0.05,  # 过滤MAF < 0.05的标记
        encoding='keep'  # 保持dosage编码 [0, 0.5, 1]
    )
    
    # 2. 加载环境数据
    print("\n【步骤2: 加载环境数据】")
    environment_data, env_ids, env_features = loader.load_environment(
        weather_file='Training_data/4_Training_Weather_Data_2014_2023_full_year.csv',
        use_3d=True,  # 使用3D时间序列数据
        handle_missing='drop'
    )
    
    # 3. 加载表型数据（目标变量）
    print("\n【步骤3: 加载表型数据 (目标变量)】")
    phenotype_data, pheno_ids, trait_names = loader.load_phenotype(
        filepath='Training_data/1_Training_Trait_Data_2014_2023.csv',
        traits=['Yield_Mg_ha'],  # 只加载产量性状作为目标
        sample_id_col='Hybrid',
        env_col='Env',
        handle_missing='drop',
        filter_by_env=None,  # 加载所有环境
        handle_outliers=True,
        outlier_method='iqr'
    )
    
    # 4. 打印数据摘要
    loader.print_summary()
    
    # 5. 对齐样本
    print("\n【步骤4: 样本对齐】")
    aligned_genetic, aligned_env, aligned_pheno, aligned_ids = loader.align_samples(
        strategy='inner'  # 只保留三者都有的样本
    )
    
    # 6. 创建数据集
    print("\n【步骤5: 创建PyTorch Dataset】")
    dataset = loader.create_dataset(
        genetic_data=aligned_genetic,
        environment_data=aligned_env,
        phenotype_data=aligned_pheno,
        sample_ids=aligned_ids,
        return_dict=True  # 返回字典格式
    )
    
    print(f"✓ 数据集创建成功")
    print(f"  - 总样本数: {len(dataset)}")
    print(f"  - 数据形状:")
    shapes = dataset.get_shapes()
    for key, shape in shapes.items():
        if shape is not None and key != 'n_samples':
            print(f"    {key}: {shape}")
    
    # 7. 划分训练/验证/测试集
    print("\n【步骤6: 数据集划分】")
    train_dataset, val_dataset, test_dataset = loader.split_dataset(
        dataset,
        train_ratio=0.7,
        val_ratio=0.15,
        test_ratio=0.15,
        random_seed=42
    )
    
    # 8. 创建DataLoader
    print("\n【步骤7: 创建DataLoader】")
    train_loader = loader.create_dataloader(
        train_dataset,
        batch_size=32,
        shuffle=True,
        num_workers=0
    )
    
    val_loader = loader.create_dataloader(
        val_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    test_loader = loader.create_dataloader(
        test_dataset,
        batch_size=32,
        shuffle=False,
        num_workers=0
    )
    
    print(f"✓ DataLoader创建成功")
    print(f"  - 训练批次数: {len(train_loader)}")
    print(f"  - 验证批次数: {len(val_loader)}")
    print(f"  - 测试批次数: {len(test_loader)}")
    
    # 9. 测试数据加载
    print("\n【步骤8: 测试批次加载】")
    batch = next(iter(train_loader))
    
    print(f"\n第一个批次:")
    print(f"  - 基因型形状: {batch['genetic'].shape}")
    print(f"  - 环境形状: {batch['environment'].shape}")
    print(f"  - 表型形状 (TARGET): {batch['phenotype'].shape}")
    print(f"  - 样本ID数量: {len(batch['sample_id'])}")
    
    print(f"\n数据类型:")
    print(f"  - 基因型: {batch['genetic'].dtype}")
    print(f"  - 环境: {batch['environment'].dtype}")
    print(f"  - 表型: {batch['phenotype'].dtype}")
    
    print(f"\n数据范围:")
    print(f"  - 基因型: [{batch['genetic'].min():.3f}, {batch['genetic'].max():.3f}]")
    print(f"  - 环境: [{batch['environment'].min():.3f}, {batch['environment'].max():.3f}]")
    print(f"  - 表型: [{batch['phenotype'].min():.3f}, {batch['phenotype'].max():.3f}]")
    
    # 10. 可视化数据分布
    print("\n【步骤9: 生成可视化】")
    fig = plt.figure(figsize=(15, 10))
    
    # 基因型分布
    ax1 = plt.subplot(2, 3, 1)
    genetic_sample = batch['genetic'][0].numpy().flatten()[:1000]  # 前1000个标记
    ax1.hist(genetic_sample, bins=50, edgecolor='black', alpha=0.7)
    ax1.set_xlabel('Genotype Value')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Genotype Distribution (First Sample)')
    
    # 环境时间序列
    ax2 = plt.subplot(2, 3, 2)
    env_sample = batch['environment'][0].numpy()  # (timesteps, features)
    for i in range(min(5, env_sample.shape[1])):  # 前5个特征
        ax2.plot(env_sample[:, i], alpha=0.7, label=f'Feature {i+1}')
    ax2.set_xlabel('Time Steps')
    ax2.set_ylabel('Value')
    ax2.set_title('Environment Time Series (First Sample)')
    ax2.legend(fontsize=8)
    
    # 表型分布
    ax3 = plt.subplot(2, 3, 3)
    pheno_all = batch['phenotype'].numpy().flatten()
    ax3.hist(pheno_all, bins=30, edgecolor='black', alpha=0.7, color='green')
    ax3.set_xlabel('Yield (Mg/ha)')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Phenotype Distribution (Target)')
    ax3.axvline(pheno_all.mean(), color='r', linestyle='--', label=f'Mean={pheno_all.mean():.2f}')
    ax3.legend()
    
    # 基因型热图
    ax4 = plt.subplot(2, 3, 4)
    genetic_batch = batch['genetic'][:16].numpy()[:, :100]  # 前16样本，前100标记
    im = ax4.imshow(genetic_batch, aspect='auto', cmap='viridis', vmin=0, vmax=1)
    ax4.set_xlabel('Marker Index')
    ax4.set_ylabel('Sample Index')
    ax4.set_title('Genotype Heatmap (Batch)')
    plt.colorbar(im, ax=ax4)
    
    # 环境热图
    ax5 = plt.subplot(2, 3, 5)
    env_batch = batch['environment'][:16].numpy()[:, :, 0]  # 前16样本，第1个特征
    im = ax5.imshow(env_batch, aspect='auto', cmap='coolwarm')
    ax5.set_xlabel('Time Steps')
    ax5.set_ylabel('Sample Index')
    ax5.set_title('Environment Heatmap (Feature 1)')
    plt.colorbar(im, ax=ax5)
    
    # 表型vs样本索引
    ax6 = plt.subplot(2, 3, 6)
    ax6.scatter(range(len(pheno_all)), pheno_all, alpha=0.6)
    ax6.set_xlabel('Sample Index in Batch')
    ax6.set_ylabel('Yield (Mg/ha)')
    ax6.set_title('Phenotype Values in Batch')
    ax6.grid(alpha=0.3)
    
    plt.tight_layout()
    output_file = 'output/full_dataloader_test.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"✓ 保存可视化: {output_file}")
    plt.close()
    
    # 11. 统计信息
    print("\n【步骤10: 数据统计】")
    print(f"\n训练集统计:")
    train_phenotypes = []
    for batch in train_loader:
        train_phenotypes.append(batch['phenotype'].numpy())
    train_phenotypes = np.concatenate(train_phenotypes)
    print(f"  - 产量均值: {train_phenotypes.mean():.2f} Mg/ha")
    print(f"  - 产量标准差: {train_phenotypes.std():.2f} Mg/ha")
    print(f"  - 产量范围: [{train_phenotypes.min():.2f}, {train_phenotypes.max():.2f}] Mg/ha")
    
    print("\n" + "=" * 80)
    print("测试完成！数据已准备好用于深度学习训练")
    print("=" * 80)
    
    return {
        'loader': loader,
        'train_loader': train_loader,
        'val_loader': val_loader,
        'test_loader': test_loader,
        'dataset': dataset
    }


if __name__ == "__main__":
    results = test_full_dataloader()
