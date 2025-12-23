#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
测试仅使用EC数据加载环境信息
"""
import os
import sys

# 添加项目路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from openge.data import GxEDataLoader

def test_ec_only():
    """测试仅使用EC数据"""
    print("=" * 80)
    print("测试: 仅使用EC数据加载环境信息")
    print("=" * 80)
    
    loader = GxEDataLoader()
    
    # 只加载EC数据，不加载天气数据
    env_data, env_ids, env_features = loader.load_environment(
        ec_file='Training_data/6_Training_EC_Data_2014_2023.csv',
        weather_file=None,  # 不使用天气数据
        soil_file=None      # 不使用土壤数据
    )
    
    print(f"\n结果:")
    print(f"  - 环境数据形状: {env_data.shape}")
    print(f"  - 环境样本数: {len(env_ids)}")
    print(f"  - 特征数: {len(env_features)}")
    print(f"\n前5个环境ID: {env_ids[:5]}")
    print(f"前10个特征名: {env_features[:10]}")
    
    return True

def test_soil_only():
    """测试仅使用土壤数据"""
    print("\n" + "=" * 80)
    print("测试: 仅使用土壤数据加载环境信息")
    print("=" * 80)
    
    loader = GxEDataLoader()
    
    # 只加载土壤数据
    env_data, env_ids, env_features = loader.load_environment(
        soil_file='Training_data/3_Training_Soil_Data_2015_2023.csv',
        weather_file=None,
        ec_file=None
    )
    
    print(f"\n结果:")
    print(f"  - 环境数据形状: {env_data.shape}")
    print(f"  - 环境样本数: {len(env_ids)}")
    print(f"  - 特征数: {len(env_features)}")
    print(f"\n前5个环境ID: {env_ids[:5]}")
    print(f"特征名: {env_features}")
    
    return True

def test_ec_and_soil():
    """测试同时使用EC和土壤数据"""
    print("\n" + "=" * 80)
    print("测试: 同时使用EC和土壤数据")
    print("=" * 80)
    
    loader = GxEDataLoader()
    
    # 同时加载EC和土壤数据
    env_data, env_ids, env_features = loader.load_environment(
        ec_file='Training_data/6_Training_EC_Data_2014_2023.csv',
        soil_file='Training_data/3_Training_Soil_Data_2015_2023.csv',
        weather_file=None
    )
    
    print(f"\n结果:")
    print(f"  - 环境数据形状: {env_data.shape}")
    print(f"  - 环境样本数: {len(env_ids)}")
    print(f"  - 特征数: {len(env_features)}")
    print(f"\n前5个环境ID: {env_ids[:5]}")
    print(f"前20个特征名: {env_features[:20]}")
    
    return True

def test_weather_and_ec():
    """测试3D天气数据与EC数据组合"""
    print("\n" + "=" * 80)
    print("测试: 3D天气数据 + EC数据")
    print("=" * 80)
    
    loader = GxEDataLoader()
    
    env_data, env_ids, env_features = loader.load_environment(
        weather_file='Training_data/4_Training_Weather_Data_2014_2023_full_year.csv',
        ec_file='Training_data/6_Training_EC_Data_2014_2023.csv',
        use_3d=True
    )
    
    print(f"\n结果:")
    print(f"  - 3D环境数据形状: {env_data.shape}")
    print(f"  - 环境样本数: {len(env_ids)}")
    print(f"  - 天气特征数: {len(env_features)}")
    
    if loader.static_environment_data is not None:
        print(f"  - 静态EC特征数: {len(loader.static_environment_features)}")
        print(f"  - 静态EC数据形状: {loader.static_environment_data.shape}")
    
    return True

def test_full_pipeline_with_ec():
    """测试完整管道（仅使用EC数据）"""
    print("\n" + "=" * 80)
    print("测试: 完整管道 - 仅使用EC作为环境数据")
    print("=" * 80)
    
    loader = GxEDataLoader()
    
    # 1. 加载基因型
    gen_data, gen_ids, gen_markers = loader.load_genetic(
        filepath='Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt'
    )
    print(f"基因型: {gen_data.shape}")
    
    # 2. 加载EC作为环境数据
    env_data, env_ids, env_features = loader.load_environment(
        ec_file='Training_data/6_Training_EC_Data_2014_2023.csv'
    )
    print(f"EC环境数据: {env_data.shape}")
    
    # 3. 加载表型
    pheno_data, pheno_ids, pheno_traits = loader.load_phenotype(
        filepath='Training_data/1_Training_Trait_Data_2014_2023.csv'
    )
    print(f"表型: {pheno_data.shape}")
    
    # 4. 样本对齐
    aligned_gen, aligned_env, aligned_pheno, aligned_ids = loader.align_samples()
    
    print(f"\n对齐后:")
    print(f"  - 样本数: {len(aligned_ids)}")
    print(f"  - 基因型: {aligned_gen.shape}")
    print(f"  - EC环境: {aligned_env.shape}")
    print(f"  - 表型: {aligned_pheno.shape}")
    
    return True

if __name__ == '__main__':
    print("\n" + "=" * 80)
    print("灵活环境数据加载测试")
    print("=" * 80 + "\n")
    
    # 测试1: 仅EC数据
    test_ec_only()
    
    # 测试2: 仅土壤数据
    test_soil_only()
    
    # 测试3: EC + 土壤
    test_ec_and_soil()
    
    # 测试4: 天气 + EC (3D + 静态)
    test_weather_and_ec()
    
    # 测试5: 完整管道（仅EC）
    test_full_pipeline_with_ec()
    
    print("\n" + "=" * 80)
    print("所有测试完成!")
    print("=" * 80)
