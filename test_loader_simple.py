"""简化的测试脚本：直接导入模块测试数据加载"""

import sys
import importlib.util
import numpy as np
import pandas as pd
from pathlib import Path

# 手动加载 preprocess 模块
spec_preprocess = importlib.util.spec_from_file_location('preprocess', 'openge/data/preprocess.py')
preprocess = importlib.util.module_from_spec(spec_preprocess)
sys.modules['openge.data.preprocess'] = preprocess
spec_preprocess.loader.exec_module(preprocess)

# 手动加载 environment 模块
spec_env = importlib.util.spec_from_file_location('environment', 'openge/data/loaders/environment.py')
environment = importlib.util.module_from_spec(spec_env)
spec_env.loader.exec_module(environment)

print("=" * 70)
print("测试 1：初始化 EnvironmentLoader")
print("=" * 70)
loader = environment.EnvironmentLoader()
print("✓ EnvironmentLoader 初始化成功\n")

print("=" * 70)
print("测试 2：加载天气数据")
print("=" * 70)
weather_file = 'Testing_data/4_Testing_Weather_Data_2024_seasons_only.csv'
try:
    df_weather = loader.load_weather_data(
        weather_file, 
        handle_missing='drop',
        missing_threshold=0.5
    )
    print(f"\n✓ 天气数据加载成功")
    print(f"  形状: {df_weather.shape}")
    print(f"  列: {list(df_weather.columns[:5])}... (共{len(df_weather.columns)}列)")
    print(f"  样本: {df_weather['Env'].unique()[:3]}... (共{df_weather['Env'].nunique()}个)")
except Exception as e:
    print(f"\n✗ 天气数据加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试 3：加载土壤数据")
print("=" * 70)
soil_file = 'Testing_data/3_Testing_Soil_Data_2024.csv'
try:
    df_soil = loader.load_soil_data(
        soil_file,
        handle_missing='drop',
        missing_threshold=0.5
    )
    print(f"\n✓ 土壤数据加载成功")
    print(f"  形状: {df_soil.shape}")
    print(f"  列: {list(df_soil.columns[:5])}... (共{len(df_soil.columns)}列)")
    print(f"  样本: {df_soil['Env'].unique()[:3]}... (共{df_soil['Env'].nunique()}个)")
except Exception as e:
    print(f"\n✗ 土壤数据加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试 4：加载 EC 数据")
print("=" * 70)
ec_file = 'Testing_data/6_Testing_EC_Data_2024.csv'
try:
    df_ec = loader.load_ec_data(
        ec_file,
        handle_missing='drop',
        missing_threshold=0.5
    )
    print(f"\n✓ EC 数据加载成功")
    print(f"  形状: {df_ec.shape}")
    print(f"  列: {list(df_ec.columns[:5])}... (共{len(df_ec.columns)}列)")
    print(f"  样本: {df_ec['Env'].unique()[:3]}... (共{df_ec['Env'].nunique()}个)")
except Exception as e:
    print(f"\n✗ EC 数据加载失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试 5：转换天气数据为 3D 数组")
print("=" * 70)
try:
    weather_3d, sample_ids, feature_names = loader.load_weather_data_3d(
        weather_file,
        handle_missing='drop'
    )
    print(f"\n✓ 3D 天气数据转换成功")
    print(f"  形状: {weather_3d.shape}")
    print(f"  样本数: {len(sample_ids)}")
    print(f"  特征数: {len(feature_names)}")
    print(f"  样本: {sample_ids[:3]}")
    print(f"  特征: {feature_names[:5]}... (共{len(feature_names)}个)")
except Exception as e:
    print(f"\n✗ 3D 转换失败: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "=" * 70)
print("测试完成")
print("=" * 70)
