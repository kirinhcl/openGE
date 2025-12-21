"""
测试脚本：用于测试 environment.py 的功能
测试数据文件：
- 3_Testing_Soil_Data_2024.csv
- 4_Testing_Weather_Data_2024_seasons_only.csv
- 6_Testing_EC_Data_2024.csv

功能：
1. 加载三个数据文件
2. 天气文件不做聚合（保留时间序列）
3. 合并三个数据源
4. 输出为 CSV 文件
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys
from datetime import datetime

# ✅ 【修改】导入 EnvironmentLoader 类
# 方式1：如果 environment.py 在 openge/data/loaders/ 目录
sys.path.insert(0, str(Path(__file__).parent / 'openge'))
from openge.data import EnvironmentLoader


def test_environment_loader():
    """测试 EnvironmentLoader 类"""
    print("\n" + "=" * 60)
    print("TEST 0: EnvironmentLoader 类初始化")
    print("=" * 60)
    
    try:
        loader = EnvironmentLoader()
        print("✓ EnvironmentLoader 类初始化成功")
        return loader
    except Exception as e:
        print(f"✗ EnvironmentLoader 类初始化失败: {e}")
        return None


def test_load_weather_data(loader):
    """测试加载天气数据"""
    print("\n" + "=" * 60)
    print("TEST 1a: 使用 EnvironmentLoader 加载天气数据")
    print("=" * 60)
    
    if loader is None:
        print("✗ Loader 未初始化")
        return None
    
    try:
        weather_file = "Testing_data/4_Testing_Weather_Data_2024_seasons_only.csv"
        df_weather = loader.load_weather_data(
            weather_file,
            handle_missing='drop',
            missing_threshold=0.5
        )
        print(f"\n✓ 天气数据加载成功")
        print(f"  - 形状: {df_weather.shape}")
        print(f"  - 列名: {list(df_weather.columns[:5])}...")
        print(f"\n数据预览:")
        print(df_weather.head())
        return df_weather
    except Exception as e:
        print(f"✗ 天气数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_load_soil_data(loader):
    """测试加载土壤数据"""
    print("\n" + "=" * 60)
    print("TEST 1b: 使用 EnvironmentLoader 加载土壤数据")
    print("=" * 60)
    
    if loader is None:
        print("✗ Loader 未初始化")
        return None
    
    try:
        soil_file = "Testing_data/3_Testing_Soil_Data_2024.csv"
        df_soil = loader.load_soil_data(
            soil_file,
            handle_missing='drop',
            missing_threshold=0.5
        )
        print(f"\n✓ 土壤数据加载成功")
        print(f"  - 形状: {df_soil.shape}")
        print(f"  - 列名: {list(df_soil.columns[:5])}...")
        print(f"\n数据预览:")
        print(df_soil.head())
        return df_soil
    except Exception as e:
        print(f"✗ 土壤数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_load_ec_data(loader):
    """测试加载 EC 数据"""
    print("\n" + "=" * 60)
    print("TEST 1c: 使用 EnvironmentLoader 加载 EC 数据")
    print("=" * 60)
    
    if loader is None:
        print("✗ Loader 未初始化")
        return None
    
    try:
        ec_file = "Testing_data/6_Testing_EC_Data_2024.csv"
        df_ec = loader.load_ec_data(
            ec_file,
            handle_missing='drop',
            missing_threshold=0.5
        )
        print(f"\n✓ EC 数据加载成功")
        print(f"  - 形状: {df_ec.shape}")
        print(f"  - 列名: {list(df_ec.columns[:5])}...")
        print(f"\n数据预览:")
        print(df_ec.head())
        return df_ec
    except Exception as e:
        print(f"✗ EC 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_load_all_environment_data(loader):
    """测试加载所有环境数据"""
    print("\n" + "=" * 60)
    print("TEST 2: 使用 EnvironmentLoader 加载所有环境数据")
    print("=" * 60)
    
    if loader is None:
        print("✗ Loader 未初始化")
        return None
    
    try:
        df_combined = loader.load_all_environment_data(
            weather_file='Testing_data/4_Testing_Weather_Data_2024_seasons_only.csv',
            soil_file='Testing_data/3_Testing_Soil_Data_2024.csv',
            ec_file='Testing_data/6_Testing_EC_Data_2024.csv',
            sample_col='Env',
            handle_missing='drop',
            missing_threshold=0.5
        )
        print(f"\n✓ 所有环境数据加载成功")
        print(f"  - 形状: {df_combined.shape}")
        print(f"  - 列名 (前10列): {list(df_combined.columns[:10])}...")
        print(f"\n数据预览:")
        print(df_combined.head())
        return df_combined
    except Exception as e:
        print(f"✗ 所有环境数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return None


def test_data_loading():
    """测试数据文件的加载"""
    print("=" * 60)
    print("TEST 3: 直接使用 Pandas 加载数据文件")
    print("=" * 60)
    
    # 定义数据文件路径
    soil_file = "Testing_data/3_Testing_Soil_Data_2024.csv"
    weather_file = "Testing_data/4_Testing_Weather_Data_2024_seasons_only.csv"
    ec_file = "Testing_data/6_Testing_EC_Data_2024.csv"
    
    try:
        # 加载土壤数据
        soil_df = pd.read_csv(soil_file)
        print(f"\n✓ 土壤数据加载成功")
        print(f"  - 文件行数: {len(soil_df)}")
        print(f"  - 文件列数: {len(soil_df.columns)}")
        print(f"  - 环境ID数量: {soil_df['Env'].nunique()}")
        print(f"  - 环境ID: {sorted(soil_df['Env'].unique())}")
        
        # 加载天气数据
        weather_df = pd.read_csv(weather_file)
        print(f"\n✓ 天气数据加载成功")
        print(f"  - 文件行数: {len(weather_df)}")
        print(f"  - 文件列数: {len(weather_df.columns)}")
        print(f"  - 环境ID数量: {weather_df['Env'].nunique()}")
        print(f"  - 环境ID: {sorted(weather_df['Env'].unique())}")
        
        # 加载EC数据
        ec_df = pd.read_csv(ec_file)
        print(f"\n✓ EC数据加载成功")
        print(f"  - 文件行数: {len(ec_df)}")
        print(f"  - 文件列数: {len(ec_df.columns)}")
        print(f"  - 环境ID数量: {ec_df['Env'].nunique()}")
        print(f"  - 环境ID: {sorted(ec_df['Env'].unique())}")
        
        return soil_df, weather_df, ec_df
        
    except FileNotFoundError as e:
        print(f"✗ 文件加载失败: {e}")
        return None, None, None


def test_soil_data_analysis(soil_df):
    """测试土壤数据分析"""
    print("\n" + "=" * 60)
    print("TEST 4: 土壤数据分析")
    print("=" * 60)
    
    if soil_df is None:
        print("✗ 土壤数据未加载")
        return
    
    # 基本统计
    numeric_cols = soil_df.select_dtypes(include=[np.number]).columns
    print(f"\n数值列: {list(numeric_cols)}")
    
    print("\n土壤数据统计:")
    print(soil_df[numeric_cols].describe())
    
    # 按环境分组统计
    print("\n按环境ID分组的平均值:")
    grouped = soil_df.groupby('Env')[numeric_cols].mean()
    print(grouped)
    
    # 检查缺失值
    print("\n缺失值统计:")
    missing = soil_df.isnull().sum()
    print(missing[missing > 0])


def test_weather_data_analysis(weather_df):
    """测试天气数据分析"""
    print("\n" + "=" * 60)
    print("TEST 5: 天气数据分析")
    print("=" * 60)
    
    if weather_df is None:
        print("✗ 天气数据未加载")
        return
    
    # 基本统计
    numeric_cols = weather_df.select_dtypes(include=[np.number]).columns
    print(f"\n数值列: {list(numeric_cols)}")
    
    print("\n天气数据统计:")
    print(weather_df[numeric_cols].describe())
    
    # 按环境分组统计
    print("\n按环境ID分组的平均值:")
    grouped = weather_df.groupby('Env')[numeric_cols].mean()
    print(grouped)
    
    # 日期范围
    print(f"\n数据日期范围:")
    print(f"  - 最早日期: {weather_df['Date'].min()}")
    print(f"  - 最晚日期: {weather_df['Date'].max()}")


def test_ec_data_analysis(ec_df):
    """测试 EC 数据分析"""
    print("\n" + "=" * 60)
    print("TEST 6: EC 数据分析")
    print("=" * 60)
    
    if ec_df is None:
        print("✗ EC 数据未加载")
        return
    
    print(f"\n环境ID及其数据行数:")
    env_counts = ec_df['Env'].value_counts().sort_index()
    print(env_counts)
    
    # 查看主要列
    print(f"\n所有列名:")
    for i, col in enumerate(ec_df.columns, 1):
        print(f"  {i:2d}. {col}")
    
    # 数值列统计
    numeric_cols = ec_df.select_dtypes(include=[np.number]).columns
    print(f"\n数值列统计:")
    print(ec_df[numeric_cols].describe())


def test_data_consistency(soil_df, weather_df, ec_df):
    """测试数据一致性"""
    print("\n" + "=" * 60)
    print("TEST 7: 数据一致性检查")
    print("=" * 60)
    
    # 获取每个数据源的环境ID
    soil_envs = set(soil_df['Env'].unique()) if soil_df is not None else set()
    weather_envs = set(weather_df['Env'].unique()) if weather_df is not None else set()
    ec_envs = set(ec_df['Env'].unique()) if ec_df is not None else set()
    
    print(f"\n土壤数据环境: {sorted(soil_envs)}")
    print(f"天气数据环境: {sorted(weather_envs)}")
    print(f"EC 数据环境: {sorted(ec_envs)}")
    
    # 检查交集
    common_envs = soil_envs & weather_envs & ec_envs
    print(f"\n三个数据源共有的环境: {sorted(common_envs)}")
    print(f"共有环境数量: {len(common_envs)}")
    
    # 检查差异
    print(f"\n仅在土壤数据中的环境: {sorted(soil_envs - weather_envs - ec_envs)}")
    print(f"仅在天气数据中的环境: {sorted(weather_envs - soil_envs - ec_envs)}")
    print(f"仅在 EC 数据中的环境: {sorted(ec_envs - soil_envs - weather_envs)}")


def test_merge_data(soil_df, weather_df, ec_df):
    """测试数据合并"""
    print("\n" + "=" * 60)
    print("TEST 8: 数据合并测试")
    print("=" * 60)
    
    # 找一个共有的环境
    soil_envs = set(soil_df['Env'].unique()) if soil_df is not None else set()
    weather_envs = set(weather_df['Env'].unique()) if weather_df is not None else set()
    ec_envs = set(ec_df['Env'].unique()) if ec_df is not None else set()
    common_envs = soil_envs & weather_envs & ec_envs
    
    if not common_envs:
        print("✗ 没有共有的环境")
        return
    
    test_env = sorted(common_envs)[0]
    
    try:
        soil_subset = soil_df[soil_df['Env'] == test_env].copy()
        weather_subset = weather_df[weather_df['Env'] == test_env].copy()
        ec_subset = ec_df[ec_df['Env'] == test_env].copy()
        
        print(f"\n环境 {test_env} 的数据行数:")
        print(f"  - 土壤数据: {len(soil_subset)} 行")
        print(f"  - 天气数据: {len(weather_subset)} 行")
        print(f"  - EC 数据: {len(ec_subset)} 行")
        
        if len(soil_subset) > 0:
            print(f"\n土壤数据样本 (前3行):")
            print(soil_subset.head(3))
        
        if len(weather_subset) > 0:
            print(f"\n天气数据样本 (前3行):")
            print(weather_subset.head(3))
        
        if len(ec_subset) > 0:
            print(f"\nEC 数据样本 (前3行):")
            print(ec_subset.head(3))
            
    except Exception as e:
        print(f"✗ 数据合并失败: {e}")
        import traceback
        traceback.print_exc()


def convert_weather_to_3d(df_weather: pd.DataFrame, 
                          sample_col: str = 'Env', 
                          date_col: str = 'Date') -> tuple:
    """
    将长格式天气 DataFrame 转换为 3D NumPy 数组
    
    Parameters:
        df_weather: 长格式天气数据
        sample_col: 样本ID列名
        date_col: 日期列名
    
    Returns:
        tuple: (weather_3d, sample_ids, feature_names)
               weather_3d 形状: (n_samples, n_timesteps, n_features)
    """
    # 获取特征列（排除样本ID和日期）
    feature_cols = [col for col in df_weather.columns 
                   if col not in [sample_col, date_col]]
    
    # 按样本分组并排序
    grouped_data = []
    sample_ids = []
    
    for sample_id, group in df_weather.groupby(sample_col):
        group_sorted = group.sort_values(date_col)
        features = group_sorted[feature_cols].values
        grouped_data.append(features)
        sample_ids.append(sample_id)
    
    # 检查时间步数
    timesteps = [len(t) for t in grouped_data]
    if len(set(timesteps)) > 1:
        print(f"⚠️ 时间步数不一致: {set(timesteps)}")
        # 填充到最大时间步数
        max_t = max(timesteps)
        padded = []
        for data in grouped_data:
            if len(data) < max_t:
                pad = np.repeat(data[-1:], max_t - len(data), axis=0)
                data = np.vstack([data, pad])
            padded.append(data)
        grouped_data = padded
    
    # 转换为 3D 数组
    weather_3d = np.array(grouped_data, dtype=np.float32)
    
    print(f"✓ 转换为 3D 数组:")
    print(f"  - 形状: {weather_3d.shape}")
    print(f"  - 维度: (样本={len(sample_ids)}, 时间步={weather_3d.shape[1]}, 特征={len(feature_cols)})")
    
    return weather_3d, sample_ids, feature_cols


def load_all_data_and_merge(loader, output_dir="output"):
    """
    ✅ 【新增】加载所有数据并合并
    
    Parameters:
        loader: EnvironmentLoader 实例
        output_dir: 输出目录
    
    Returns:
        tuple: (df_weather, df_soil, df_ec, df_combined)
    """
    
    print("\n" + "=" * 70)
    print("加载三个数据文件（天气文件不做聚合）")
    print("=" * 70)
    
    # 创建输出目录
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    try:
        # ✅ 【关键】加载天气数据，reshape_to_temporal=True 保留原始时间序列
        print("\n📍 第1步：加载天气数据（保留时间序列，不做聚合）")
        print("-" * 70)
        df_weather = loader.load_weather_data(
            filepath="Testing_data/4_Testing_Weather_Data_2024_seasons_only.csv",
            reshape_to_temporal=True,  # ✅ 保留时间序列
            sample_col="Env",
            date_col="Date",
            handle_missing='drop',
            missing_threshold=0.5
        )
        print(f"\n✓ 天气数据加载完成")
        print(f"  - 形状: {df_weather.shape}")
        
        # ✅ 【新增】转换为 3D 数组
        print("\n  转换为 3D 数组...")
        weather_3d, sample_ids, feature_names = convert_weather_to_3d(
            df_weather, 
            sample_col='Env', 
            date_col='Date'
        )
        print(f"  - 3D 形状: {weather_3d.shape}")
        print(f"  - 样本数: {len(sample_ids)}")
        print(f"  - 时间步: {weather_3d.shape[1]}")
        print(f"  - 特征数: {weather_3d.shape[2]}")
        
        # ✅ 加载土壤数据
        print("\n\n📍 第2步：加载土壤数据")
        print("-" * 70)
        df_soil = loader.load_soil_data(
            filepath="Testing_data/3_Testing_Soil_Data_2024.csv",
            handle_missing='drop',
            missing_threshold=0.5,
            sample_col='Env'
        )
        print(f"\n✓ 土壤数据加载完成")
        print(f"  - 形状: {df_soil.shape}")
        print(f"  - 列数: {len(df_soil.columns)}")
        print(f"  - 环境数: {df_soil['Env'].nunique()}")
        print(f"  - 样本ID: {sorted(df_soil['Env'].unique())}")
        print(f"\n土壤数据预览（前3行）:")
        print(df_soil.head(3))
        
        # ✅ 加载 EC 数据
        print("\n\n📍 第3步：加载 EC 数据")
        print("-" * 70)
        df_ec = loader.load_ec_data(
            filepath="Testing_data/6_Testing_EC_Data_2024.csv",
            handle_missing='drop',
            missing_threshold=0.5,
            sample_col='Env'
        )
        print(f"\n✓ EC 数据加载完成")
        print(f"  - 形状: {df_ec.shape}")
        print(f"  - 列数: {len(df_ec.columns)}")
        print(f"  - 环境数: {df_ec['Env'].nunique()}")
        print(f"  - 样本ID: {sorted(df_ec['Env'].unique())}")
        print(f"\nEC 数据预览（前3行）:")
        print(df_ec.head(3))
        
        # ✅ 数据一致性检查
        print("\n\n📍 第4步：数据一致性检查")
        print("-" * 70)
        weather_envs = set(df_weather['Env'].unique())
        soil_envs = set(df_soil['Env'].unique())
        ec_envs = set(df_ec['Env'].unique())
        
        print(f"\n天气数据环境: {sorted(weather_envs)}")
        print(f"土壤数据环境: {sorted(soil_envs)}")
        print(f"EC 数据环境: {sorted(ec_envs)}")
        
        common_envs = weather_envs & soil_envs & ec_envs
        print(f"\n✓ 三个数据源共有环境: {sorted(common_envs)}")
        print(f"  共有环境数量: {len(common_envs)}")
        
        # ✅ 合并数据
        print("\n\n📍 第5步：合并三个数据源")
        print("-" * 70)
        
        # 先合并土壤和 EC 数据（静态数据，1行/样本）
        print("\n  5.1 合并土壤和 EC 数据...")
        df_merged = df_soil.merge(df_ec, on='Env', how='inner')
        print(f"      ✓ 合并后形状: {df_merged.shape}")
        
        # 再合并天气数据（时间序列数据，多行/样本）
        print("\n  5.2 合并天气数据...")
        df_combined = df_weather.merge(df_merged, on='Env', how='inner')
        print(f"      ✓ 合并后形状: {df_combined.shape}")
        
        print(f"\n✓ 数据合并完成")
        print(f"  - 总行数: {len(df_combined)}")
        print(f"  - 总列数: {len(df_combined.columns)}")
        print(f"  - 环境数: {df_combined['Env'].nunique()}")
        
        print(f"\n合并后数据预览（前5行）:")
        print(df_combined.head(5))
        
        # ✅ 保存为 CSV 文件
        print("\n\n📍 第6步：保存为 CSV 文件")
        print("-" * 70)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file = output_path / f"combined_environment_data_{timestamp}.csv"
        
        df_combined.to_csv(output_file, index=False)
        
        print(f"\n✓ 数据已保存")
        print(f"  - 文件路径: {output_file}")
        print(f"  - 文件大小: {output_file.stat().st_size / 1024:.2f} KB")
        print(f"  - 行数: {len(df_combined)}")
        print(f"  - 列数: {len(df_combined.columns)}")
        
        # ✅ 显示数据摘要
        print("\n\n📋 数据摘要")
        print("=" * 70)
        print(f"\n天气数据:")
        print(f"  - 行数: {len(df_weather)}")
        print(f"  - 列数: {len(df_weather.columns)}")
        print(f"  - 时间范围: {df_weather['Date'].min()} 到 {df_weather['Date'].max()}")
        print(f"  - 环境数: {df_weather['Env'].nunique()}")
        
        print(f"\n土壤数据:")
        print(f"  - 行数: {len(df_soil)}")
        print(f"  - 列数: {len(df_soil.columns)}")
        print(f"  - 环境数: {df_soil['Env'].nunique()}")
        
        print(f"\nEC 数据:")
        print(f"  - 行数: {len(df_ec)}")
        print(f"  - 列数: {len(df_ec.columns)}")
        print(f"  - 环境数: {df_ec['Env'].nunique()}")
        
        print(f"\n合并后数据:")
        print(f"  - 行数: {len(df_combined)}")
        print(f"  - 列数: {len(df_combined.columns)}")
        print(f"  - 环境数: {df_combined['Env'].nunique()}")
        print(f"  - 输出文件: {output_file.name}")
        
        return df_weather, df_soil, df_ec, df_combined
        
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return None, None, None, None


def main():
    """主测试函数"""
    print("\n")
    print("╔" + "=" * 68 + "╗")
    print("║" + " " * 68 + "║")
    print("║" + "  加载三个数据文件并输出为 CSV".center(68) + "║")
    print("║" + " " * 68 + "║")
    print("╚" + "=" * 68 + "╝")
    
    # 初始化 EnvironmentLoader
    print("\n初始化 EnvironmentLoader...")
    try:
        loader = EnvironmentLoader()
        print("✓ EnvironmentLoader 初始化成功")
    except Exception as e:
        print(f"✗ EnvironmentLoader 初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ✅ 【关键】加载数据并合并
    df_weather, df_soil, df_ec, df_combined = load_all_data_and_merge(
        loader, 
        output_dir="output"
    )
    
    if df_combined is not None:
        print("\n\n" + "=" * 70)
        print("✓ 成功！数据已保存到 output 目录")
        print("=" * 70)
        
        # 显示输出文件列表
        output_path = Path("output")
        csv_files = list(output_path.glob("*.csv"))
        if csv_files:
            print(f"\n生成的 CSV 文件:")
            for i, file in enumerate(csv_files, 1):
                print(f"  {i}. {file.name} ({file.stat().st_size / 1024:.2f} KB)")
    else:
        print("\n❌ 数据加载失败")


if __name__ == "__main__":
    main()