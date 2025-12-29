# Environment Data Loader - 使用说明

## 📋 概述

`EnvironmentLoader` 是一个强大的环境数据加载器，支持加载和预处理天气、土壤和 EC 数据。

## 🏗️ 架构

```
openge/data/
├── preprocess.py          # 预处理函数
│   ├── check_and_handle_missing()
│   ├── aggregate_temporal_to_static()
│   ├── aggregate_temporal_features()
│   ├── get_growth_stages()
│   └── CROP_GROWTH_STAGES
└── loaders/
    └── environment.py     # 数据加载器
        └── EnvironmentLoader
```

## ✨ 主要功能

### 1. 数据加载

#### 天气数据
- ✅ 自动检测日期格式（YYYYMMDD 整数或字符串）
- ✅ 处理不同时间步长度（自动填充到最大值）
- ✅ 支持 2D 静态聚合或 3D 时间序列
- ✅ 缺失值处理（删除/填充）

```python
# 加载静态天气数据（聚合后）
df_weather = loader.load_weather_data(
    'weather.csv',
    handle_missing='drop',
    missing_threshold=0.5
)

# 加载 3D 时间序列天气数据
weather_3d, sample_ids, features = loader.load_weather_data_3d(
    'weather.csv',
    handle_missing='drop'
)
```

#### 土壤数据
- ✅ 自动检测日期列
- ✅ 过滤数值列
- ✅ 缺失值处理

```python
df_soil = loader.load_soil_data(
    'soil.csv',
    handle_missing='drop',
    missing_threshold=0.5
)
```

#### EC 数据
- ✅ 支持宽格式数据
- ✅ 缺失值处理

```python
df_ec = loader.load_ec_data(
    'ec.csv',
    handle_missing='drop',
    missing_threshold=0.5
)
```

### 2. 数据合并

```python
# 合并多个数据源（按样本ID内连接）
df_combined = loader._merge_dataframes(
    [df_weather, df_soil, df_ec],
    sample_col='Env',
    how='inner'
)
```

### 3. 预处理函数

#### 缺失值处理
```python
from openge.data.preprocess import check_and_handle_missing

df_clean = check_and_handle_missing(
    df,
    method='drop',  # 'drop', 'mean', 'forward_fill', 'backward_fill', 'interpolate'
    threshold=0.5,  # 删除缺失率超过50%的列
    name='数据名称'
)
```

#### 时间序列聚合
```python
from openge.data.preprocess import aggregate_temporal_to_static

# 3D → 2D（mean, max, min, std）
static_data = aggregate_temporal_to_static(weather_3d)
```

#### 生育期聚合
```python
from openge.data.preprocess import (
    aggregate_temporal_features,
    CROP_GROWTH_STAGES
)

# 按作物生育期聚合
features, names = aggregate_temporal_features(
    weather_3d,
    temporal_windows='maize',  # 或 'wheat', 'rice'
    return_feature_names=True
)

# 自定义时间窗口
features = aggregate_temporal_features(
    weather_3d,
    temporal_windows=[30, 60, 90]  # 30天、60天、90天窗口
)
```

## 📊 测试结果

### 测试数据统计
- **天气数据**: 23 样本 × 64 特征（聚合后）
- **土壤数据**: 20 样本 × 24 特征
- **EC 数据**: 22 样本 × 654 特征
- **合并数据**: 17 样本 × 742 特征（14个共同环境）
- **3D 天气**: (23, 259, 11) - 23样本 × 259时间步 × 11特征

### 数据特点

#### 天气数据
- 时间序列数据，每个环境有不同的时间步数（173-262步）
- 自动填充到最大时间步数（262步）
- 部分特征缺失率超过50%（如 GWETTOP），被自动删除
- 日期格式：YYYYMMDD 整数（20240411）

#### 土壤数据
- 静态数据，可能有重复环境（多次测量）
- 有日期列（'Date Received'）但用于记录而非时间序列
- 部分列缺失率很高（>95%），自动删除
- 列名包含空格和特殊字符

#### EC 数据
- 宽格式数据，655列特征
- 无缺失值
- 22个环境

## 🚀 快速开始

### 完整示例

```python
import sys
import importlib.util

# 手动加载模块（如果无法直接 import）
spec_preprocess = importlib.util.spec_from_file_location(
    'preprocess', 
    'openge/data/preprocess.py'
)
preprocess = importlib.util.module_from_spec(spec_preprocess)
sys.modules['openge.data.preprocess'] = preprocess
spec_preprocess.loader.exec_module(preprocess)

spec_env = importlib.util.spec_from_file_location(
    'environment', 
    'openge/data/loaders/environment.py'
)
environment = importlib.util.module_from_spec(spec_env)
spec_env.loader.exec_module(environment)

# 初始化
loader = environment.EnvironmentLoader()

# 加载数据
df_weather = loader.load_weather_data('weather.csv', handle_missing='drop')
df_soil = loader.load_soil_data('soil.csv', handle_missing='drop')
df_ec = loader.load_ec_data('ec.csv', handle_missing='drop')

# 合并
df_combined = loader._merge_dataframes(
    [df_weather, df_soil, df_ec],
    sample_col='Env'
)

# 保存
df_combined.to_csv('combined_data.csv', index=False)

# 加载 3D 天气数据
weather_3d, sample_ids, features = loader.load_weather_data_3d(
    'weather.csv',
    handle_missing='drop'
)

# 保存 3D 数据
import numpy as np
np.savez('weather_3d.npz', 
         weather_3d=weather_3d,
         sample_ids=sample_ids,
         feature_names=features)
```

## 🔧 测试脚本

提供了两个测试脚本：

### 1. `test_loader_simple.py`
- 测试各个数据加载功能
- 显示详细的数据信息

### 2. `test_loader_full.py`
- 完整的加载、合并、保存流程
- 生成输出文件到 `output/` 目录
- 输出文件：
  - `combined_environment_data_YYYYMMDD_HHMMSS.csv`
  - `weather_3d_YYYYMMDD_HHMMSS.npz`
  - `weather_static_YYYYMMDD_HHMMSS.csv`

运行测试：
```bash
python test_loader_simple.py
python test_loader_full.py
```

## ⚙️ 配置选项

### 缺失值处理方法
- `'drop'`: 删除有缺失值的行（默认）
- `'mean'`: 用均值填充
- `'forward_fill'`: 向前填充
- `'backward_fill'`: 向后填充
- `'interpolate'`: 线性插值
- `'none'`: 不处理

### 缺失值阈值
- `missing_threshold=0.5`: 删除缺失率超过50%的列

### 合并方式
- `how='inner'`: 内连接，只保留共同环境（默认）
- `how='outer'`: 外连接，保留所有环境

## 📝 注意事项

1. **日期格式**: 自动检测 YYYYMMDD 整数格式（如 20240411）
2. **时间步不一致**: 自动填充到最大时间步数
3. **高缺失率列**: 默认删除缺失率>50%的列
4. **样本ID**: 默认使用 'Env' 列作为样本标识
5. **内存占用**: 3D 数据较大，注意内存使用

## 🐛 已知问题

1. **TXH1_2024 缺失**: 在 EC 数据中有此环境，但在最终合并时丢失（需检查原因）
2. **土壤数据重复**: 某些环境有多行数据（不同深度或时间），当前取所有行

## 📚 数据结构

### 天气数据 CSV 格式
```csv
Env,Date,PS,RH2M,T2MWET,T2M,...
DEH1_2024,20240411,100.78,92.38,15.23,15.85,...
DEH1_2024,20240412,99.87,79.44,14.12,15.99,...
```

### 土壤数据 CSV 格式
```csv
Year,Env,LabID,Date Received,1:1 Soil pH,...
2024,GAH1_2024,Ward Laboratories Inc,3/18/2024,6.9,...
```

### EC 数据 CSV 格式
```csv
Env,HI30_pGerEme,HI30_pEmeEnJ,...
DEH1_2024,0.123,0.456,...
```

## 🎯 下一步计划

- [ ] 优化合并逻辑（处理重复环境）
- [ ] 添加更多缺失值处理方法
- [ ] 支持外连接合并
- [ ] 添加数据验证功能
- [ ] 优化内存使用
- [ ] 添加数据可视化
