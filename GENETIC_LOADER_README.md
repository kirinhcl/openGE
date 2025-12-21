# 基因型数据加载器使用说明

## 概述

`GeneticLoader` 是一个用于加载和处理基因型SNP标记数据的工具类，支持多种文件格式和数据预处理功能。

## 测试结果

✅ **成功加载数据集**: 5,899个样本 × 2,425个SNP标记
✅ **缺失值处理**: 3.18%缺失率，自动清理和填充
✅ **数据质量**: 范围[0, 1]，均值0.7064，标准差0.3179
✅ **染色体覆盖**: 10条染色体 (S1-S10)
✅ **MAF分析**: 2.75%标记MAF<0.05，已自动过滤
✅ **输出文件**: 生成2.65MB的NPZ压缩文件和6图可视化

## 功能特性

### 1. 加载数值化基因型文件
- **格式**: 制表符/空格分隔的文本文件
- **编码**: 0 (纯合参考), 0.5 (杂合), 1 (纯合替代)
- **缺失值**: 自动识别NA并处理

### 2. 缺失值处理策略
- `mean`: 用每个标记的均值填充（推荐）
- `zero`: 用0填充
- `drop`: 删除有缺失值的样本

### 3. 质量控制
- 自动删除缺失率>50%的标记
- MAF过滤 (默认阈值0.05)
- 数据范围验证

### 4. 数据输出
- NPZ压缩格式 (包含基因型矩阵、样本ID、标记名称)
- 综合可视化 (6图分析)

## 使用示例

### 基本用法

```python
from openge.data.loaders.genetic import GeneticLoader

# 初始化加载器
loader = GeneticLoader()

# 加载基因型数据
genotypes, sample_ids, marker_names = loader.load_from_numerical_file(
    filepath='Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt',
    sample_col='<Marker>',         # 样本ID列名
    handle_missing='mean',         # 缺失值处理方法
    missing_threshold=0.5          # 标记缺失率阈值
)

# 结果
# genotypes: (5899, 2397) numpy数组
# sample_ids: ['01CSI6/LH287', '01DIB2/LH287', ...]
# marker_names: ['S1_1007742', 'S1_1020677', ...]
```

### MAF过滤

```python
# 过滤低频标记
filtered_genotypes, filtered_markers = loader.filter_markers(
    genotype_matrix=genotypes,
    marker_names=marker_names,
    maf_threshold=0.05  # MAF < 0.05的标记将被移除
)

# 过滤后: (5899, 2331)
```

### 保存到NPZ

```python
loader.save_to_numpy(
    genotype_matrix=genotypes,
    sample_ids=sample_ids,
    marker_names=marker_names,
    output_path='output/genotypes.npz'
)
```

### 加载NPZ文件

```python
import numpy as np

# 加载保存的基因型数据
data = np.load('output/genotypes.npz')
genotypes = data['genotypes']       # (5899, 2397)
sample_ids = data['sample_ids']     # 样本ID列表
marker_names = data['marker_names'] # 标记名称列表
```

## 数据文件格式

### 输入格式

```
<Marker>        S1_1007742  S1_1020677  S1_2018002  ...
01CSI6/LH287    0.5         1.0         0.0         ...
01DIB2/LH287    1.0         0.5         NA          ...
...
```

- 第一列: 样本ID (杂交组合格式 "亲本1/亲本2")
- 其他列: SNP标记值
- 标记命名: `S{染色体号}_{位置}` (例如: S1_1007742 = 1号染色体1007742位点)
- 值: 0, 0.5, 1, NA

### 输出格式

NPZ文件包含3个数组:
- `genotypes`: float32数组 (样本数 × 标记数)
- `sample_ids`: 字符串数组 (样本数,)
- `marker_names`: 字符串数组 (标记数,)

## 统计信息

### 数据集概况
- **样本数**: 5,899
- **原始标记数**: 2,425
- **过滤后标记数**: 2,397 (删除28个高缺失率标记)
- **MAF过滤后**: 2,331 (删除66个低频标记)

### 染色体分布
| 染色体 | 标记数 |
|--------|--------|
| S1     | 369    |
| S2     | 339    |
| S3     | 244    |
| S4     | 214    |
| S5     | 316    |
| S6     | 157    |
| S7     | 201    |
| S8     | 261    |
| S9     | 171    |
| S10    | 125    |

### 数据质量
- **范围**: [0.0, 1.0]
- **均值**: 0.7064 ± 0.3179
- **缺失率**: 3.18% (原始) → 0% (处理后)
- **MAF均值**: 0.2936
- **MAF < 0.05**: 2.75%

### 遗传多样性
- **等位基因频率**: 0.7064 ± 0.1262
- **杂合度**: 根据样本变化 (见可视化图)

## 可视化输出

测试脚本生成6张分析图:

1. **基因型热图**: 展示样本×标记矩阵
2. **基因型值分布**: 0, 0.5, 1三种基因型的频次
3. **等位基因频率分布**: 标记的等位基因频率
4. **MAF分布**: 次要等位基因频率分布
5. **样本间遗传相似度**: 相关系数矩阵
6. **杂合度分析**: 每个样本的杂合位点比例

## VCF文件支持

加载器还支持标准VCF格式:

```python
genotypes, sample_ids, marker_names = loader.load_from_vcf(
    filepath='path/to/file.vcf',
    encoding='dosage'  # 'dosage' 或 'additive'
)
```

### 编码方式
- `dosage`: 0/0→0, 0/1→0.5, 1/1→1 (推荐)
- `additive`: 0/0→0, 0/1→1, 1/1→2

## 运行测试

```bash
# 完整测试（包含可视化）
python test_genetic_loader_standalone.py

# 快速测试（仅加载）
python -c "from openge.data.loaders.genetic import GeneticLoader; \
           loader = GeneticLoader(); \
           g, s, m = loader.load_from_numerical_file('Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt'); \
           print(f'Loaded: {g.shape}')"
```

## 性能指标

- **加载速度**: ~2-3秒 (5899样本 × 2425标记)
- **内存占用**: ~55 MB (float32数组)
- **压缩比**: 原文件 → NPZ 约10:1
- **NPZ文件大小**: 2.65 MB

## 常见问题

### Q1: 如何处理不同的样本ID列名?
A: 使用 `sample_col` 参数指定列名，例如 `sample_col='Sample_ID'`

### Q2: 缺失值太多怎么办?
A: 调整 `missing_threshold` 参数，例如 `missing_threshold=0.3` 会删除缺失率>30%的标记

### Q3: 如何只加载部分染色体?
A: 加载后根据标记名称过滤:
```python
chr1_mask = [m.startswith('S1_') for m in marker_names]
chr1_genotypes = genotypes[:, chr1_mask]
```

### Q4: 如何转换为additive编码(0/1/2)?
A: 使用 `encode_genotypes` 方法:
```python
additive = loader.encode_genotypes(genotypes, encoding='additive')
```

## 相关文件

- 主代码: [openge/data/loaders/genetic.py](openge/data/loaders/genetic.py)
- 测试脚本: [test_genetic_loader_standalone.py](test_genetic_loader_standalone.py)
- 训练数据: `Training_data/5_Genotype_Data_All_2014_2025_Hybrids_numerical.txt`
- 输出示例: `output/genotypes_*.npz`, `output/genetic_visualization_*.png`

## 下一步

- [ ] 整合到数据集类 (`openge/data/dataset.py`)
- [ ] 添加基因型×环境交互分析
- [ ] 支持PLINK格式 (.bed/.bim/.fam)
- [ ] 实现基因组预测 (GBLUP/RKHS)
- [ ] 添加连锁不平衡(LD)分析

## 更新日志

### 2024-12-21
- ✅ 实现 `load_from_numerical_file()` 方法
- ✅ 添加缺失值处理和质量控制
- ✅ 实现MAF过滤
- ✅ 添加NPZ保存功能
- ✅ 创建综合测试脚本
- ✅ 成功加载5899样本数据集
- ✅ 生成6图可视化分析
