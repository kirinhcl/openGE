"""测试加载全年天气数据并可视化"""

import sys
import importlib.util
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from datetime import datetime

# 手动加载 preprocess 模块
spec_preprocess = importlib.util.spec_from_file_location('preprocess', 'openge/data/preprocess.py')
preprocess = importlib.util.module_from_spec(spec_preprocess)
sys.modules['openge.data.preprocess'] = preprocess
spec_preprocess.loader.exec_module(preprocess)

# 手动加载 environment 模块
spec_env = importlib.util.spec_from_file_location('environment', 'openge/data/loaders/environment.py')
environment = importlib.util.module_from_spec(spec_env)
spec_env.loader.exec_module(environment)

print("\n" + "=" * 70)
print("全年天气数据加载与可视化测试")
print("=" * 70)

# 创建输出目录
output_dir = Path("output")
output_dir.mkdir(exist_ok=True)

# 初始化 loader
loader = environment.EnvironmentLoader()
print("✓ EnvironmentLoader 初始化成功\n")

# 文件路径 - 使用全年数据
weather_file = 'Testing_data/4_Testing_Weather_Data_2024_full_year.csv'

print("=" * 70)
print("步骤 1: 加载全年天气数据（3D格式）")
print("=" * 70)

# 加载 3D 天气数据
weather_3d, sample_ids, feature_names = loader.load_weather_data_3d(
    weather_file,
    handle_missing='drop'
)

print(f"\n✓ 3D 天气数据加载成功")
print(f"  - 形状: {weather_3d.shape}")
print(f"  - 样本数: {len(sample_ids)}")
print(f"  - 时间步: {weather_3d.shape[1]} 天")
print(f"  - 特征数: {len(feature_names)}")
print(f"\n样本ID: {sample_ids}")
print(f"\n特征名称: {feature_names}")

print("\n" + "=" * 70)
print("步骤 2: 数据统计信息")
print("=" * 70)

print(f"\n数据类型: {weather_3d.dtype}")
print(f"数据范围: [{weather_3d.min():.2f}, {weather_3d.max():.2f}]")
print(f"缺失值数量: {np.isnan(weather_3d).sum()}")
print(f"缺失值比例: {np.isnan(weather_3d).sum() / weather_3d.size * 100:.2f}%")

# 计算每个特征的统计信息
print("\n各特征统计:")
for i, feat_name in enumerate(feature_names):
    feat_data = weather_3d[:, :, i]
    print(f"  {feat_name:15s}: mean={np.nanmean(feat_data):8.2f}, "
          f"std={np.nanstd(feat_data):8.2f}, "
          f"min={np.nanmin(feat_data):8.2f}, "
          f"max={np.nanmax(feat_data):8.2f}")

print("\n" + "=" * 70)
print("步骤 3: 保存 NPZ 文件")
print("=" * 70)

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
npz_file = output_dir / f"weather_full_year_3d_{timestamp}.npz"

np.savez(
    npz_file,
    weather_3d=weather_3d,
    sample_ids=sample_ids,
    feature_names=feature_names
)

print(f"\n✓ 保存 NPZ 文件: {npz_file}")
print(f"  - 文件大小: {npz_file.stat().st_size / 1024:.1f} KB")

print("\n" + "=" * 70)
print("步骤 4: 可视化数据")
print("=" * 70)

# 创建可视化
fig = plt.figure(figsize=(16, 10))

# 1. 显示第一个样本的所有特征时间序列
ax1 = plt.subplot(3, 2, 1)
sample_idx = 0
for feat_idx in range(min(5, len(feature_names))):  # 显示前5个特征
    ax1.plot(weather_3d[sample_idx, :, feat_idx], 
             label=feature_names[feat_idx], alpha=0.7, linewidth=1)
ax1.set_title(f'样本 {sample_ids[sample_idx]} - 前5个特征的时间序列', fontsize=10)
ax1.set_xlabel('时间步（天）')
ax1.set_ylabel('数值')
ax1.legend(fontsize=8)
ax1.grid(True, alpha=0.3)

# 2. 温度特征（如果存在）的热力图
ax2 = plt.subplot(3, 2, 2)
temp_keywords = ['T2M', 'temp', 'temperature']
temp_idx = None
for i, name in enumerate(feature_names):
    if any(kw in name for kw in temp_keywords):
        temp_idx = i
        break

if temp_idx is not None:
    im = ax2.imshow(weather_3d[:, :, temp_idx], aspect='auto', cmap='RdYlBu_r')
    ax2.set_title(f'所有样本的 {feature_names[temp_idx]} 热力图', fontsize=10)
    ax2.set_xlabel('时间步（天）')
    ax2.set_ylabel('样本索引')
    ax2.set_yticks(range(len(sample_ids)))
    ax2.set_yticklabels(sample_ids, fontsize=6)
    plt.colorbar(im, ax=ax2)
else:
    ax2.text(0.5, 0.5, '未找到温度特征', ha='center', va='center')
    ax2.set_title('温度热力图（未找到）', fontsize=10)

# 3. 对比多个样本的某一特征
ax3 = plt.subplot(3, 2, 3)
feat_idx = 0  # 第一个特征
for sample_idx in range(min(5, len(sample_ids))):  # 显示前5个样本
    ax3.plot(weather_3d[sample_idx, :, feat_idx], 
             label=sample_ids[sample_idx], alpha=0.7, linewidth=1)
ax3.set_title(f'多样本对比 - {feature_names[feat_idx]}', fontsize=10)
ax3.set_xlabel('时间步（天）')
ax3.set_ylabel(feature_names[feat_idx])
ax3.legend(fontsize=8)
ax3.grid(True, alpha=0.3)

# 4. 特征间相关性（使用第一个样本）
ax4 = plt.subplot(3, 2, 4)
sample_data = weather_3d[0, :, :]  # (time, features)
# 计算特征间的相关系数
corr_matrix = np.corrcoef(sample_data.T)
im = ax4.imshow(corr_matrix, aspect='auto', cmap='coolwarm', vmin=-1, vmax=1)
ax4.set_title(f'特征相关性矩阵 ({sample_ids[0]})', fontsize=10)
ax4.set_xticks(range(len(feature_names)))
ax4.set_yticks(range(len(feature_names)))
ax4.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=7)
ax4.set_yticklabels(feature_names, fontsize=7)
plt.colorbar(im, ax=ax4)

# 5. 时间序列分布箱线图
ax5 = plt.subplot(3, 2, 5)
feat_idx = 0  # 第一个特征
# 选择几个时间点的数据分布
time_points = np.linspace(0, weather_3d.shape[1]-1, 5, dtype=int)
data_for_boxplot = [weather_3d[:, t, feat_idx] for t in time_points]
ax5.boxplot(data_for_boxplot, labels=[f'Day {t}' for t in time_points])
ax5.set_title(f'不同时间点的 {feature_names[feat_idx]} 分布', fontsize=10)
ax5.set_xlabel('时间点')
ax5.set_ylabel(feature_names[feat_idx])
ax5.grid(True, alpha=0.3)

# 6. 全局统计信息文本
ax6 = plt.subplot(3, 2, 6)
ax6.axis('off')
stats_text = f"""
数据统计信息

基本信息:
  • 样本数: {len(sample_ids)}
  • 时间步: {weather_3d.shape[1]} 天
  • 特征数: {len(feature_names)}
  • 数据类型: {weather_3d.dtype}

数据范围:
  • 最小值: {weather_3d.min():.2f}
  • 最大值: {weather_3d.max():.2f}
  • 平均值: {np.nanmean(weather_3d):.2f}
  • 标准差: {np.nanstd(weather_3d):.2f}

缺失值:
  • 数量: {np.isnan(weather_3d).sum()}
  • 比例: {np.isnan(weather_3d).sum() / weather_3d.size * 100:.2f}%

输出文件:
  • {npz_file.name}
"""
ax6.text(0.1, 0.9, stats_text, fontsize=9, family='monospace',
         verticalalignment='top', transform=ax6.transAxes)

plt.tight_layout()

# 保存图片
plot_file = output_dir / f"weather_full_year_visualization_{timestamp}.png"
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"\n✓ 保存可视化图片: {plot_file}")

plt.show()

print("\n" + "=" * 70)
print("测试完成！")
print("=" * 70)
print(f"\n输出文件:")
print(f"  1. NPZ 数据文件: {npz_file}")
print(f"  2. 可视化图片: {plot_file}")
print(f"\n可以使用 view.ipynb 笔记本进一步探索 NPZ 数据")
print("=" * 70)
