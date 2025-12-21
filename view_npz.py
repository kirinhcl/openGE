"""查看 NPZ 文件内容的实用脚本"""

import numpy as np
import sys
from pathlib import Path


def view_npz(filepath):
    """查看 NPZ 文件内容
    
    Args:
        filepath: NPZ 文件路径
    """
    path = Path(filepath)
    
    if not path.exists():
        print(f"❌ 文件不存在: {filepath}")
        return
    
    if path.suffix != '.npz':
        print(f"⚠️ 警告: 文件扩展名不是 .npz")
    
    print("=" * 70)
    print(f"📦 NPZ 文件: {path.name}")
    print("=" * 70)
    
    # 加载 NPZ 文件
    data = np.load(filepath, allow_pickle=True)
    
    # 显示包含的数组名称
    print(f"\n包含的数组: {len(data.files)} 个")
    print("-" * 70)
    
    for i, key in enumerate(data.files, 1):
        arr = data[key]
        
        print(f"\n{i}. '{key}'")
        print(f"   类型: {type(arr).__name__}")
        
        if isinstance(arr, np.ndarray):
            print(f"   形状: {arr.shape}")
            print(f"   数据类型: {arr.dtype}")
            print(f"   大小: {arr.size:,} 个元素")
            print(f"   内存: {arr.nbytes / 1024:.2f} KB")
            
            # 显示数据范围（如果是数值类型）
            if np.issubdtype(arr.dtype, np.number):
                print(f"   最小值: {np.nanmin(arr):.4f}")
                print(f"   最大值: {np.nanmax(arr):.4f}")
                print(f"   平均值: {np.nanmean(arr):.4f}")
                
                # 检查 NaN
                n_nan = np.isnan(arr).sum()
                if n_nan > 0:
                    print(f"   ⚠️ NaN 数量: {n_nan:,} ({100*n_nan/arr.size:.2f}%)")
            
            # 显示预览
            if arr.ndim == 1:
                preview = arr[:5]
                print(f"   前5个值: {preview}")
            elif arr.ndim == 2:
                print(f"   前3行:")
                for row in arr[:3]:
                    print(f"      {row[:5]}...")
            elif arr.ndim == 3:
                print(f"   第1个样本前3行:")
                for row in arr[0][:3]:
                    print(f"      {row[:5]}...")
            else:
                print(f"   数据维度: {arr.ndim}D")
        else:
            # 非数组类型（如列表、字符串等）
            if hasattr(arr, '__len__'):
                print(f"   长度: {len(arr)}")
                if len(arr) <= 10:
                    print(f"   内容: {arr}")
                else:
                    print(f"   前10项: {arr[:10]}")
            else:
                print(f"   值: {arr}")
    
    print("\n" + "=" * 70)
    print("✓ 查看完成")
    print("=" * 70)
    
    data.close()


def list_npz_files(directory="."):
    """列出目录中的所有 NPZ 文件
    
    Args:
        directory: 目录路径
    """
    path = Path(directory)
    npz_files = list(path.glob("**/*.npz"))
    
    if not npz_files:
        print(f"未找到 NPZ 文件: {directory}")
        return
    
    print("=" * 70)
    print(f"📁 目录: {path.absolute()}")
    print(f"找到 {len(npz_files)} 个 NPZ 文件")
    print("=" * 70)
    
    for i, file in enumerate(npz_files, 1):
        size = file.stat().st_size / 1024  # KB
        print(f"{i}. {file.relative_to(path)}")
        print(f"   大小: {size:.2f} KB")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python view_npz.py <npz_file>           # 查看指定文件")
        print("  python view_npz.py --list [directory]   # 列出目录中的所有 NPZ 文件")
        print("\n示例:")
        print("  python view_npz.py output/weather_3d_20251221_024235.npz")
        print("  python view_npz.py --list output/")
        sys.exit(1)
    
    if sys.argv[1] == "--list":
        directory = sys.argv[2] if len(sys.argv) > 2 else "."
        list_npz_files(directory)
    else:
        view_npz(sys.argv[1])
