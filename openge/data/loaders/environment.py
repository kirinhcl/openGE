"""Loader for environmental data (weather, soil, EC data)."""

import numpy as np
import pandas as pd
from typing import List, Union, Tuple, Optional
from pathlib import Path

# 从 preprocess 模块导入预处理函数
from openge.data.preprocess import (
    check_and_handle_missing,
    aggregate_temporal_to_static,
    aggregate_temporal_features
)


class EnvironmentLoader:
    """Loader for environmental data (weather, soil, ec)."""
    
    def __init__(self):
        """Initialize environment data loader."""
        self.feature_names: Optional[List[str]] = None
    
    def _detect_data_format(self, data: Union[np.ndarray, pd.DataFrame]) -> str:
        """Detect if data is static or temporal."""
        if isinstance(data, pd.DataFrame):
            temporal_cols = [col for col in data.columns 
                           if any(keyword in col.lower() for keyword in ['date', 'time', 'day'])]
            return 'temporal' if temporal_cols else 'static'
        
        elif isinstance(data, np.ndarray):
            return 'temporal' if data.ndim == 3 else 'static'
        
        return 'static'
    
    def _is_long_format(self, df: pd.DataFrame) -> bool:
        """检查是否是长格式"""
        cols_lower = set(df.columns.str.lower())
        has_sample_col = any(col in cols_lower for col in ['sample_id', 'env', 'location'])
        has_feature_col = any(col in cols_lower for col in ['feature', 'variable', 'value'])
        
        return has_sample_col and has_feature_col
    
    def _is_temporal_wide_format(self, df: pd.DataFrame, 
                                sample_col: str = "Env",
                                date_col: str = "Date") -> Tuple[bool, Optional[str], Optional[str]]:
        """
        检查是否是时间序列的宽格式，并返回实际的列名。
        
        Returns:
            (is_temporal, actual_sample_col, actual_date_col)
        """
        cols_lower = {col.lower(): col for col in df.columns}
        sample_col_lower = sample_col.lower()
        date_col_lower = date_col.lower()
        
        if sample_col_lower in cols_lower and date_col_lower in cols_lower:
            return True, cols_lower[sample_col_lower], cols_lower[date_col_lower]
        
        return False, None, None
    
    def _convert_long_to_wide(self, df: pd.DataFrame) -> pd.DataFrame:
        """转换长格式到宽格式"""
        cols_lower = {col.lower(): col for col in df.columns}
        
        # 识别关键列
        sample_col = cols_lower.get('sample_id') or cols_lower.get('env') or cols_lower.get('location')
        feature_col = cols_lower.get('feature') or cols_lower.get('variable')
        value_col = cols_lower.get('value')
        date_col = cols_lower.get('date')
        
        if not (sample_col and feature_col and value_col):
            print("⚠️ 警告：缺少必需列，无法转换长格式")
            return df
        
        try:
            if date_col:
                result = df.pivot_table(
                    index=[sample_col, date_col],
                    columns=feature_col,
                    values=value_col,
                    aggfunc='first'
                )
            else:
                result = df.pivot_table(
                    index=sample_col,
                    columns=feature_col,
                    values=value_col,
                    aggfunc='first'
                )
            return result.reset_index()
        except Exception as e:
            print(f"⚠️ 转换失败: {e}")
            return df
    
    def _ensure_datetime_column(self, df: pd.DataFrame, date_col: str) -> pd.DataFrame:
        """
        确保日期列的数据类型正确

        Parameters:
            df: 输入 DataFrame
            date_col: 日期列名

        Returns:
            pd.DataFrame: 日期列已转换的 DataFrame
        """
        if date_col not in df.columns:
            raise ValueError(f"列 '{date_col}' 不存在")
        
        df = df.copy()
        
        # 检查日期列的当前数据类型
        current_dtype = df[date_col].dtype
        print(f"\n📅 日期列 '{date_col}' 处理:")
        print(f"   原始数据类型: {current_dtype}")
        
        # 获取样本值用于诊断
        sample_values = df[date_col].head(3).tolist()
        print(f"   样本值: {sample_values}")

        # 如果已经是 datetime 类型，直接返回
        if pd.api.types.is_datetime64_any_dtype(current_dtype):
            print(f"   ✓ 已经是 datetime 类型，无需转换")
            return df

        # 尝试转换为 datetime
        try:
            first_value = df[date_col].iloc[0]
            
            # ✅ 【修复】检测整数格式的日期（如 20240411）
            if pd.api.types.is_integer_dtype(current_dtype):
                # 检查是否是 YYYYMMDD 格式（8位整数）
                if 10000000 <= first_value <= 99999999:
                    print(f"   检测到 YYYYMMDD 整数格式")
                    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%Y%m%d')
                # 检查是否是 YYMMDD 格式（6位整数）
                elif 100000 <= first_value <= 999999:
                    print(f"   检测到 YYMMDD 整数格式")
                    df[date_col] = pd.to_datetime(df[date_col].astype(str), format='%y%m%d')
                else:
                    # 尝试自动解析
                    df[date_col] = pd.to_datetime(df[date_col].astype(str))
            
            # ✅ 【修复】检测字符串格式的日期
            elif pd.api.types.is_string_dtype(current_dtype) or current_dtype == object:
                first_str = str(first_value)
                
                # 尝试常见的日期格式
                date_formats = [
                    ('%Y%m%d', '20240411'),           # YYYYMMDD
                    ('%Y-%m-%d', '2024-04-11'),       # YYYY-MM-DD
                    ('%Y/%m/%d', '2024/04/11'),       # YYYY/MM/DD
                    ('%d-%m-%Y', '11-04-2024'),       # DD-MM-YYYY
                    ('%d/%m/%Y', '11/04/2024'),       # DD/MM/YYYY
                    ('%m-%d-%Y', '04-11-2024'),       # MM-DD-YYYY
                    ('%m/%d/%Y', '04/11/2024'),       # MM/DD/YYYY
                    ('%Y-%m-%d %H:%M:%S', '2024-04-11 00:00:00'),  # with time
                ]
                
                converted = False
                for fmt, example in date_formats:
                    try:
                        df[date_col] = pd.to_datetime(df[date_col], format=fmt)
                        print(f"   使用格式 '{fmt}' 转换成功")
                        converted = True
                        break
                    except (ValueError, TypeError):
                        continue
                
                # 如果所有格式都失败，尝试自动解析
                if not converted:
                    print(f"   尝试自动解析日期格式...")
                    df[date_col] = pd.to_datetime(df[date_col], infer_datetime_format=True)
            
            else:
                # 其他类型，尝试自动解析
                df[date_col] = pd.to_datetime(df[date_col])
            
            print(f"   ✓ 成功转换为 datetime 类型")
            
            # 显示日期范围
            date_min = df[date_col].min()
            date_max = df[date_col].max()
            print(f"   ✓ 日期范围: {date_min.strftime('%Y-%m-%d')} 到 {date_max.strftime('%Y-%m-%d')}")
            
            # ✅ 【新增】验证日期是否合理（1900-2100年之间）
            if date_min.year < 1900 or date_max.year > 2100:
                print(f"   ❌ 警告：日期范围异常！可能解析错误")
                raise ValueError(f"日期范围异常: {date_min} 到 {date_max}")
            
            return df

        except Exception as e:
            print(f"   ❌ 转换失败: {e}")
            print(f"   样本值: {df[date_col].head().tolist()}")
            raise ValueError(f"无法将列 '{date_col}' 转换为 datetime 类型: {e}")
    
    def _process_temporal_weather(self, df: pd.DataFrame,
                                 sample_col: str,
                                 date_col: str,
                                 reshape_to_temporal: bool) -> pd.DataFrame:
        """
        处理时间序列天气数据，返回 DataFrame
        """
        # 验证列名存在
        if sample_col not in df.columns:
            raise ValueError(f"列 '{sample_col}' 不存在。可用列: {list(df.columns)}")
        if date_col not in df.columns:
            raise ValueError(f"列 '{date_col}' 不存在。可用列: {list(df.columns)}")
        
        # 验证和转换日期列
        df = self._ensure_datetime_column(df, date_col)
        
        feature_cols = [col for col in df.columns 
                       if col not in [sample_col, date_col]]
        
        if not feature_cols:
            raise ValueError(f"没有特征列！排除了 {sample_col} 和 {date_col} 后没有其他列")
        
        grouped_data = []
        sample_ids = []
        
        for sample_id, group in df.groupby(sample_col):
            try:
                group_sorted = group.sort_values(date_col)
            except Exception as e:
                print(f"❌ 错误：无法按 '{date_col}' 排序样本 '{sample_id}': {e}")
                raise
        
            features = group_sorted[feature_cols].values
            grouped_data.append(features)
            sample_ids.append(sample_id)
        
        timesteps = [len(t) for t in grouped_data]
        
        if reshape_to_temporal:
            result_df = df.copy()
            return result_df
            
        else:
            if len(set(timesteps)) > 1:
                print(f"\n⚠️ 警告：样本的时间步数不同：{set(timesteps)}")
                print(f"   样本时间步数详情：")
                for sample_id, n_steps in zip(sample_ids, timesteps):
                    print(f"      • {sample_id}: {n_steps} 步")
                
                max_timesteps = max(timesteps)
                min_timesteps = min(timesteps)
                
                print(f"\n   处理方案：")
                print(f"   1. 删除时间步数过少的样本")
                print(f"   2. 截断到最小时间步数 ({min_timesteps})")
                print(f"   3. 填充最后值到最大时间步数 ({max_timesteps}) ⭐ 默认")
                print(f"\n   采用方案3：填充最后值到最大时间步数 ({max_timesteps})")
                
                padded_data = []
                for i, data in enumerate(grouped_data):
                    if len(data) < max_timesteps:
                        last_row = data[-1:, :]
                        n_pad = max_timesteps - len(data)
                        pad_rows = np.repeat(last_row, n_pad, axis=0)
                        padded = np.vstack([data, pad_rows])
                        padded_data.append(padded)
                        print(f"   ✓ {sample_ids[i]}: {len(data)} → {max_timesteps} 步 (填充 {n_pad} 行)")
                    else:
                        padded_data.append(data)
                
                grouped_data = padded_data
                print(f"   ✓ 所有样本已填充到 {max_timesteps} 步")
            
            temporal_array = np.array(grouped_data)
            
            n_nan_before = np.isnan(temporal_array).sum()
            if n_nan_before > 0:
                print(f"\n⚠️ 警告：原始数据中有 {n_nan_before} 个 NaN，会被忽略")
            
            aggregated = aggregate_temporal_to_static(temporal_array)
            
            n_features = len(feature_cols)
            feature_names = []
            for feat in feature_cols:
                feature_names.extend([f"{feat}_mean", f"{feat}_max", f"{feat}_min", f"{feat}_std"])
            
            result_df = pd.DataFrame(aggregated, columns=feature_names)
            result_df.insert(0, sample_col, sample_ids)
            
            return result_df

    def load_weather_data(self, filepath: str, 
                         reshape_to_temporal: bool = False,
                         sample_col: str = "Env",
                         date_col: str = "Date",
                         handle_missing: str = 'drop',
                         missing_threshold: float = 0.5) -> pd.DataFrame:
        """加载天气数据，返回 DataFrame"""
        path = Path(filepath)
        
        try:
            if path.suffix == ".csv":
                data = pd.read_csv(filepath)
            elif path.suffix in [".xlsx", ".xls"]:
                data = pd.read_excel(filepath)
            elif path.suffix == ".npz":
                loaded = np.load(filepath)
                weather_array = loaded["weather"].astype(np.float32)
                print("✓ NPZ 文件已加载（无列名信息）")
                return pd.DataFrame(weather_array)
            elif path.suffix == ".npy":
                weather_array = np.load(filepath).astype(np.float32)
                print("✓ NPY 文件已加载（无列名信息）")
                return pd.DataFrame(weather_array)
            else:
                raise ValueError(f"不支持的文件格式: {path.suffix}")
            
            if self._is_long_format(data):
                data = self._convert_long_to_wide(data)
            
            is_temporal, actual_sample_col, actual_date_col = self._is_temporal_wide_format(
                data, sample_col, date_col
            )
            if is_temporal:
                result_df = self._process_temporal_weather(
                    data, actual_sample_col, actual_date_col, reshape_to_temporal
                )
            else:
                numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
                if not numeric_cols:
                    raise ValueError("天气数据中没有数值列")
                result_df = data[numeric_cols].copy().astype(np.float32)
            
            result_df = check_and_handle_missing(
                result_df,
                method=handle_missing,
                threshold=missing_threshold,
                name='天气数据'
            )
            
            print(f"✓ 天气数据加载完成: {result_df.shape}")
            return result_df
    
        except Exception as e:
            print(f"❌ 加载天气数据失败: {e}")
            raise
    
    def load_soil_data(self, filepath: str,
                      handle_missing: str = 'drop',
                      missing_threshold: float = 0.5,
                      sample_col: str = 'Env',
                      date_col: str = None) -> pd.DataFrame:
        """加载土壤数据，返回 DataFrame"""
        path = Path(filepath)
        
        try:
            # 【步骤1】加载文件
            if path.suffix == ".csv":
                data = pd.read_csv(filepath)
            elif path.suffix in [".xlsx", ".xls"]:
                data = pd.read_excel(filepath)
            elif path.suffix == ".npz":
                loaded = np.load(filepath)
                soil_array = loaded["soil"].astype(np.float32)
                print("✓ NPZ 文件已加载（无列名信息）")
                return pd.DataFrame(soil_array)
            elif path.suffix == ".npy":
                soil_array = np.load(filepath).astype(np.float32)
                print("✓ NPY 文件已加载（无列名信息）")
                return pd.DataFrame(soil_array)
            else:
                raise ValueError(f"不支持的文件格式: {path.suffix}")
        
            # 【步骤2】自动检测日期列
            if date_col is None:
                possible_date_cols = [col for col in data.columns 
                                    if any(keyword in col.lower() for keyword in ['date', 'time', 'day'])]
                if possible_date_cols:
                    date_col = possible_date_cols[0]
                    print(f"📅 自动检测到日期列: '{date_col}'")
        
            # 【步骤3】保留样本ID列，提取数值列
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("土壤数据中没有数值列")
        
            # 【步骤4】构建包含样本ID列的完整 DataFrame
            if sample_col in data.columns:
                result_df = data[[sample_col] + numeric_cols].copy()
                result_df[numeric_cols] = result_df[numeric_cols].astype(np.float32)
            else:
                print(f"⚠️ 警告：未找到 '{sample_col}' 列")
                result_df = data[numeric_cols].copy().astype(np.float32)
        
            # 【步骤5】处理缺失值
            result_df = check_and_handle_missing(
                result_df,
                method=handle_missing,
                threshold=missing_threshold,
                name='土壤数据'
            )
        
            # 【步骤6】确保样本ID列在第一列
            if sample_col in result_df.columns:
                cols = [sample_col] + [c for c in result_df.columns if c != sample_col]
                result_df = result_df[cols]
        
            print(f"✓ 土壤数据加载完成: {result_df.shape}")
            return result_df

        except Exception as e:
            print(f"❌ 加载土壤数据失败: {e}")
            raise

    def load_ec_data(self, filepath: str,
                    handle_missing: str = 'drop',
                    missing_threshold: float = 0.5,
                    sample_col: str = 'Env',
                    date_col: str = None) -> pd.DataFrame:
        """加载 EC 数据，返回 DataFrame"""
        path = Path(filepath)
        
        try:
            # 【步骤1】加载文件
            if path.suffix == ".csv":
                data = pd.read_csv(filepath)
            elif path.suffix in [".xlsx", ".xls"]:
                data = pd.read_excel(filepath)
            elif path.suffix == ".npz":
                loaded = np.load(filepath)
                ec_array = loaded["ec"].astype(np.float32)
                print("✓ NPZ 文件已加载（无列名信息）")
                return pd.DataFrame(ec_array)
            elif path.suffix == ".npy":
                ec_array = np.load(filepath).astype(np.float32)
                print("✓ NPY 文件已加载（无列名信息）")
                return pd.DataFrame(ec_array)
            else:
                raise ValueError(f"不支持的文件格式: {path.suffix}")
            
            # 【步骤2】自动检测日期列
            if date_col is None:
                possible_date_cols = [col for col in data.columns 
                                    if any(keyword in col.lower() for keyword in ['date', 'time', 'day'])]
                if possible_date_cols:
                    date_col = possible_date_cols[0]
                    print(f"📅 自动检测到日期列: '{date_col}'")
            
            # 【步骤3】保留样本ID列，提取数值列
            numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
            if not numeric_cols:
                raise ValueError("EC 数据中没有数值列")
            
            # 【步骤4】构建包含样本ID列的完整 DataFrame
            if sample_col in data.columns:
                result_df = data[[sample_col] + numeric_cols].copy()
                result_df[numeric_cols] = result_df[numeric_cols].astype(np.float32)
            else:
                print(f"⚠️ 警告：未找到 '{sample_col}' 列")
                result_df = data[numeric_cols].copy().astype(np.float32)
            
            # 【步骤5】处理缺失值
            result_df = check_and_handle_missing(
                result_df,
                method=handle_missing,
                threshold=missing_threshold,
                name='EC数据'
            )
            
            # 【步骤6】确保样本ID列在第一列
            if sample_col in result_df.columns:
                cols = [sample_col] + [c for c in result_df.columns if c != sample_col]
                result_df = result_df[cols]
            
            print(f"✓ EC 数据加载完成: {result_df.shape}")
            return result_df

        except Exception as e:
            print(f"❌ 加载 EC 数据失败: {e}")
            raise
    
    def load_all_environment_data(self, 
                                  weather_file: str,
                                  soil_file: str,
                                  ec_file: str,
                                  temporal_weather_file: str = None,
                                  temporal_windows: Union[list, str, dict] = None,
                                  crop_name: str = 'maize',
                                  sample_col: str = 'Env',
                                  handle_missing: str = 'drop',
                                  missing_threshold: float = 0.5) -> pd.DataFrame:
        """一次性加载所有环境数据，返回合并后的 DataFrame"""
        print(f"正在加载环境数据...")
        
        df_weather = self.load_weather_data(
            weather_file, 
            sample_col=sample_col,
            handle_missing=handle_missing,
            missing_threshold=missing_threshold
        )
        df_soil = self.load_soil_data(
            soil_file,
            handle_missing=handle_missing,
            missing_threshold=missing_threshold
        )
        df_ec = self.load_ec_data(
            ec_file,
            handle_missing=handle_missing,
            missing_threshold=missing_threshold
        )
        
        print(f"\n✓ 基础数据加载完成: weather{df_weather.shape}, soil{df_soil.shape}, ec{df_ec.shape}")
        
        df_combined = self._merge_dataframes(
            [df_weather, df_soil, df_ec], 
            sample_col=sample_col
        )
        
        print(f"✓ 基础数据合并完成: {df_combined.shape}")
        
        if temporal_weather_file is not None and temporal_windows is not None:
            temporal_data = np.load(temporal_weather_file)
            
            if temporal_data.ndim != 3:
                raise ValueError(f"时间序列数据应为 3D，但得到 {temporal_data.ndim}D")
            
            n_samples_combined = len(df_combined)
            n_samples_temporal = temporal_data.shape[0]
            if n_samples_temporal != n_samples_combined:
                raise ValueError(f"时间序列数据样本数不匹配！")
            
            aggregated, feature_names = aggregate_temporal_features(
                temporal_data, temporal_windows, crop_name, return_feature_names=True
            )
            
            df_temporal = pd.DataFrame(aggregated, columns=feature_names)
            df_combined = pd.concat([df_combined, df_temporal], axis=1)
            print(f"✓ 时间聚合完成: 添加 {df_temporal.shape[1]} 列")
        
        df_combined = check_and_handle_missing(
            df_combined,
            method=handle_missing,
            threshold=missing_threshold,
            name='合并后的环境数据'
        )
        
        print(f"✓ 合并完成: {df_combined.shape}\n")
        return df_combined

    def _merge_dataframes(self, 
                         dataframes: List[pd.DataFrame],
                         sample_col: str = 'Env',
                         how: str = 'inner') -> pd.DataFrame:
        """按样本ID列合并多个 DataFrame"""
        if not dataframes:
            raise ValueError("dataframes 列表不能为空")
        
        dfs_with_col = []
        dfs_without_col = []
        
        for i, df in enumerate(dataframes):
            if sample_col in df.columns:
                dfs_with_col.append((i, df))
            else:
                dfs_without_col.append((i, df))
        
        if dfs_without_col:
            print(f"⚠️ 警告：以下 DataFrame 没有 '{sample_col}' 列，将使用行索引对齐:")
            for idx, _ in dfs_without_col:
                print(f"   - dataframes[{idx}]")
        
        if not dfs_with_col:
            print(f"❌ 错误：所有 DataFrame 都没有 '{sample_col}' 列！")
            return pd.concat(dataframes, axis=1)
        
        result = dfs_with_col[0][1].copy()
        
        for i in range(1, len(dfs_with_col)):
            other_df = dfs_with_col[i][1]
            
            overlap_cols = set(result.columns) & set(other_df.columns)
            overlap_cols.discard(sample_col)
            
            if overlap_cols:
                print(f"⚠️ 警告：发现重复列名 {overlap_cols}")
            
            result = result.merge(other_df, on=sample_col, how=how)
        
        for idx, df_no_col in dfs_without_col:
            if len(df_no_col) != len(result):
                print(f"⚠️ 警告：dataframes[{idx}] 的样本数不匹配")
            
            df_no_col_reset = df_no_col.reset_index(drop=True)
            result_reset = result.reset_index(drop=True)
            result = pd.concat([result_reset, df_no_col_reset], axis=1)
        
        n_missing = result.isna().sum().sum()
        if n_missing > 0:
            print(f"⚠️ 警告：合并后有 {n_missing} 个缺失值")
        
        return result
    
    def convert_to_3d_array(self, df: pd.DataFrame, 
                           sample_col: str = 'Env', 
                           date_col: str = 'Date') -> Tuple[np.ndarray, List[str], List[str]]:
        """
        将长格式 DataFrame 转换为 3D NumPy 数组
        
        Parameters:
            df: 长格式天气数据 DataFrame
            sample_col: 样本ID列名
            date_col: 日期列名
        
        Returns:
            tuple: (weather_3d, sample_ids, feature_names)
                   weather_3d 形状: (n_samples, n_timesteps, n_features)
        """
        # 获取特征列（排除样本ID和日期）
        feature_cols = [col for col in df.columns 
                       if col not in [sample_col, date_col]]
        
        if not feature_cols:
            raise ValueError(f"没有特征列！排除了 {sample_col} 和 {date_col} 后没有其他列")
        
        # 按样本分组并排序
        grouped_data = []
        sample_ids = []
        
        for sample_id, group in df.groupby(sample_col):
            group_sorted = group.sort_values(date_col)
            features = group_sorted[feature_cols].values
            grouped_data.append(features)
            sample_ids.append(sample_id)
        
        # 检查时间步数是否一致
        timesteps = [len(t) for t in grouped_data]
        if len(set(timesteps)) > 1:
            print(f"\n⚠️ 时间步数不一致: {set(timesteps)}")
            # 填充到最大时间步数
            max_t = max(timesteps)
            padded = []
            for i, data in enumerate(grouped_data):
                if len(data) < max_t:
                    n_pad = max_t - len(data)
                    pad = np.repeat(data[-1:], n_pad, axis=0)
                    data = np.vstack([data, pad])
                    print(f"   ✓ {sample_ids[i]}: 填充 {n_pad} 行")
                padded.append(data)
            grouped_data = padded
            print(f"   ✓ 所有样本已填充到 {max_t} 步")
        
        # 转换为 3D 数组
        weather_3d = np.array(grouped_data, dtype=np.float32)
        
        print(f"\n✓ 转换为 3D 数组:")
        print(f"   - 形状: {weather_3d.shape}")
        print(f"   - 维度: (样本={len(sample_ids)}, 时间步={weather_3d.shape[1]}, 特征={len(feature_cols)})")
        
        return weather_3d, sample_ids, feature_cols

    def load_weather_data_3d(self, filepath: str,
                            sample_col: str = "Env",
                            date_col: str = "Date",
                            handle_missing: str = 'drop',
                            missing_threshold: float = 0.5,
                            required_features: Optional[List[str]] = None) -> Tuple[np.ndarray, List[str], List[str]]:
        """
        加载天气数据并返回 3D 数组
        
        Parameters:
            filepath: 数据文件路径
            sample_col: 样本ID列名
            date_col: 日期列名
            handle_missing: 缺失值处理方法
            missing_threshold: 缺失率阈值
            required_features: 必须包含的特征列表（用于推理时保持一致性）
        
        Returns:
            tuple: (weather_3d, sample_ids, feature_names)
                   weather_3d 形状: (n_samples, n_timesteps, n_features)
        """
        # 先加载为 DataFrame（保留时间序列）
        # If required_features specified, use 'mean' instead of 'drop' to keep all columns
        if required_features:
            df_weather = self.load_weather_data(
                filepath=filepath,
                reshape_to_temporal=True,
                sample_col=sample_col,
                date_col=date_col,
                handle_missing='mean',  # Don't drop columns when we need specific features
                missing_threshold=1.0   # Don't drop any columns
            )
        else:
            df_weather = self.load_weather_data(
                filepath=filepath,
                reshape_to_temporal=True,  # 保留时间序列
                sample_col=sample_col,
                date_col=date_col,
                handle_missing=handle_missing,
                missing_threshold=missing_threshold
            )
        
        # 转换为 3D 数组
        weather_3d, sample_ids, feature_names = self.convert_to_3d_array(
            df_weather, 
            sample_col=sample_col, 
            date_col=date_col
        )
        
        # Reorder features to match required_features if specified
        if required_features:
            available_features = set(feature_names)
            required_set = set(required_features)
            
            if not required_set.issubset(available_features):
                missing = required_set - available_features
                print(f"⚠️ 缺少必需的特征: {missing}")
                raise ValueError(f"Required features not available: {missing}")
            
            if feature_names != required_features:
                print(f"📌 重新排序特征以匹配训练顺序...")
                feature_indices = [feature_names.index(f) for f in required_features]
                weather_3d = weather_3d[:, :, feature_indices]
                feature_names = required_features.copy()
                print(f"   ✓ 特征已重新排序: {len(feature_names)} 个特征")
        
        return weather_3d, sample_ids, feature_names
