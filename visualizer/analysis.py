"""数据质量分析工具。

提供数据质量检查功能：
- 离散值检测（outliers）
- 跳帧检测（frame drops）
- 缺失值检测（missing values）
- 异常值检测（anomalies）
- 统计信息提取（statistics）
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum


class AnomalyType(Enum):
    """异常类型枚举。"""
    OUTLIER = "outlier"
    FRAME_DROP = "frame_drop"
    MISSING_VALUE = "missing_value"
    ABNORMAL_STAT = "abnormal_stat"


@dataclass
class Anomaly:
    """异常记录。"""
    frame: int
    key: str
    type: AnomalyType
    value: Any
    severity: float  # 0-1，1 为最严重
    description: str


@dataclass
class FieldStatistics:
    """单个字段的统计信息。"""
    key: str
    dtype: str
    shape: Tuple[int, ...]
    min_val: float
    max_val: float
    mean: float
    std: float
    median: float
    missing_count: int
    outlier_count: int
    frame_drop_count: int
    has_anomalies: bool


class DataQualityChecker:
    """数据质量检查器。"""

    def __init__(
        self,
        outlier_threshold: float = 3.0,  # 标准差倍数
        frame_drop_threshold: float = 2.0,  # 标准差倍数，用于检测时间戳间隔
        missing_value_threshold: float = 0.01,  # 缺失值比例阈值
    ):
        """初始化检查器。
        
        Args:
            outlier_threshold: 离散值检测的标准差倍数阈值（z-score）。
            frame_drop_threshold: 跳帧检测的标准差倍数阈值。
            missing_value_threshold: 缺失值比例警告阈值。
        """
        self.outlier_threshold = outlier_threshold
        self.frame_drop_threshold = frame_drop_threshold
        self.missing_value_threshold = missing_value_threshold
        self.anomalies: List[Anomaly] = []
        self.statistics: Dict[str, FieldStatistics] = {}

    def check_sequence(
        self,
        data: Dict[str, np.ndarray],
        timestamps: Optional[np.ndarray] = None,
    ) -> Dict[str, Any]:
        """检查时间序列数据质量。
        
        Args:
            data: 数据字典，键为字段名，值为 numpy 数组。
            timestamps: 可选的时间戳数组，用于检测跳帧。
            
        Returns:
            包含检查结果的字典。
        """
        self.anomalies = []
        self.statistics = {}

        # 检查每个字段
        for key, arr in data.items():
            if arr is None or (isinstance(arr, dict) and not arr):
                continue
            
            # 处理嵌套字典（如 robot_state）
            if isinstance(arr, dict):
                for subkey, subarr in arr.items():
                    if subarr is not None:
                        self._check_field(f"{key}/{subkey}", subarr)
            else:
                self._check_field(key, arr)

        # 检查时间戳（跳帧）
        if timestamps is not None:
            self._check_timestamps(timestamps)

        return {
            "num_frames": self._infer_num_frames(data, timestamps),
            "num_anomalies": len(self.anomalies),
            "anomalies": self.anomalies,
            "statistics": self.statistics,
        }

    def _infer_num_frames(self, data: Dict[str, Any], timestamps: Optional[np.ndarray]) -> int:
        """推断帧数。"""
        if timestamps is not None:
            return len(timestamps)
        for arr in data.values():
            if isinstance(arr, np.ndarray):
                return len(arr)
            elif isinstance(arr, dict):
                for subarr in arr.values():
                    if isinstance(subarr, np.ndarray):
                        return len(subarr)
        return 0

    def _check_field(self, key: str, arr: np.ndarray) -> None:
        """检查单个字段。"""
        if not isinstance(arr, np.ndarray):
            return

        # 处理非数值数组（如图像）
        if arr.dtype.kind not in ('f', 'i', 'u'):
            self.statistics[key] = FieldStatistics(
                key=key,
                dtype=str(arr.dtype),
                shape=arr.shape,
                min_val=np.nan,
                max_val=np.nan,
                mean=np.nan,
                std=np.nan,
                median=np.nan,
                missing_count=0,
                outlier_count=0,
                frame_drop_count=0,
                has_anomalies=False,
            )
            return

        # 展平并检查
        if arr.ndim > 1:
            flat_arr = arr.reshape(arr.shape[0], -1)
        else:
            flat_arr = arr.reshape(-1, 1)

        # 计算统计信息（忽略 NaN）
        valid_data = flat_arr[~np.isnan(flat_arr)]
        
        missing_count = np.isnan(flat_arr).sum()
        missing_ratio = missing_count / flat_arr.size if flat_arr.size > 0 else 0

        if len(valid_data) > 0:
            mean_val = float(np.nanmean(flat_arr, axis=0).mean())
            std_val = float(np.nanstd(flat_arr, axis=0).mean())
            min_val = float(np.nanmin(flat_arr))
            max_val = float(np.nanmax(flat_arr))
            median_val = float(np.nanmedian(flat_arr))
        else:
            mean_val = std_val = min_val = max_val = median_val = np.nan

        # 检测离散值（outliers）
        outlier_count = 0
        if std_val > 0 and len(valid_data) > 0:
            z_scores = np.abs((flat_arr - mean_val) / (std_val + 1e-8))
            outliers = np.where(z_scores > self.outlier_threshold)
            outlier_count = len(outliers[0])
            
            # 记录离散值
            for frame_idx, col_idx in zip(outliers[0], outliers[1]):
                self.anomalies.append(Anomaly(
                    frame=frame_idx,
                    key=key,
                    type=AnomalyType.OUTLIER,
                    value=flat_arr[frame_idx, col_idx],
                    severity=min(1.0, (z_scores[frame_idx, col_idx] - self.outlier_threshold) / 5.0),
                    description=f"Z-score: {z_scores[frame_idx, col_idx]:.2f}",
                ))

        # 记录缺失值
        if missing_ratio > self.missing_value_threshold:
            self.anomalies.append(Anomaly(
                frame=-1,
                key=key,
                type=AnomalyType.MISSING_VALUE,
                value=missing_ratio,
                severity=min(1.0, missing_ratio / 0.5),
                description=f"Missing ratio: {missing_ratio:.2%}",
            ))

        self.statistics[key] = FieldStatistics(
            key=key,
            dtype=str(arr.dtype),
            shape=arr.shape,
            min_val=min_val,
            max_val=max_val,
            mean=mean_val,
            std=std_val,
            median=median_val,
            missing_count=int(missing_count),
            outlier_count=outlier_count,
            frame_drop_count=0,
            has_anomalies=outlier_count > 0 or missing_count > 0,
        )

    def _check_timestamps(self, timestamps: np.ndarray) -> None:
        """检查时间戳，检测跳帧。"""
        if len(timestamps) < 2:
            return

        # 计算时间间隔
        diffs = np.diff(timestamps)
        valid_diffs = diffs[~np.isnan(diffs)]

        if len(valid_diffs) == 0:
            return

        mean_diff = np.mean(valid_diffs)
        std_diff = np.std(valid_diffs)

        if std_diff > 0:
            z_scores = np.abs((diffs - mean_diff) / (std_diff + 1e-8))
            frame_drops = np.where(z_scores > self.frame_drop_threshold)[0]
            
            frame_drop_count = len(frame_drops)
            for drop_idx in frame_drops:
                actual_interval = diffs[drop_idx]
                expected_interval = mean_diff
                severity = min(1.0, (actual_interval - expected_interval) / (5 * expected_interval + 1e-8))
                self.anomalies.append(Anomaly(
                    frame=drop_idx,
                    key="timestamps",
                    type=AnomalyType.FRAME_DROP,
                    value=actual_interval,
                    severity=severity,
                    description=f"Interval: {actual_interval:.4f}s (expected ~{expected_interval:.4f}s)",
                ))

            if "timestamps" in self.statistics:
                self.statistics["timestamps"].frame_drop_count = frame_drop_count

    def print_summary(self) -> None:
        """打印检查摘要。"""
        print("\n" + "=" * 80)
        print("数据质量检查摘要")
        print("=" * 80)
        
        print(f"\n总异常数: {len(self.anomalies)}")
        
        # 按类型统计
        anomaly_counts = {}
        for anom in self.anomalies:
            atype = anom.type.value
            anomaly_counts[atype] = anomaly_counts.get(atype, 0) + 1
        
        for atype, count in anomaly_counts.items():
            print(f"  {atype}: {count}")
        
        print(f"\n统计字段数: {len(self.statistics)}")
        print("\n字段统计:")
        for key, stat in sorted(self.statistics.items()):
            if np.isnan(stat.mean):
                print(f"  {key:30s} | dtype={stat.dtype:10s} shape={stat.shape}")
            else:
                print(f"  {key:30s} | mean={stat.mean:8.3f} std={stat.std:8.3f} "
                      f"min={stat.min_val:8.3f} max={stat.max_val:8.3f} "
                      f"anomalies={stat.outlier_count:3d}")
        
        print("\n异常详情（严重程度 > 0.5）:")
        severe_anomalies = [a for a in self.anomalies if a.severity > 0.5]
        if severe_anomalies:
            for anom in sorted(severe_anomalies, key=lambda x: x.severity, reverse=True)[:10]:
                print(f"  Frame {anom.frame:4d} | {anom.key:30s} | "
                      f"{anom.type.value:15s} | severity={anom.severity:.2f} | {anom.description}")
        else:
            print("  无严重异常")
        
        print("=" * 80 + "\n")


def check_episode_quality(
    adapter,
    episode_id: str,
    timestamps_key: str = "timestamps",
    checker: Optional[DataQualityChecker] = None,
) -> Dict[str, Any]:
    """检查单个 episode 的数据质量。
    
    Args:
        adapter: DatasetAdapter 实例。
        episode_id: Episode ID。
        timestamps_key: 时间戳字段名。
        checker: 可选的 DataQualityChecker 实例。
        
    Returns:
        检查结果。
    """
    if checker is None:
        checker = DataQualityChecker()
    
    seq = adapter.read_sequence(episode_id)
    timestamps = seq.get(timestamps_key)
    
    # 提取数值数据
    data = {}
    for key, val in seq.items():
        if key == timestamps_key:
            continue
        if isinstance(val, dict):
            data[key] = val
        elif isinstance(val, np.ndarray):
            data[key] = val
    
    result = checker.check_sequence(data, timestamps)
    result["episode_id"] = episode_id
    return result
