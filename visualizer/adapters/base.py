from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional


class DatasetAdapter(ABC):
    """抽象数据集适配器接口。

    目的：将不同存储格式映射为统一的内部表示。
    实现应返回 numpy-friendly 的数据结构（ndarray / list / dict）。
    """

    @abstractmethod
    def list_episodes(self) -> List[str]:
        """返回数据集中可用的 episode id 列表（字符串）。"""

    @abstractmethod
    def get_episode_meta(self, episode_id: str) -> Dict[str, Any]:
        """返回 episode 级别的元数据（例如采样率、坐标系、camera extrinsics）。"""

    @abstractmethod
    def read_sequence(
        self,
        episode_id: str,
        fields: Optional[List[str]] = None,
        start: Optional[int] = None,
        end: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        读取指定 episode 的时间序列数据。

        Args:
            episode_id: 要读取的 episode 名称或 id。
            fields: 可选字段列表，例如 ['images', 'robot_state/joint_positions', 'actions']。
            start, end: 可选的帧区间（半开区间 [start, end)）。

        Returns:
            字典，键是字段名，值为 numpy 数组或嵌套字典。
        """

    def close(self) -> None:
        """可选：关闭资源（例如文件句柄）。"""
        return None
