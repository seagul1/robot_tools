"""
多轨迹汇总可视化工具。

功能：
- 加载数据集中的所有轨迹
- 显示轨迹长度分布直方图
- 显示关键数据（joint_positions, ee_pose 等）的分布热力图
- 轨迹质量对比（异常数、缺失值、跳帧等）
- 导出统计报告

用法示例：
python batch_viewer.py --file path/to/data.h5 --schema path/to/schema.yaml
python batch_viewer.py --dir path/to/data_dir --schema path/to/schema.yaml
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict
import glob

# 导入自定义模块
VIS_DIR = os.path.dirname(os.path.abspath(__file__))
if VIS_DIR not in sys.path:
    sys.path.insert(0, VIS_DIR)

try:
    from adapters.hdf5_adapter import HDF5Adapter
    from schema_loader import load_schema, extract_visualization_fields
    from analysis import DataQualityChecker, check_episode_quality
except Exception:
    try:
        from .adapters.hdf5_adapter import HDF5Adapter  # type: ignore
        from .schema_loader import load_schema, extract_visualization_fields  # type: ignore
        from .analysis import DataQualityChecker, check_episode_quality  # type: ignore
    except Exception:
        raise


class BatchAnalyzer:
    """多轨迹批量分析工具。"""

    def __init__(self, schema_path: Optional[str] = None):
        """初始化分析器。
        
        Args:
            schema_path: YAML schema 配置文件路径。
        """
        self.schema = None
        if schema_path and os.path.exists(schema_path):
            self.schema = load_schema(schema_path)
            print(f"[INFO] 已加载 schema: {schema_path}")
        
        self.episode_stats = {}
        self.episode_qualities = {}
        self.dataset_stats = {}

    def analyze_file(self, h5_path: str) -> Dict[str, Any]:
        """分析单个 HDF5 文件中的所有 episode。
        
        Args:
            h5_path: HDF5 文件路径。
            
        Returns:
            分析结果字典。
        """
        adapter = HDF5Adapter(h5_path)
        if self.schema:
            adapter.set_schema(self.schema)
        
        episodes = adapter.list_episodes()
        print(f"[INFO] 文件 {os.path.basename(h5_path)} 包含 {len(episodes)} 个 episode")
        
        for episode_id in episodes:
            self._analyze_episode(adapter, episode_id, h5_path)
        
        adapter.close()
        return self.episode_stats

    def _analyze_episode(self, adapter, episode_id: str, source_file: str) -> None:
        """分析单个 episode。"""
        print(f"  分析 episode: {episode_id}...", end="")
        
        try:
            seq = adapter.read_sequence(episode_id)
            timestamps = seq.get("timestamps")
            
            # 基本统计
            n_frames = None
            for v in seq.values():
                if isinstance(v, dict):
                    for subv in v.values():
                        if isinstance(subv, np.ndarray):
                            n_frames = len(subv)
                            break
                elif isinstance(v, np.ndarray):
                    n_frames = len(v)
                    break
            
            stats = {
                "episode_id": episode_id,
                "source_file": source_file,
                "n_frames": n_frames,
                "duration": float(timestamps[-1] - timestamps[0]) if timestamps is not None and len(timestamps) > 1 else 0.0,
                "fps": n_frames / (float(timestamps[-1] - timestamps[0]) + 1e-8) if timestamps is not None and len(timestamps) > 1 else 0.0,
            }
            
            # 数据质量检查
            quality_result = check_episode_quality(adapter, episode_id, checker=DataQualityChecker())
            stats["num_anomalies"] = quality_result["num_anomalies"]
            stats["anomaly_types"] = defaultdict(int)
            for anom in quality_result["anomalies"]:
                stats["anomaly_types"][anom.type.value] += 1
            
            # 提取数据特征（均值、方差）
            features = {}
            robot_state = seq.get("robot_state", {})
            
            for key in ["joint_positions", "joint_angles"]:
                if key in robot_state:
                    data = robot_state[key]
                    if isinstance(data, np.ndarray) and data.size > 0:
                        features[f"{key}_mean"] = float(np.nanmean(data))
                        features[f"{key}_std"] = float(np.nanstd(data))
                        features[f"{key}_min"] = float(np.nanmin(data))
                        features[f"{key}_max"] = float(np.nanmax(data))
            
            for key in ["ee_pose"]:
                if key in robot_state:
                    data = robot_state[key]
                    if isinstance(data, np.ndarray) and data.ndim >= 2 and data.shape[1] >= 3:
                        ee_xyz = data[:, :3]
                        features[f"{key}_xyz_mean"] = float(np.nanmean(ee_xyz))
                        features[f"{key}_xyz_std"] = float(np.nanstd(ee_xyz))
            
            stats.update(features)
            self.episode_stats[episode_id] = stats
            self.episode_qualities[episode_id] = quality_result
            print(" ✓")
        
        except Exception as e:
            print(f" ✗ ({e})")
            self.episode_stats[episode_id] = {"episode_id": episode_id, "error": str(e)}

    def generate_report(self, output_dir: Optional[str] = None) -> None:
        """生成分析报告并可视化。
        
        Args:
            output_dir: 输出目录（可选）。
        """
        if not self.episode_stats:
            print("[WARNING] 没有分析结果。")
            return
        
        print("\n" + "=" * 80)
        print("多轨迹汇总分析报告")
        print("=" * 80)
        
        # 基本统计
        valid_episodes = [s for s in self.episode_stats.values() if "error" not in s]
        print(f"\n总 Episode 数: {len(self.episode_stats)}")
        print(f"有效 Episode 数: {len(valid_episodes)}")
        
        if valid_episodes:
            n_frames_list = [s["n_frames"] for s in valid_episodes if "n_frames" in s]
            durations = [s["duration"] for s in valid_episodes if "duration" in s]
            anomaly_counts = [s["num_anomalies"] for s in valid_episodes if "num_anomalies" in s]
            
            print(f"\n轨迹长度 (frames):")
            print(f"  最小: {min(n_frames_list)}, 最大: {max(n_frames_list)}, 平均: {np.mean(n_frames_list):.0f}")
            
            print(f"\n轨迹时长 (seconds):")
            print(f"  最小: {min(durations):.2f}, 最大: {max(durations):.2f}, 平均: {np.mean(durations):.2f}")
            
            print(f"\n异常统计:")
            print(f"  总异常数: {sum(anomaly_counts)}")
            print(f"  平均异常数/轨迹: {np.mean(anomaly_counts):.2f}")
            print(f"  最大异常数: {max(anomaly_counts)}")

        # 生成可视化
        self._visualize_report()
        
        # 保存报告
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            report_path = os.path.join(output_dir, "batch_analysis_report.txt")
            with open(report_path, "w") as f:
                f.write(self._format_report())
            print(f"\n[INFO] 报告已保存到: {report_path}")

    def _format_report(self) -> str:
        """格式化报告为文本。"""
        lines = []
        lines.append("=" * 80)
        lines.append("多轨迹批量分析报告")
        lines.append("=" * 80)
        lines.append("")
        
        valid_episodes = [s for s in self.episode_stats.values() if "error" not in s]
        lines.append(f"总 Episode: {len(self.episode_stats)}, 有效: {len(valid_episodes)}\n")
        
        lines.append("Episode 详情:")
        for episode_id, stats in sorted(self.episode_stats.items()):
            if "error" in stats:
                lines.append(f"  {episode_id}: ERROR - {stats['error']}")
            else:
                lines.append(f"  {episode_id}:")
                lines.append(f"    Frames: {stats.get('n_frames', 'N/A')}")
                lines.append(f"    Duration: {stats.get('duration', 'N/A'):.2f}s")
                lines.append(f"    FPS: {stats.get('fps', 'N/A'):.2f}")
                lines.append(f"    Anomalies: {stats.get('num_anomalies', 0)}")
        
        return "\n".join(lines)

    def _visualize_report(self) -> None:
        """生成可视化图表。"""
        valid_episodes = [s for s in self.episode_stats.values() if "error" not in s]
        if not valid_episodes:
            print("[WARNING] 没有有效的分析结果用于可视化。")
            return
        
        fig = plt.figure(figsize=(16, 12))
        fig.suptitle("多轨迹汇总分析", fontsize=16, fontweight='bold')
        
        # 1. 轨迹长度分布
        ax1 = fig.add_subplot(3, 3, 1)
        n_frames_list = [s["n_frames"] for s in valid_episodes if "n_frames" in s]
        ax1.hist(n_frames_list, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax1.set_xlabel("Frames")
        ax1.set_ylabel("Count")
        ax1.set_title("轨迹长度分布")
        ax1.grid(True, alpha=0.3)
        
        # 2. 轨迹时长分布
        ax2 = fig.add_subplot(3, 3, 2)
        durations = [s["duration"] for s in valid_episodes if "duration" in s]
        ax2.hist(durations, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax2.set_xlabel("Duration (seconds)")
        ax2.set_ylabel("Count")
        ax2.set_title("轨迹时长分布")
        ax2.grid(True, alpha=0.3)
        
        # 3. FPS 分布
        ax3 = fig.add_subplot(3, 3, 3)
        fps_list = [s["fps"] for s in valid_episodes if "fps" in s]
        ax3.hist(fps_list, bins=20, color='lightgreen', edgecolor='black', alpha=0.7)
        ax3.set_xlabel("FPS")
        ax3.set_ylabel("Count")
        ax3.set_title("帧率分布")
        ax3.grid(True, alpha=0.3)
        
        # 4. 异常数分布
        ax4 = fig.add_subplot(3, 3, 4)
        anomaly_counts = [s["num_anomalies"] for s in valid_episodes if "num_anomalies" in s]
        ax4.hist(anomaly_counts, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        ax4.set_xlabel("Anomaly Count")
        ax4.set_ylabel("Count")
        ax4.set_title("异常数分布")
        ax4.grid(True, alpha=0.3)
        
        # 5. Episode 轨迹长度 vs 异常数散点图
        ax5 = fig.add_subplot(3, 3, 5)
        n_frames_scatter = [s["n_frames"] for s in valid_episodes if "n_frames" in s and "num_anomalies" in s]
        anomalies_scatter = [s["num_anomalies"] for s in valid_episodes if "n_frames" in s and "num_anomalies" in s]
        ax5.scatter(n_frames_scatter, anomalies_scatter, alpha=0.6, s=80, color='purple')
        ax5.set_xlabel("Frames")
        ax5.set_ylabel("Anomalies")
        ax5.set_title("轨迹长度 vs 异常数")
        ax5.grid(True, alpha=0.3)
        
        # 6. 关节位置范围分布
        ax6 = fig.add_subplot(3, 3, 6)
        joint_mins = []
        joint_maxs = []
        for s in valid_episodes:
            if "joint_positions_min" in s:
                joint_mins.append(s["joint_positions_min"])
            if "joint_positions_max" in s:
                joint_maxs.append(s["joint_positions_max"])
        
        if joint_mins and joint_maxs:
            positions = np.arange(len(joint_mins))
            ax6.bar(positions, joint_maxs, label='Max', alpha=0.7, color='green')
            ax6.bar(positions, joint_mins, label='Min', alpha=0.7, color='red')
            ax6.set_ylabel("Position Range")
            ax6.set_title("关节位置范围")
            ax6.legend()
            ax6.grid(True, alpha=0.3, axis='y')
        
        # 7. 异常类型分布
        ax7 = fig.add_subplot(3, 3, 7)
        anomaly_type_counts = defaultdict(int)
        for quality_result in self.episode_qualities.values():
            for anom in quality_result["anomalies"]:
                anomaly_type_counts[anom.type.value] += 1
        
        if anomaly_type_counts:
            types = list(anomaly_type_counts.keys())
            counts = list(anomaly_type_counts.values())
            ax7.barh(types, counts, color='skyblue', edgecolor='black')
            ax7.set_xlabel("Count")
            ax7.set_title("异常类型分布")
            ax7.grid(True, alpha=0.3, axis='x')
        
        # 8. 质量热力图（每个 episode 的异常/帧数比）
        ax8 = fig.add_subplot(3, 3, 8)
        episode_ids = []
        quality_scores = []
        for episode_id, stats in sorted(self.episode_stats.items()):
            if "error" not in stats and "num_anomalies" in stats:
                episode_ids.append(episode_id)
                # 计算质量分数：更低的异常比表示更高的质量
                n_frames = stats.get("n_frames", 1)
                anomaly_ratio = stats["num_anomalies"] / (n_frames + 1e-8)
                quality_scores.append(1.0 / (1.0 + anomaly_ratio))  # 归一化到 0-1
        
        if episode_ids:
            colors = plt.cm.RdYlGn(quality_scores)
            ax8.barh(range(len(episode_ids)), quality_scores, color=colors, edgecolor='black')
            ax8.set_yticks(range(len(episode_ids)))
            ax8.set_yticklabels([e[-10:] for e in episode_ids], fontsize=8)  # 显示最后 10 个字符
            ax8.set_xlabel("Quality Score")
            ax8.set_title("轨迹质量评分")
            ax8.set_xlim([0, 1.1])
            ax8.grid(True, alpha=0.3, axis='x')
        
        # 9. 统计信息文本
        ax9 = fig.add_subplot(3, 3, 9)
        ax9.axis("off")
        
        stats_text = f"""总轨迹数: {len(valid_episodes)}
        
平均轨迹长度: {np.mean(n_frames_list):.0f} frames
平均时长: {np.mean(durations):.2f} s
平均异常数: {np.mean(anomaly_counts):.2f}
        
最长轨迹: {max(n_frames_list)} frames
最短轨迹: {min(n_frames_list)} frames
        
最多异常: {max(anomaly_counts)} anomalies
无异常轨迹: {sum(1 for a in anomaly_counts if a == 0)}
"""
        ax9.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
                verticalalignment='center', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        plt.show()


def batch_viewer_main(
    data_path: str,
    schema_path: Optional[str] = None,
    is_directory: bool = False,
    output_dir: Optional[str] = None,
) -> None:
    """主批量分析函数。
    
    Args:
        data_path: HDF5 文件路径或包含 HDF5 文件的目录。
        schema_path: YAML schema 配置文件路径。
        is_directory: 是否为目录模式。
        output_dir: 输出目录。
    """
    analyzer = BatchAnalyzer(schema_path)
    
    if is_directory:
        # 查找所有 .h5 文件
        h5_files = glob.glob(os.path.join(data_path, "**/*.h5"), recursive=True)
        if not h5_files:
            h5_files = glob.glob(os.path.join(data_path, "*.h5"))
        
        print(f"[INFO] 在目录 {data_path} 中找到 {len(h5_files)} 个 HDF5 文件")
        for h5_file in h5_files:
            analyzer.analyze_file(h5_file)
    else:
        # 单个文件
        if not os.path.exists(data_path):
            print(f"[ERROR] 文件不存在: {data_path}")
            return
        analyzer.analyze_file(data_path)
    
    # 生成报告
    analyzer.generate_report(output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="多轨迹汇总可视化工具")
    parser.add_argument("--file", default=None, help="单个 HDF5 文件路径")
    parser.add_argument("--dir", default=None, help="包含 HDF5 文件的目录")
    parser.add_argument("--schema", default=None, help="YAML schema 配置文件路径")
    parser.add_argument("--output", default=None, help="输出目录（保存报告）")
    args = parser.parse_args()
    
    if args.file:
        batch_viewer_main(args.file, args.schema, is_directory=False, output_dir=args.output)
    elif args.dir:
        batch_viewer_main(args.dir, args.schema, is_directory=True, output_dir=args.output)
    else:
        parser.print_help()
