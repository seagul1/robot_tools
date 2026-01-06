"""
单轨迹可视化工具（增强版）。

功能：
- 支持 YAML schema 配置文件指导数据读取
- 集成数据质量检查（离散值、跳帧、缺失值）
- 改进的时间序列图表（显示统计信息、异常标记）
- 支持多传感器可视化
- 交互式播放控件

用法示例：
python enhanced_simple_viewer.py --file path/to/data.h5 --schema path/to/schema.yaml --episode episode_0
python enhanced_simple_viewer.py --file path/to/data.h5 --episode episode_0  # 自动检测 schema
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button
from matplotlib.patches import Rectangle
from typing import Dict, List, Optional, Any

# 导入自定义模块
VIS_DIR = os.path.dirname(os.path.abspath(__file__))
if VIS_DIR not in sys.path:
    sys.path.insert(0, VIS_DIR)

try:
    from adapters.hdf5_adapter import HDF5Adapter
    from schema_loader import load_schema, extract_visualization_fields
    from analysis import DataQualityChecker, check_episode_quality
except Exception as e:
    try:
        from .adapters.hdf5_adapter import HDF5Adapter  # type: ignore
        from .schema_loader import load_schema, extract_visualization_fields  # type: ignore
        from .analysis import DataQualityChecker, check_episode_quality  # type: ignore
    except Exception:
        raise


def _show_images_axes(images_dict: Dict[str, np.ndarray], frame_idx: int, axes: List) -> None:
    """更新图像子图显示。"""
    views = list(images_dict.keys())
    for i, view in enumerate(views):
        if i >= len(axes):
            break
        im_arr = images_dict[view]
        if im_arr is None:
            axes[i].clear()
            axes[i].set_title(view)
            continue
        # im_arr shape: (N, H, W, C) or (N, H, W)
        if frame_idx < 0 or frame_idx >= im_arr.shape[0]:
            axes[i].clear()
            axes[i].set_title(view)
            continue
        img = im_arr[frame_idx]
        axes[i].imshow(img)
        axes[i].axis("off")
        axes[i].set_title(view, fontsize=10)


def _plot_timeseries(
    ax,
    data: Optional[np.ndarray],
    label_prefix: str = "",
    anomalies: Optional[List[int]] = None,
    current_frame: Optional[int] = None,
) -> None:
    """绘制时间序列数据。
    
    Args:
        ax: matplotlib 轴对象。
        data: 形状为 (N,) 或 (N, D) 的数据。
        label_prefix: 标签前缀。
        anomalies: 异常帧索引列表。
        current_frame: 当前帧索引。
    """
    if data is None:
        ax.clear()
        return

    # 绘制曲线
    if data.ndim == 1:
        ax.plot(data, label=label_prefix, linewidth=1.5, alpha=0.8)
    else:
        for d in range(min(data.shape[1], 10)):  # 最多显示 10 条曲线
            ax.plot(data[:, d], label=f"{label_prefix}{d}", linewidth=1.5, alpha=0.7)

    # 标记异常
    if anomalies:
        ax.scatter(anomalies, data[anomalies] if data.ndim == 1 else data[anomalies, 0],
                  color='red', s=50, marker='x', label='Anomaly', zorder=5, linewidth=2)

    ax.legend(loc="upper right", fontsize="small", ncol=2)
    ax.set_xlabel("frame", fontsize=9)
    ax.set_ylabel("value", fontsize=9)
    ax.grid(True, alpha=0.3)


def _extract_anomaly_frames(anomalies_list, key: str) -> List[int]:
    """从异常列表中提取指定字段的异常帧。"""
    frames = []
    for anom in anomalies_list:
        if anom.key == key and anom.frame >= 0:
            frames.append(anom.frame)
    return list(set(frames))


def viewer_main(
    h5_path: str,
    schema_path: Optional[str] = None,
    episode: Optional[str] = None,
    save_plots: bool = False,
) -> None:
    """主可视化函数。
    
    Args:
        h5_path: HDF5 文件路径。
        schema_path: YAML schema 配置文件路径。
        episode: Episode ID。
        save_plots: 是否保存图表到本地。
    """
    # 加载适配器和 schema
    adapter = HDF5Adapter(h5_path)
    
    if schema_path and os.path.exists(schema_path):
        schema = load_schema(schema_path)
        adapter.set_schema(schema)
        print(f"[INFO] 已加载 schema: {schema_path}")
    else:
        schema = None
        print("[INFO] 未加载 schema，使用自动检测模式")

    # 列出可用 episodes
    episodes = adapter.list_episodes()
    if not episodes:
        print("[ERROR] 未找到任何 episode。")
        return

    if episode is None:
        episode = episodes[0]
        print(f"[INFO] 未指定 episode，使用第一个: {episode}")
    elif episode not in episodes:
        print(f"[ERROR] Episode {episode} 不存在。可用: {episodes}")
        return

    print(f"[INFO] 读取 episode: {episode}")

    # 读取序列数据
    seq = adapter.read_sequence(episode)
    timestamps = seq.get("timestamps", None)
    images = seq.get("images", None) or {}
    robot_state = seq.get("robot_state", {})
    actions = seq.get("actions", None)

    # 推断帧数
    n_frames = None
    if timestamps is not None:
        n_frames = int(len(timestamps))
    else:
        for v in images.values():
            n_frames = v.shape[0]
            break
    if n_frames is None:
        for key in list(robot_state.keys()):
            data = robot_state[key]
            if isinstance(data, np.ndarray):
                n_frames = data.shape[0]
                break
    if n_frames is None and actions is not None:
        n_frames = actions.shape[0]
    if n_frames is None:
        print("[ERROR] 无法推断帧数。")
        return

    print(f"[INFO] 帧数: {n_frames}")

    # 数据质量检查
    print("\n[INFO] 执行数据质量检查...")
    checker = DataQualityChecker(outlier_threshold=3.0, frame_drop_threshold=2.0)
    quality_result = check_episode_quality(adapter, episode, checker=checker)
    checker.print_summary()

    anomalies_dict = {}
    for anom in quality_result["anomalies"]:
        if anom.key not in anomalies_dict:
            anomalies_dict[anom.key] = []
        anomalies_dict[anom.key].append(anom.frame)

    # 准备绘图布局
    n_views = max(1, len(images))
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle(f"单轨迹可视化: {os.path.basename(h5_path)} :: {episode}", fontsize=14, fontweight='bold')

    # 图像子图（顶部行）
    img_axes = []
    for i, view in enumerate(images.keys()):
        ax = fig.add_subplot(4, max(3, n_views), i + 1)
        img_axes.append(ax)
    if not img_axes:
        img_axes.append(fig.add_subplot(4, 1, 1))

    # 时间序列子图（第 2-4 行）
    ax_joints = fig.add_subplot(4, 2, max(3, n_views) + 1)
    ax_ee = fig.add_subplot(4, 2, max(3, n_views) + 2)
    ax_actions = fig.add_subplot(4, 2, max(3, n_views) + 3)
    ax_tactile = fig.add_subplot(4, 2, max(3, n_views) + 4)

    # 绘制关节数据
    joint_data = None
    joint_anomalies = None
    if "joint_positions" in robot_state:
        joint_data = robot_state["joint_positions"]
        joint_anomalies = _extract_anomaly_frames(quality_result["anomalies"], "robot_state/joint_positions")
    elif "joint_angles" in robot_state:
        joint_data = robot_state["joint_angles"]
        joint_anomalies = _extract_anomaly_frames(quality_result["anomalies"], "robot_state/joint_angles")
    
    _plot_timeseries(ax_joints, joint_data, label_prefix="joint_", anomalies=joint_anomalies)
    ax_joints.set_title("Joint Positions", fontweight='bold')

    # 绘制末端位姿（EE）
    ee_data = robot_state.get("ee_pose", None)
    ee_anomalies = None
    if ee_data is not None and ee_data.ndim == 2 and ee_data.shape[1] >= 3:
        # 仅绘制 x, y, z
        ee_xyz = ee_data[:, :3]
        ee_anomalies = _extract_anomaly_frames(quality_result["anomalies"], "robot_state/ee_pose")
        _plot_timeseries(ax_ee, ee_xyz, label_prefix="ee_", anomalies=ee_anomalies)
    else:
        ax_ee.clear()
    ax_ee.set_title("End-Effector Position", fontweight='bold')

    # 绘制动作
    actions_anomalies = None
    if actions is not None:
        actions_anomalies = _extract_anomaly_frames(quality_result["anomalies"], "actions")
        _plot_timeseries(ax_actions, actions, label_prefix="action_", anomalies=actions_anomalies)
    else:
        ax_actions.clear()
    ax_actions.set_title("Actions", fontweight='bold')

    # 绘制其他传感器数据（如力觉）
    tactile_found = False
    for key in robot_state.keys():
        if "tactile" in key.lower() or "force" in key.lower():
            tactile_data = robot_state[key]
            if isinstance(tactile_data, np.ndarray) and tactile_data.ndim >= 1:
                tactile_anomalies = _extract_anomaly_frames(quality_result["anomalies"], f"robot_state/{key}")
                _plot_timeseries(ax_tactile, tactile_data, label_prefix=key, anomalies=tactile_anomalies)
                tactile_found = True
                break
    if not tactile_found:
        ax_tactile.clear()
    ax_tactile.set_title("Tactile/Other Sensors", fontweight='bold')

    # 初始化图像显示
    current_frame = 0
    for ax in img_axes:
        ax.axis("off")
    if images:
        _show_images_axes(images, current_frame, img_axes)

    # 添加垂直线以标记当前帧
    vlines = []
    for ax in [ax_joints, ax_ee, ax_actions, ax_tactile]:
        if ax.lines or ax.collections:
            vline = ax.axvline(current_frame, color='r', linestyle='--', linewidth=2, alpha=0.7)
            vlines.append((ax, vline))

    # 当前动作文本显示
    ax_action_text = fig.add_axes([0.72, 0.08, 0.25, 0.08])
    ax_action_text.axis("off")
    action_text = ax_action_text.text(0, 1, "", transform=ax_action_text.transAxes,
                                     fontsize=9, verticalalignment='top', family='monospace')

    # 数据质量信息显示
    ax_quality_text = fig.add_axes([0.72, 0.16, 0.25, 0.08])
    ax_quality_text.axis("off")
    quality_text = ax_quality_text.text(0, 1, "", transform=ax_quality_text.transAxes,
                                       fontsize=8, verticalalignment='top', family='monospace')

    # 滑条
    ax_slider = fig.add_axes([0.2, 0.02, 0.5, 0.03])
    slider = Slider(ax_slider, "frame", 0, max(0, n_frames - 1), valinit=current_frame, valfmt="%0.0f")

    def update(val):
        frame = int(slider.val)
        
        # 更新图像
        if images:
            _show_images_axes(images, frame, img_axes)
        
        # 更新垂直线
        for ax, vline in vlines:
            vline.set_xdata(frame)
        
        # 更新当前动作文本
        if actions is not None and frame < actions.shape[0]:
            action_val = actions[frame]
            action_str = np.array2string(action_val, precision=3, separator=", ", max_line_width=60)
            action_text.set_text(f"Action[{frame}]:\n{action_str}")
        else:
            action_text.set_text("")
        
        # 更新数据质量信息
        quality_info = f"Frame: {frame}/{n_frames}\n"
        if timestamps is not None:
            quality_info += f"Time: {timestamps[frame]:.3f}s\n"
        
        anomalies_at_frame = [a for a in quality_result["anomalies"] if a.frame == frame]
        if anomalies_at_frame:
            quality_info += f"Anomalies: {len(anomalies_at_frame)}\n"
            for a in anomalies_at_frame[:3]:
                quality_info += f"  - {a.key}: {a.type.value}\n"
        
        quality_text.set_text(quality_info)
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # 前/后按钮
    axprev = fig.add_axes([0.02, 0.02, 0.08, 0.04])
    axnext = fig.add_axes([0.11, 0.02, 0.08, 0.04])
    bprev = Button(axprev, "Prev")
    bnext = Button(axnext, "Next")

    def prev(event):
        val = int(slider.val)
        if val > 0:
            slider.set_val(val - 1)

    def next_(event):
        val = int(slider.val)
        if val < n_frames - 1:
            slider.set_val(val + 1)

    bprev.on_clicked(prev)
    bnext.on_clicked(next_)

    plt.tight_layout(rect=[0, 0.07, 0.7, 0.97])
    
    if save_plots:
        output_dir = os.path.dirname(h5_path)
        output_path = os.path.join(output_dir, f"{episode}_visualization.png")
        fig.savefig(output_path, dpi=100, bbox_inches='tight')
        print(f"[INFO] 图表已保存到: {output_path}")
    
    plt.show()
    adapter.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单轨迹可视化工具（增强版）")
    parser.add_argument("--file", required=True, help="HDF5 数据文件路径")
    parser.add_argument("--schema", default=None, help="YAML schema 配置文件路径（可选）")
    parser.add_argument("--episode", default=None, help="Episode ID（可选，默认为第一个）")
    parser.add_argument("--save", action="store_true", help="保存图表到本地")
    args = parser.parse_args()
    
    viewer_main(args.file, args.schema, args.episode, args.save)
