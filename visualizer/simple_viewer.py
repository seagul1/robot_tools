"""
简单模式可视化脚本（HDF5 + matplotlib）。

功能：
- 读取一个 HDF5 文件（支持本项目 HDF5 适配器定义的组织），
- 显示多视角图像（grid），
- 显示机器人末端位姿（x,y,z）和关节角随时间的曲线，
- 显示动作（actions）随时间的曲线，
- 使用滑条选择帧并支持前后步进按钮。

用法示例：
python simple_viewer.py --file path/to/data.h5 --episode episode_0

当前为 MVP 级别，交互简洁，基于 matplotlib，仅作本地快速调试。
"""

import os
import sys
import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button

# 使得可以直接从本目录的 adapters 导入
VIS_DIR = os.path.dirname(os.path.abspath(__file__))
if VIS_DIR not in sys.path:
    sys.path.insert(0, VIS_DIR)

try:
    from adapters.hdf5_adapter import HDF5Adapter
except Exception:
    # 尝试相对导入（当作为包运行时）
    try:
        from .adapters.hdf5_adapter import HDF5Adapter  # type: ignore
    except Exception:
        raise


def _show_images_axes(images_dict, frame_idx, axes):
    """更新图像子图显示。"""
    views = list(images_dict.keys())
    for i, view in enumerate(views):
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
        axes[i].set_title(view)


def _plot_timeseries(ax, data, label_prefix=""):
    # data: (N, D)
    if data is None:
        ax.clear()
        return None
    if data.ndim == 1:
        ax.plot(data, label=label_prefix)
    else:
        for d in range(data.shape[1]):
            ax.plot(data[:, d], label=f"{label_prefix}{d}")
    ax.legend(loc="upper right", fontsize="small")
    ax.set_xlabel("frame")


def viewer_main(h5_path: str, episode: str = None):
    adapter = HDF5Adapter(h5_path)
    episodes = adapter.list_episodes()
    if not episodes:
        print("No episodes found in HDF5 file.")
        return
    if episode is None:
        episode = episodes[0]
        print(f"No episode specified, using first: {episode}")
    elif episode not in episodes:
        print(f"Episode {episode} not in file. Available: {episodes}")
        return

    seq = adapter.read_sequence(episode)

    timestamps = seq.get("timestamps", None)
    images = seq.get("images", None) or {}
    robot_state = seq.get("robot_state", {})
    actions = seq.get("actions", None)

    # infer n_frames
    n_frames = None
    if timestamps is not None:
        n_frames = int(len(timestamps))
    else:
        # try images
        for v in images.values():
            n_frames = v.shape[0]
            break
    if n_frames is None:
        # try robot_state
        for key in ["joint_positions", "joint_angles"]:
            if key in robot_state:
                n_frames = robot_state[key].shape[0]
                break
    if n_frames is None and actions is not None:
        n_frames = actions.shape[0]
    if n_frames is None:
        print("无法推断帧数（timestamps/images/robot_state/actions 均不存在或为空）")
        return

    # prepare plot layout
    n_views = max(1, len(images))
    fig = plt.figure(figsize=(4 * n_views, 8))

    # images axes (top row)
    img_axes = []
    for i, view in enumerate(images.keys()):
        ax = fig.add_subplot(3, max(3, n_views), i + 1)
        img_axes.append(ax)
    if not img_axes:
        # create at least one placeholder
        img_axes.append(fig.add_subplot(3, 1, 1))

    # timeseries axes
    ax_joints = fig.add_subplot(3, 1, 2)
    ax_ee = fig.add_subplot(3, 1, 3)

    # plot static curves
    joint_data = None
    if "joint_positions" in robot_state:
        joint_data = robot_state["joint_positions"]
    elif "joint_angles" in robot_state:
        joint_data = robot_state["joint_angles"]

    _plot_timeseries(ax_joints, joint_data, label_prefix="joint_")

    ee_data = robot_state.get("ee_pose", None)
    # ee_data expected shape (N, 7) -> plot x,y,z
    if ee_data is not None and ee_data.ndim == 2 and ee_data.shape[1] >= 3:
        _plot_timeseries(ax_ee, ee_data[:, :3], label_prefix="ee_")
    else:
        ax_ee.clear()

    # actions
    ax_actions = fig.add_axes([0.75, 0.05, 0.2, 0.15])
    ax_actions.set_title("actions (current)")
    action_text = ax_actions.text(0.05, 0.5, "", transform=ax_actions.transAxes)
    ax_actions.axis("off")

    # initial images
    current_frame = 0
    for ax in img_axes:
        ax.axis("off")

    if images:
        _show_images_axes(images, current_frame, img_axes)

    # vertical lines for current frame on time series
    vline_joints = None
    vline_ee = None
    if ax_joints.lines:
        vline_joints = ax_joints.axvline(current_frame, color="r")
    if ax_ee.lines:
        vline_ee = ax_ee.axvline(current_frame, color="r")

    # Slider
    ax_slider = fig.add_axes([0.2, 0.01, 0.55, 0.03])
    slider = Slider(ax_slider, "frame", 0, n_frames - 1, valinit=current_frame, valfmt="%0.0f")

    def update(val):
        frame = int(slider.val)
        # update images
        if images:
            _show_images_axes(images, frame, img_axes)
        # update vlines
        if vline_joints is not None:
            vline_joints.set_xdata(frame)
        if vline_ee is not None:
            vline_ee.set_xdata(frame)
        # update current actions text
        if actions is not None:
            if frame < actions.shape[0]:
                a = actions[frame]
                action_text.set_text(np.array2string(a, precision=3, separator=", "))
            else:
                action_text.set_text("")
        fig.canvas.draw_idle()

    slider.on_changed(update)

    # Prev / Next buttons
    axprev = fig.add_axes([0.02, 0.01, 0.08, 0.04])
    axnext = fig.add_axes([0.12, 0.01, 0.08, 0.04])
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

    plt.suptitle(f"Simple Viewer: {os.path.basename(h5_path)} :: {episode}")
    plt.tight_layout(rect=[0, 0.05, 1, 0.97])
    plt.show()

    adapter.close()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--file", required=True, help="HDF5 dataset path")
    p.add_argument("--episode", default=None, help="Episode id (optional)")
    args = p.parse_args()
    viewer_main(args.file, args.episode)
