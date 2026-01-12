# ROBOT TOOLS
用来放机器人相关的代码

## Piper Collect 使用说明（piper_collect.py）

此脚本用于交互式采集 Piper 机械臂与多路 RealSense 摄像头的数据，按空格开始/结束一次轨迹采集，生成 HDF5 文件保存观测与动作信息。

快速示例：

```bash
conda activate piper_tools
python piper_collect.py --use_head2_cam --use_right_cam --outdir data --img_size 480 --sample_dt 0.05
```

- 按 `SPACE` 开始录制轨迹，按 `SPACE` 停止并保存 HDF5（及旁侧 JSON 元数据）。
- 按 `q` 退出程序。

可选参数说明（简要）：
- `--outdir, -o`: 保存目录（默认 `data`）。
-- `--img_size`: 方形图片尺寸（默认 480）。注意：`--img_size` 在对图像启用 resize 时生效，用于指定缩放后的方形尺寸；若不需要对图像进行 resize，请不要使用该选项以保留相机原始分辨率。
- `--sample_dt`: 采样周期（秒，默认 0.05）。
- `--use_head2_cam`/`--use_right_cam`/`--use_left_cam`: 是否启用对应摄像头。
- `--camera_extrinsics`: JSON 文件路径或 JSON 字符串，用于手动传入 camera 外参（可选）。

实现细节与注意事项：
- 程序启动后会对 Piper SDK 进行短暂的 `warmup`，等待非零的反馈数据以避免初始样本全 0。若 3 秒内仍无有效反馈会打印警告并继续。
- 程序默认使用 `deployment.Robot.Piper.PiperInterface` 获取机械臂状态，使用 `deployment.Camera.rs_camera.MultiRealSense` 获取多路相机图像。
- 为减少运行时日志噪声，采样/动作填充类中使用 `logging`，默认 INFO 级别；如需更详细调试信息请设置 `logging` 为 DEBUG。

## HDF5 数据格式解读（piper_collect 产生的文件）

HDF5 文件（例如 `data/traj_<ts>.h5`）包含以下主要路径和数据集：

- `observation/images/<cam>/color` : (N, H, W, C) uint8 — 彩色图像序列。
- `observation/images/<cam>/depth` : (N, H, W) float32 — 深度图（单位：米）。
- `observation/images/<cam>/timestamp` : (N,) int64 — 每帧图像时间戳（毫秒，自 epoch）。

- `observation/proprioception/joint_timestamp` : (N,) int64 — 机械臂关节反馈的时间戳（毫秒，自 epoch，由 SDK 提供并转换为 ms）。
- `observation/proprioception/joints` : (N,6) float64 — 6 个关节角（弧度）。
- `observation/proprioception/eef` : (N,6) float64 — 末端位姿（X,Y,Z 米；RX,RY,RZ 弧度）。
- `observation/proprioception/gripper` : (N,) float64 — 夹爪归一化位置（0-1）。

- `action_joint_abs` : (N,7) float64 — 记录“后一步”绝对目标：6 个关节角 + 夹爪（即 action[t] == joints[t+1] + gripper[t+1] 的组合）。
- `action_joint_rel` : (N,7) float64 — 记录后一步与当前的差分（next - current）：前 6 个为关节增量（rad），最后一列为夹爪增量。
- `action_eef_abs` : (N,7) float64 — 后一步末端绝对目标（X,Y,Z, RX,RY,RZ + gripper）。
- `action_eef_rel` : (N,7) float64 — 后一步与当前末端的差分（米/弧度 + gripper 增量）。

语义说明：为了能将每一帧样本与“执行动作”对齐，文件中 `action_*_abs[t]` 保存的是时间 t+1 的观测（即下一步目标），而 `*_rel[t]` = `*_abs[t] - observation[t]`（若连续两帧无变化则差分为 0）。

元数据（`meta/`）：
- `meta/camera_intrinsics/<cam>`：为每个启用摄像头保存 `width,height,fx,fy,cx,cy,scale` 等数值数据集，属性 `provided` 表示是否从摄像头 sentinel 获取到有效内参。
- `meta/camera_extrinsics`：如启动时通过 `--camera_extrinsics` 提供，会以 4x4 矩阵或 JSON 属性存储，否则为每个相机创建占位组并标注 `provided=False`。

注意事项：
- HDF5 中的时间戳已统一为毫秒整数（int64），可直接用于对齐或转换为秒（除以 1000）。
- h5 浏览器（如 h5web）可能会以科学计数法展示数值或省略多维数组的完整显示，但文件内数据未被截断；如要确认数据完整性请使用 `h5py` 或脚本导出为 PNG/CSV。
- `action_*` 当前使用 float64。若需要节省空间，可考虑将其与 `joints/eef` 改为 float32（非向后兼容更改）。

示例：快速读取首帧 joints 与 action

```python
import h5py
f = h5py.File('data/traj_<ts>.h5','r')
j0 = f['observation/proprioception/joints'][0]
a0 = f['action_joint_abs'][0]
print('j0', j0)
print('a0', a0)
f.close()
```

如需我把 README 中的该部分另存为单独文档或生成 API 使用示例脚本，告诉我就行。 
