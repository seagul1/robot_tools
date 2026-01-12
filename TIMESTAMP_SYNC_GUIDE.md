# 多视角相机与机械臂时间戳同步策略指南

## 概述

在采集多视角 RealSense 图像与机械臂状态数据时，确保时间戳对齐至关重要，以便后续进行准确的运动学和图像处理。本文档介绍如何诊断和解决时间戳同步问题。

---

## 1. 当前采集系统中的时间戳信息

### 1.1 时间戳来源

根据 `piper_collect.py`，系统采集以下时间戳信息：

| 来源 | 路径 | 说明 |
|------|------|------|
| **RealSense 相机** | `/observation/images/<cam>/timestamp` | 每个相机的帧时间戳（由相机硬件提供，单位：毫秒） |
| **机械臂关节状态** | `/observation/proprioception/joint_timestamp` | 关节位置反馈的时间戳（由 Piper SDK 提供，单位：毫秒） |
| **数据采集**| 采集循环中的 `now = time.time()` | 采集线程的采样时间（秒，用于采样控制） |

### 1.2 时间戳单位

- **RealSense 硬件时间戳**：一般为相机内部计时器，与系统时钟**不同步**，精度通常为微秒。
- **Piper SDK 时间戳**：Piper 的关节反馈时间戳，与相机时间戳不在同一坐标系。
- **系统采样时间**：采集循环中的 `time.time()`，基于系统时钟（秒）。

---

## 2. 诊断同步问题

### 2.1 使用 `analyze_timestamp_sync.py` 进行离线分析

采集完成后，使用以下命令分析单个轨迹文件的时间戳对齐：

```bash
# 分析单个文件
python3 analyze_timestamp_sync.py --traj_file data/traj_1234567890.h5 --verbose

# 分析整个任务文件夹（多个轨迹）
python3 analyze_timestamp_sync.py --task_dir data/task_pick_and_place --verbose

# 生成时间戳曲线图（需要 matplotlib）
python3 analyze_timestamp_sync.py --task_dir data/task_pick_and_place --plot
```

### 2.2 报告解读

该脚本会输出以下信息：

#### 2.2.1 Per-Stream Statistics

```
--- Per-Stream Statistics (in milliseconds) ---

  camera_head1:
    Frames: 500
    Mean inter-frame (ΔT): 33.45 ms        # 平均帧间隔
    Std inter-frame: 1.23 ms              # 帧间隔标准差（越小越好）
    Min/Max inter-frame: 30.12 / 35.67 ms # 最小/最大帧间隔
    Duration: 16723.50 ms

  robot_joint:
    Frames: 500
    Mean inter-frame (ΔT): 5.00 ms        # 关节状态更新频率（200 Hz）
    Std inter-frame: 0.15 ms
    Min/Max inter-frame: 4.85 / 5.20 ms
    Duration: 2498.50 ms
```

**解读**：
- **Mean inter-frame (ΔT)**：理想情况下应为 1000/FPS（如 30 fps → 33.3 ms）。
- **Std inter-frame**：标准差越小，采样越规律。大于 ±10% 的变异表示存在卡顿或丢帧。
- **Duration**：总采样时长（ms）。

#### 2.2.2 Cross-Stream Time Offsets

```
--- Cross-Stream Time Offsets (camera - robot) ---

  head1_vs_robot:
    First frame offset: +45.32 ms         # 起始时的相机时间 - 机械臂时间
    Last frame offset: +48.67 ms          # 最后一帧的偏移
    Mean offset: +46.50 ms                # 平均偏移（应该相对稳定）
```

**解读**：
- **Mean offset**：两个数据流之间的**恒定时间差**。一般由硬件时钟不同步或网络延迟引起。
  - 如果恒定 ±50 ms 以内，**可以接受**（通过后处理对齐）。
  - 如果随时间线性增长，表示**时钟频率不匹配**（需要调整）。
  - 如果抖动很大，表示**网络或驱动问题**。

#### 2.2.3 Issues & Warnings

```
--- Issues & Warnings ---
  1. High jitter on camera_head2: normalized_std=0.145
  2. Frame count mismatch: cameras=[500, 495], robot=500
```

**常见问题**：
| 问题 | 原因 | 解决方案 |
|------|------|---------|
| Frame count mismatch | 采集期间掉帧 | 检查相机/机械臂连接，降低采样率 |
| High jitter (>0.1) | 网络延迟、驱动问题 | 检查 USB/网络连接，增加缓冲 |
| Large offset drift | 时钟频率不匹配 | 使用时钟同步工具（NTP）或后处理补偿 |
| Offset >100 ms | 严重的延迟差异 | 检查硬件驱动，可能需要主从同步 |

---

## 3. 时间戳同步策略

### 3.1 软时间戳同步（推荐用于开发/演示）

**原理**：使用采集线程中的系统时钟 (`time.time()`) 作为参考，将所有数据映射到同一时间坐标系。

**步骤**：

1. 记录采集线程的采样时刻 `t_sample = time.time()`（秒）。
2. 查询相机时间戳 `cam_ts`（毫秒）和机械臂时间戳 `robot_ts`（毫秒）。
3. 计算相对于采样时刻的时间差：
   ```python
   t_sample_ms = t_sample * 1000
   cam_delay = cam_ts - t_sample_ms
   robot_delay = robot_ts - t_sample_ms
   ```
4. 在 HDF5 中存储这些**相对时间戳**或进行事后对齐。

**优点**：
- 易于实现。
- 不需要额外硬件。

**缺点**：
- 精度有限（±10 ms 范围内）。
- 对于时序敏感的应用（如精密控制）不够。

---

### 3.2 硬件时间同步（高精度方案）

**原理**：利用 PTP（精密时间协议）或 NTP 同步所有设备的系统时钟。

**步骤**：

1. 启用网络中的 PTP 主时钟（如网络交换机或专用设备）。
2. 在采集计算机和相机/机械臂所在的设备上启用 PTP 客户端：
   ```bash
   sudo apt install linuxptp
   sudo ptp4l -i eth0 -m -p /var/run/ptp4l.pid
   ```
3. 验证时钟同步：
   ```bash
   phc_ctl eth0 get
   ```
4. 一旦时钟同步，所有时间戳将共享同一参考帧。

**优点**：
- 精度可达微秒级。
- 适合工业应用和精密控制。

**缺点**：
- 需要网络基础设施支持。
- 配置复杂。

---

### 3.3 时间戳对齐后处理（离线处理）

**原理**：采集时保存原始时间戳，后处理时进行时间对齐和插值。

**步骤**（在 `visualize_data.py` 或自定义脚本中）：

```python
import h5py
import numpy as np
from scipy import interpolate

def align_timestamps(h5_file):
    with h5py.File(h5_file, 'r') as f:
        obs = f['observation']
        
        # 获取所有时间戳（单位：毫秒）
        robot_ts = obs['proprioception']['joint_timestamp'][:]
        camera_names = list(obs['images'].keys())
        cam_ts_dict = {cam: obs['images'][cam]['timestamp'][:] for cam in camera_names}
        
        # 时间基准：取机械臂时间戳作为基准
        t_ref = robot_ts
        
        # 为每个相机创建插值函数
        interp_funcs = {}
        for cam_name, cam_ts in cam_ts_dict.items():
            # 创建 (cam_ts -> 帧索引) 的映射
            frame_indices = np.arange(len(cam_ts))
            # 创建反向映射：给定 robot_ts，查找对应的相机帧
            interp_funcs[cam_name] = interpolate.interp1d(
                cam_ts, frame_indices, 
                kind='nearest',  # 最近邻插值
                fill_value='extrapolate'
            )
        
        # 对于每个机械臂时间戳，找到对应的相机帧
        aligned_indices = {}
        for cam_name, interp_func in interp_funcs.items():
            indices = np.clip(interp_func(t_ref).astype(int), 0, len(cam_ts_dict[cam_name]) - 1)
            aligned_indices[cam_name] = indices
        
        print(f"Time alignment complete. Aligned {len(t_ref)} samples.")
        print(f"Camera frame indices aligned to robot timestamps:")
        for cam_name, indices in aligned_indices.items():
            print(f"  {cam_name}: {indices[:5]} ... {indices[-5:]}")
        
        return aligned_indices

# 使用
aligned = align_timestamps('data/traj_1234567890.h5')
```

**优点**：
- 灵活，支持多种对齐策略。
- 不需要改动采集代码。

**缺点**：
- 需要事后处理。
- 可能损失细节信息。

---

## 4. 实时监控与调试

### 4.1 集成 `TimestampMonitor` 到采集脚本

在 `piper_collect.py` 的采集循环中添加：

```python
from timestamp_monitor import TimestampMonitor

# 在 run_collector 中初始化
monitor = TimestampMonitor(max_history=200)

# 在采集循环中每次 append 后调用
if recording and (now - last_sample >= sample_dt):
    # ... 现有的采样逻辑 ...
    writer.append(colors, depths, cam_ts, ts, joints_arr, float(joint_ts), eef_arr, grip)
    
    # 添加监控
    monitor.add_sample(cam_ts, float(joint_ts), writer.length)
    
    # 每 100 帧打印报告
    if writer.length % 100 == 0:
        monitor.report()
        warnings = monitor.check_sync_quality(max_offset_ms=100.0, max_jitter=0.2)
        for w in warnings:
            logging.warning(f"Sync issue: {w}")
```

### 4.2 实时输出示例

```
  [Frame 100] Timestamp Synchronization Monitor
  ------------------------------------------------------------
  Frame Rates (FPS):
    camera_head1_fps: 29.8
    camera_right_fps: 30.2
    robot_fps: 199.5

  Time Offsets (ms, camera - robot):
    head1_offset_ms:
      mean: +45.32, std: 2.15, range: [40.12, 51.23]
    right_offset_ms:
      mean: +48.67, std: 1.98, range: [43.45, 53.12]

  Inter-Frame Jitter (normalized):
    camera_head1_jitter: 0.032
    camera_right_jitter: 0.041
    robot_jitter: 0.012
  ------------------------------------------------------------
```

---

## 5. 最佳实践建议

| 场景 | 推荐方案 |
|------|---------|
| **开发/演示** | 软时间戳同步 + 离线分析 |
| **中等精度需求** | 系统时钟 NTP 同步 + 监控抖动 |
| **高精度工业应用** | PTP 同步 + 硬件时间戳 |
| **离线处理训练数据** | 时间戳对齐插值 + 验证一致性 |

---

## 6. 常见问题 (FAQ)

### Q1: 为什么相机时间戳和机械臂时间戳完全不同？
**A:** 相机和机械臂使用不同的时钟源。相机通常使用内部计时器，而机械臂（Piper）使用系统时钟。建议：
- 记录采集时的系统时间作为参考。
- 事后通过对齐相对时间差来同步。

### Q2: 帧间隔标准差很大怎么办？
**A:** 可能原因：
- USB 网络拥塞：减少其他数据传输。
- 相机掉帧：检查 USB 功率、线缆质量。
- 驱动问题：更新 RealSense SDK。

解决方案：降低采样率或增加缓冲区大小。

### Q3: 时间偏移在增加（线性漂移）怎么办？
**A:** 时钟频率不匹配（时钟跑快或跑慢）。解决方案：
- 启用 NTP 或 PTP 同步。
- 估计频率比（漂移速率），事后进行线性补偿：
  ```python
  drift_rate = (final_offset - initial_offset) / num_frames
  corrected_ts = original_ts + drift_rate * frame_index
  ```

### Q4: 采集过程中如何动态调整采样率以保持同步？
**A:** 监控帧计数和时间戳偏移：
```python
if offset > max_offset_threshold:
    sample_dt *= 1.05  # 减速 5%
elif jitter > max_jitter_threshold:
    # 增加缓冲或检查硬件
```

---

## 7. 文件结构参考

采集系统中时间戳相关的文件：

```
/home/zyj/robot_tools/
├── piper_collect.py                 # 采集脚本（含时间戳记录）
├── analyze_timestamp_sync.py         # 离线时间戳分析工具
├── timestamp_monitor.py              # 实时监控模块
├── tests/test_writer.py             # 写入器单元测试
└── data/
    └── task_pick_and_place/
        ├── traj_1234567890.h5       # 轨迹数据（含时间戳）
        └── traj_1234567890_timestamps.png  # 时间戳可视化（使用 --plot 生成）
```

---

## 总结

- **采集阶段**：使用 `TimestampMonitor` 实时监控同步状态。
- **验证阶段**：用 `analyze_timestamp_sync.py` 离线分析时间戳质量。
- **处理阶段**：根据同步偏移情况选择软同步、硬同步或事后对齐。
- **长期改进**：收集多个轨迹数据，分析同步模式，逐步优化系统参数。

有问题或需要自定义同步策略，请参考相应模块的源代码和文档。

