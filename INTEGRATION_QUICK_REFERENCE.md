# 采集脚本时间戳同步集成快速参考

## 集成内容

`piper_collect.py` 已集成 `TimestampMonitor` 实时时间戳同步监控功能。

### 新增功能

1. **实时 FPS 监控**：在采集过程中每 100 帧打印一次各数据流的帧率
2. **时间偏移检测**：监控相机与机械臂间的时间戳偏移（ms）
3. **抖动分析**：检测高频抖动并预警
4. **最终报告**：采集完成时输出完整的同步性能统计

### 运行示例

```bash
python3 piper_collect.py --outdir data/task_pick --sample_dt 0.05
```

### 控制台输出示例

**采集过程（每 100 帧打印一次）**：
```
[SYNC]   [Frame 100] Timestamp Synchronization Monitor
[SYNC]   ============================================================
[SYNC]   Frame Rates (FPS):
[SYNC]     camera_head_fps: 29.8
[SYNC]     camera_right_fps: 30.1
[SYNC]     robot_fps: 199.5
[SYNC]   
[SYNC]   Time Offsets (ms, camera - robot):
[SYNC]     head_offset_ms:
[SYNC]       mean: +45.23, std: 1.45, range: [41.50, 49.67]
[SYNC]     right_offset_ms:
[SYNC]       mean: +48.56, std: 1.32, range: [44.89, 52.34]
[SYNC]   
[SYNC]   Inter-Frame Jitter (normalized):
[SYNC]     camera_head_jitter: 0.032
[SYNC]     camera_right_jitter: 0.028
[SYNC]     robot_jitter: 0.008
[SYNC]   ============================================================
```

**采集完成时（最终报告）**：
```
[FINAL]   [Frame 500] Timestamp Synchronization Monitor
[FINAL]   ============================================================
[FINAL]   Frame Rates (FPS):
[FINAL]     camera_head_fps: 29.7
[FINAL]     camera_right_fps: 30.0
[FINAL]     robot_fps: 200.0
[FINAL]   ... (完整统计) ...
```

### 告警示例

如果检测到同步问题会输出警告：
```
WARNING Sync warning: Large time offset on head_vs_robot: 145.3 ms
WARNING Sync warning: High jitter on camera_right: 0.185
```

---

## 配置说明

### 调整监控参数

在 `piper_collect.py` 的 `run_collector` 函数中修改：

```python
# 初始化时（第 ~515 行）
monitor = TimestampMonitor(max_history=300)  # 保留最近 300 帧的统计

# 采样循环中（第 ~748 行）
if monitor is not None and writer.length % 100 == 0:  # 每 100 帧打印
    monitor.report(prefix="[SYNC] ")
    warnings = monitor.check_sync_quality(
        max_offset_ms=100.0,    # 警告阈值：时间偏移 > 100 ms
        max_jitter=0.2          # 警告阈值：抖动 > 0.2（20%）
    )
```

### 禁用实时监控

如果不需要实时输出，注释掉报告代码：

```python
# if monitor is not None and writer.length % 100 == 0:
#     monitor.report(prefix="[SYNC] ")
#     ...
```

---

## 后续分析

采集完成后，进一步分析时间戳数据：

```bash
# 详细离线分析
python3 analyze_timestamp_sync.py --task_dir data/task_pick --plot --verbose

# 生成时间戳曲线图（需要 matplotlib）
# 输出文件：traj_*.png
```

---

## 典型诊断场景

| 场景 | 实时输出提示 | 后续处理 |
|------|---------|---------|
| **正常同步** | Jitter < 0.1，offset 稳定 ±20 ms | 直接使用数据 |
| **轻微抖动** | Jitter 0.1-0.15，offset 波动 | 运行 analyze_timestamp_sync.py 详细检查 |
| **严重不同步** | Offset > 100 ms 或 Jitter > 0.2 | 检查硬件连接，降低采样率 |
| **时钟漂移** | Offset 逐帧增长 | 检查 NTP 同步，或事后校正 |

---

## 故障排查

### 问题 1：采集过程中没有看到 [SYNC] 输出

**原因**：可能是还没有采集 100 帧
- **解决**：继续录制，或降低 `sample_dt` 加快采样速度

### 问题 2：监控显示 "Frame 0" 或值为 0

**原因**：初期数据不足，统计不准确
- **解决**：等待至少 50-100 帧后数据会稳定

### 问题 3：看到大量的 "Error adding sample to monitor" 日志

**原因**：相机命名不匹配或时间戳格式不兼容
- **解决**：检查相机名称是否为 'head', 'right', 'left' 等

---

## 集成代码位置

- **导入**（第 ~37 行）：`from timestamp_monitor import TimestampMonitor`
- **初始化**（第 ~508 行）：`monitor = None`
- **创建**（第 ~631 行）：采集开始时 `monitor = TimestampMonitor(...)`
- **更新**（第 ~750 行）：每次 append 后 `monitor.add_sample(...)`
- **清理**（第 ~702 行）：采集结束时 `monitor = None`

---

更多详见 `TIMESTAMP_SYNC_GUIDE.md`。
