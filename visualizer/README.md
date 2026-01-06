# 轨迹数据统一可视化工具

## 项目概述

这是一个为 **RLinf** 项目构建的轨迹数据统一可视化工具，支持：
- ✅ **多格式数据适配**：HDF5、Zarr、Parquet 等（目前主要支持 HDF5）
- ✅ **Schema-Guided 数据读取**：通过 YAML 配置文件指导数据提取
- ✅ **数据质量检查**：离散值、跳帧、缺失值检测
- ✅ **单轨迹增强可视化**：图像、传感器、动作时间序列
- ✅ **多轨迹批量分析**：轨迹分布、质量对比、统计报告
- ✅ **统一命令行接口**：易用的 CLI 工具

## 核心组件

```
toolkits/visualizer/
├── analysis.py                      # 数据质量检查模块
│   ├── DataQualityChecker           # 检查器类
│   ├── AnomalyType                  # 异常类型枚举
│   └── Anomaly                      # 异常记录数据类
├── adapters/
│   ├── base.py                      # 抽象适配器接口
│   └── hdf5_adapter.py              # HDF5 适配器实现
├── schema_loader.py                 # YAML Schema 解析（现有）
├── enhanced_simple_viewer.py        # 单轨迹增强可视化
├── batch_viewer.py                  # 多轨迹批量分析
├── visualizer_main.py               # 统一命令行入口
├── test_integration.py              # 集成测试脚本
├── USAGE_GUIDE.md                   # 详细使用指南
└── schema/
    └── hdf5_example.yaml            # Schema 配置示例
```

## 快速开始

### 1. 查看数据集信息

```bash
python visualizer_main.py info --file data.h5
```

输出数据集中的所有 episode 和数据字段结构。

### 2. 单轨迹增强可视化

```bash
python visualizer_main.py single --file data.h5 \
    --schema schema.yaml \
    --episode episode_0
```

显示：
- 📸 多视角图像（grid 布局）
- 📈 时间序列曲线（关节、末端位姿、动作）
- 🚨 异常标记（红色 X 表示检测到的异常）
- 📊 实时质量信息

### 3. 多轨迹批量分析

```bash
python visualizer_main.py batch --file data.h5 \
    --schema schema.yaml \
    --output reports/
```

生成：
- 📊 9 个分析图表（长度分布、质量评分、异常统计等）
- 📝 文本报告（batch_analysis_report.txt）

### 4. 数据质量检查

```bash
python visualizer_main.py check --file data.h5 \
    --output reports/ \
    --outlier-threshold 3.0 \
    --frame-drop-threshold 2.0
```

生成：
- 📋 JSON 格式的详细报告

## 主要功能特性

### 数据质量检查

#### 离散值检测 (Outlier Detection)
- 方法：Z-score 统计分析
- 公式：`Z-score = |x - mean| / std`
- 默认阈值：3.0（标准差倍数）
- 应用：检测数据中的异常波动

#### 跳帧检测 (Frame Drop Detection)
- 方法：时间戳间隔异常检测
- 流程：计算相邻帧时间差，检测显著偏离
- 默认阈值：2.0（倍数）
- 应用：发现数据采集中断或同步失败

#### 缺失值检测 (Missing Value Detection)
- 方法：NaN 值比例计算
- 默认阈值：1%
- 应用：评估数据完整性

### 可视化特性

#### 单轨迹视图
```
┌─────────────────────────────────────┐
│ 📸 多摄像头图像（实时显示）         │
├─────────────────────────────────────┤
│ 📈 关节位置曲线 │ 📈 末端位姿曲线    │
├─────────────────────────────────────┤
│ 📈 动作曲线    │ 📈 其他传感器       │
├─────────────────────────────────────┤
│ ◄ Prev  [====●================] Next ►
└─────────────────────────────────────┘
```

#### 多轨迹分析视图
```
┌──────────────────────────┬──────────────────────────┐
│ 轨迹长度分布             │ 轨迹时长分布             │
├──────────────────────────┼──────────────────────────┤
│ 帧率分布                 │ 异常数分布               │
├──────────────────────────┼──────────────────────────┤
│ 轨迹长度 vs 异常数       │ 关节位置范围             │
├──────────────────────────┼──────────────────────────┤
│ 异常类型分布             │ 质量热力图               │
└──────────────────────────┴──────────────────────────┘
```

## YAML Schema 配置示例

```yaml
# 基本配置
mode: single_episode
fps: 30

# 图像传感器
vision_sensor:
  type: rgb
  prefix: observations/images
  key: ["head", "left_wrist", "right_wrist"]

# 关节传感器（本体感受）
proprioception_sensor:
  type: proprioception
  prefix: observations
  key: ["qpos", "qvel"]

# 末端执行器
end_effector:
  type: pose
  prefix: observations
  key: ["ee_pose"]

# 动作
action:
  type: joint_position
  key: ["action"]
```

## 使用场景

### 场景 1：新数据集验证
```bash
# 快速检查数据是否正确读取和质量如何
python visualizer_main.py info --file new_data.h5
python visualizer_main.py single --file new_data.h5 --episode episode_0
```

### 场景 2：数据清理与筛选
```bash
# 批量检查所有轨迹，找出有问题的轨迹
python visualizer_main.py batch --dir data_dir/ --schema schema.yaml --output analysis/

# 查看分析报告，识别异常轨迹
cat analysis/batch_analysis_report.txt
```

### 场景 3：数据集统计分析
```bash
# 生成详细的统计信息
python visualizer_main.py batch --file data.h5 --schema schema.yaml

# 通过可视化图表评估数据分布的均衡性和质量
# - 轨迹长度是否均匀？
# - 是否存在大量异常？
# - 关键数据的分布是否合理？
```

## HDF5 文件布局支持

工具自动检测并支持以下常见布局：

### 1. 单文件单轨迹
```
/
├── timestamps [N]
├── images/
│   ├── head [N, H, W, C]
│   └── wrist [N, H, W, C]
├── observations/
│   ├── qpos [N, D]
│   └── qvel [N, D]
└── actions [N, A]
```

### 2. 多轨迹在顶层
```
/
├── episode_0/
│   ├── timestamps [N1]
│   ├── images/...
│   └── actions [N1, A]
├── episode_1/
│   └── ...
└── episode_N/
    └── ...
```

### 3. 容器式（LIBERO 风格）
```
/
└── data/
    ├── demo_0/
    │   └── ...
    ├── demo_1/
    │   └── ...
    └── demo_N/
        └── ...
```

## Python API 使用

### 简单读取
```python
from adapters.hdf5_adapter import HDF5Adapter

adapter = HDF5Adapter("data.h5")
seq = adapter.read_sequence("episode_0")
print(seq.keys())  # ['timestamps', 'images', 'robot_state', 'actions']

# 访问数据
images = seq["images"]["head"]  # (N, H, W, C)
qpos = seq["robot_state"]["qpos"]  # (N, D)

adapter.close()
```

### 质量检查
```python
from analysis import DataQualityChecker

checker = DataQualityChecker(outlier_threshold=3.0)
result = checker.check_sequence(data, timestamps)

print(f"异常数: {result['num_anomalies']}")
for anom in result['anomalies']:
    print(f"  {anom.key} at frame {anom.frame}: {anom.description}")

checker.print_summary()
```

### 批量分析
```python
from batch_viewer import BatchAnalyzer

analyzer = BatchAnalyzer(schema_path="schema.yaml")
analyzer.analyze_file("data.h5")
analyzer.generate_report(output_dir="./reports")
```

## 测试验证

运行集成测试：
```bash
python test_integration.py
```

预期输出：
```
✓ PASS: Schema Loader
✓ PASS: Data Quality Checker
✓ PASS: HDF5 Adapter
✓ PASS: Integration Test

总体: 4/4 测试通过
```

## 性能特点

| 操作 | 时间复杂度 | 空间复杂度 | 备注 |
|------|-----------|-----------|------|
| 列出 episodes | O(1) | O(n) | n = episode 数 |
| 读取单个轨迹 | O(m) | O(m) | m = 轨迹数据量 |
| 质量检查 | O(m) | O(1) | 流式处理 |
| 批量分析 | O(n*m) | O(n) | 可扩展处理 |

## 扩展指南

### 添加新的数据格式适配器
1. 在 `adapters/` 目录创建新文件
2. 继承 `DatasetAdapter` 基类
3. 实现 `list_episodes()`, `get_episode_meta()`, `read_sequence()`

### 自定义异常检测
1. 扩展 `DataQualityChecker` 类
2. 添加新的 `_check_*` 方法
3. 在 `check_sequence()` 中调用

### 自定义可视化
1. 修改 `enhanced_simple_viewer.py` 中的 `_plot_timeseries()` 等函数
2. 或在 `batch_viewer.py` 中扩展 `_visualize_report()` 方法

## 依赖项

```
h5py>=3.0.0
numpy>=1.22
matplotlib>=3.0
PyYAML>=5.3
```

## 文档与帮助

- **详细使用指南**：查看 `USAGE_GUIDE.md`
- **命令行帮助**：`python visualizer_main.py -h`
- **子命令帮助**：`python visualizer_main.py single -h`

## 常见问题

**Q: 如何处理不同采样率的多个传感器？**
A: 在 schema.yaml 中为每个传感器指定 `timestamp` 字段，工具会自动同步。

**Q: 可视化很慢怎么办？**
A: 使用 `--episode` 指定单个轨迹，或在 `enhanced_simple_viewer.py` 中降低图像分辨率。

**Q: 如何批量导出图表？**
A: 在 `single` 命令中加 `--save` 参数，或在 `batch_viewer.py` 中调整输出格式。

## 许可证

遵循 RLinf 项目许可证

## 更新日志

### v1.0 (2025-12-02)
- ✨ 初始发布
- ✨ HDF5 适配器实现
- ✨ 数据质量检查模块
- ✨ 单轨迹增强可视化
- ✨ 多轨迹批量分析
- ✨ 统一命令行接口
- ✨ 完整测试覆盖

## 作者与贡献

RLinf 可视化团队

---

**开始使用：** `python visualizer_main.py --help`
