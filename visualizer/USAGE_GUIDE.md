"""
轨迹数据统一可视化工具使用指南

概述
====
这是一个为支持不同数据格式（HDF5、Parquet 等）的统一可视化工具，通过 YAML 配置文件
指导数据读取，允许对单条轨迹和多条轨迹进行数据质量检查和可视化。

核心特性
========

1. **Schema-Guided 数据读取**
   - 使用 YAML 配置文件描述数据结构
   - 自动提取指定的数据字段（图像、传感器、动作等）
   - 支持多种 HDF5 文件布局

2. **数据质量检查**
   - 离散值检测（Outlier Detection）- 基于 Z-score
   - 跳帧检测（Frame Drop Detection）- 检测时间戳异常
   - 缺失值检测（Missing Value Detection）
   - 异常标记和严重程度评分

3. **单轨迹可视化**
   - 多视角图像显示（grid 布局）
   - 时间序列数据曲线（关节位置、末端位姿、动作等）
   - 异常帧标记（红色 X 标记）
   - 交互式播放控件（前/后/滑条）
   - 实时数据质量信息显示

4. **多轨迹汇总分析**
   - 轨迹长度分布直方图
   - 轨迹时长和帧率分布
   - 异常数分布和类型统计
   - 关键数据分布（关节范围、末端位姿等）
   - 轨迹质量评分热力图
   - 生成详细的分析报告

快速开始
========

### 1. 查看数据集信息

首先，查看 HDF5 文件的结构和包含的 episode：

    python visualizer_main.py info --file /path/to/data.h5

输出示例：
    Episode: episode_0
      images: ['head', 'left_wrist', 'right_wrist']
      robot_state: ['qpos', 'ee_pose']
      actions: shape=(1000, 7)

### 2. 单轨迹可视化（推荐）

使用增强版 simple_viewer 查看单个轨迹，包括数据质量检查：

    python visualizer_main.py single --file /path/to/data.h5 \\
        --schema /path/to/schema.yaml \\
        --episode episode_0

参数说明：
    --file          HDF5 数据文件路径（必需）
    --schema        YAML 配置文件路径（可选）
    --episode       Episode ID（可选，默认为第一个）
    --save          保存图表为 PNG（可选）

界面说明：
    - 顶部：多视角图像（根据 schema 指定）
    - 中部：关节位置曲线、末端位姿曲线、动作曲线
    - 下部：滑条控制播放位置，Prev/Next 按钮步进
    - 右侧：当前动作值、数据质量信息

红色 X 标记表示检测到的异常（离散值、缺失值等）。

### 3. 多轨迹批量分析

分析单个 HDF5 文件中的所有轨迹，并生成汇总报告：

    python visualizer_main.py batch --file /path/to/data.h5 \\
        --schema /path/to/schema.yaml \\
        --output /path/to/output_dir

或分析整个目录中的所有 HDF5 文件：

    python visualizer_main.py batch --dir /path/to/data_dir \\
        --schema /path/to/schema.yaml \\
        --output /path/to/output_dir

输出：
    - 可视化图表（9 个子图）
    - batch_analysis_report.txt（文本报告）

### 4. 数据质量检查（详细）

执行深度数据质量检查，生成 JSON 格式报告：

    python visualizer_main.py check --file /path/to/data.h5 \\
        --schema /path/to/schema.yaml \\
        --output /path/to/output_dir \\
        --outlier-threshold 3.0 \\
        --frame-drop-threshold 2.0

参数说明：
    --outlier-threshold         离散值检测阈值（Z-score）
    --frame-drop-threshold      跳帧检测阈值（倍数）
    --missing-value-threshold   缺失值比例阈值

YAML 配置文件格式
=================

示例：schema.yaml

    # 数据集配置
    mode: single_episode
    fps: 30
    
    # 图像配置
    vision_sensor:
      type: rgb
      timestamp: none
      prefix: observations/images
      key: ["head", "left_wrist", "right_wrist"]
    
    # 传感器配置
    proprioception_sensor:
      type: proprioception
      timestamp: none
      prefix: observations
      key: ["qpos", "qvel"]
    
    # 末端位姿
    end_effector:
      type: pose
      timestamp: none
      prefix: observations
      key: ["ee_pose", "ee_twist"]
    
    # 动作
    action:
      type: joint_position
      timestamp: none
      prefix: none
      key: ["action"]

关键字说明：

    vision_sensor:           图像传感器配置
      type:                  图像类型：rgb / depth / pointcloud
      key:                   需要读取的图像 key 列表
      prefix:                HDF5 文件中的路径前缀

    proprioception_sensor:   本体感受传感器（关节位置等）
      key:                   ["qpos", "qvel", ...]

    action:                  机器人动作
      type:                  动作类型

    timestamp:               时间戳 key（用于时间同步）

数据质量检查说明
===============

### 离散值检测（Outlier Detection）
使用 Z-score 方法检测数据中的离散值：
    Z-score = |x - mean| / std
    如果 Z-score > threshold（默认 3.0），则标记为异常

### 跳帧检测（Frame Drop Detection）
检测时间戳序列中的异常间隔：
    检查相邻时间戳的差值，如果显著偏离平均值，则标记为跳帧
    threshold 默认为 2.0（倍数）

### 缺失值检测（Missing Value Detection）
检测数据中的 NaN 值：
    如果缺失比例超过阈值（默认 1%），则标记警告

### 异常严重程度评分
异常由严重程度评分（0-1）标记，1 为最严重。

HDF5 文件布局
=============

工具支持三种常见的 HDF5 文件布局：

1. **单文件单轨迹**：
       /
       ├── timestamps [N]
       ├── images/
       │   ├── head [N, H, W, C]
       │   ├── left_wrist [N, H, W, C]
       │   └── right_wrist [N, H, W, C]
       ├── observations/
       │   ├── qpos [N, D]
       │   └── qvel [N, D]
       └── actions [N, A]

2. **多轨迹在顶层**：
       /
       ├── episode_0/
       │   ├── timestamps [N1]
       │   ├── images/...
       │   └── actions [N1, A]
       ├── episode_1/
       │   └── ...
       └── episode_N/
           └── ...

3. **容器式**（如 LIBERO）：
       /
       └── data/
           ├── demo_0/
           │   └── ...
           ├── demo_1/
           │   └── ...
           └── demo_N/
               └── ...

Python API 使用
===============

### 使用 HDF5Adapter 读取数据

    from adapters.hdf5_adapter import HDF5Adapter
    
    adapter = HDF5Adapter("data.h5")
    episodes = adapter.list_episodes()
    
    # 读取数据
    seq = adapter.read_sequence("episode_0")
    images = seq["images"]  # Dict[str, np.ndarray]
    robot_state = seq["robot_state"]  # Dict[str, np.ndarray]
    actions = seq["actions"]  # np.ndarray
    
    adapter.close()

### 使用 DataQualityChecker 检查质量

    from analysis import DataQualityChecker
    
    checker = DataQualityChecker(outlier_threshold=3.0)
    result = checker.check_sequence(data, timestamps)
    
    print(f"异常数: {len(result['anomalies'])}")
    for anom in result['anomalies']:
        print(f"  {anom.key}: {anom.type.value} at frame {anom.frame}")
    
    checker.print_summary()

### 批量分析

    from batch_viewer import BatchAnalyzer
    
    analyzer = BatchAnalyzer(schema_path="schema.yaml")
    analyzer.analyze_file("data.h5")
    analyzer.generate_report(output_dir="./reports")

常见问题
========

### Q1: 如何处理不同的 HDF5 文件布局？
A: 工具会自动检测文件布局。如果检测失败，可以修改 HDF5Adapter._detect_layout() 方法
   或手动指定正确的路径前缀在 schema.yaml 中。

### Q2: 如何处理多个摄像头的图像？
A: 在 schema.yaml 的 vision_sensor.key 列表中列出所有摄像头名称：
    key: ["camera_0", "camera_1", "camera_2"]

### Q3: 时间戳不同步怎么办？
A: 在 schema.yaml 中为每个传感器指定 timestamp 字段，工具会根据时间戳进行同步。
   目前 HDF5Adapter 是按帧索引对齐，后续可扩展为时间戳对齐。

### Q4: 如何自定义异常检测参数？
A: 使用 check 命令时指定参数：
    --outlier-threshold 2.5         更敏感的离散值检测
    --frame-drop-threshold 1.5      更敏感的跳帧检测

### Q5: 可视化显示很慢，如何加速？
A: 
    1. 使用 --episode 指定单个轨迹进行快速预览
    2. 在 enhanced_simple_viewer.py 中调整采样频率或降低图像分辨率
    3. 使用 batch viewer 而非逐个加载轨迹

扩展指南
========

### 添加新的适配器

1. 在 adapters/ 目录中创建新文件，实现 DatasetAdapter 接口
2. 实现以下方法：
    - list_episodes()
    - get_episode_meta()
    - read_sequence()
3. 在 visualizer_main.py 中添加相应的适配器选择逻辑

### 添加新的传感器类型

1. 在 schema_loader.py 中的 extract_visualization_fields() 函数中
   添加新的 sensor_name_sensor 处理逻辑
2. 在 enhanced_simple_viewer.py 中添加相应的可视化子图

### 自定义异常检测算法

1. 在 analysis.py 中扩展 DataQualityChecker 类
2. 添加新的检测方法（例如 _check_acceleration_anomaly）
3. 在 check_sequence() 中调用新方法

文件结构
========

    toolkits/visualizer/
    ├── schema_loader.py              # YAML schema 解析
    ├── analysis.py                   # 数据质量检查
    ├── adapters/
    │   ├── base.py                   # 抽象基类
    │   └── hdf5_adapter.py            # HDF5 适配器
    ├── enhanced_simple_viewer.py     # 单轨迹可视化
    ├── batch_viewer.py                # 多轨迹批量分析
    ├── visualizer_main.py             # 主入口脚本
    ├── schema/
    │   └── hdf5_example.yaml          # 配置示例
    └── requirements.txt               # 依赖

版本信息
========

Version: 1.0
Author: RLinf Visualization Team
Last Updated: 2025-12-02

更新日志
========

### v1.0 (2025-12-02)
- 初始版本发布
- 支持 HDF5 单文件/多文件读取
- 实现数据质量检查（离散值、跳帧、缺失值）
- 单轨迹增强可视化
- 多轨迹批量分析
- 统一命令行接口

许可证
======

本工具遵循 RLinf 项目许可证。
"""

# 本文件为文档文件，无可执行代码
