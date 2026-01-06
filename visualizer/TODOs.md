# Visualizer TODO 列表

> 生成于：2025-11-19

这是为轨迹数据可视化工具规划的可跟踪任务清单。每项包含：目的（Purpose）、要点（Key points）、交付物（Deliverables）和子任务（Subtasks）。

---

## 目录

- [1. 定义通用数据模型与元数据规范](#1-定义通用数据模型与元数据规范)
- [2. 实现数据后端适配器接口（HDF5/Zarr/Lerobot/ROS）](#2-实现数据后端适配器接口hdf5zarrlerobotros)
- [3. 实现时间戳与坐标系同步模块](#3-实现时间戳与坐标系同步模块)
- [4. 图像与多视角可视化组件](#4-图像与多视角可视化组件)
- [5. 机械臂状态可视化（末端位姿/关节曲线/力觉）](#5-机械臂状态可视化末端位姿关节曲线力觉)
- [6. 3D 场景与机器人轨迹重放模块](#6-3d-场景与机器人轨迹重放模块)
- [7. 交互控件（播放/暂停/拖动/速度/步进/过滤）](#7-交互控件播放暂停拖动速度步进过滤)
- [8. 大数据处理与性能优化（懒加载/分块/缓存）](#8-大数据处理与性能优化懒加载分块缓存)
- [9. 导出、快照、标注与测量工具](#9-导出快照标注与测量工具)
- [10. 插件/扩展 API 与数据集注册机制](#10-插件扩展-api-与数据集注册机制)
- [11. 测试用例与示例数据集（单元测试/端到端）](#11-测试用例与示例数据集单元测试端到端)
- [12. 文档、使用指南与用户体验设计](#12-文档使用指南与用户体验设计)

---

## 1. 定义通用数据模型与元数据规范

- 目的：建立统一的内部表示（schema），把不同来源/格式的数据映射为统一接口，便于上层可视化与分析复用。
- 要点：
	- 时间序列基元：`timestamp`、`frame_index`、`episode_id`。
	- 多模态字段：`images[{view_id, data, cam_intrinsics, cam_extrinsics}]`、`robot_state{joint_positions, joint_velocities, ee_pose, ee_twist}`、`actions`、`tactile{taxel_array, force_torque}`、`annotations`、`metadata`（采样频率、坐标系、单位、标定信息）。
	- 明确坐标系与单位（world, base_link, camera），并在 `metadata` 中保存标定矩阵与时间同步偏移。
	- 提供版本化 schema（JSON Schema / Protobuf），并包含字段映射规则以兼容命名差异。
- 交付物：
	- `visualizer/schema/trajectory_schema.json`（JSON Schema）
	- 字段映射示例 `visualizer/schema/field_mappings.md`
	- 简短文档 `visualizer/docs/data_model.md`
- 子任务：
	1. 列出常见字段与命名变体（根据已有数据集）。
	2. 设计 schema（初版 JSON Schema）。
	3. 编写字段映射与版本说明。 
	4. 定义最小可用数据子集（MVP 必需字段）。

---

## 2. 实现数据后端适配器接口（HDF5/Zarr/Lerobot/ROS）

- 目的：统一读取不同存储格式并映射到通用数据模型，支持按需读取与随机访问。
- 要点：
	- 抽象接口 `DatasetAdapter`：`list_episodes()`, `get_episode_meta()`, `read_sequence(episode, fields, time_range, chunk)`。
	- 支持 HDF5、Zarr、Lerobot、ROS bag 等后端，优先实现 HDF5 与 Zarr。
	- 处理变体：图像是单文件目录还是打包数组，压缩格式，索引/帧表。
- 交付物：
	- `visualizer/adapters/base.py`（接口定义）
	- `visualizer/adapters/hdf5_adapter.py`、`zarr_adapter.py`（示例实现）
	- 适配器使用说明 `visualizer/docs/adapters.md`
- 子任务：
	1. 定义 `DatasetAdapter` 抽象类。
	2. 实现 `HDF5` 适配器（支持按-frame 与按-chunk 读）。
	3. 实现 `Zarr` 适配器（利用 zarr.chunking）。
	4. 编写适配器单元测试与示例数据加载脚本。

---

## 3. 实现时间戳与坐标系同步模块

- 目的：对齐多传感器时间与坐标系，支持插值与变换，降低多源数据同步误差对可视化的影响。
- 要点：
	- 时间同步：处理不同采样率、时间偏移、丢帧；提供插值策略（nearest/linear/spline）。
	- 坐标转换：使用 `cam_extrinsics`, `ee_to_base` 等元数据进行变换，支持从任意坐标系投影到目标坐标系。
	- 提供可视化校验工具（时间差曲线、坐标投影叠加）。
- 交付物：
	- `visualizer/sync/time_sync.py`、`visualizer/sync/pose_transform.py`
	- 同步策略文档 `visualizer/docs/sync.md`
- 子任务：
	1. 设计时间对齐 API（resample_to_timestamps, align_on_events）。
	2. 实现坐标变换工具（基于 numpy/pytransform）。
	3. 编写插值单元并测试常见失真场景。

---

## 4. 图像与多视角可视化组件

- 目的：交互式查看多视角图像序列，支持 overlay（深度/语义/掩码/关键点）。
- 要点：
	- 可配置网格布局，显示摄像头 ID、时间戳、内参信息。
	- Overlay 支持（深度、语义分割、关键点、bounding-box）。
	- 导出视频/帧（png、mp4、gif）。
- 交付物：
	- 前端组件 `visualizer/ui/image_grid.py`（或 Web 前端实现目录）
	- Overlay 插件示例 `visualizer/plugins/overlays/*`
	- 使用与导出说明 `visualizer/docs/image_viewer.md`
- 子任务：
	1. 选择原型技术栈（Streamlit/Dash/React+three.js）。
	2. 实现本地原型（image grid + single view overlay）。
	3. 添加导出与基本交互（窗口缩放、ROI）。

---

## 5. 机械臂状态可视化（末端位姿/关节曲线/力觉）

- 目的：展示机器人状态随时间变化，便于调试控制策略与检测异常。
- 要点：
	- 时序曲线视图（每个关节/力传感器一条曲线），支持缩放与选择时间窗口。
	- 数值/表格视图显示当前帧信息（位置、姿态、关节角）。
	- 异常检测告警（NaN、超限、突变）。
- 交付物：
	- `visualizer/ui/timeseries.py`（时序图表组件）
	- 告警/校验工具 `visualizer/utils/quality_checks.py`
- 子任务：
	1. 选用绘图库（plotly/bokeh/matplotlib）。
	2. 实现时序组件与交互（缩放、导出 CSV）。
	3. 集成异常检测规则并显示告警。

---

## 6. 3D 场景与机器人轨迹重放模块

- 目的：在 3D 场景中重放机器人、物体与传感器数据，直观验证空间动作与相互关系。
- 要点：
	- 渲染选项：机器人 URDF/mesh、点云（深度/雷达）、物体网格、抓取接触点、力向量。
	- 同步播放与 2D 视图（图像/时序）联动。
	- 推荐渲染引擎：Web three.js（便于远程）或 `meshcat`/`open3d`（Python）。
- 交付物：
	- `visualizer/3d/renderer.py`（统一小型 API）
	- URDF/mesh 加载示例与 demo 数据
- 子任务：
	1. 选择首版渲染后端（meshcat/three.js）。
	2. 实现 URDF/mesh 加载与简单场景渲染。
	3. 实现轨迹重放与同步接口。

---

## 7. 交互控件（播放/暂停/拖动/速度/步进/过滤）

- 目的：提供便捷的时间轴与过滤操作，提升探索效率。
- 要点：
	- 时间轴控制（播放速度、步进、跳转到事件）。
	- 过滤器（按 episode、标签、传感器类型、数值阈值筛选）。
	- 键盘快捷键与布局保存（工作区复现）。
- 交付物：
	- `visualizer/ui/timeline.py`
	- 过滤配置示例 `visualizer/config/filters.yaml`
- 子任务：
	1. 实现基本时间轴控件。
	2. 支持事件跳转（例如接触开始/结束）。
	3. 添加过滤与快捷键支持。

---

## 8. 大数据处理与性能优化（懒加载/分块/缓存）

- 目的：保证在长序列/多视角/高分辨率数据上仍能流畅交互。
- 要点：
	- 后端支持 chunked read（Zarr、HDF5 chunk），仅加载当前视图所需数据。
	- 内存缓存（LRU）与磁盘缓存缩略图/预览。
	- 低分辨率优先加载，必要时再加载全分辨率。
- 交付物：
	- `visualizer/io/cache.py`（缓存策略实现）
	- 性能测试脚本 `visualizer/tests/perf_test.py`
- 子任务：
	1. 设计缓存接口与 LRU 策略。
	2. 支持缩略图生成与预取。
	3. 性能基准与优化项列表。

---

## 9. 导出、快照、标注与测量工具

- 目的：支持数据标注、测量与导出以便离线分析与训练集构建。
- 要点：
	- 标注工具（ROI、事件标签、关键帧），并将标注写回 `annotations`。
	- 测量工具（距离、角度、力峰值）并导出为 CSV/Parquet。
	- 导出视频/重放脚本/状态记录（JSON）。
- 交付物：
	- `visualizer/tools/annotator.py`
	- 导出脚本 `visualizer/tools/exporters.py`
- 子任务：
	1. 设计标注格式与保存机制。
	2. 实现基本标注 UI 与导出功能。
	3. 添加测量工具并写入示例结果。

---

## 10. 插件/扩展 API 与数据集注册机制

- 目的：让团队或社区能方便扩展新适配器、可视化组件或分析插件。
- 要点：
	- 定义插件生命周期（register/init/render/teardown）。
	- 数据集注册表，支持 dataset descriptor（名称、类型、路径、adapter）。
- 交付物：
	- `visualizer/plugins/api.py`
	- `visualizer/datasets/registry.py`
- 子任务：
	1. 设计插件接口与示例插件。
	2. 实现数据集注册表与 CLI/配置加载。

---

## 11. 测试用例与示例数据集（单元测试/端到端）

- 目的：保证适配器、同步、可视化组件的正确性，并便于演示与回归测试。
- 要点：
	- 单元测试覆盖适配器、时间同步、坐标变换、导出/导入逻辑。
	- 端到端示例（小型数据包，含多视角图像、关节与力传感器）。
- 交付物：
	- `visualizer/tests/`（单元测试与 e2e 脚本）
	- 示例数据 `visualizer/examples/sample_dataset/`
- 子任务：
	1. 准备示例数据（或合成数据）。
	2. 编写核心单元测试与 e2e 场景。
	3. 集成到 CI（可选）。

---

## 12. 文档、使用指南与用户体验设计

- 目的：降低接入成本，帮助用户快速上手并遵循最佳实践。
- 要点：
	- 快速开始（加载不同格式示例）、架构说明、插件开发指南。
	- UX 关注点：默认布局、主题、性能提示、键位说明。
- 交付物：
	- `visualizer/README.md`（快速开始）
	- `visualizer/docs/`（架构与开发文档）
- 子任务：
	1. 编写快速开始与示例加载流程。
	2. 撰写插件开发与适配器指南。
	3. 收集常见问题与故障排查。

---

## MVP 建议（优先级）

- 第一阶段（MVP）优先实现：
	1. 数据模型（任务 1）
	2. HDF5 适配器（任务 2）
	3. 时间同步（任务 3）
	4. 图像网格 + 时间轴 + 时序可视化（任务 4、5、7）
	5. 示例数据与入门文档（任务 11、12）

---

如果需要，我可以：

- 继续把每条任务转为更细的 Jira/Issue 风格子任务（含估时与验收标准）。
- 现在生成 `visualizer/schema/trajectory_schema.json` 的初版（JSON Schema）。
- 直接实现 `HDF5` 适配器原型并提供示例脚本。

请选择你下一步想要我做的具体动作。
