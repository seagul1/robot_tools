# 轨迹可视化工具快速参考

## 核心命令

### 1️⃣ 查看数据集信息
```bash
python visualizer_main.py info --file data.h5
python visualizer_main.py info --file data.h5 --limit 10  # 显示前10个episode
```

### 2️⃣ 单轨迹可视化（推荐首先使用）
```bash
python visualizer_main.py single --file data.h5

# 使用配置文件
python visualizer_main.py single --file data.h5 --schema schema.yaml

# 指定特定episode
python visualizer_main.py single --file data.h5 --episode episode_0

# 保存图表
python visualizer_main.py single --file data.h5 --save
```

### 3️⃣ 多轨迹批量分析
```bash
# 单个文件
python visualizer_main.py batch --file data.h5 --schema schema.yaml

# 整个目录
python visualizer_main.py batch --dir /path/to/data --schema schema.yaml

# 保存报告
python visualizer_main.py batch --file data.h5 --output reports/
```

### 4️⃣ 数据质量检查
```bash
python visualizer_main.py check --file data.h5 --output reports/

# 自定义检测参数
python visualizer_main.py check --file data.h5 \
    --outlier-threshold 2.5 \
    --frame-drop-threshold 1.5 \
    --missing-value-threshold 0.05
```

## 快速工作流

### 工作流 A：验证新数据集
```bash
# 1. 检查文件结构
python visualizer_main.py info --file new_data.h5

# 2. 快速预览
python visualizer_main.py single --file new_data.h5

# 3. 详细检查
python visualizer_main.py check --file new_data.h5 --output check_results/
```

### 工作流 B：批量数据清理
```bash
# 1. 分析所有轨迹
python visualizer_main.py batch --dir data_dir/ --output analysis/

# 2. 查看报告
cat analysis/batch_analysis_report.txt

# 3. 检查有问题的轨迹
python visualizer_main.py single --file data_dir/problem_file.h5 --schema schema.yaml
```

### 工作流 C：完整数据评估
```bash
# 运行完整分析流程
python visualizer_main.py batch --dir data_dir/ --schema schema.yaml --output full_report/

# 检查单个异常轨迹
python visualizer_main.py single --file data_dir/episode_X.h5 --schema schema.yaml
```

## Schema YAML 模板

```yaml
# 基础配置
mode: single_episode
fps: 30

# 图像
vision_sensor:
  type: rgb
  prefix: observations/images
  key: ["camera_0", "camera_1"]
  timestamp: none

# 关节状态
proprioception_sensor:
  type: proprioception
  prefix: observations
  key: ["qpos", "qvel", "qaccel"]
  timestamp: none

# 末端执行器
end_effector:
  type: pose
  prefix: observations
  key: ["ee_pose", "ee_twist"]
  timestamp: none

# 动作
action:
  type: joint_position
  key: ["action"]
  timestamp: none

# 力觉反馈（可选）
force_sensor:
  type: force_torque
  prefix: observations
  key: ["ee_force", "ee_torque"]
  timestamp: none
```

## 参数速查

### outlier-threshold (离散值检测)
- **默认**: 3.0（标准差倍数）
- **取值**: 2.0-4.0
- **调整建议**:
  - 2.0: 更敏感，可能误报
  - 3.0: 推荐值
  - 4.0: 保守，可能漏报

### frame-drop-threshold (跳帧检测)
- **默认**: 2.0（倍数）
- **取值**: 1.5-3.0
- **调整建议**:
  - 1.5: 严格检查
  - 2.0: 推荐值
  - 3.0: 宽松检查

### missing-value-threshold (缺失值检测)
- **默认**: 0.01 (1%)
- **取值**: 0.001-0.1
- **调整建议**:
  - 0.001: 严格
  - 0.01: 推荐值
  - 0.05: 宽松

## 输出文件说明

### single 命令输出
```
└── {episode_id}_visualization.png  # 当使用 --save 时
```

### batch 命令输出
```
output_dir/
├── batch_analysis_report.txt       # 文本报告
└── 可视化图表（显示在窗口）
```

### check 命令输出
```
output_dir/
└── quality_check_report.json       # 详细 JSON 报告
```

## 交互控制

### 单轨迹可视化窗口
| 操作 | 功能 |
|------|------|
| **Prev** 按钮 | 上一帧 |
| **Next** 按钮 | 下一帧 |
| **滑条拖动** | 快速跳转 |
| **红色 X** 标记 | 异常检测位置 |
| **右侧面板** | 实时数据显示 |

## 数据流向图

```
┌─────────────────┐
│   data.h5       │
│   data_dir/     │
└────────┬────────┘
         │
         ▼
    ┌──────────┐
    │ Adapter  │ ◄─── schema.yaml
    └────┬─────┘
         │
         ▼
  ┌─────────────┐
  │ read_seq()  │
  └────┬────────┘
       │
       ├──────────────────────┐
       │                      │
       ▼                      ▼
  ┌─────────┐         ┌──────────────┐
  │Visualize│         │QualityChecker│
  └────┬────┘         └──────┬───────┘
       │                     │
       ▼                     ▼
  ┌─────────┐         ┌─────────────┐
  │Matplotlib│        │ Anomalies   │
  │  Plots  │         │ Statistics  │
  └─────────┘         └─────────────┘
       │                     │
       └──────────┬──────────┘
                  ▼
          ┌──────────────┐
          │ Report Files │
          │ JSON / Text  │
          └──────────────┘
```

## 故障排除

### 问题：ModuleNotFoundError
```bash
# 解决：确保在 visualizer 目录运行
cd toolkits/visualizer
python visualizer_main.py ...
```

### 问题：FileNotFoundError: data.h5
```bash
# 解决：使用完整路径或相对路径
python visualizer_main.py info --file /full/path/to/data.h5
```

### 问题：matplotlib 显示问题
```bash
# 解决：设置后端
export MPLBACKEND=TkAgg  # Linux/Mac
set MPLBACKEND=TkAgg     # Windows
python visualizer_main.py single --file data.h5
```

### 问题：内存溢出（大文件）
```bash
# 解决：
# 1. 指定 episode 减少加载
python visualizer_main.py single --file huge.h5 --episode episode_0

# 2. 使用采样（修改代码）
# 在 enhanced_simple_viewer.py 中添加采样步长
```

## 性能提示

⚡ **加速技巧**：
- 使用 `--episode` 指定单个轨迹
- 使用 `--limit 5` 只检查前5个
- 在 HDF5 中使用压缩存储
- 使用 SSD 存储 HDF5 文件

💾 **内存优化**：
- 流式读取而非全量加载（支持中）
- 降低图像分辨率
- 分批处理多个文件

## 获取帮助

```bash
# 查看所有命令
python visualizer_main.py --help

# 查看子命令帮助
python visualizer_main.py single --help
python visualizer_main.py batch --help
python visualizer_main.py check --help
python visualizer_main.py info --help

# 查看详细文档
cat USAGE_GUIDE.md
cat README_ENHANCED.md
```

## 常用组合命令

### 快速诊断新数据
```bash
python visualizer_main.py info --file data.h5 && \
python visualizer_main.py single --file data.h5
```

### 完整检查报告
```bash
python visualizer_main.py check --file data.h5 --output report && \
cat report/quality_check_report.json
```

### 批量处理整个目录
```bash
for f in data_dir/*.h5; do
  echo "Processing $f"
  python visualizer_main.py info --file "$f" --limit 1
done
```

---

**提示**：将此文件打印或保存到桌面以便快速参考！

**最后更新**：2025-12-02
