#!/bin/bash

# 轨迹可视化工具 - 快速启动脚本
# 这个脚本演示如何使用可视化工具的各个功能

set -e

VISUALIZER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "╔════════════════════════════════════════════════════════════════════════════════════════╗"
echo "║                                                                                        ║"
echo "║              轨迹数据统一可视化工具 - 快速启动脚本                                    ║"
echo "║                                                                                        ║"
echo "╚════════════════════════════════════════════════════════════════════════════════════════╝"
echo ""

# 显示菜单
show_menu() {
    echo "请选择要执行的操作："
    echo ""
    echo "  1. 验证环境和依赖"
    echo "  2. 查看数据集信息"
    echo "  3. 单轨迹可视化"
    echo "  4. 多轨迹批量分析"
    echo "  5. 数据质量检查"
    echo "  6. 查看完整帮助"
    echo "  7. 查看快速参考"
    echo "  8. 查看使用教程"
    echo "  0. 退出"
    echo ""
}

# 验证环境
verify_env() {
    echo "正在验证环境..."
    cd "$VISUALIZER_DIR"
    python check_completion.py
}

# 查看帮助
show_help() {
    echo ""
    echo "完整的命令行帮助："
    echo "cd $VISUALIZER_DIR"
    python visualizer_main.py --help
}

# 查看快速参考
show_quick_ref() {
    if [ -f "$VISUALIZER_DIR/QUICK_REFERENCE.md" ]; then
        less "$VISUALIZER_DIR/QUICK_REFERENCE.md" || cat "$VISUALIZER_DIR/QUICK_REFERENCE.md"
    else
        echo "找不到 QUICK_REFERENCE.md 文件"
    fi
}

# 查看教程
show_tutorial() {
    if [ -f "$VISUALIZER_DIR/USAGE_GUIDE.md" ]; then
        less "$VISUALIZER_DIR/USAGE_GUIDE.md" || cat "$VISUALIZER_DIR/USAGE_GUIDE.md"
    else
        echo "找不到 USAGE_GUIDE.md 文件"
    fi
}

# 单轨迹可视化
single_viz() {
    echo ""
    echo "单轨迹可视化用法："
    echo "=================================================="
    echo ""
    echo "基础命令："
    echo "  python visualizer_main.py single --file data.h5"
    echo ""
    echo "使用 schema 配置文件："
    echo "  python visualizer_main.py single --file data.h5 --schema schema.yaml"
    echo ""
    echo "指定特定 episode："
    echo "  python visualizer_main.py single --file data.h5 --episode episode_0"
    echo ""
    echo "保存图表："
    echo "  python visualizer_main.py single --file data.h5 --save"
    echo ""
    echo "请输入你的 HDF5 文件路径 (按 Enter 跳过)："
    read -p "> " h5_file
    
    if [ -n "$h5_file" ]; then
        cd "$VISUALIZER_DIR"
        python visualizer_main.py single --file "$h5_file"
    fi
}

# 多轨迹分析
batch_analysis() {
    echo ""
    echo "多轨迹批量分析用法："
    echo "=================================================="
    echo ""
    echo "分析单个文件："
    echo "  python visualizer_main.py batch --file data.h5 --output reports/"
    echo ""
    echo "分析整个目录："
    echo "  python visualizer_main.py batch --dir /path/to/data_dir --output reports/"
    echo ""
    echo "请输入 HDF5 文件或目录路径 (按 Enter 跳过)："
    read -p "> " data_path
    
    if [ -n "$data_path" ]; then
        cd "$VISUALIZER_DIR"
        if [ -d "$data_path" ]; then
            python visualizer_main.py batch --dir "$data_path" --output "./reports"
        else
            python visualizer_main.py batch --file "$data_path" --output "./reports"
        fi
    fi
}

# 数据质量检查
quality_check() {
    echo ""
    echo "数据质量检查用法："
    echo "=================================================="
    echo ""
    echo "基础检查："
    echo "  python visualizer_main.py check --file data.h5 --output reports/"
    echo ""
    echo "自定义参数："
    echo "  python visualizer_main.py check --file data.h5 \\"
    echo "    --outlier-threshold 2.5 \\"
    echo "    --frame-drop-threshold 1.5 \\"
    echo "    --output reports/"
    echo ""
    echo "请输入 HDF5 文件路径 (按 Enter 跳过)："
    read -p "> " h5_file
    
    if [ -n "$h5_file" ]; then
        cd "$VISUALIZER_DIR"
        python visualizer_main.py check --file "$h5_file" --output "./check_report"
    fi
}

# 查看数据集信息
info_dataset() {
    echo ""
    echo "数据集信息查看用法："
    echo "=================================================="
    echo ""
    echo "查看文件结构："
    echo "  python visualizer_main.py info --file data.h5"
    echo ""
    echo "限制显示数量："
    echo "  python visualizer_main.py info --file data.h5 --limit 10"
    echo ""
    echo "请输入 HDF5 文件路径 (按 Enter 跳过)："
    read -p "> " h5_file
    
    if [ -n "$h5_file" ]; then
        cd "$VISUALIZER_DIR"
        python visualizer_main.py info --file "$h5_file"
    fi
}

# 主循环
cd "$VISUALIZER_DIR"

while true; do
    echo ""
    show_menu
    read -p "请输入选项 (0-8): " choice
    
    case $choice in
        1)
            verify_env
            ;;
        2)
            info_dataset
            ;;
        3)
            single_viz
            ;;
        4)
            batch_analysis
            ;;
        5)
            quality_check
            ;;
        6)
            show_help
            ;;
        7)
            show_quick_ref
            ;;
        8)
            show_tutorial
            ;;
        0)
            echo "再见！"
            exit 0
            ;;
        *)
            echo "无效的选项，请重试"
            ;;
    esac
done
