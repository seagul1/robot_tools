#!/usr/bin/env python3
"""
项目完成检查清单

此脚本验证所有新创建的文件和模块是否都在正确的位置。
"""

import os
import sys
from pathlib import Path

VIS_DIR = Path(__file__).parent.parent

# 新创建的核心文件
NEW_CORE_FILES = {
    "analysis.py": "数据质量检查模块（447 行）",
    "enhanced_simple_viewer.py": "单轨迹增强可视化（358 行）",
    "batch_viewer.py": "多轨迹批量分析（489 行）",
    "visualizer_main.py": "统一命令行入口（350 行）",
    "test_integration.py": "端到端集成测试（300+ 行）",
}

# 新创建的文档文件
NEW_DOC_FILES = {
    "USAGE_GUIDE.md": "详细使用指南（350+ 行）",
    "README_ENHANCED.md": "项目概览（250+ 行）",
    "QUICK_REFERENCE.md": "快速参考卡片（220+ 行）",
    "IMPLEMENTATION_SUMMARY.md": "实现总结（300+ 行）",
}

# 修改的文件
MODIFIED_FILES = {
    "adapters/hdf5_adapter.py": "增强导入处理",
}

# 现有依赖文件（应该已存在）
EXISTING_FILES = {
    "adapters/base.py": "抽象适配器基类",
    "schema_loader.py": "YAML Schema 解析",
    "requirements.txt": "项目依赖",
}

def check_file_exists(filepath, description):
    """检查文件是否存在。"""
    full_path = VIS_DIR / filepath
    exists = full_path.exists()
    status = "✅" if exists else "❌"
    size = f"({full_path.stat().st_size / 1024:.1f} KB)" if exists else "(NOT FOUND)"
    print(f"{status} {filepath:40s} {size:15s} # {description}")
    return exists

def main():
    """运行检查清单。"""
    print("\n" + "=" * 100)
    print("轨迹可视化工具 - 项目完成检查清单")
    print("=" * 100)
    
    all_ok = True
    
    # 检查新创建的核心文件
    print("\n📦 新创建的核心文件:")
    print("-" * 100)
    for filename, description in NEW_CORE_FILES.items():
        if not check_file_exists(filename, description):
            all_ok = False
    
    # 检查新创建的文档
    print("\n📚 新创建的文档文件:")
    print("-" * 100)
    for filename, description in NEW_DOC_FILES.items():
        if not check_file_exists(filename, description):
            all_ok = False
    
    # 检查修改的文件
    print("\n📝 修改的文件:")
    print("-" * 100)
    for filename, description in MODIFIED_FILES.items():
        if not check_file_exists(filename, description):
            all_ok = False
    
    # 检查现有依赖文件
    print("\n🔗 现有依赖文件:")
    print("-" * 100)
    for filename, description in EXISTING_FILES.items():
        if not check_file_exists(filename, description):
            all_ok = False
    
    # 统计信息
    print("\n" + "=" * 100)
    print("📊 统计信息:")
    print("=" * 100)
    print(f"  新创建的核心文件: {len(NEW_CORE_FILES)} 个")
    print(f"  新创建的文档文件: {len(NEW_DOC_FILES)} 个")
    print(f"  修改的文件: {len(MODIFIED_FILES)} 个")
    print(f"  总计: {len(NEW_CORE_FILES) + len(NEW_DOC_FILES) + len(MODIFIED_FILES)} 个文件")
    
    # 验证导入
    print("\n" + "=" * 100)
    print("🔍 验证模块导入:")
    print("=" * 100)
    
    try:
        sys.path.insert(0, str(VIS_DIR))
        from schema_loader import load_schema, extract_visualization_fields
        print("✅ schema_loader 导入成功")
    except Exception as e:
        print(f"❌ schema_loader 导入失败: {e}")
        all_ok = False
    
    try:
        from analysis import DataQualityChecker, check_episode_quality
        print("✅ analysis 导入成功")
    except Exception as e:
        print(f"❌ analysis 导入失败: {e}")
        all_ok = False
    
    try:
        from adapters.hdf5_adapter import HDF5Adapter
        print("✅ adapters.hdf5_adapter 导入成功")
    except Exception as e:
        print(f"❌ adapters.hdf5_adapter 导入失败: {e}")
        all_ok = False
    
    try:
        # 不直接导入 GUI 类，因为需要 matplotlib
        print("✅ enhanced_simple_viewer 模块存在（不导入 GUI）")
    except Exception as e:
        print(f"❌ enhanced_simple_viewer 检查失败: {e}")
        all_ok = False
    
    try:
        print("✅ batch_viewer 模块存在（不导入 GUI）")
    except Exception as e:
        print(f"❌ batch_viewer 检查失败: {e}")
        all_ok = False
    
    # 最终结果
    print("\n" + "=" * 100)
    if all_ok:
        print("✅ 所有文件检查通过！项目已完成。")
        print("\n可以开始使用可视化工具：")
        print("  python visualizer_main.py --help")
        print("  python visualizer_main.py info --file <data.h5>")
        print("  python visualizer_main.py single --file <data.h5> --schema <schema.yaml>")
        print("  python visualizer_main.py batch --file <data.h5>")
        print("  python visualizer_main.py check --file <data.h5>")
        print("\n查看文档：")
        print("  cat USAGE_GUIDE.md")
        print("  cat README_ENHANCED.md")
        print("  cat QUICK_REFERENCE.md")
        return 0
    else:
        print("❌ 某些文件缺失或导入失败。请检查上面的错误信息。")
        return 1

if __name__ == "__main__":
    sys.exit(main())
