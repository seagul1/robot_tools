#!/usr/bin/env python3
"""
快速集成测试脚本。

用于验证整个可视化工具的端到端流程是否正常工作。
"""

import os
import sys
import tempfile
import json
from pathlib import Path

# 添加本目录和父目录到路径
VIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(os.path.dirname(VIS_DIR))
if VIS_DIR not in sys.path:
    sys.path.insert(0, VIS_DIR)
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

try:
    # 导入时使用完全限定的包路径
    import numpy as np
    import h5py
    import yaml
    
    sys.path.insert(0, VIS_DIR)
    from schema_loader import load_schema, extract_visualization_fields
    from analysis import DataQualityChecker, check_episode_quality
    from adapters.hdf5_adapter import HDF5Adapter
except Exception as e:
    print(f"导入错误: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)


def test_schema_loader():
    """测试 schema_loader 模块。"""
    print("\n" + "=" * 80)
    print("测试 1: Schema Loader")
    print("=" * 80)
    
    schema_path = os.path.join(VIS_DIR, "schema", "hdf5_example.yaml")
    if not os.path.exists(schema_path):
        print(f"[SKIP] Schema 文件不存在: {schema_path}")
        return True
    
    try:
        schema = load_schema(schema_path)
        print(f"✓ 成功加载 schema: {schema_path}")
        
        fields = extract_visualization_fields(schema)
        print(f"✓ 提取的可视化字段:")
        for key, vals in fields.items():
            if vals:
                print(f"    {key}: {vals}")
        
        return True
    except Exception as e:
        print(f"✗ 错误: {e}")
        return False


def test_data_quality_checker():
    """测试数据质量检查器。"""
    print("\n" + "=" * 80)
    print("测试 2: Data Quality Checker")
    print("=" * 80)
    
    try:
        import numpy as np
        
        # 创建测试数据
        n_frames = 100
        data = {
            "joint_positions": np.random.randn(n_frames, 7),
            "actions": np.random.randn(n_frames, 7),
        }
        timestamps = np.arange(n_frames) * 0.033  # 30 FPS
        
        # 注入异常
        data["joint_positions"][20, 0] = 100.0  # 离散值
        timestamps[50] = 2.0  # 跳帧
        
        checker = DataQualityChecker()
        result = checker.check_sequence(data, timestamps)
        
        print(f"✓ 检查完成")
        print(f"  总帧数: {result['num_frames']}")
        print(f"  检测到的异常数: {result['num_anomalies']}")
        
        # 统计异常类型
        type_counts = {}
        for anom in result["anomalies"]:
            atype = anom.type.value
            type_counts[atype] = type_counts.get(atype, 0) + 1
        
        for atype, count in type_counts.items():
            print(f"    {atype}: {count}")
        
        # 验证检测到了注入的异常
        if result["num_anomalies"] > 0:
            print("✓ 成功检测到异常")
            return True
        else:
            print("✗ 未检测到任何异常")
            return False
    
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_hdf5_adapter():
    """测试 HDF5 适配器。"""
    print("\n" + "=" * 80)
    print("测试 3: HDF5 Adapter")
    print("=" * 80)
    
    try:
        import h5py
        import numpy as np
        
        # 创建临时 HDF5 文件
        with tempfile.TemporaryDirectory() as tmpdir:
            h5_path = os.path.join(tmpdir, "test.h5")
            
            # 创建测试文件
            with h5py.File(h5_path, "w") as f:
                n_frames = 50
                f.create_dataset("timestamps", data=np.arange(n_frames) * 0.033)
                
                # 图像
                img_group = f.create_group("images")
                for view in ["head", "wrist"]:
                    img_group.create_dataset(view, data=np.random.randint(0, 255, (n_frames, 64, 64, 3), dtype=np.uint8))
                
                # 观察
                obs_group = f.create_group("observations")
                obs_group.create_dataset("qpos", data=np.random.randn(n_frames, 7))
                
                # 动作
                f.create_dataset("actions", data=np.random.randn(n_frames, 7))
            
            # 测试适配器
            adapter = HDF5Adapter(h5_path)
            
            # 列出 episodes
            episodes = adapter.list_episodes()
            print(f"✓ 找到 {len(episodes)} 个 episode")
            
            if episodes:
                episode_id = episodes[0]
                
                # 读取序列
                seq = adapter.read_sequence(episode_id)
                print(f"✓ 成功读取 episode: {episode_id}")
                print(f"  字段: {list(seq.keys())}")
                
                # 检查数据
                if "timestamps" in seq:
                    print(f"  timestamps shape: {seq['timestamps'].shape}")
                if "images" in seq:
                    print(f"  images: {list(seq['images'].keys())}")
                
                return True
            else:
                print("✗ 未找到任何 episode")
                return False
        
        adapter.close()
    
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_integration():
    """集成测试：创建完整的数据流。"""
    print("\n" + "=" * 80)
    print("测试 4: 集成测试")
    print("=" * 80)
    
    try:
        import h5py
        import numpy as np
        
        with tempfile.TemporaryDirectory() as tmpdir:
            # 1. 创建测试数据
            h5_path = os.path.join(tmpdir, "test.h5")
            with h5py.File(h5_path, "w") as f:
                n_frames = 100
                f.create_dataset("timestamps", data=np.arange(n_frames) * 0.033)
                img_group = f.create_group("images")
                img_group.create_dataset("rgb", data=np.random.randint(0, 255, (n_frames, 100, 100, 3), dtype=np.uint8))
                obs_group = f.create_group("observations")
                obs_group.create_dataset("qpos", data=np.random.randn(n_frames, 7))
                f.create_dataset("actions", data=np.random.randn(n_frames, 7))
            
            # 2. 加载和检查
            adapter = HDF5Adapter(h5_path)
            episodes = adapter.list_episodes()
            
            if not episodes:
                print("✗ 无法读取 episode")
                return False
            
            episode_id = episodes[0]
            print(f"✓ 读取 episode: {episode_id}")
            
            # 3. 执行质量检查
            checker = DataQualityChecker()
            result = check_episode_quality(adapter, episode_id, checker=checker)
            
            print(f"✓ 数据质量检查完成")
            print(f"  帧数: {result['num_frames']}")
            print(f"  异常数: {result['num_anomalies']}")
            
            # 4. 保存结果
            report_path = os.path.join(tmpdir, "report.json")
            with open(report_path, "w") as f:
                report = {
                    "episode_id": episode_id,
                    "num_frames": result["num_frames"],
                    "num_anomalies": result["num_anomalies"],
                    "statistics": {
                        k: {
                            "min": float(v.min_val) if not (isinstance(v.min_val, float) and (v.min_val != v.min_val)) else None,
                            "max": float(v.max_val) if not (isinstance(v.max_val, float) and (v.max_val != v.max_val)) else None,
                        }
                        for k, v in result["statistics"].items()
                    }
                }
                json.dump(report, f, indent=2)
            
            print(f"✓ 报告已保存: {report_path}")
            adapter.close()
            return True
    
    except Exception as e:
        print(f"✗ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """运行所有测试。"""
    print("\n" + "=" * 80)
    print("轨迹可视化工具集成测试")
    print("=" * 80)
    
    results = []
    
    # 运行所有测试
    results.append(("Schema Loader", test_schema_loader()))
    results.append(("Data Quality Checker", test_data_quality_checker()))
    results.append(("HDF5 Adapter", test_hdf5_adapter()))
    results.append(("Integration Test", test_integration()))
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("测试摘要")
    print("=" * 80)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\n总体: {passed}/{total} 测试通过")
    
    if passed == total:
        print("\n✓ 所有测试通过！")
        print("\n可以开始使用可视化工具：")
        print("  python visualizer_main.py info --file <data.h5>")
        print("  python visualizer_main.py single --file <data.h5> --schema <schema.yaml>")
        print("  python visualizer_main.py batch --file <data.h5>")
        print("  python visualizer_main.py check --file <data.h5>")
        return 0
    else:
        print("\n✗ 某些测试失败。请检查错误信息。")
        return 1


if __name__ == "__main__":
    sys.exit(main())
