"""
可视化工具主入口脚本。

支持以下使用模式：

1. 单轨迹可视化（增强版）：
   python visualizer_main.py single --file data.h5 --schema schema.yaml --episode episode_0
   
2. 多轨迹批量分析：
   python visualizer_main.py batch --file data.h5 --schema schema.yaml
   python visualizer_main.py batch --dir /path/to/data_dir --schema schema.yaml
   
3. 数据质量检查报告：
   python visualizer_main.py check --file data.h5 --schema schema.yaml --output report_dir

"""

import os
import sys
import argparse
import json
from typing import Optional

# 导入自定义模块
VIS_DIR = os.path.dirname(os.path.abspath(__file__))
if VIS_DIR not in sys.path:
    sys.path.insert(0, VIS_DIR)

try:
    from enhanced_simple_viewer import viewer_main
    from batch_viewer import batch_viewer_main
    from analysis import check_episode_quality, DataQualityChecker
    from adapters.hdf5_adapter import HDF5Adapter
except Exception as e:
    try:
        from .enhanced_simple_viewer import viewer_main  # type: ignore
        from .batch_viewer import batch_viewer_main  # type: ignore
        from .analysis import check_episode_quality, DataQualityChecker  # type: ignore
        from .adapters.hdf5_adapter import HDF5Adapter  # type: ignore
    except Exception:
        raise


def cmd_single(args):
    """处理单轨迹可视化命令。"""
    print("\n" + "=" * 80)
    print("单轨迹增强可视化")
    print("=" * 80)
    
    if not os.path.exists(args.file):
        print(f"[ERROR] 文件不存在: {args.file}")
        return
    
    viewer_main(args.file, args.schema, args.episode, args.save)


def cmd_batch(args):
    """处理多轨迹批量分析命令。"""
    print("\n" + "=" * 80)
    print("多轨迹批量分析")
    print("=" * 80)
    
    if args.file:
        if not os.path.exists(args.file):
            print(f"[ERROR] 文件不存在: {args.file}")
            return
        batch_viewer_main(args.file, args.schema, is_directory=False, output_dir=args.output)
    elif args.dir:
        if not os.path.isdir(args.dir):
            print(f"[ERROR] 目录不存在: {args.dir}")
            return
        batch_viewer_main(args.dir, args.schema, is_directory=True, output_dir=args.output)
    else:
        print("[ERROR] 请指定 --file 或 --dir")


def cmd_check(args):
    """处理数据质量检查命令。"""
    print("\n" + "=" * 80)
    print("数据质量检查")
    print("=" * 80)
    
    if not os.path.exists(args.file):
        print(f"[ERROR] 文件不存在: {args.file}")
        return
    
    adapter = HDF5Adapter(args.file)
    if args.schema and os.path.exists(args.schema):
        from schema_loader import load_schema
        schema = load_schema(args.schema)
        adapter.set_schema(schema)
        print(f"[INFO] 已加载 schema: {args.schema}")
    
    episodes = adapter.list_episodes()
    print(f"[INFO] 文件包含 {len(episodes)} 个 episode")
    
    checker = DataQualityChecker(
        outlier_threshold=args.outlier_threshold,
        frame_drop_threshold=args.frame_drop_threshold,
        missing_value_threshold=args.missing_value_threshold,
    )
    
    all_results = {}
    for episode_id in episodes:
        print(f"\n分析 {episode_id}...")
        result = check_episode_quality(adapter, episode_id, checker=checker)
        all_results[episode_id] = {
            "num_frames": result["num_frames"],
            "num_anomalies": result["num_anomalies"],
            "statistics": {
                k: {
                    "min": float(v.min_val) if not (isinstance(v.min_val, float) and \
                        (v.min_val != v.min_val)) else None,
                    "max": float(v.max_val) if not (isinstance(v.max_val, float) and \
                        (v.max_val != v.max_val)) else None,
                    "mean": float(v.mean) if not (isinstance(v.mean, float) and \
                        (v.mean != v.mean)) else None,
                    "std": float(v.std) if not (isinstance(v.std, float) and \
                        (v.std != v.std)) else None,
                    "outlier_count": v.outlier_count,
                    "missing_count": v.missing_count,
                }
                for k, v in result["statistics"].items()
            }
        }
    
    adapter.close()
    
    # 保存报告
    if args.output:
        os.makedirs(args.output, exist_ok=True)
        report_path = os.path.join(args.output, "quality_check_report.json")
        with open(report_path, "w") as f:
            json.dump(all_results, f, indent=2)
        print(f"\n[INFO] 报告已保存到: {report_path}")
    
    # 打印摘要
    print("\n" + "=" * 80)
    print("质量检查摘要")
    print("=" * 80)
    total_anomalies = sum(r["num_anomalies"] for r in all_results.values())
    print(f"总 Episode: {len(all_results)}")
    print(f"总异常数: {total_anomalies}")
    print(f"平均异常数/Episode: {total_anomalies / max(1, len(all_results)):.2f}")


def cmd_info(args):
    """处理信息查询命令。"""
    print("\n" + "=" * 80)
    print("数据集信息")
    print("=" * 80)
    
    if not os.path.exists(args.file):
        print(f"[ERROR] 文件不存在: {args.file}")
        return
    
    adapter = HDF5Adapter(args.file)
    episodes = adapter.list_episodes()
    
    print(f"\n文件: {args.file}")
    print(f"Episode 数: {len(episodes)}\n")
    
    for episode_id in episodes[:args.limit]:
        print(f"Episode: {episode_id}")
        
        try:
            meta = adapter.get_episode_meta(episode_id)
            seq = adapter.read_sequence(episode_id)
            
            # 显示元数据
            if meta:
                print(f"  Metadata: {meta}")
            
            # 显示数据字段
            for key, val in seq.items():
                if isinstance(val, dict):
                    print(f"  {key}:")
                    for subkey, subval in val.items():
                        if isinstance(subval, dict):
                            print(f"    {subkey}: (nested dict)")
                        elif hasattr(subval, 'shape'):
                            print(f"    {subkey}: shape={subval.shape}, dtype={subval.dtype}")
                        else:
                            print(f"    {subkey}: {type(subval)}")
                elif hasattr(val, 'shape'):
                    print(f"  {key}: shape={val.shape}, dtype={val.dtype}")
                else:
                    print(f"  {key}: {type(val)}")
        except Exception as e:
            print(f"  Error: {e}")
        
        print()
    
    if len(episodes) > args.limit:
        print(f"... 还有 {len(episodes) - args.limit} 个 episode")
    
    adapter.close()


def main():
    """主函数。"""
    parser = argparse.ArgumentParser(
        description="轨迹数据统一可视化工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例：
  # 单轨迹可视化
  python visualizer_main.py single --file data.h5 --schema schema.yaml
  
  # 多轨迹分析
  python visualizer_main.py batch --file data.h5 --schema schema.yaml
  
  # 数据质量检查
  python visualizer_main.py check --file data.h5 --schema schema.yaml
  
  # 查看数据集信息
  python visualizer_main.py info --file data.h5
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="可用命令")
    
    # ===== single 命令 =====
    single_parser = subparsers.add_parser("single", help="单轨迹增强可视化")
    single_parser.add_argument("--file", required=True, help="HDF5 文件路径")
    single_parser.add_argument("--schema", default=None, help="YAML schema 配置文件路径")
    single_parser.add_argument("--episode", default=None, help="Episode ID（可选，默认为第一个）")
    single_parser.add_argument("--save", action="store_true", help="保存图表到文件")
    single_parser.set_defaults(func=cmd_single)
    
    # ===== batch 命令 =====
    batch_parser = subparsers.add_parser("batch", help="多轨迹批量分析")
    batch_parser.add_argument("--file", default=None, help="单个 HDF5 文件路径")
    batch_parser.add_argument("--dir", default=None, help="包含 HDF5 文件的目录")
    batch_parser.add_argument("--schema", default=None, help="YAML schema 配置文件路径")
    batch_parser.add_argument("--output", default=None, help="输出目录（保存报告）")
    batch_parser.set_defaults(func=cmd_batch)
    
    # ===== check 命令 =====
    check_parser = subparsers.add_parser("check", help="数据质量检查")
    check_parser.add_argument("--file", required=True, help="HDF5 文件路径")
    check_parser.add_argument("--schema", default=None, help="YAML schema 配置文件路径")
    check_parser.add_argument("--output", default=None, help="输出目录（保存报告）")
    check_parser.add_argument("--outlier-threshold", type=float, default=3.0, 
                            help="离散值检测阈值（标准差倍数）")
    check_parser.add_argument("--frame-drop-threshold", type=float, default=2.0,
                            help="跳帧检测阈值（标准差倍数）")
    check_parser.add_argument("--missing-value-threshold", type=float, default=0.01,
                            help="缺失值比例阈值")
    check_parser.set_defaults(func=cmd_check)
    
    # ===== info 命令 =====
    info_parser = subparsers.add_parser("info", help="查看数据集信息")
    info_parser.add_argument("--file", required=True, help="HDF5 文件路径")
    info_parser.add_argument("--limit", type=int, default=5, help="显示的 episode 最大数（默认 5）")
    info_parser.set_defaults(func=cmd_info)
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        args.func(args)
    except KeyboardInterrupt:
        print("\n\n[INFO] 用户中断")
        sys.exit(0)
    except Exception as e:
        print(f"\n[ERROR] {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
