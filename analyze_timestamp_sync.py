#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Timestamp synchronization analysis tool for multi-camera + robot data.

This tool analyzes HDF5 files saved by piper_collect.py to:
1. Check timestamp consistency across multiple camera streams.
2. Identify time delays/offsets between cameras and robot state.
3. Detect dropped frames or irregular sampling.
4. Provide summary statistics and visualizations.

Usage:
  python3 analyze_timestamp_sync.py --traj_file traj_123.h5 [--plot] [--verbose]
  python3 analyze_timestamp_sync.py --task_dir task_folder [--plot] [--verbose]
"""

import os
import glob
import argparse
import numpy as np
import h5py
import json
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


class TimestampAnalyzer:
    """Analyze timestamp synchronization in trajectory data."""

    def __init__(self, h5_path):
        self.h5_path = h5_path
        self.data = {}
        self.load_data()

    def load_data(self):
        """Load all timestamps from HDF5 file."""
        with h5py.File(self.h5_path, 'r') as f:
            try:
                obs = f['observation']
            except KeyError:
                print(f"ERROR: No /observation group in {self.h5_path}")
                return

            # load camera timestamps per camera
            try:
                imgs = obs['images']
                self.data['cameras'] = {}
                for cam_name in imgs.keys():
                    cam_group = imgs[cam_name]
                    if 'timestamp' in cam_group:
                        ts = cam_group['timestamp'][:]
                        self.data['cameras'][cam_name] = ts
            except Exception as e:
                print(f"WARNING: Could not load camera timestamps: {e}")
                self.data['cameras'] = {}

            # load robot timestamps
            try:
                proprio = obs['proprioception']
                if 'joint_timestamp' in proprio:
                    self.data['robot_joint_ts'] = proprio['joint_timestamp'][:]
                else:
                    self.data['robot_joint_ts'] = None

                if 'joints' in proprio:
                    self.data['joints'] = proprio['joints'][:]
                if 'eef' in proprio:
                    self.data['eef'] = proprio['eef'][:]
            except Exception as e:
                print(f"WARNING: Could not load robot timestamps: {e}")
                self.data['robot_joint_ts'] = None

    def compute_stats(self, ts_array):
        """Compute basic statistics for a timestamp array."""
        if ts_array is None or len(ts_array) == 0:
            return None

        # convert to milliseconds if needed for readability
        ts = np.asarray(ts_array, dtype=np.float64)
        
        # compute differences (inter-frame intervals)
        if len(ts) > 1:
            diffs = np.diff(ts)
            mean_dt = np.mean(diffs)
            std_dt = np.std(diffs)
            min_dt = np.min(diffs)
            max_dt = np.max(diffs)
            n_frames = len(ts)
        else:
            mean_dt = std_dt = min_dt = max_dt = 0
            n_frames = len(ts)

        return {
            'n_frames': n_frames,
            'mean_dt_ms': mean_dt,
            'std_dt_ms': std_dt,
            'min_dt_ms': min_dt,
            'max_dt_ms': max_dt,
            'ts_first_ms': ts[0] if len(ts) > 0 else None,
            'ts_last_ms': ts[-1] if len(ts) > 0 else None,
            'total_duration_ms': ts[-1] - ts[0] if len(ts) > 1 else 0,
        }

    def analyze_sync(self):
        """Analyze cross-stream synchronization."""
        results = {
            'per_stream': {},
            'cross_stream_stats': {},
            'issues': []
        }

        # per-camera stats
        for cam_name, ts in self.data['cameras'].items():
            stats = self.compute_stats(ts)
            results['per_stream'][f'camera_{cam_name}'] = stats

        # robot joint timestamp stats
        if self.data['robot_joint_ts'] is not None:
            stats = self.compute_stats(self.data['robot_joint_ts'])
            results['per_stream']['robot_joint'] = stats

        # cross-stream analysis
        if len(self.data['cameras']) > 0 and self.data['robot_joint_ts'] is not None:
            cam_names = list(self.data['cameras'].keys())
            robot_ts = self.data['robot_joint_ts']

            # check frame counts match
            cam_counts = [len(self.data['cameras'][cam]) for cam in cam_names]
            robot_count = len(robot_ts)

            if not all(c == robot_count for c in cam_counts):
                results['issues'].append(
                    f"Frame count mismatch: cameras={cam_counts}, robot={robot_count}"
                )

            # compute time offset between first camera and robot
            for cam_name in cam_names:
                cam_ts = self.data['cameras'][cam_name]
                if len(cam_ts) > 0 and len(robot_ts) > 0:
                    offset_first = cam_ts[0] - robot_ts[0]
                    offset_last = cam_ts[-1] - robot_ts[-1]
                    offset_mean = np.mean(cam_ts) - np.mean(robot_ts)
                    results['cross_stream_stats'][f'{cam_name}_vs_robot'] = {
                        'offset_first_ms': float(offset_first),
                        'offset_last_ms': float(offset_last),
                        'offset_mean_ms': float(offset_mean),
                    }

            # check timestamp jitter (inter-frame interval variance)
            for cam_name in cam_names:
                cam_ts = self.data['cameras'][cam_name]
                if len(cam_ts) > 2:
                    diffs = np.diff(cam_ts)
                    jitter = np.std(diffs) / np.mean(diffs)  # normalized std of intervals
                    if jitter > 0.1:  # threshold: >10% jitter is suspicious
                        results['issues'].append(
                            f"High jitter on {cam_name}: normalized_std={jitter:.3f}"
                        )

            # check robot timestamp jitter
            if len(robot_ts) > 2:
                diffs = np.diff(robot_ts)
                jitter = np.std(diffs) / np.mean(diffs)
                if jitter > 0.1:
                    results['issues'].append(
                        f"High jitter on robot: normalized_std={jitter:.3f}"
                    )

        return results

    def print_report(self):
        """Print a human-readable synchronization report."""
        analysis = self.analyze_sync()

        print("\n" + "=" * 70)
        print(f"Timestamp Synchronization Analysis: {os.path.basename(self.h5_path)}")
        print("=" * 70)

        # per-stream statistics
        print("\n--- Per-Stream Statistics (in milliseconds) ---")
        for stream_name, stats in analysis['per_stream'].items():
            if stats is None:
                print(f"  {stream_name}: [No data]")
            else:
                print(f"\n  {stream_name}:")
                print(f"    Frames: {stats['n_frames']}")
                print(f"    Mean inter-frame (ΔT): {stats['mean_dt_ms']:.2f} ms")
                print(f"    Std inter-frame: {stats['std_dt_ms']:.2f} ms")
                print(f"    Min/Max inter-frame: {stats['min_dt_ms']:.2f} / {stats['max_dt_ms']:.2f} ms")
                print(f"    Duration: {stats['total_duration_ms']:.2f} ms")

        # cross-stream synchronization
        if analysis['cross_stream_stats']:
            print("\n--- Cross-Stream Time Offsets (camera - robot) ---")
            for pair_name, offsets in analysis['cross_stream_stats'].items():
                print(f"\n  {pair_name}:")
                print(f"    First frame offset: {offsets['offset_first_ms']:.2f} ms")
                print(f"    Last frame offset: {offsets['offset_last_ms']:.2f} ms")
                print(f"    Mean offset: {offsets['offset_mean_ms']:.2f} ms")

        # issues and warnings
        if analysis['issues']:
            print("\n--- Issues & Warnings ---")
            for i, issue in enumerate(analysis['issues'], 1):
                print(f"  {i}. {issue}")
        else:
            print("\n--- No major synchronization issues detected. ---")

        print("\n" + "=" * 70)

        return analysis

    def plot_timestamps(self, save_path=None):
        """Plot timestamp data for visual inspection."""
        if not HAS_MATPLOTLIB:
            print("ERROR: matplotlib not available. Skipping plot.")
            return

        analysis = self.analyze_sync()

        # create subplots: one row per camera + robot
        n_streams = len(self.data['cameras']) + (1 if self.data['robot_joint_ts'] is not None else 0)
        if n_streams == 0:
            print("No timestamp data to plot.")
            return

        fig, axes = plt.subplots(n_streams, 2, figsize=(12, 3 * n_streams))
        if n_streams == 1:
            axes = axes.reshape(1, -1)

        row = 0

        # plot camera timestamps
        for cam_name, ts in self.data['cameras'].items():
            if ts is None or len(ts) == 0:
                continue

            ts = np.asarray(ts, dtype=np.float64)

            # left: raw timestamps
            axes[row, 0].plot(ts, marker='.')
            axes[row, 0].set_ylabel(f'{cam_name} (ms)')
            axes[row, 0].set_title(f'{cam_name} - Raw Timestamps')
            axes[row, 0].grid(True, alpha=0.3)

            # right: inter-frame intervals
            if len(ts) > 1:
                diffs = np.diff(ts)
                axes[row, 1].plot(diffs, marker='.', color='orange')
                axes[row, 1].set_ylabel(f'{cam_name} ΔT (ms)')
                axes[row, 1].set_title(f'{cam_name} - Inter-Frame Intervals')
                axes[row, 1].axhline(y=np.mean(diffs), color='r', linestyle='--', label='mean')
                axes[row, 1].legend()
                axes[row, 1].grid(True, alpha=0.3)

            row += 1

        # plot robot timestamps
        if self.data['robot_joint_ts'] is not None:
            ts_robot = np.asarray(self.data['robot_joint_ts'], dtype=np.float64)

            axes[row, 0].plot(ts_robot, marker='.', color='green')
            axes[row, 0].set_ylabel('robot (ms)')
            axes[row, 0].set_title('Robot Joint - Raw Timestamps')
            axes[row, 0].grid(True, alpha=0.3)

            if len(ts_robot) > 1:
                diffs_robot = np.diff(ts_robot)
                axes[row, 1].plot(diffs_robot, marker='.', color='green')
                axes[row, 1].set_ylabel('robot ΔT (ms)')
                axes[row, 1].set_title('Robot Joint - Inter-Frame Intervals')
                axes[row, 1].axhline(y=np.mean(diffs_robot), color='r', linestyle='--', label='mean')
                axes[row, 1].legend()
                axes[row, 1].grid(True, alpha=0.3)

        for ax in axes.flat:
            ax.set_xlabel('Frame Index')

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=100)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        return fig


def main(args):
    h5_files = []

    if args.traj_file:
        if os.path.exists(args.traj_file):
            h5_files.append(args.traj_file)
        else:
            print(f"ERROR: {args.traj_file} not found")
            return

    if args.task_dir:
        pattern = os.path.join(args.task_dir, "*.h5")
        h5_files.extend(sorted(glob.glob(pattern)))

    if not h5_files:
        print("No HDF5 files found.")
        return

    print(f"Found {len(h5_files)} trajectory file(s).")

    all_analyses = {}
    for h5_path in h5_files:
        print(f"\n\nAnalyzing: {h5_path}")
        try:
            analyzer = TimestampAnalyzer(h5_path)
            analysis = analyzer.print_report()
            all_analyses[h5_path] = analysis

            if args.plot:
                plot_path = h5_path.replace('.h5', '_timestamps.png')
                analyzer.plot_timestamps(save_path=plot_path)

        except Exception as e:
            print(f"ERROR analyzing {h5_path}: {e}")

    # summary across all files
    if len(all_analyses) > 1:
        print("\n\n" + "=" * 70)
        print("SUMMARY ACROSS ALL TRAJECTORIES")
        print("=" * 70)

        issue_count = sum(len(a.get('issues', [])) for a in all_analyses.values())
        print(f"Total issues across all files: {issue_count}")

        if args.verbose:
            print("\nDetailed issues by file:")
            for fpath, analysis in all_analyses.items():
                if analysis.get('issues'):
                    print(f"\n  {os.path.basename(fpath)}:")
                    for issue in analysis['issues']:
                        print(f"    - {issue}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--traj_file', help='Path to a single trajectory HDF5 file')
    p.add_argument('--task_dir', help='Path to a task directory containing multiple HDF5 files')
    p.add_argument('--plot', action='store_true', help='Generate plots for each trajectory')
    p.add_argument('--verbose', action='store_true', help='Verbose output')
    args = p.parse_args()

    if not args.traj_file and not args.task_dir:
        p.print_help()
    else:
        main(args)
