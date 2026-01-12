#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Real-time timestamp synchronization monitor for live data collection.

This script monitors timestamp alignment during data collection and reports:
1. Frame rate (FPS) per camera and robot
2. Time offsets between cameras and robot in real-time
3. Synchronization quality metrics
4. Warnings if synchronization drifts beyond thresholds

Usage:
  Integrate into piper_collect.py or run as a standalone monitor by loading
  the HDF5 file periodically during recording (if using incremental HDF5 writing).

Integration example in piper_collect.py:
  from analyze_timestamp_sync import TimestampMonitor
  monitor = TimestampMonitor()
  # on each append:
  monitor.add_sample(cam_timestamps, robot_joint_ts, frame_index)
  # periodic report:
  monitor.report(interval=100)  # every 100 frames
"""

import numpy as np
from collections import deque, defaultdict
from typing import Optional, Dict, List
import time


class TimestampMonitor:
    """Real-time timestamp monitor for multi-camera + robot synchronization."""

    def __init__(self, max_history=500):
        """
        Args:
            max_history: keep the last N samples for statistics
        """
        self.max_history = max_history
        self.frame_index = 0

        # per-camera buffers: deque of (ts_ms, frame_idx)
        self.cam_buffers: Dict[str, deque] = defaultdict(lambda: deque(maxlen=max_history))
        # robot buffer: deque of (ts_ms, frame_idx)
        self.robot_buffer: deque = deque(maxlen=max_history)

        # cumulative stats for averaging
        self.cumulative_stats = {
            'n_samples': 0,
            'camera_fps': defaultdict(list),
            'robot_fps': [],
            'cam_vs_robot_offset_ms': defaultdict(list),
        }

    def add_sample(self, cam_timestamps: Dict[str, float], robot_timestamp: float, frame_idx: int):
        """
        Add a new sample (from piper_collect.py append loop).

        Args:
            cam_timestamps: dict mapping camera name to timestamp (ms or seconds)
            robot_timestamp: robot joint timestamp (ms or seconds)
            frame_idx: frame index (optional, for debugging)
        """
        self.frame_index = frame_idx

        # normalize timestamps to milliseconds if needed (assuming input is in seconds or ms)
        for cam_name, ts in cam_timestamps.items():
            ts_ms = float(ts)
            # heuristic: if ts is very large, assume it's already in ms
            if ts_ms > 1e8:  # rough cutoff for seconds -> ms epoch
                ts_ms = ts_ms  # already in ms
            self.cam_buffers[cam_name].append((ts_ms, frame_idx))

        robot_ts_ms = float(robot_timestamp)
        if robot_ts_ms > 1e8:
            robot_ts_ms = robot_ts_ms
        self.robot_buffer.append((robot_ts_ms, frame_idx))

        self.cumulative_stats['n_samples'] += 1

    def get_fps_stats(self) -> Dict:
        """Compute per-stream FPS from recent history."""
        stats = {}

        # camera FPS
        for cam_name, buf in self.cam_buffers.items():
            if len(buf) > 1:
                ts_vals = [t[0] for t in buf]
                diffs = np.diff(ts_vals)  # ms
                mean_dt = np.mean(diffs)
                if mean_dt > 0:
                    fps = 1000.0 / mean_dt
                    stats[f'{cam_name}_fps'] = fps

        # robot FPS
        if len(self.robot_buffer) > 1:
            ts_vals = [t[0] for t in self.robot_buffer]
            diffs = np.diff(ts_vals)  # ms
            mean_dt = np.mean(diffs)
            if mean_dt > 0:
                fps = 1000.0 / mean_dt
                stats['robot_fps'] = fps

        return stats

    def get_offset_stats(self) -> Dict:
        """Compute time offset (camera - robot) statistics."""
        stats = {}

        if len(self.robot_buffer) == 0:
            return stats

        robot_ts_vals = np.array([t[0] for t in self.robot_buffer])

        for cam_name, buf in self.cam_buffers.items():
            if len(buf) == 0:
                continue

            # align by frame index: match camera and robot samples at same frame
            cam_ts_vals = np.array([t[0] for t in buf])

            # compute offset: camera_ts - robot_ts
            # ideally these should be close if synchronized
            if len(cam_ts_vals) == len(robot_ts_vals):
                offset = cam_ts_vals - robot_ts_vals
                stats[f'{cam_name}_offset_ms'] = {
                    'mean': float(np.mean(offset)),
                    'std': float(np.std(offset)),
                    'min': float(np.min(offset)),
                    'max': float(np.max(offset)),
                }
            else:
                # different lengths, just report current offset
                if len(cam_ts_vals) > 0 and len(robot_ts_vals) > 0:
                    offset = cam_ts_vals[-1] - robot_ts_vals[-1]
                    stats[f'{cam_name}_offset_current_ms'] = float(offset)

        return stats

    def get_jitter(self) -> Dict:
        """Compute inter-frame interval jitter (normalized std)."""
        stats = {}

        for cam_name, buf in self.cam_buffers.items():
            if len(buf) > 2:
                ts_vals = np.array([t[0] for t in buf])
                diffs = np.diff(ts_vals)
                mean_dt = np.mean(diffs)
                if mean_dt > 0:
                    jitter = np.std(diffs) / mean_dt
                    stats[f'{cam_name}_jitter'] = float(jitter)

        if len(self.robot_buffer) > 2:
            ts_vals = np.array([t[0] for t in self.robot_buffer])
            diffs = np.diff(ts_vals)
            mean_dt = np.mean(diffs)
            if mean_dt > 0:
                jitter = np.std(diffs) / mean_dt
                stats['robot_jitter'] = float(jitter)

        return stats

    def report(self, prefix: str = "  "):
        """Print a real-time status report."""
        fps_stats = self.get_fps_stats()
        offset_stats = self.get_offset_stats()
        jitter_stats = self.get_jitter()

        print(f"\n{prefix}[Frame {self.frame_index}] Timestamp Synchronization Monitor")
        print(f"{prefix}" + "-" * 60)

        # FPS
        if fps_stats:
            print(f"{prefix}Frame Rates (FPS):")
            for key, val in sorted(fps_stats.items()):
                print(f"{prefix}  {key}: {val:.1f}")

        # Offsets
        if offset_stats:
            print(f"\n{prefix}Time Offsets (ms, camera - robot):")
            for key, val in sorted(offset_stats.items()):
                if isinstance(val, dict):
                    print(f"{prefix}  {key}:")
                    print(f"{prefix}    mean: {val['mean']:+.2f}, std: {val['std']:.2f}, range: [{val['min']:.2f}, {val['max']:.2f}]")
                else:
                    print(f"{prefix}  {key}: {val:+.2f}")

        # Jitter
        if jitter_stats:
            print(f"\n{prefix}Inter-Frame Jitter (normalized):")
            for key, val in sorted(jitter_stats.items()):
                if val > 0.1:
                    print(f"{prefix}  {key}: {val:.3f} ⚠️  (HIGH)")
                else:
                    print(f"{prefix}  {key}: {val:.3f}")

        print(f"{prefix}" + "-" * 60)

    def check_sync_quality(self, max_offset_ms: float = 50.0, max_jitter: float = 0.15) -> List[str]:
        """
        Check synchronization quality against thresholds.

        Returns:
            List of warning messages if synchronization is poor.
        """
        warnings = []

        offset_stats = self.get_offset_stats()
        for key, val in offset_stats.items():
            if isinstance(val, dict):
                if abs(val['mean']) > max_offset_ms:
                    warnings.append(f"Large time offset on {key}: {val['mean']:.1f} ms")
            else:
                if abs(val) > max_offset_ms:
                    warnings.append(f"Large time offset on {key}: {val:.1f} ms")

        jitter_stats = self.get_jitter()
        for key, val in jitter_stats.items():
            if val > max_jitter:
                warnings.append(f"High jitter on {key}: {val:.3f}")

        return warnings


# Example usage in piper_collect.py (add to run_collector function)
def example_integration():
    """
    Example of how to integrate TimestampMonitor into piper_collect.py:

    In run_collector():
        from analyze_timestamp_sync import TimestampMonitor
        monitor = TimestampMonitor(max_history=200)

        # ... in the append loop ...
        if recording:
            # after collecting cam_ts, joint_ts, etc:
            monitor.add_sample(cam_ts, float(joint_ts), writer.length)

            # every 100 frames, print report
            if writer.length % 100 == 0:
                monitor.report()
                warnings = monitor.check_sync_quality()
                for w in warnings:
                    logging.warning(w)
    """
    pass
