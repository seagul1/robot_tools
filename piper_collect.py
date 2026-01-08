#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Interactive data collection script for Piper robot + RealSense.

Press SPACE to start a trajectory recording, press SPACE again to stop
and save the collected trajectory to an HDF5 file.

Requirements:
- Uses `deployment.Robot.Piper.PiperRobot.PiperInterface` for robot state.
- Uses `deployment.Camera.rs_camera.MultiRealSense` for images.

Saved HDF5 layout (per trajectory group):
  /traj_<ts>/observation/images/color   (N,H,W,3) uint8
  /traj_<ts>/observation/images/depth   (N,H,W) float32 (meters)
  /traj_<ts>/observation/proprioception/timestamp  (N,) float64
  /traj_<ts>/observation/proprioception/joints     (N,6) float64 (radian)
  /traj_<ts>/observation/proprioception/eef        (N,6) float64 (m + rad)
  /traj_<ts>/observation/proprioception/gripper    (N,) float64 (0-1)
  /traj_<ts>/action                              (N,7) float64 (placeholder)
  /traj_<ts>/meta/camera_intrinsics              (dict fields)

This script shows the camera preview and prints basic robot status.
"""

import os
import time
import argparse
from collections import deque
import numpy as np
import h5py
import json
import cv2

from deployment.Robot.Piper.PiperRobot import PiperInterface
from deployment.Camera.rs_camera import MultiRealSense


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


class H5IncrementalWriter:
    """Helper for incremental writing of trajectory data into HDF5.

    Creates extendable datasets and appends rows as samples arrive.
    Stores unit attributes on datasets and camera intrinsics/extrinsics as attributes.
    """
    def __init__(self, out_path, traj_name, img_shape=(0,0,3), depth_shape=(0,0), camera_info=None, enabled_cams=None, attrs=None):
        self.out_path = out_path
        self.traj_name = traj_name
        self.file = h5py.File(out_path, 'w')
        # self.group = self.file.create_group(traj_name)

        obs = self.file.create_group('observation')
        imgs = obs.create_group('images')

        # determine enabled cameras
        # camera_info expected to be a dict like {'head_camera_info': CameraInfo, ...}
        if enabled_cams is None:
            if camera_info is not None:
                # derive enabled cams from keys
                enabled = []
                for k in camera_info.keys():
                    if k.endswith('_camera_info'):
                        enabled.append(k.replace('_camera_info', ''))
                self.enabled_cams = enabled if enabled else ['head']
            else:
                self.enabled_cams = ['head']
        else:
            self.enabled_cams = enabled_cams

        # create per-camera datasets under images/<cam>/{color,depth,timestamp}
        self.cam_dsets = {}
        for cam in self.enabled_cams:
            cam_group = imgs.create_group(cam)
            # try to get camera-specific shape
            ci = None
            if camera_info is not None:
                key = f'{cam}_camera_info'
                ci = camera_info.get(key, None)

            if ci is not None:
                h, w = int(ci.height), int(ci.width)
            else:
                h, w = img_shape[0], img_shape[1]

            # color dataset
            maxshape_color = (None, h, w, img_shape[2])
            ds_color = cam_group.create_dataset('color', shape=(0, h, w, img_shape[2]), maxshape=maxshape_color,
                                               dtype=np.uint8, chunks=(1, h, w, img_shape[2]), compression='gzip')
            ds_color.attrs['unit'] = 'uint8'

            # depth dataset
            maxshape_depth = (None, h, w)
            ds_depth = cam_group.create_dataset('depth', shape=(0, h, w), maxshape=maxshape_depth,
                                               dtype=np.float32, chunks=(1, h, w), compression='gzip')
            ds_depth.attrs['unit'] = 'meters'

            # per-camera timestamps
            ds_ts = cam_group.create_dataset('timestamp', shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))
            ds_ts.attrs['unit'] = 'seconds'

            self.cam_dsets[cam] = {'color': ds_color, 'depth': ds_depth, 'timestamp': ds_ts}

        # proprioception datasets
        proprio = obs.create_group('proprioception')
        # self.ds_timestamp = proprio.create_dataset('timestamp', shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))
        # self.ds_timestamp.attrs['unit'] = 'seconds'

        # joint timestamp (robot-provided timestamp for joint state)
        self.ds_joint_timestamp = proprio.create_dataset('joint_timestamp', shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))
        self.ds_joint_timestamp.attrs['unit'] = 'seconds'

        self.ds_joints = proprio.create_dataset('joints', shape=(0,6), maxshape=(None,6), dtype=np.float64, chunks=(1024,6))
        self.ds_joints.attrs['unit'] = 'radian'

        self.ds_eef = proprio.create_dataset('eef', shape=(0,6), maxshape=(None,6), dtype=np.float64, chunks=(1024,6))
        # eef: XYZ meters, RX RY RZ radians
        self.ds_eef.attrs['unit'] = 'm+rad (X,Y,Z in m; RX,RY,RZ in rad)'

        self.ds_gripper = proprio.create_dataset('gripper', shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))
        self.ds_gripper.attrs['unit'] = 'normalized (0-1)'

        # actions
        self.ds_action = self.file.create_dataset('action', shape=(0,7), maxshape=(None,7), dtype=np.float64, chunks=(1024,7))
        # action is stored as absolute joint angles (6) + gripper (1)
        # By design `action[t]` == joints[t-1] + gripper[t-1]. For the first sample
        # where no previous exists we store the initial joints+gripper (i.e. a zero-lag
        # fallback). Downstream consumers should account for this 1-step lag.
        self.ds_action.attrs['unit'] = 'radian + normalized (gripper)'
        self.ds_action.attrs['note'] = 'action[t] = joints[t-1] (6) concatenated with gripper[t-1] (1). first action==initial joints+gripper'

        # camera metadata (store per-camera intrinsics under meta/camera_intrinsics/<cam>)
        meta = self.file.create_group('meta')
        cammeta = meta.create_group('camera_intrinsics')
        if camera_info is not None:
            for k, v in camera_info.items():
                # camera_info keys like 'head_camera_info'
                cammeta.attrs[k] = str(v)
        cammeta.attrs['enabled_cams'] = json.dumps(self.enabled_cams)

        if attrs:
            for k, v in attrs.items():
                self.file.attrs[k] = v

        # track current length (common proprio length)
        self.length = 0
        # pending action index: when we append a sample at index n we don't yet
        # know the next-step joints (which define action[n]). We append a placeholder
        # and remember the index here; on the next append we fill that index with
        # the newly observed joints+gripper. On close we fill the final pending
        # action with the last observed joints+gripper as a fallback.
        self._pending_action_index = None
        self._last_joints = None
        self._last_gripper = None

    def append(self, colors: dict, depths: dict, cam_timestamps: dict, timestamp, joints, joint_timestamp, eef, gripper):
        """Append a sample.

        colors, depths, cam_timestamps are dicts keyed by cam name (e.g., 'head','right','left').
        timestamp is the proprio timestamp (float).
        """
        n = self.length
        # per-camera datasets
        for cam in self.enabled_cams:
            ds = self.cam_dsets[cam]
            # color
            ds['color'].resize((n+1,) + ds['color'].shape[1:])
            c = colors.get(cam)
            ds['color'][n] = c
            # depth
            ds['depth'].resize((n+1,) + ds['depth'].shape[1:])
            d = depths.get(cam)
            ds['depth'][n] = d
            # cam timestamp
            ds['timestamp'].resize((n+1,))
            ts_cam = cam_timestamps.get(cam, float(timestamp))
            ds['timestamp'][n] = ts_cam

        # # proprio (common)
        # self.ds_timestamp.resize((n+1,))
        # self.ds_timestamp[n] = timestamp

        # joint timestamp (robot-provided)
        self.ds_joint_timestamp.resize((n+1,))
        self.ds_joint_timestamp[n] = joint_timestamp

        self.ds_joints.resize((n+1,6))
        self.ds_joints[n] = joints
        # eef
        self.ds_eef.resize((n+1,6))
        self.ds_eef[n] = eef
        # gripper
        self.ds_gripper.resize((n+1,))
        self.ds_gripper[n] = gripper

        # action: the dataset stores absolute joint+gripper values such that
        # joints[t+1] == action[t]. Therefore when appending sample at index n
        # we fill the previously pending action (index n-1) with the current
        # joints+gripper, then append a placeholder for action[n] and mark it
        # pending. On close(), the final pending action is filled with the
        # last observed joints+gripper as a fallback.

        # Prepare current joints+gripper value
        current_act = np.concatenate([np.asarray(joints, dtype=np.float64).reshape(6,), np.asarray([gripper], dtype=np.float64)])

        # if there is a pending action index (from previous append), fill it
        if self._pending_action_index is not None:
            idx = self._pending_action_index
            # ensure dataset big enough
            if self.ds_action.shape[0] <= idx:
                self.ds_action.resize((idx+1, 7))
            self.ds_action[idx] = current_act

        # append placeholder for current action (will be filled on next append)
        self.ds_action.resize((n+1, 7))
        # initialize placeholder with zeros
        self.ds_action[n] = np.zeros((7,), dtype=np.float64)
        # mark this index as pending to be filled by the next observed joints
        self._pending_action_index = n

        # remember last observed joints/gripper for possible fill on close()
        self._last_joints = np.asarray(joints, dtype=np.float64).reshape(6,)
        self._last_gripper = float(gripper)

        self.length += 1

    def close(self):
        try:
            # finalize pending action: if there is a pending index, fill it with
            # the last observed joints+gripper as a fallback
            if self._pending_action_index is not None and self._last_joints is not None:
                idx = self._pending_action_index
                if self.ds_action.shape[0] <= idx:
                    self.ds_action.resize((idx+1, 7))
                fill_val = np.concatenate([self._last_joints, np.asarray([self._last_gripper], dtype=np.float64)])
                self.ds_action[idx] = fill_val

            self.file.flush()
            self.file.close()
        except Exception:
            pass


def run_collector(args: argparse.Namespace):
    outdir = args.outdir
    img_size = args.img_size
    sample_dt = args.sample_dt

    ensure_outdir(outdir)

    print('Starting Piper interface...')
    piper = PiperInterface(can_name='can0')
    # ask for response params once at start (non-blocking)
    try:
        piper.request_all_params()
    except Exception:
        pass

    print('Starting camera(s)...')
    cam = MultiRealSense(use_head1_cam=True,
                         use_head2_cam=getattr(args, 'use_head2_cam', False),
                         use_right_cam=getattr(args, 'use_right_cam', False),
                         use_left_cam=getattr(args, 'use_left_cam', False),
                         img_size=img_size)
    time.sleep(0.2)

    recording = False
    collected = None
    traj_idx = 1
    writer = None

    window_name = 'Piper Collect - press SPACE to start/stop, q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_sample = 0.0
    print('Ready. Press SPACE to start a trajectory.')

    try:
        while True:
            cam_dict = cam()
            # preview color: choose first available camera color (head -> right -> left)
            preview_color = None
            try:
                cam_list_for_preview = (writer.enabled_cams if writer is not None else getattr(cam, 'enabled_cams', ['head']))
            except Exception:
                cam_list_for_preview = getattr(cam, 'enabled_cams', ['head'])
            for cam_name in cam_list_for_preview:
                pc_key = f'{cam_name}_color'
                if pc_key in cam_dict and cam_dict.get(pc_key) is not None:
                    preview_color = cam_dict.get(pc_key)
                    break
            color = preview_color
            # depth for preview (from same cam if available)
            depth = None
            if preview_color is not None:
                # attempt to get corresponding depth
                try:
                    depth = cam_dict.get(f'{cam_list_for_preview[0]}_depth', None)
                except Exception:
                    depth = None

            # obtain robot quick status
            try:
                joints, joint_ts = piper.get_joint_positions()  # returns (positions, timestamp)
                eef = piper.get_ee_pose()  # meters + radians
                gripper = piper.get_gripper_width()
                arm_status = piper.get_arm_status()
            except Exception:
                joints = np.zeros(6, dtype=np.float64)
                joint_ts = time.time()
                eef = np.zeros(6, dtype=np.float64)
                gripper = 0.0
                arm_status = None

            # display preview
            if color is None:
                preview = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            else:
                preview = color.copy()
            label = 'REC' if recording else 'IDLE'
            cv2.putText(preview, f'STATUS: {label}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if recording else (0,255,0), 2)
            # overlay joints
            jtxt = 'j:' + ','.join([f'{x:.2f}' for x in joints])
            cv2.putText(preview, jtxt, (10, img_size-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print('Quitting.')
                break
            if key == 32:  # SPACE pressed
                if not recording:
                    print('Start recording trajectory')
                    recording = True
                    # create writer immediately so we can append samples
                    start_ts = int(time.time())
                    traj_name = f'traj_{start_ts}'
                    out_path = os.path.join(outdir, f'{traj_name}.h5')
                    # get camera_info and enabled cameras from MultiRealSense
                    try:
                        caminfo = cam.camera_info
                    except Exception:
                        caminfo = None
                    try:
                        enabled = cam.enabled_cams
                    except Exception:
                        enabled = None

                    writer = H5IncrementalWriter(out_path, traj_name, img_shape=(img_size, img_size, 3), depth_shape=(img_size, img_size), camera_info=caminfo, enabled_cams=enabled, attrs={'created': time.time()})
                    last_sample = 0.0
                else:
                    print('Stop recording. Finalizing save...')
                    recording = False
                    # finalize writer and write small metadata JSON
                    try:
                        frames = writer.length
                        writer.close()
                        meta = {
                            'traj_name': traj_name,
                            'frames': frames,
                            'created': time.time(),
                        }
                    except Exception:
                        frames = 0
                        meta = {'traj_name': traj_name, 'frames': frames, 'created': time.time()}

                    # write metadata JSON alongside HDF5
                    try:
                        json_path = os.path.join(outdir, f'{traj_name}.json')
                        with open(json_path, 'w') as jf:
                            json.dump(meta, jf, indent=2)
                    except Exception:
                        pass

                    print(f'Saved {out_path}  (frames={frames})')

            # sampling while recording
            now = time.time()
            if recording and (now - last_sample >= sample_dt):
                last_sample = now
                # prepare sample for all enabled cameras
                colors = {}
                depths = {}
                cam_ts = {}
                cams_to_iterate = (writer.enabled_cams if writer is not None else getattr(cam, 'enabled_cams', ['head']))
                for cam_name in cams_to_iterate:
                    c_key = f'{cam_name}_color'
                    d_key = f'{cam_name}_depth'
                    t_key = f'{cam_name}_timestamp'
                    cval = cam_dict.get(c_key, None)
                    if cval is None:
                        cval = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                    dval = cam_dict.get(d_key, None)
                    if dval is None:
                        dval = np.zeros((img_size, img_size), dtype=np.float32)
                    tval = cam_dict.get(t_key, now)
                    colors[cam_name] = cval
                    depths[cam_name] = dval.astype(np.float32)
                    cam_ts[cam_name] = float(tval)

                ts = now
                joints_arr = np.asarray(joints, dtype=np.float64)
                eef_arr = np.asarray(eef, dtype=np.float64)
                grip = float(gripper)
                try:
                    writer.append(colors, depths, cam_ts, ts, joints_arr, float(joint_ts), eef_arr, grip)
                except Exception:
                    # fallback: ignore append errors and continue
                    pass

    finally:
        # if recorder still open, finalize it
        try:
            if writer is not None:
                try:
                    frames = writer.length
                    writer.close()
                    json_path = os.path.join(outdir, f'{traj_name}.json')
                    with open(json_path, 'w') as jf:
                        json.dump({'traj_name': traj_name, 'frames': frames, 'created': time.time()}, jf, indent=2)
                except Exception:
                    pass
        except Exception:
            pass

        cv2.destroyAllWindows()
        try:
            cam.finalize()
        except Exception:
            pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--outdir', '-o', default='data', help='Directory to save trajectories')
    parser.add_argument('--img_size', type=int, default=None, help='Image size (square) to collect')
    # 增加外部传入参数，用于控制启用多少摄像头
    parser.add_argument('--use_head2_cam', action='store_true', help='Whether to use head 2 camera')
    parser.add_argument('--use_right_cam', action='store_true', help='Whether to use right camera')
    parser.add_argument('--use_left_cam', action='store_true', help='Whether to use left camera')
    parser.add_argument('--sample_dt', type=float, default=0.05, help='Sampling interval in seconds')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_collector(args)
