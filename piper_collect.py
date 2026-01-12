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
import logging

from deployment.Robot.Piper.PiperRobot import PiperInterface
from deployment.Camera.rs_camera import MultiRealSense

# configure logging: INFO by default, DEBUG for verbose
logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')


def ensure_outdir(path):
    os.makedirs(path, exist_ok=True)


class H5IncrementalWriter:
    """Helper for incremental writing of trajectory data into HDF5.

    Creates extendable datasets and appends rows as samples arrive.
    Stores unit attributes on datasets and camera intrinsics/extrinsics as attributes.
    """
    def __init__(self, out_path, traj_name, img_shape=(0,0,3), depth_shape=(0,0), camera_info=None, enabled_cams=None, attrs=None, camera_extrinsics=None):
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
            ds_ts = cam_group.create_dataset('timestamp', shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
            ds_ts.attrs['unit'] = 'milliseconds since epoch'

            self.cam_dsets[cam] = {'color': ds_color, 'depth': ds_depth, 'timestamp': ds_ts}

        # proprioception datasets
        proprio = obs.create_group('proprioception')
        # self.ds_timestamp = proprio.create_dataset('timestamp', shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))
        # self.ds_timestamp.attrs['unit'] = 'seconds'

        # joint timestamp (robot-provided timestamp for joint state)
        self.ds_joint_timestamp = proprio.create_dataset('joint_timestamp', shape=(0,), maxshape=(None,), dtype=np.int64, chunks=(1024,))
        self.ds_joint_timestamp.attrs['unit'] = 'milliseconds since epoch'

        self.ds_joints = proprio.create_dataset('joints', shape=(0,6), maxshape=(None,6), dtype=np.float64, chunks=(1024,6))
        self.ds_joints.attrs['unit'] = 'radian'

        self.ds_eef = proprio.create_dataset('eef', shape=(0,6), maxshape=(None,6), dtype=np.float64, chunks=(1024,6))
        # eef: XYZ meters, RX RY RZ radians
        self.ds_eef.attrs['unit'] = 'm+rad (X,Y,Z in m; RX,RY,RZ in rad)'

        self.ds_gripper = proprio.create_dataset('gripper', shape=(0,), maxshape=(None,), dtype=np.float64, chunks=(1024,))
        self.ds_gripper.attrs['unit'] = 'normalized (0-1)'

        # actions: we store several action variants below (do not create legacy 'action')

        # camera metadata (store per-camera intrinsics under meta/camera_intrinsics/<cam>)
        meta = self.file.create_group('meta')
        cammeta = meta.create_group('camera_intrinsics')
        # camera_info: store numeric intrinsics as datasets under each camera subgroup
        for cam in self.enabled_cams:
            grp = cammeta.create_group(cam)
            grp.attrs['provided'] = False
            # default placeholders
            grp.create_dataset('width', data=np.array([np.nan], dtype=np.float64))
            grp.create_dataset('height', data=np.array([np.nan], dtype=np.float64))
            grp.create_dataset('fx', data=np.array([np.nan], dtype=np.float64))
            grp.create_dataset('fy', data=np.array([np.nan], dtype=np.float64))
            grp.create_dataset('cx', data=np.array([np.nan], dtype=np.float64))
            grp.create_dataset('cy', data=np.array([np.nan], dtype=np.float64))
            grp.create_dataset('scale', data=np.array([np.nan], dtype=np.float64))

        # populate if camera_info provided
        if camera_info is not None:
            for k, v in camera_info.items():
                if k.endswith('_camera_info'):
                    cam_name = k.replace('_camera_info', '')
                else:
                    cam_name = k
                try:
                    if cam_name not in cammeta:
                        grp = cammeta.create_group(cam_name)
                    else:
                        grp = cammeta[cam_name]
                    grp.attrs['provided'] = True
                    # overwrite datasets with numeric values
                    try:
                        grp['width'][0] = float(v.width)
                    except Exception:
                        grp['width'][0] = np.nan
                    try:
                        grp['height'][0] = float(v.height)
                    except Exception:
                        grp['height'][0] = np.nan
                    try:
                        grp['fx'][0] = float(v.fx)
                    except Exception:
                        grp['fx'][0] = np.nan
                    try:
                        grp['fy'][0] = float(v.fy)
                    except Exception:
                        grp['fy'][0] = np.nan
                    try:
                        grp['cx'][0] = float(v.cx)
                    except Exception:
                        grp['cx'][0] = np.nan
                    try:
                        grp['cy'][0] = float(v.cy)
                    except Exception:
                        grp['cy'][0] = np.nan
                    try:
                        grp['scale'][0] = float(getattr(v, 'scale', 1.0))
                    except Exception:
                        grp['scale'][0] = np.nan
                except Exception:
                    # fallback: store string representation as attribute
                    g = cammeta.require_group(cam_name)
                    g.attrs['provided'] = False
                    g.attrs['info_str'] = str(v)
        # record enabled camera list
        cammeta.attrs['enabled_cams'] = json.dumps(self.enabled_cams)

        # extrinsics group: always create and populate if provided, otherwise create placeholders
        ext_grp = meta.create_group('camera_extrinsics')
        if camera_extrinsics is not None:
            for cam_name, ext in camera_extrinsics.items():
                try:
                    arr = np.asarray(ext, dtype=np.float64)
                    if arr.ndim == 2 and arr.shape == (4, 4):
                        ext_grp.create_dataset(cam_name, data=arr)
                    else:
                        ext_grp.attrs[cam_name] = json.dumps(ext)
                except Exception:
                    try:
                        ext_grp.attrs[cam_name] = json.dumps(ext)
                    except Exception:
                        ext_grp.attrs[cam_name] = str(ext)
        else:
            # create empty placeholder entries for enabled cams
            for cam in self.enabled_cams:
                g = ext_grp.create_group(cam)
                g.attrs['provided'] = False

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
        self._last_eef = None

        # additional action dataset variants (joint/eef x absolute/relative)
        # joint absolute (redundant with ds_action but kept for clarity)
        self.ds_action_joint_abs = self.file.create_dataset('action_joint_abs', shape=(0,7), maxshape=(None,7), dtype=np.float64, chunks=(1024,7))
        # joint relative (delta = next_joints - prev_joints, gripper delta)
        self.ds_action_joint_rel = self.file.create_dataset('action_joint_rel', shape=(0,7), maxshape=(None,7), dtype=np.float64, chunks=(1024,7))
        # eef absolute (XYZ+RXRYRZ + gripper)
        self.ds_action_eef_abs = self.file.create_dataset('action_eef_abs', shape=(0,7), maxshape=(None,7), dtype=np.float64, chunks=(1024,7))
        # eef relative (delta between subsequent eef poses + gripper delta)
        self.ds_action_eef_rel = self.file.create_dataset('action_eef_rel', shape=(0,7), maxshape=(None,7), dtype=np.float64, chunks=(1024,7))

    def append(self, colors: dict, depths: dict, cam_timestamps: dict, timestamp, joints, joint_timestamp, eef, gripper):
        """Append a sample.

        colors, depths, cam_timestamps are dicts keyed by cam name (e.g., 'head','right','left').
        timestamp is the proprio timestamp (float).
        """
        n = self.length
        # per-camera datasets
        for cam in self.enabled_cams:
            ds = self.cam_dsets[cam]
            # expected shapes
            expected_h = ds['color'].shape[1]
            expected_w = ds['color'].shape[2]

            # color
            ds['color'].resize((n+1,) + ds['color'].shape[1:])
            c = colors.get(cam)
            if c is None:
                c = np.zeros((expected_h, expected_w, ds['color'].shape[3]), dtype=np.uint8)
            else:
                # ensure proper channels
                if c.ndim == 2:
                    c = cv2.cvtColor(c, cv2.COLOR_GRAY2BGR)
                if c.shape[2] != ds['color'].shape[3]:
                    # convert or trim/expand channels
                    if c.shape[2] > ds['color'].shape[3]:
                        c = c[..., :ds['color'].shape[3]]
                    else:
                        # replicate channels if needed
                        c = np.repeat(c[..., :1], ds['color'].shape[3], axis=2)
                # resize if needed (cv2.resize expects (w,h))
                if (c.shape[0], c.shape[1]) != (expected_h, expected_w):
                    c = cv2.resize(c, (expected_w, expected_h), interpolation=cv2.INTER_LINEAR)
                c = c.astype(np.uint8)

            ds['color'][n] = c

            # depth
            ds['depth'].resize((n+1,) + ds['depth'].shape[1:])
            d = depths.get(cam)
            if d is None:
                d = np.zeros((expected_h, expected_w), dtype=np.float32)
            else:
                if d.ndim == 3:
                    # if depth has channel dim, squeeze it
                    d = d[..., 0]
                if (d.shape[0], d.shape[1]) != (expected_h, expected_w):
                    d = cv2.resize(d, (expected_w, expected_h), interpolation=cv2.INTER_NEAREST)
                d = d.astype(np.float32)

            ds['depth'][n] = d

            # cam timestamp
            ds['timestamp'].resize((n+1,))
            ts_cam = cam_timestamps.get(cam, float(timestamp))
            try:
                ds['timestamp'][n] = int(round(float(ts_cam) * 1000.0))
            except Exception:
                ds['timestamp'][n] = np.int64(0)

        # # proprio (common)
        # self.ds_timestamp.resize((n+1,))
        # self.ds_timestamp[n] = timestamp

        # joint timestamp (robot-provided)
        self.ds_joint_timestamp.resize((n+1,))
        try:
            self.ds_joint_timestamp[n] = int(round(float(joint_timestamp) * 1000.0))
        except Exception:
            self.ds_joint_timestamp[n] = np.int64(0)

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

        # Prepare current joints+gripper and eef+gripper values
        joints_arr = np.asarray(joints, dtype=np.float64).reshape(6,)
        eef_arr = np.asarray(eef, dtype=np.float64).reshape(6,)
        current_joint_act = np.concatenate([joints_arr, np.asarray([gripper], dtype=np.float64)])
        current_eef_act = np.concatenate([eef_arr, np.asarray([gripper], dtype=np.float64)])

        # if there is a pending action index (from previous append), fill it with "next" observations
        if self._pending_action_index is not None:
            idx = self._pending_action_index
            # ensure datasets big enough
            if self.ds_action_joint_abs.shape[0] <= idx:
                self.ds_action_joint_abs.resize((idx+1, 7))
            if self.ds_action_joint_rel.shape[0] <= idx:
                self.ds_action_joint_rel.resize((idx+1, 7))
            if self.ds_action_eef_abs.shape[0] <= idx:
                self.ds_action_eef_abs.resize((idx+1, 7))
            if self.ds_action_eef_rel.shape[0] <= idx:
                self.ds_action_eef_rel.resize((idx+1, 7))

            # joint absolute
            try:
                logging.debug('Filling pending action at idx=%d', idx)
            except Exception:
                pass
            self.ds_action_joint_abs[idx] = current_joint_act

            # joint relative: requires last observed joints
            if self._last_joints is not None:
                joint_rel = np.concatenate([joints_arr - self._last_joints, np.asarray([gripper - self._last_gripper], dtype=np.float64)])
            else:
                joint_rel = np.concatenate([np.zeros(6, dtype=np.float64), np.asarray([0.0], dtype=np.float64)])
            self.ds_action_joint_rel[idx] = joint_rel

            # eef absolute
            self.ds_action_eef_abs[idx] = current_eef_act

            # eef relative
            if self._last_eef is not None:
                eef_rel = np.concatenate([eef_arr - self._last_eef, np.asarray([gripper - self._last_gripper], dtype=np.float64)])
            else:
                eef_rel = np.concatenate([np.zeros(6, dtype=np.float64), np.asarray([0.0], dtype=np.float64)])
            self.ds_action_eef_rel[idx] = eef_rel

        # append placeholder for current action (will be filled on next append)
        self.ds_action_joint_abs.resize((n+1, 7))
        self.ds_action_joint_abs[n] = np.zeros((7,), dtype=np.float64)
        self.ds_action_joint_rel.resize((n+1, 7))
        self.ds_action_joint_rel[n] = np.zeros((7,), dtype=np.float64)
        self.ds_action_eef_abs.resize((n+1, 7))
        self.ds_action_eef_abs[n] = np.zeros((7,), dtype=np.float64)
        self.ds_action_eef_rel.resize((n+1, 7))
        self.ds_action_eef_rel[n] = np.zeros((7,), dtype=np.float64)
        try:
            logging.debug('Appended placeholder action at idx=%d', n)
        except Exception:
            pass
        # mark this index as pending to be filled by the next observed joints
        self._pending_action_index = n

        # remember last observed joints/gripper for possible fill on close()
        self._last_joints = np.asarray(joints, dtype=np.float64).reshape(6,)
        self._last_gripper = float(gripper)
        self._last_eef = np.asarray(eef, dtype=np.float64).reshape(6,)

        self.length += 1

    def close(self):

        try:
            # finalize pending action: if there is a pending index, fill it with
            # the last observed joints+gripper / eef as a fallback
            if self._pending_action_index is not None:
                idx = self._pending_action_index
                # ensure datasets big enough
                if self.ds_action_joint_abs.shape[0] <= idx:
                    self.ds_action_joint_abs.resize((idx+1, 7))
                if self.ds_action_joint_rel.shape[0] <= idx:
                    self.ds_action_joint_rel.resize((idx+1, 7))
                if self.ds_action_eef_abs.shape[0] <= idx:
                    self.ds_action_eef_abs.resize((idx+1, 7))
                if self.ds_action_eef_rel.shape[0] <= idx:
                    self.ds_action_eef_rel.resize((idx+1, 7))

                if self._last_joints is not None:
                    fill_joint_abs = np.concatenate([self._last_joints, np.asarray([self._last_gripper], dtype=np.float64)])
                else:
                    fill_joint_abs = np.zeros((7,), dtype=np.float64)
                try:
                    logging.debug('Finalizing pending action at idx=%d', idx)
                except Exception:
                    pass
                self.ds_action_joint_abs[idx] = fill_joint_abs

                # joint relative fallback
                fill_joint_rel = np.zeros((7,), dtype=np.float64)
                self.ds_action_joint_rel[idx] = fill_joint_rel

                # eef absolute
                if self._last_eef is not None:
                    fill_eef_abs = np.concatenate([self._last_eef, np.asarray([self._last_gripper], dtype=np.float64)])
                else:
                    fill_eef_abs = np.zeros((7,), dtype=np.float64)
                self.ds_action_eef_abs[idx] = fill_eef_abs

                # eef relative fallback
                fill_eef_rel = np.zeros((7,), dtype=np.float64)
                self.ds_action_eef_rel[idx] = fill_eef_rel

            self.file.flush()
            self.file.close()
        except Exception:
            pass


def run_collector(args: argparse.Namespace):
    outdir = args.outdir
    img_size = args.img_size
    sample_dt = args.sample_dt

    ensure_outdir(outdir)

    # parse optional camera extrinsics passed via args (JSON file path or JSON string)
    camera_extrinsics = None
    try:
        ext_arg = getattr(args, 'camera_extrinsics', None)
        if ext_arg:
            # if ext_arg is a path to a file, load it
            if os.path.exists(ext_arg):
                with open(ext_arg, 'r') as ef:
                    camera_extrinsics = json.load(ef)
            else:
                # try parse JSON string
                try:
                    camera_extrinsics = json.loads(ext_arg)
                except Exception:
                    camera_extrinsics = None
    except Exception:
        camera_extrinsics = None

    logging.info('Starting Piper interface...')
    piper = PiperInterface(can_name='can0')
    # ask for response params once at start (non-blocking)
    try:
        piper.request_all_params()
    except Exception:
        pass

    # give piper SDK some time to start receiving feedback messages
    try:
        logging.info('Warming up Piper interface for feedback...')
        warmup_start = time.time()
        warmup_timeout = 3.0
        while time.time() - warmup_start < warmup_timeout:
            try:
                j, jt = piper.get_joint_positions()
                e = piper.get_ee_pose()
                # break early if we have non-zero readings
                if not (np.allclose(j, 0.0) and np.allclose(e, 0.0)):
                    break
            except Exception:
                pass
            time.sleep(0.05)
        else:
            logging.warning('Piper feedback still zero after warmup. Continuing anyway.')
    except Exception:
        pass

    logging.info('Starting camera(s)...')
    cam = MultiRealSense(use_head1_cam=True,
                         use_head2_cam=getattr(args, 'use_head2_cam', False),
                         use_right_cam=getattr(args, 'use_right_cam', False),
                         use_left_cam=getattr(args, 'use_left_cam', False),
                         img_size=img_size)
    time.sleep(1.0)

    recording = False
    collected = None
    traj_idx = 1
    writer = None

    window_name = 'Piper Collect - press SPACE to start/stop, q to quit'
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    last_sample = 0.0
    logging.info('Ready. Press SPACE to start a trajectory.')

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

            # display preview: tile multiple camera images horizontally
            imgs_for_tile = []
            try:
                cams_for_preview = cam_list_for_preview
            except Exception:
                cams_for_preview = getattr(cam, 'enabled_cams', ['head'])

            for cam_name in cams_for_preview:
                pc_key = f'{cam_name}_color'
                cimg = cam_dict.get(pc_key, None)
                if cimg is None:
                    tile = np.zeros((img_size, img_size, 3), dtype=np.uint8)
                else:
                    tile = cimg.copy()
                    if tile.ndim == 2:
                        tile = cv2.cvtColor(tile, cv2.COLOR_GRAY2BGR)
                    try:
                        tile = cv2.resize(tile, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
                    except Exception:
                        tile = cv2.resize(tile, (img_size, img_size))
                imgs_for_tile.append(tile)

            if len(imgs_for_tile) == 0:
                preview = np.zeros((img_size, img_size, 3), dtype=np.uint8)
            elif len(imgs_for_tile) == 1:
                preview = imgs_for_tile[0]
            else:
                preview = np.hstack(imgs_for_tile)

            label = 'REC' if recording else 'IDLE'
            cv2.putText(preview, f'STATUS: {label}', (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255) if recording else (0,255,0), 2)
            # overlay joints (position vertically relative to tile height)
            jtxt = 'j:' + ','.join([f'{x:.2f}' for x in joints])
            cv2.putText(preview, jtxt, (10, img_size-10), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1)
            cv2.imshow(window_name, preview)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                logging.info('Quitting.')
                break
            if key == 32:  # SPACE pressed
                if not recording:
                    logging.info('Start recording trajectory')
                    recording = True
                    # create writer immediately so we can append samples
                    start_ts = int(time.time())
                    traj_name = f'traj_{start_ts}'
                    out_path = os.path.join(outdir, f'{traj_name}.h5')
                    # ensure we read sentinel camera_info messages from child processes
                    try:
                        _ = cam()  # flush any '__camera_info__' messages into cam._camera_info
                        # wait briefly for camera_info to be populated (max 1s)
                        wait_start = time.time()
                        while time.time() - wait_start < 1.0:
                            ci = cam.camera_info
                            if ci and len(ci) > 0:
                                break
                            time.sleep(0.05)
                    except Exception:
                        pass

                    # get camera_info and enabled cameras from MultiRealSense
                    try:
                        caminfo = cam.camera_info
                    except Exception:
                        caminfo = None
                    try:
                        enabled = cam.enabled_cams
                    except Exception:
                        enabled = None

                    writer = H5IncrementalWriter(out_path, traj_name, img_shape=(img_size, img_size, 3), depth_shape=(img_size, img_size), camera_info=caminfo, enabled_cams=enabled, attrs={'created': time.time()}, camera_extrinsics=camera_extrinsics)
                    # force an immediate sample on start
                    last_sample = time.time() - sample_dt
                    # attempt an immediate capture and write one sample so trajectory isn't empty
                    try:
                        cam_dict = cam()
                        # collect robot state
                        try:
                            joints, joint_ts = piper.get_joint_positions()
                            eef = piper.get_ee_pose()
                            gripper = piper.get_gripper_width()
                        except Exception:
                            joints = np.zeros(6, dtype=np.float64)
                            joint_ts = time.time()
                            eef = np.zeros(6, dtype=np.float64)
                            gripper = 0.0

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
                            tval = cam_dict.get(t_key)
                            if tval is None:
                                tval = time.time()
                            colors[cam_name] = cval
                            depths[cam_name] = dval.astype(np.float32)
                            cam_ts[cam_name] = float(tval)

                        ts = time.time()
                        try:
                            writer.append(colors, depths, cam_ts, ts, np.asarray(joints, dtype=np.float64), float(joint_ts), np.asarray(eef, dtype=np.float64), float(gripper))
                        except Exception as e:
                            logging.exception('Error appending initial sample:')
                            import traceback
                            traceback.print_exc()
                    except Exception:
                        pass
                else:
                    logging.info('Stop recording. Finalizing save...')
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

                    logging.info(f'Saved {out_path}  (frames={frames})')

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
                    tval = cam_dict.get(t_key)
                    if tval is None:
                        tval = now
                    colors[cam_name] = cval
                    depths[cam_name] = dval.astype(np.float32)
                    cam_ts[cam_name] = float(tval)

                ts = now
                joints_arr = np.asarray(joints, dtype=np.float64)
                eef_arr = np.asarray(eef, dtype=np.float64)
                grip = float(gripper)
                try:
                    logging.debug('Appending sample at t=%f (joint_ts=%s)', ts, str(joint_ts))
                    logging.debug('cams: %s', list(colors.keys()))
                    logging.debug('joints: %s', np.array2string(joints_arr, precision=6))
                    writer.append(colors, depths, cam_ts, ts, joints_arr, float(joint_ts), eef_arr, grip)
                except Exception:
                    logging.exception('Error appending sampled frame:')

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
    parser.add_argument('--img_size', type=int, default=480, help='Image size (square) to collect')
    # 增加外部传入参数，用于控制启用多少摄像头
    parser.add_argument('--use_head2_cam', action='store_true', help='Whether to use head 2 camera')
    parser.add_argument('--use_right_cam', action='store_true', help='Whether to use right camera')
    parser.add_argument('--use_left_cam', action='store_true', help='Whether to use left camera')
    parser.add_argument('--sample_dt', type=float, default=0.05, help='Sampling interval in seconds')
    parser.add_argument('--camera_extrinsics', type=str, default=None, help='Path to JSON file or JSON string containing camera extrinsics per camera')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    run_collector(args)
