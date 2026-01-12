import os
import tempfile
import json

import numpy as np
import h5py

import importlib.util
import pathlib
import sys
import types

# import the module under test (load by file path to avoid import path issues)
repo_root = pathlib.Path(__file__).resolve().parents[1]
module_path = repo_root / 'piper_collect.py'
spec = importlib.util.spec_from_file_location('piper_collect', str(module_path))
pc = importlib.util.module_from_spec(spec)
# inject lightweight stubs for hardware-dependent modules so module import succeeds
if 'deployment.Camera.rs_camera' not in sys.modules:
    mod_cam = types.ModuleType('deployment.Camera.rs_camera')
    # dummy class
    class MultiRealSense:
        def __init__(self, *args, **kwargs):
            self.front_process = types.SimpleNamespace(camera_info=None)
        def __call__(self):
            return {'color': None, 'depth': None}
        def finalize(self):
            pass
    mod_cam.MultiRealSense = MultiRealSense
    sys.modules['deployment.Camera.rs_camera'] = mod_cam

if 'deployment.Robot.Piper.PiperRobot' not in sys.modules:
    mod_piper = types.ModuleType('deployment.Robot.Piper.PiperRobot')
    class PiperInterface:
        def __init__(self, *args, **kwargs):
            pass
        def request_all_params(self):
            pass
    mod_piper.PiperInterface = PiperInterface
    sys.modules['deployment.Robot.Piper.PiperRobot'] = mod_piper

spec.loader.exec_module(pc)


def test_h5incremental_writer_basic_append_and_attrs():
    # create a small writer, append two frames, close and inspect file
    with tempfile.TemporaryDirectory() as td:
        out_path = os.path.join(td, 'traj_1.h5')
        # provide camera_info keyed by '<cam>_camera_info' so writer detects enabled cams
        cam_info = {'head_camera_info': types.SimpleNamespace(width=8, height=8, fx=1.0, fy=1.0, cx=4.0, cy=4.0, scale=0.001)}
        writer = pc.H5IncrementalWriter(out_path, 'traj_1', img_shape=(8,8,3), depth_shape=(8,8), camera_info=cam_info, attrs={'created': 123.0})

        # prepare two simple frames (per-camera dicts)
        color0 = np.zeros((8,8,3), dtype=np.uint8)
        depth0 = np.zeros((8,8), dtype=np.float32)
        joints0 = np.zeros(6, dtype=np.float64)
        eef0 = np.zeros(6, dtype=np.float64)
        colors = {'head': color0}
        depths = {'head': depth0}
        cam_ts = {'head': 0.0}
        writer.append(colors, depths, cam_ts, 0.0, joints0, 0.0, eef0, 0.0, np.zeros(7, dtype=np.float64))

        color1 = np.ones((8,8,3), dtype=np.uint8)*255
        depth1 = np.ones((8,8), dtype=np.float32)*2.5
        joints1 = np.arange(6).astype(np.float64)
        eef1 = np.arange(6).astype(np.float64)*0.1
        colors = {'head': color1}
        depths = {'head': depth1}
        cam_ts = {'head': 0.1}
        writer.append(colors, depths, cam_ts, 0.1, joints1, 0.1, eef1, 1.0, np.ones(7, dtype=np.float64))

        assert writer.length == 2
        writer.close()

        # open file and inspect datasets and attributes
        with h5py.File(out_path, 'r') as f:
            assert 'traj_1' in f
            g = f['traj_1']
            # observations
            obs = g['observation']
            imgs = obs['images']
            # now images are grouped per camera
            assert 'head' in imgs
            head = imgs['head']
            assert head['color'].shape == (2,8,8,3)
            assert head['depth'].shape == (2,8,8)
            # dataset units
            assert head['color'].attrs.get('unit') == 'uint8'
            assert head['depth'].attrs.get('unit') == 'meters'

            proprio = obs['proprioception']
            assert proprio['timestamp'].shape[0] == 2
            assert proprio['joints'].shape == (2,6)
            assert proprio['joints'].attrs.get('unit') == 'radian'
            assert proprio['eef'].attrs.get('unit') is not None
            assert proprio['gripper'].shape[0] == 2

            # actions
            assert g['action'].shape == (2,7)

            # meta / camera intrinsics
            meta = g['meta']
            cammeta = meta['camera_intrinsics']
            # camera info string stored
            assert 'head_camera_info' in cammeta.attrs
            enabled = json.loads(cammeta.attrs['enabled_cams'])
            assert 'head' in enabled

            # file-level attr
            assert g.attrs.get('created') == 123.0

