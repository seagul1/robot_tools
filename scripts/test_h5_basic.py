#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Basic HDF5 test script: prints dataset shapes/dtypes and samples; exports first color frame per camera.
Usage: python3 scripts/test_h5_basic.py [path/to/file.h5]
If no path provided, uses newest .h5 under data/.
"""
import os
import sys
import h5py
import numpy as np

try:
    from PIL import Image
    _HAVE_PIL = True
except Exception:
    _HAVE_PIL = False


def latest_h5(folder='data'):
    if not os.path.isdir(folder):
        return None
    files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.h5')]
    if not files:
        return None
    files.sort(key=os.path.getmtime, reverse=True)
    return files[0]


def inspect(h5path):
    print('Inspecting', h5path)
    with h5py.File(h5path, 'r') as f:
        print('\nFile attrs:')
        for k,v in f.attrs.items():
            print(' ', k, ':', v)

        # cameras
        cams = []
        if 'observation' in f and 'images' in f['observation']:
            imgs = f['observation']['images']
            cams = list(imgs.keys())
            print('\nCameras:', cams)
            base = os.path.splitext(os.path.basename(h5path))[0]
            for cam in cams:
                grp = imgs[cam]
                if 'color' in grp:
                    ds = grp['color']
                    print(f" camera '{cam}' color: shape={ds.shape} dtype={ds.dtype}")
                    if ds.shape[0] > 0:
                        img0 = ds[0]
                        out = os.path.join('data', f'{base}_{cam}_frame0')
                        if _HAVE_PIL and img0.ndim == 3 and img0.dtype == np.uint8:
                            im = Image.fromarray(img0)
                            outp = out + '.png'
                            im.save(outp)
                            print('  Saved', outp)
                        else:
                            outp = out + '.npy'
                            np.save(outp, img0)
                            print('  Saved raw array', outp)
                else:
                    print(f" camera '{cam}' has no color dataset")
        else:
            print('\nNo observation/images group')

        # proprioception
        prop = 'observation/proprioception'
        if prop in f:
            g = f[prop]
            print('\nProprioception datasets:')
            for name in ['joint_timestamp','joints','eef','gripper']:
                if name in g:
                    ds = g[name]
                    print(f'  {name}: shape={ds.shape} dtype={ds.dtype}')
                    try:
                        sample = ds[:10]
                        if np.issubdtype(ds.dtype, np.integer):
                            sample = [int(x) for x in sample]
                        print('    sample:', sample)
                    except Exception as e:
                        print('    could not read sample:', e)
                else:
                    print('  missing', name)
        else:
            print('\nNo proprioception group')

        # actions (may be at root or under observation)
        print('\nAction datasets (root-level or observation/):')
        action_names = ['action','action_joint_abs','action_joint_rel','action_eef_abs','action_eef_rel']
        for name in action_names:
            # check root first
            if name in f:
                ds = f[name]
                print(f'  {name} (root): shape={ds.shape} dtype={ds.dtype}')
                try:
                    s = ds[:10]
                    print('    sample:', s.tolist() if hasattr(s, 'tolist') else s)
                except Exception as e:
                    print('    could not read sample:', e)
            else:
                # check under observation
                obs_path = 'observation/' + name
                if obs_path in f:
                    ds = f[obs_path]
                    print(f"  {name} (observation): shape={ds.shape} dtype={ds.dtype}")
                    try:
                        s = ds[:10]
                        print('    sample:', s.tolist() if hasattr(s, 'tolist') else s)
                    except Exception as e:
                        print('    could not read sample:', e)
                else:
                    print(f'  {name}: MISSING')

if __name__ == '__main__':
    if len(sys.argv) > 1 and os.path.exists(sys.argv[1]):
        path = sys.argv[1]
    else:
        path = latest_h5('data')
        if path is None:
            print('No h5 file found in data/. Provide path as arg.')
            sys.exit(2)
    inspect(path)
    print('\nDone')
