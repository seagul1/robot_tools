#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Visualization tools for trajectories saved by `piper_collect.py`.

Features:
- Read all `.h5` files in a task folder (one trajectory per file).
- Plot all trajectories' end-effector (EEF) positions in 3D (lines + scatter).
- Plot joint trajectories (6 joints) across all trajectories (one subplot per joint).
- For each trajectory, draw a small coordinate frame at the final EEF pose to show orientation.

Usage examples:
  python3 visualize_data.py --task_dir data/task_xyz --show
  python3 visualize_data.py --task_dir data/task_xyz --save figures.png

Notes:
- This script assumes each HDF5 file follows the collector layout: e.g. `/observation/proprioception/eef` (N,6)
  with first 3 columns (x,y,z) in meters and last 3 columns as rotations in radians. The rotation
  format is assumed to be local Euler angles (rx,ry,rz). If your dataset uses another convention,
  adjust `draw_coord_frame` accordingly.
"""

import os
import argparse
import glob
import math
import numpy as np
import h5py
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def find_h5_files(task_dir):
	p = os.path.abspath(task_dir)
	files = sorted(glob.glob(os.path.join(p, "*.h5")))
	return files


def load_trajectory(path):
	"""Load EEF and joints from a trajectory HDF5 file.

	Returns dict with keys: 'path', 'eef' (N,6), 'joints' (N,6), 'gripper' (N,), 'length'
	Missing datasets will be replaced by zeros of appropriate shape.
	"""
	data = {'path': path}
	with h5py.File(path, 'r') as f:
		# navigate to proprioception
		try:
			pj = f['observation']['proprioception']
		except Exception:
			pj = None

		if pj is not None and 'eef' in pj:
			eef = pj['eef'][:]
		else:
			eef = np.zeros((0,6), dtype=np.float64)

		if pj is not None and 'joints' in pj:
			joints = pj['joints'][:]
		else:
			joints = np.zeros((0,6), dtype=np.float64)

		if pj is not None and 'gripper' in pj:
			gripper = pj['gripper'][:]
		else:
			gripper = np.zeros((len(joints),), dtype=np.float64)

	data['eef'] = eef
	data['joints'] = joints
	data['gripper'] = gripper
	data['length'] = max(len(eef), len(joints))
	return data


def draw_coord_frame(ax, origin, rx, ry, rz, size=0.05):
	"""Draw a small 3D coordinate frame at origin with rotations rx,ry,rz (radians).

	This uses the convention that rx,ry,rz are Euler rotations applied in order X, Y, Z.
	If your data uses a different convention replace this with the correct rotation.
	"""
	# build rotation matrix from Euler XYZ
	cx, sx = math.cos(rx), math.sin(rx)
	cy, sy = math.cos(ry), math.sin(ry)
	cz, sz = math.cos(rz), math.sin(rz)

	# rotation matrices
	Rx = np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]])
	Ry = np.array([[cy, 0, sy], [0, 1, 0], [-sy, 0, cy]])
	Rz = np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])

	R = Rz @ Ry @ Rx

	origin = np.asarray(origin).reshape(3,)
	x_axis = origin + R[:, 0] * size
	y_axis = origin + R[:, 1] * size
	z_axis = origin + R[:, 2] * size

	# draw arrows
	ax.plot([origin[0], x_axis[0]], [origin[1], x_axis[1]], [origin[2], x_axis[2]], color='r')
	ax.plot([origin[0], y_axis[0]], [origin[1], y_axis[1]], [origin[2], y_axis[2]], color='g')
	ax.plot([origin[0], z_axis[0]], [origin[1], z_axis[1]], [origin[2], z_axis[2]], color='b')


def plot_all_eef_3d(trajs, ax=None, show_start_end=True):
	"""Plot all trajectories' EEF positions in 3D.

	trajs: list of loaded trajectory dicts (from load_trajectory)
	"""
	if ax is None:
		fig = plt.figure(figsize=(9, 7))
		ax = fig.add_subplot(111, projection='3d')
	else:
		fig = None

	cmap = plt.get_cmap('tab10')

	for i, t in enumerate(trajs):
		eef = t['eef']
		if eef is None or len(eef) == 0:
			continue
		xyz = eef[:, :3]
		color = cmap(i % 10)
		ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, label=os.path.basename(t['path']))
		ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], color=color, s=6)

		if show_start_end:
			# mark start and end
			ax.scatter([xyz[0, 0]], [xyz[0, 1]], [xyz[0, 2]], color=color, s=40, marker='o', edgecolors='k')
			ax.scatter([xyz[-1, 0]], [xyz[-1, 1]], [xyz[-1, 2]], color=color, s=80, marker='X', edgecolors='k')

	ax.set_xlabel('X (m)')
	ax.set_ylabel('Y (m)')
	ax.set_zlabel('Z (m)')
	ax.set_title('End-effector positions (all trajectories)')
	ax.legend(fontsize='small')
	return fig, ax


def plot_joint_trajectories(trajs, savepath=None):
	"""Plot each joint's trajectories. One row per joint, x axis is timestep index."""
	joints_count = 6
	fig, axes = plt.subplots(joints_count, 1, figsize=(10, 2 * joints_count), sharex=True)
	cmap = plt.get_cmap('tab10')

	for i, t in enumerate(trajs):
		joints = t['joints']
		if joints is None or len(joints) == 0:
			continue
		indices = np.arange(joints.shape[0])
		color = cmap(i % 10)
		for j in range(joints_count):
			axes[j].plot(indices, joints[:, j], color=color, label=os.path.basename(t['path']) if j == 0 else None)
			axes[j].set_ylabel(f'joint_{j}')

	axes[-1].set_xlabel('timestep')
	axes[0].legend(fontsize='small')
	fig.suptitle('Joint trajectories (all trajectories)')
	fig.tight_layout(rect=[0, 0.03, 1, 0.97])
	if savepath:
		fig.savefig(savepath)
	return fig, axes


def plot_ee_poses_at_end(trajs, ax=None, size=0.05):
	"""For each trajectory, draw the end effector coordinate frame at the last timestep."""
	if ax is None:
		fig = plt.figure(figsize=(9, 7))
		ax = fig.add_subplot(111, projection='3d')
	else:
		fig = None

	cmap = plt.get_cmap('tab10')
	for i, t in enumerate(trajs):
		eef = t['eef']
		if eef is None or len(eef) == 0:
			continue
		xyz = eef[:, :3]
		rx, ry, rz = eef[-1, 3], eef[-1, 4], eef[-1, 5]
		origin = xyz[-1]
		draw_coord_frame(ax, origin, rx, ry, rz, size=size)
		ax.text(origin[0], origin[1], origin[2], os.path.basename(t['path']), color=cmap(i % 10))

	ax.set_xlabel('X (m)')
	ax.set_ylabel('Y (m)')
	ax.set_zlabel('Z (m)')
	ax.set_title('End-effector poses (final frames)')
	return fig, ax


def main(args):
	files = find_h5_files(args.task_dir)
	if not files:
		print('No .h5 files found in', args.task_dir)
		return

	trajs = [load_trajectory(p) for p in files]

	# plot EEF 3D
	fig1, ax1 = plot_all_eef_3d(trajs)
	if args.save:
		fig1.savefig(args.save)
	if args.show:
		plt.show()

	# plot joints
	fig2, axes = plot_joint_trajectories(trajs, savepath=None)
	if args.save_joints:
		fig2.savefig(args.save_joints)
	if args.show:
		plt.show()

	# plot final poses
	fig3, ax3 = plot_ee_poses_at_end(trajs)
	if args.save_poses:
		fig3.savefig(args.save_poses)
	if args.show:
		plt.show()


if __name__ == '__main__':
	p = argparse.ArgumentParser()
	p.add_argument('--task_dir', required=True, help='Directory containing trajectory .h5 files for one task')
	p.add_argument('--show', action='store_true', help='Show plots interactively')
	p.add_argument('--save', help='Path to save combined EEF 3D figure')
	p.add_argument('--save_joints', help='Path to save joint trajectories figure')
	p.add_argument('--save_poses', help='Path to save final EEF poses figure')
	args = p.parse_args()
	main(args)

