import os
import tempfile
import h5py
import numpy as np
import sys
import types
sys.path.append('.')  # ensure repo root is in import path

# Inject lightweight stubs for cv2 and hardware modules so tests run without GUI/hardware
if 'cv2' not in sys.modules:
    cv2_stub = types.SimpleNamespace()
    cv2_stub.namedWindow = lambda *a, **k: None
    cv2_stub.imshow = lambda *a, **k: None
    cv2_stub.waitKey = lambda *a, **k: -1
    cv2_stub.destroyAllWindows = lambda *a, **k: None
    cv2_stub.putText = lambda *a, **k: None
    cv2_stub.WINDOW_NORMAL = 0
    sys.modules['cv2'] = cv2_stub

# stub Piper interface
if 'deployment.Robot.Piper.PiperRobot' not in sys.modules:
    piper_mod = types.ModuleType('deployment.Robot.Piper.PiperRobot')
    class DummyPiper:
        def __init__(self, *a, **k):
            pass
        def request_all_params(self):
            return None
        def get_joint_positions(self):
            return (np.zeros(6, dtype=np.float64), float(0.0))
        def get_ee_pose(self):
            return np.zeros(6, dtype=np.float64)
        def get_gripper_width(self):
            return 0.0
        def get_arm_status(self):
            return None
    piper_mod.PiperInterface = DummyPiper
    sys.modules['deployment.Robot.Piper.PiperRobot'] = piper_mod

# stub camera module
if 'deployment.Camera.rs_camera' not in sys.modules:
    cam_mod = types.ModuleType('deployment.Camera.rs_camera')
    class DummyCam:
        def __init__(self, *a, **k):
            self.enabled_cams = ['head']
            self.camera_info = None
        def __call__(self):
            return {}
        def finalize(self):
            return None
    cam_mod.MultiRealSense = DummyCam
    sys.modules['deployment.Camera.rs_camera'] = cam_mod

from piper_collect import H5IncrementalWriter


def test_action_is_previous_joint_and_gripper():
    tmp = tempfile.NamedTemporaryFile(suffix='.h5', delete=False)
    tmp.close()
    path = tmp.name
    print(path)
    try:
        writer = H5IncrementalWriter("test.h5", 'test_traj', img_shape=(64,64,3), depth_shape=(64,64), enabled_cams=['head'])

        # first sample
        joints0 = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64)
        grip0 = 0.1
        colors = {'head': np.zeros((64,64,3), dtype=np.uint8)}
        depths = {'head': np.zeros((64,64), dtype=np.float32)}
        cam_ts = {'head': 0.01}
        writer.append(colors, depths, cam_ts, 0.01, joints0, 0.01, np.zeros(6, dtype=np.float64), grip0)

        # second sample (different joints/gripper)
        joints1 = np.array([1.1, 2.1, 3.1, 4.1, 5.1, 6.1], dtype=np.float64)
        grip1 = 0.2
        writer.append(colors, depths, cam_ts, 0.02, joints1, 0.02, np.zeros(6, dtype=np.float64), grip1)

        writer.close()

        # validate in HDF5
        with h5py.File(path, 'r') as f:
            pj = f['observation']['proprioception']
            acts = f['action'][:]
            joints_ds = pj['joints'][:]
            gr = pj['gripper'][:]

            # joints stored properly
            assert np.allclose(joints_ds[0], joints0)
            assert np.allclose(joints_ds[1], joints1)

            # New semantics: action[t] == joints[t+1] + gripper[t+1]
            expected_next = np.concatenate([joints1, np.array([grip1])])
            # action[0] should equal joints1+grip1
            assert np.allclose(acts[0], expected_next)
            # final action (index 1) is filled on close() with last observed joints+gripper
            assert np.allclose(acts[1], expected_next)

    finally:
        try:
            os.unlink(path)
        except Exception:
            pass


if __name__ == '__main__':
    test_action_is_previous_joint_and_gripper()
    print('TEST_OK')
# import h5py
# import numpy as np
# import sys
# sys.path.append('.')  # ensure repo root is in import path
# from piper_collect import H5IncrementalWriter
# import time

# if __name__ == "__main__":
#     writer = H5IncrementalWriter('test_traj.h5', 'test_traj', img_shape=(480,640,3), depth_shape=(480,640), enabled_cams=['head1', 'head2', 'right'])

#     for _ in range(10):
#         color = np.random.randint(0, 255, (480,640,3), dtype=np.uint8)
#         depth = np.random.rand(480,640).astype(np.float32)
#         colors = {'head1': color, 'head2': color, 'right': color}
#         depths = {'head1': depth, 'head2': depth, 'right': depth}
#         cam_ts = {'head1': 1.23, 'head2': 1.23, 'right': 1.24}
#         joints = np.array([10.0, 20.0, 30.0, 40.0, 50.0, 60.0], dtype=np.float64)
#         eef = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=np.float64)
#         gripper = 0.05
#         action = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], dtype=np.float64)
#         writer.append(colors, depths, cam_ts, 1.25, joints, 1.26, eef, gripper, action)
#         time.sleep(0.1)
#     writer.close()