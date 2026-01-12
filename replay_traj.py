import h5py
import time
import numpy as np
from deployment.Robot.Piper.PiperRobot import PiperInterface


ACTION_TYPE_TO_HDF5_KEY = {
    "joint_abs": "action/joint_abs",
    "joint_rel": "action/joint_rel",
    "eef_abs":   "action/eef_abs",
    "eef_rel":   "action/eef_rel",
}

FALLBACK_KEY = {
    "joint_abs": "observation/proprioception/joint",
    "eef_abs":   "observation/proprioception/eef",
}


def replay_hdf5_trajectory(
    robot,
    hdf5_path: str,
    traj_name: str = "traj_1",
    action_type: str = "joint_abs",
    sleep_dt: float = 0.15,
    init_wait: float = 2.0,
):
    """
    Replay a trajectory from HDF5 using user-specified action_type.

    Args:
        robot: robot interface with update_command()
        hdf5_path: path to hdf5 file
        traj_name: group name, e.g. traj_1
        action_type: one of
            ['joint_abs', 'joint_rel', 'eef_abs', 'eef_rel']
        sleep_dt: sleep between steps
        init_wait: wait after first command
    """

    assert action_type in ACTION_TYPE_TO_HDF5_KEY, \
        f"Invalid action_type: {action_type}"

    print("=" * 70)
    print(f"[Replay] File       : {hdf5_path}")
    print(f"[Replay] Trajectory: {traj_name}")
    print(f"[Replay] ActionType: {action_type}")
    print("=" * 70)

    with h5py.File(hdf5_path, "r") as f:
        assert traj_name in f, f"Trajectory '{traj_name}' not found"
        traj = f[traj_name]

        # ======================================================
        # Load trajectory according to action_type
        # ======================================================
        traj_data = None

        primary_key = ACTION_TYPE_TO_HDF5_KEY[action_type]
        if primary_key in traj:
            traj_data = np.array(traj[primary_key])
            print(f"[INFO] Use {primary_key}")

        elif action_type in FALLBACK_KEY and FALLBACK_KEY[action_type] in traj:
            traj_data = np.array(traj[FALLBACK_KEY[action_type]])
            print(f"[WARN] {primary_key} not found, "
                  f"fallback to {FALLBACK_KEY[action_type]}")

        else:
            raise KeyError(
                f"Cannot find data for action_type '{action_type}'. "
                f"Tried: {primary_key}"
            )

    # ======================================================
    # Sanity check
    # ======================================================
    assert traj_data.ndim == 2, "Trajectory must be (T, D)"
    assert traj_data.shape[1] == 7, \
        f"Expected action dim 7, got {traj_data.shape[1]}"

    T = traj_data.shape[0]
    print(f"[Replay] Trajectory length: {T}")

    # ======================================================
    # Move to initial pose (only for abs)
    # ======================================================
    if action_type.endswith("_abs"):
        print("[Replay] Move to initial absolute pose")
        robot.update_command(traj_data[0], action_type)
        time.sleep(init_wait)

    # ======================================================
    # Replay loop
    # ======================================================
    print("[Replay] Start replay")
    for t in range(T):
        action = traj_data[t]
        print(f"[Step {t:04d}] {action}")

        robot.update_command(action, action_type)
        time.sleep(sleep_dt)

    print("[Replay] Finished")
    print("=" * 70)


# ==========================================================
# Example main
# ==========================================================

if __name__ == "__main__":

    robot = PiperInterface()

    hdf5_path = "/home/liu/data/traj_001.hdf5"

    # =========================
    # 用户自由指定 action_type
    # =========================
    replay_hdf5_trajectory(
        robot=robot,
        hdf5_path=hdf5_path,
        traj_name="traj_1",
        action_type="joint_abs",   
        sleep_dt=0.15,
    )

    # 其他合法示例：
    # action_type="eef_abs"
    # action_type="joint_rel"
    # action_type="eef_rel"