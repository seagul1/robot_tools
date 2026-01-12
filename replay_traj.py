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
    "joint_abs": "observation/proprioception/joints",
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
        # assert traj_name in f, f"Trajectory '{traj_name}' not found"
        # 修改开始：增加根目录回退逻辑
        if traj_name in f:
            traj = f[traj_name]
        else:
            print(f"[WARN] Trajectory group '{traj_name}' not found in HDF5.")
            print(f"[WARN] Assuming data is located at the file root (created by piper_collect.py).")
            traj = f
        # 修改结束
        # traj = f[traj_name]

        # ======================================================
        # Load trajectory according to action_type
        # ======================================================
        traj_data = None

        primary_key = ACTION_TYPE_TO_HDF5_KEY[action_type]
        if primary_key in traj:
            traj_data = np.array(traj[primary_key])
            print(f"[INFO] Use {primary_key}")

        elif action_type in FALLBACK_KEY and FALLBACK_KEY[action_type] in traj:
            # 读取 6维关节数据
            joints_data = np.array(traj[FALLBACK_KEY[action_type]])
            # 尝试读取夹爪数据并拼接
            if 'observation/proprioception/gripper' in traj:
                gripper_data = np.array(traj['observation/proprioception/gripper'])
                
                # 确保 gripper 是 (N, 1) 的形状
                if gripper_data.ndim == 1:
                    gripper_data = gripper_data[:, np.newaxis]
                
                # 拼接成 (N, 7)
                traj_data = np.hstack([joints_data, gripper_data])
                print(f"[WARN] {primary_key} not found, fallback to {FALLBACK_KEY[action_type]} + gripper")
            else:
                #如果没有夹爪数据，就只能抛出异常或者暂时填充0
                print(f"[WARN] joints found but no gripper data found. Cannot form 7-dim action.")
                traj_data = joints_data
            # traj_data = np.array(traj[FALLBACK_KEY[action_type]])
            # print(f"[WARN] {primary_key} not found, "
            #       f"fallback to {FALLBACK_KEY[action_type]}")

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

    # 修改开始：将脚本的 action_type 转换为 robot 接口支持的字符串
    robot_action_type = "joint" if "joint" in action_type else "eef"
    # 修改结束

    # ======================================================
    # Move to initial pose (only for abs)
    # ======================================================
    if action_type.endswith("_abs"):
        print("[Replay] Move to initial absolute pose")
        # robot.update_command(traj_data[0], action_type)
        robot.update_command(traj_data[0], robot_action_type)
        time.sleep(init_wait)

    # ======================================================
    # Replay loop
    # ======================================================
    print("[Replay] Start replay")

    # --- 调试代码开始 ---
    # 计算整个轨迹的变化幅度（最大值 - 最小值），看看是否有运动
    diff = np.max(traj_data, axis=0) - np.min(traj_data, axis=0)
    print(f"[Debug] Range of motion (Max-Min) for each joint + gripper: \n{diff}")
    
    if np.all(diff < 0.01):
        print("[Debug] WARNING: The trajectory data seems almost static! Did you move the robot during recording?")
    # --- 调试代码结束 ---

    for t in range(T):
        action = traj_data[t]
        # print(f"[Step {t:04d}] {action}")
        # 修改：使用 robot_action_type
        robot.update_command(action, robot_action_type)
        # robot.update_command(action, action_type)
        time.sleep(sleep_dt)

    print("[Replay] Finished")
    print("=" * 70)


# ==========================================================
# Example main
# ==========================================================

if __name__ == "__main__":

    robot = PiperInterface()

    hdf5_path = "/home/zmy/Project/robot_tools/data/traj_1768215712.h5"

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