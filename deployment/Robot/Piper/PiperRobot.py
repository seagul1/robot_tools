from typing import Dict, List

from piper_sdk import C_PiperInterface_V2
import numpy as np
import time
 

def enable_fun(piper: C_PiperInterface_V2, timeout: float = 5.0):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    start_time = time.time()
    elapsed_time_flag = False
    while not enable_flag:
        elapsed_time = time.time() - start_time
        print("--------------------")
        enable_status = piper.GetArmEnableStatus()
        enable_flag = all(enable_status)
        print("使能状态:",enable_flag)
        piper.EnableArm(7)
        piper.GripperCtrl(0,1000,0x01, 0)
        print("--------------------")
        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if(elapsed_time_flag):
        print("程序自动使能超时,退出程序")
        exit(0)

def eef_conversion_p2r(eef_pose: np.ndarray):
    """
    convert policy output(meter and radian) to piper input(0.001mm and 0.001degree)
    """
    eef_pose = np.asarray(eef_pose, dtype=np.float64).copy()
    if len(eef_pose) != 6:
        raise ValueError("The length of eef_pose should be 6.")
    # 将前三位从米转换为0.001毫米单位表示
    eef_pose[:3] = eef_pose[:3] * 1_000_000
    # 将后三位从弧度转换为单位0.001度的角度表示
    eef_pose[3:6] = np.rad2deg(eef_pose[3:6]) * 1_000

    return eef_pose.astype(np.float64)

def eef_conversion_r2p(eef_pose: np.ndarray):
    """
    Convert piper data(0.001mm and 0.001degree) to data suited policy(meter and radian)
    """
    eef_pose = np.asarray(eef_pose, dtype=np.float64).copy()
    # 将前三位从0.001毫米转换为米
    eef_pose[:3] = eef_pose[:3] / 1_000_000
    # 将后三位从0.001角度表示转换为弧度表示
    eef_pose[3:6] = np.deg2rad(eef_pose[3:6] / 1_000)

    return eef_pose.astype(np.float64)

def joint_conversion_p2r(joints: np.ndarray):
    """
    convert policy output(radian) to piper input(0.001degree)
    """
    joints = np.asarray(joints, dtype=np.float64).copy()
    # 将弧度转换为角度
    joints = np.rad2deg(joints) * 1000

    return joints.astype(np.float64)

def joint_conversion_r2p(joints):
    """
    Convert piper robot data(0.001degree) to data suited policy(radian)
    """
    joints = np.asarray(joints, dtype=np.float64).copy()
    # 将角度转换为弧度
    joints = np.deg2rad(joints / 1000)

    return joints.astype(np.float64)

def gripper_conversion_p2r(gripper):
    """
    convert policy output(0-1) to piper input(0 - 70_000)
    """
    # 将0-1转换为0-70_000
    gripper = gripper * 70_000

    return gripper

def gripper_conversion_r2p(gripper):
    """
    Convert piper gripper data(0 - 70_000) to data suited policy(0-1)
    """
    # 将0-70_000转换为0-1
    gripper = gripper / 70_000

    return float(gripper)

class PiperInterface:
    
    def __init__(self, can_name: str = "can0", auto_enable: bool = True, enable_timeout: float = 5.0) -> None:
        # 创建 C_PiperInterface 实例
        self.robot: C_PiperInterface_V2 = C_PiperInterface_V2(can_name=can_name)

        # 连接端口
        self.robot.ConnectPort()
        if auto_enable:
            self.robot.EnableArm(7)
            # 启用机械臂并检测使能状态
            enable_fun(piper=self.robot, timeout=enable_timeout)

    def get_robot(self):
        return self.robot

    def get_arm_status(self):
        """
        get arm status feedback
        """
        return self.robot.GetArmStatus()

    def get_arm_mode(self):
        """
        get current arm mode control feedback
        """
        return self.robot.GetArmModeCtrl()

    def get_can_fps(self) -> int:
        """
        get CAN fps
        """
        return self.robot.GetCanFps()

    def get_arm_enable_status(self) -> List[bool]:
        """
        get enable status for all joints
        """
        return self.robot.GetArmEnableStatus()

    def get_high_speed_info(self):
        """
        get high speed feedback message for all motors
        """
        return self.robot.GetArmHighSpdInfoMsgs()

    def get_low_speed_info(self):
        """
        get low speed feedback message for all motors
        """
        return self.robot.GetArmLowSpdInfoMsgs()

    def get_joint_ctrl_feedback(self):
        """
        get joint control message feedback
        """
        return self.robot.GetArmJointCtrl()

    def get_gripper_ctrl_feedback(self):
        """
        get gripper control message feedback
        """
        return self.robot.GetArmGripperCtrl()
    
    def get_gripper_width(self):
        """
        get gripper width
        """
        gripper_state = self.robot.GetArmGripperMsgs()
        gripper_width: int = gripper_state.gripper_state.grippers_angle
        gripper_width = gripper_conversion_r2p(gripper_width)
        # avoid noisy printing in tight loops; callers may log if needed
        return gripper_width

    def get_joint_positions(self):
        """
        get current joint data(radian format)
        """
        joint_feedback = self.robot.GetArmJointMsgs()
        joint_positions = joint_feedback.joint_state
        joint_timestamp = joint_feedback.time_stamp
        joint_positions = np.asarray([joint_positions.joint_1, joint_positions.joint_2, joint_positions.joint_3,
                                      joint_positions.joint_4, joint_positions.joint_5, joint_positions.joint_6])
        joint_positions = joint_conversion_r2p(joint_positions)
        return joint_positions, joint_timestamp
    
    def get_ee_pose(self):  # 返回末端执行器和夹爪信息
        # 获取末端执行器的位姿和夹爪状态
        eef_fb = self.robot.GetArmEndPoseMsgs()
        # gripper_fb = self.robot.GetArmGripperMsgs()

        eef_state = np.array([eef_fb.end_pose.X_axis, eef_fb.end_pose.Y_axis, eef_fb.end_pose.Z_axis,
                                eef_fb.end_pose.RX_axis, eef_fb.end_pose.RY_axis, eef_fb.end_pose.RZ_axis], dtype=np.float64)

        # gripper_angle = gripper_fb.gripper_state.grippers_angle

        # eef_state.append(gripper_angle)
        eef_state = eef_conversion_r2p(eef_state)

        return eef_state

    def get_fk(self, mode: str = "feedback"):
        """
        get forward kinematics result
        """
        return self.robot.GetFK(mode=mode)

    def get_robot_state(self):
        """
        get robot state
        """
        # robot_state = self.get_joint_positions()
        eef_state = self.get_ee_pose()
        gripper_state = self.get_gripper_width()
        robot_state = np.concatenate((eef_state, [gripper_state]))
        return robot_state

    def get_joint_velocities(self):
        """
        get current joint velocities
        """
        high_spd = self.robot.GetArmHighSpdInfoMsgs()
        motor_speeds = np.array(
            [
                high_spd.motor_1.motor_speed,
                high_spd.motor_2.motor_speed,
                high_spd.motor_3.motor_speed,
                high_spd.motor_4.motor_speed,
                high_spd.motor_5.motor_speed,
                high_spd.motor_6.motor_speed,
            ],
            dtype=np.float64,
        )
        return motor_speeds / 1000.0

    def get_joint_torques(self):
        """
        get current joint torques (raw value in N*m)
        """
        high_spd = self.robot.GetArmHighSpdInfoMsgs()
        motor_effort = np.array(
            [
                high_spd.motor_1.effort,
                high_spd.motor_2.effort,
                high_spd.motor_3.effort,
                high_spd.motor_4.effort,
                high_spd.motor_5.effort,
                high_spd.motor_6.effort,
            ],
            dtype=np.float64,
        )
        return motor_effort / 1000.0

    def get_joint_currents(self):
        """
        get current joint motor currents (A)
        """
        high_spd = self.robot.GetArmHighSpdInfoMsgs()
        motor_current = np.array(
            [
                high_spd.motor_1.current,
                high_spd.motor_2.current,
                high_spd.motor_3.current,
                high_spd.motor_4.current,
                high_spd.motor_5.current,
                high_spd.motor_6.current,
            ],
            dtype=np.float64,
        )
        return motor_current / 1000.0

    def get_joint_motor_positions(self):
        """
        get motor position from high speed feedback (rad)
        """
        high_spd = self.robot.GetArmHighSpdInfoMsgs()
        motor_pos = np.array(
            [
                high_spd.motor_1.pos,
                high_spd.motor_2.pos,
                high_spd.motor_3.pos,
                high_spd.motor_4.pos,
                high_spd.motor_5.pos,
                high_spd.motor_6.pos,
            ],
            dtype=np.float64,
        )
        return motor_pos / 1000.0

    def get_driver_temperatures(self):
        """
        get driver temperatures (C)
        """
        low_spd = self.robot.GetArmLowSpdInfoMsgs()
        driver_temps = np.array(
            [
                low_spd.motor_1.foc_temp,
                low_spd.motor_2.foc_temp,
                low_spd.motor_3.foc_temp,
                low_spd.motor_4.foc_temp,
                low_spd.motor_5.foc_temp,
                low_spd.motor_6.foc_temp,
            ],
            dtype=np.float64,
        )
        return driver_temps

    def get_motor_temperatures(self):
        """
        get motor temperatures (C)
        """
        low_spd = self.robot.GetArmLowSpdInfoMsgs()
        motor_temps = np.array(
            [
                low_spd.motor_1.motor_temp,
                low_spd.motor_2.motor_temp,
                low_spd.motor_3.motor_temp,
                low_spd.motor_4.motor_temp,
                low_spd.motor_5.motor_temp,
                low_spd.motor_6.motor_temp,
            ],
            dtype=np.float64,
        )
        return motor_temps

    def get_bus_currents(self):
        """
        get driver bus currents (A)
        """
        low_spd = self.robot.GetArmLowSpdInfoMsgs()
        bus_currents = np.array(
            [
                low_spd.motor_1.bus_current,
                low_spd.motor_2.bus_current,
                low_spd.motor_3.bus_current,
                low_spd.motor_4.bus_current,
                low_spd.motor_5.bus_current,
                low_spd.motor_6.bus_current,
            ],
            dtype=np.float64,
        )
        return bus_currents / 1000.0

    def get_driver_voltages(self):
        """
        get driver voltages (V)
        """
        low_spd = self.robot.GetArmLowSpdInfoMsgs()
        voltages = np.array(
            [
                low_spd.motor_1.vol,
                low_spd.motor_2.vol,
                low_spd.motor_3.vol,
                low_spd.motor_4.vol,
                low_spd.motor_5.vol,
                low_spd.motor_6.vol,
            ],
            dtype=np.float64,
        )
        return voltages / 10.0

    def get_driver_status(self):
        """
        get driver status flags for each motor
        """
        low_spd = self.robot.GetArmLowSpdInfoMsgs()
        return [
            low_spd.motor_1.foc_status,
            low_spd.motor_2.foc_status,
            low_spd.motor_3.foc_status,
            low_spd.motor_4.foc_status,
            low_spd.motor_5.foc_status,
            low_spd.motor_6.foc_status,
        ]

    def get_gripper_state(self) -> Dict[str, float]:
        """
        get gripper feedback (width and effort)
        """
        gripper_state = self.robot.GetArmGripperMsgs()
        width = gripper_state.gripper_state.grippers_angle / 1000.0
        effort = gripper_state.gripper_state.grippers_effort / 1000.0
        return {
            "width_mm": width,
            "effort_nm": effort,
            "foc_status": gripper_state.gripper_state.foc_status,
        }

    def get_all_states(self) -> Dict[str, object]:
        """
        collect all arm feedback states from piper sdk
        """
        # passive reads
        states = {
            "arm_status": self.get_arm_status(),
            "arm_mode": self.get_arm_mode(),
            "can_fps": self.get_can_fps(),
            "enable_status": self.get_arm_enable_status(),
            "end_pose": self.robot.GetArmEndPoseMsgs(),
            "joint_state": self.robot.GetArmJointMsgs(),
            "gripper_state": self.robot.GetArmGripperMsgs(),
            "high_speed_info": self.robot.GetArmHighSpdInfoMsgs(),
            "low_speed_info": self.robot.GetArmLowSpdInfoMsgs(),
            "motor_states": self.robot.GetMotorStates(),
            "driver_states": self.robot.GetDriverStates(),
            "joint_ctrl_feedback": self.get_joint_ctrl_feedback(),
            "gripper_ctrl_feedback": self.get_gripper_ctrl_feedback(),
            "gripper_ctrl": self.robot.GetArmGripperCtrl(),
            "joint_ctrl": self.robot.GetArmJointCtrl(),
            "ctrl_code_151": self.robot.GetArmCtrlCode151(),
            "mode_ctrl": self.robot.GetArmModeCtrl(),
            "arm_ctrl_151": self.robot.GetArmCtrlCode151(),
            "arm_mode_ctrl": self.robot.GetArmModeCtrl(),
            "high_speed_info_all": self.robot.GetArmHighSpdInfoMsgs(),
            "low_speed_info_all": self.robot.GetArmLowSpdInfoMsgs(),
        }

        # response-type reads (may be empty until requested)
        try:
            states.update({
                "current_motor_angle_limit_max_vel": self.robot.GetCurrentMotorAngleLimitMaxVel(),
                "current_end_vel_acc_param": self.robot.GetCurrentEndVelAndAccParam(),
                "crash_protection_level": self.robot.GetCrashProtectionLevelFeedback(),
                "gripper_teaching_param": self.robot.GetGripperTeachingPendantParamFeedback(),
                "current_motor_max_acc_limit": self.robot.GetCurrentMotorMaxAccLimit(),
                "all_motor_max_acc_limit": self.robot.GetAllMotorMaxAccLimit(),
                "all_motor_angle_limit_max_spd": self.robot.GetAllMotorAngleLimitMaxSpd(),
                "firmware_version_raw": self.robot.GetPiperFirmwareVersion(),
                "resp_instruction": self.robot.GetRespInstruction(),
                "instruction_response": self.robot.GetRespInstruction(),
            })
        except Exception:
            # Some SDK getters may raise if not yet available; ignore here
            pass

        # meta / identification
        try:
            states.update({
                "can_name": self.robot.GetCanName(),
                "can_bus": self.robot.GetCanBus(),
                "sdk_version": self.robot.GetCurrentSDKVersion(),
                "protocol_version": self.robot.GetCurrentProtocolVersion(),
            })
        except Exception:
            pass

        return states

    def request_all_params(self):
        """主动请求那些需要应答的参数（电机限制、最大加速度、固件信息、夹爪/示教器参数等）。

        SDK 对于部分参数需要先发送查询指令，随后通过对应 Get* 方法读取应答。
        这里统一发送这些查询以便随后从 `get_all_states` 中读取到值。
        """
        # 查询所有电机最大角速度/角度限制
        try:
            self.robot.SearchAllMotorMaxAngleSpd()
        except Exception:
            pass
        # 查询所有电机最大加速度限制
        try:
            self.robot.SearchAllMotorMaxAccLimit()
        except Exception:
            pass
        # 查询固件版本（会把固件数据拼接到内部 buffer）
        try:
            self.robot.SearchPiperFirmwareVersion()
        except Exception:
            pass
        # 使用 ArmParamEnquiryAndConfig 请求多种参数（0x01 0x02 0x04）
        try:
            # 0x01 -> 当前末端速度/加速度参数
            self.robot.ArmParamEnquiryAndConfig(param_enquiry=0x01)
            # 0x02 -> 碰撞防护等级
            self.robot.ArmParamEnquiryAndConfig(param_enquiry=0x02)
            # 0x04 -> 夹爪/示教器参数
            self.robot.ArmParamEnquiryAndConfig(param_enquiry=0x04)
        except Exception:
            pass

    def move_to_eef_positions(self, pose: np.ndarray):
        """
        Control Robot in eef position mode
        Args:
            pose: np.ndarray, shape(7,), [x, y, z, rx, ry, rz, gripper_width]
        """

        eef_positions = pose[:6]
        gripper_width = pose[6] #夹爪角度，单位0.001度

        eef_positions = eef_conversion_p2r(eef_positions)
        gripper_width = gripper_conversion_p2r(gripper_width)

        X = round(eef_positions[0])
        Y = round(eef_positions[1])
        Z = round(eef_positions[2])
        RX = round(eef_positions[3])
        RY = round(eef_positions[4])
        RZ = round(eef_positions[5])
        gripper_width = round(gripper_width)

        # 设置机械臂进入适当模式，使用位置速度模式
        self.robot.MotionCtrl_2(0x01, 0x00, 30, 0x00)  # 位置速度模式
        # 控制末端执行器移动到指定位置
        self.robot.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        # 控制机械臂夹爪，调整为需要的状态  
        self.robot.GripperCtrl(abs(gripper_width), 1000, 0x01, 0)

    def move_to_joint_positions(self, joint_positions: np.ndarray):
        """
        Control Robot in joint space mode
        Args:
            joint_positions: np.ndarray, shape(6,), [j1, j2, j3, j4, j5, j6]
        """
        if len(joint_positions) == 6:
            joint_positions = joint_conversion_p2r(joint_positions)
            gripper = None
            j1 = round(joint_positions[0])
            j2 = round(joint_positions[1])
            j3 = round(joint_positions[2])
            j4 = round(joint_positions[3])
            j5 = round(joint_positions[4])
            j6 = round(joint_positions[5])
            # 设置机械臂进入适当模式，使用位置速度模式

        elif len(joint_positions) == 7:
            joints = joint_positions[:6]
            gripper = joint_positions[6]
            joints = joint_conversion_p2r(joints)
            gripper = gripper_conversion_p2r(gripper)

            j1 = round(joints[0])
            j2 = round(joints[1])
            j3 = round(joints[2])
            j4 = round(joints[3])
            j5 = round(joints[4])
            j6 = round(joints[5])

            gripper = round(gripper)
        else:
            raise ValueError("The length of joint_positions should be 6 or 7.")
            
        self.robot.MotionCtrl_2(0x01, 0x01, 30, 0x00)
        self.robot.JointCtrl(j1, j2, j3, j4, j5, j6)
        if gripper is not None:
            self.robot.GripperCtrl(abs(gripper), 1000, 0x01, 0)
        self.robot.MotionCtrl_2(0x01, 0x01, 30, 0x00)

    def update_command(self, action, action_type: str = "eef",):
        """
        Execute Robot action.
        Args:
            action: np.ndarray, shape(7,), 
            action_type: str, "eef" or "joint"
        """
        assert action_type in ["eef", "joint"], "Invalid action type"
        
        if action_type == "eef":
            self.move_to_eef_positions(action)
        elif action_type == "joint":
            self.move_to_joint_positions(action)
        else:
            print("Invalid action type")




if __name__ == "__main__":
    robot = PiperInterface()
    cur_joint, cur_joint_ts = robot.get_joint_positions()
    print("init joint positions:", cur_joint)

    init_eef = robot.get_ee_pose()
    print("init eef positions:", init_eef)

    curr_gripper = robot.get_gripper_width()
    print(curr_gripper)
    robot.robot.MotionCtrl_2(0x01, 0x01, 30, 0x00)
    robot.robot.GripperCtrl(int(0.08 * 1000 *1000), 1000, 0x01, 0)

    position = [
                55.0, \
                0.0, \
                260.0, \
                0, \
                85.0, \
                0, \
                0]
    factor = 1000
    X = round(position[0]*factor)
    Y = round(position[1]*factor)
    Z = round(position[2]*factor)
    RX = round(position[3]*factor)
    RY = round(position[4]*factor)
    RZ = round(position[5]*factor)
    joint_6 = round(position[6]*factor)
    print(X,Y,Z,RX,RY,RZ)
        # piper.MotionCtrl_1()
    robot.robot.MotionCtrl_2(0x01, 0x00, 30, 0x00)
    robot.robot.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
    robot.robot.GripperCtrl(abs(joint_6), 1000, 0x01, 0)
    print(robot.robot.GetArmEndPoseMsgs())
    cur_eef = robot.get_ee_pose()
    print("current eef positions:", cur_eef)
    time.sleep(0.5)


    action = [0.045, 0.02, 0.260, 0, 1.48340769, 0, 0]
    robot.update_command(action, action_type="eef")
    time.sleep(0.5)
    cur_eef = robot.get_ee_pose()
    print("current eef positions:", cur_eef)

    
