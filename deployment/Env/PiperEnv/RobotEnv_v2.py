import numpy as np
import time
import math
import pathlib
import tqdm
import torch
import pdb

from Robot.Piper.PiperRobot import PiperInterface
from Camera.realsense import SingleCameraConfig, MultiCameraConfig, MultiRealSense
from utils.transformation_util import *

import cv2
from collections import deque 
import imageio
import matplotlib.pyplot as plt
import pyrealsense2 as rs
# import gym

def time_ms():
    return time.time_ns() // 1_000_000




class RobotEnv:
    def __init__(self, 
                 # Camera parameters
                 camera_cfg: MultiCameraConfig,

                 # Robot parameters
                 action_space: str = "eef",
                 # Other parameters

                 ) -> None:
        self.action_space = action_space
        self.camera_config = camera_cfg


        self.robot = PiperInterface()   # 机械臂控制接口
        self.camera = MultiRealSense(self.camera_config)
        
        self.init_joint_state = np.array([0, np.pi/6, -np.pi/6, 0, np.pi/6, 0, 1]) 
        self.curr_path_length = 0

    def reset_joint(self, joints: np.ndarray):
        self.init_joint_state = joints

    def reset(self):
        self.curr_path_length = 0
        self.robot.update_command(self.reset_joint, "joint")
        time.sleep(2)
        print("Finished reset!!!")
        return self.get_observation()
        

    def step(self, action):
        current_state = self.get_state()
        curr_eef = current_state["ee_pose"]
        curr_gripper = current_state["gripper_width"]

        action[:6] = curr_eef + action[:6]
        action = np.array(action, dtype=np.float64)
        print(action)


        self.robot.update_command(action, self.action_space)
        time.sleep(0.1)
        self.curr_path_length += 1
        return self.get_observation()


    def get_image(self):
        return self.camera()


    def get_state(self):
        robot_state = {}
        robot_state["joint_positions"] = self.robot.get_joint_positions()
        robot_state["ee_pose"] = self.robot.get_ee_pose()
        robot_state["gripper_width"] = self.robot.get_gripper_width()
        return  robot_state
    
    def get_observation(self):
        obs_dict = {}

        # Robot State #
        robot_state = self.get_state()
        obs_dict["robot_state"] = robot_state

        # Camera Readings #
        camera_obs = self.get_image()
        obs_dict["image"] = camera_obs

        return obs_dict
    


class DualRobotEnv:
    def __init__(self) -> None:
        pass 
        
    def reset(self):
        raise NotImplementedError

    def step(self):
        raise NotImplementedError
    
    def get_image(self):
        raise NotImplementedError

    def get_state(self):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError
