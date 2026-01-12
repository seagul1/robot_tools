import numpy as np
import time
import math
import pathlib
import tqdm
import torch
import pdb

from Robot.Piper.PiperRobot import PiperInterface
from Camera.rs_capture import RSCapture
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
                 external_camera_sn: str = None,
                 wrist_camera_sn: str = None,
                 use_pointcloud: bool = False,
                 img_size:int = 224,

                 # Robot parameters
                 action_space: str = "eef",
                 # Other parameters

                 ) -> None:
        self.action_space = action_space


        self.robot = PiperInterface()   # 机械臂控制接口

        if external_camera_sn is not None:
            self.external_camera = RSCapture(external_camera_sn,
                                            dim=(640, 480),
                                            fps=30,
                                            depth=False,
                                            ) # 外部相机
        if wrist_camera_sn is not None:
            self.external_camera = RSCapture(wrist_camera_sn,
                                            dim=(224, 224),
                                            fps=30,
                                            depth=False,
                                            ) # 外部相机
        
        self.reset_joint = np.array([0, np.pi/6, -np.pi/6, 0, np.pi/6, 0, 1]) 
        self.curr_path_length = 0

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
        img_dict = {}
        if hasattr(self, "wrist_camera"):
            img_dict["wrist_image"] = self.wrist_camera.read()[1][:, :, ::-1]
        if hasattr(self, "external_camera"):
            img_dict["external_image"] = self.external_camera.read()[1][:, :, ::-1]

        return img_dict


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

if __name__ == "__main__":
    from transformers import AutoModelForVision2Seq, AutoProcessor
    from PIL import Image
    import torch
    torch.manual_seed(42)

    # pour
    roll_out_length_dict = {
        "pour": 300,
        "grasp": 1000,
        "wipe": 300,
    }
    # task = "wipe"
    task = "grasp"
    # task = "pour"
    roll_out_length = roll_out_length_dict[task]
    

    env = RobotEnv(external_camera_sn="001622071252")
    # env = RobotEnv(external_camera_sn="146222250351")   # D455
    obs_dict = env.reset()
    print(obs_dict["robot_state"])
    # image = Image.fromarray(obs_dict['image']["external_image"][1])
    # image.show()

    # action = [0, 0, 0.05, 0, 0, 0, 1]
    # obs1 = env.step(action)
    # print(obs1["robot_state"])
    # image1 = Image.fromarray(obs1['image']["external_image"][1])
    # image1.show()
    # time.sleep(1)

    # action1 = [0, 0, 0.05, 0, 0, 0, 1]
    # obs2 = env.step(action1)
    # print(obs2["robot_state"])
    # image2 = Image.fromarray(obs2['image']["external_image"][1])
    # image2.show()
    # time.sleep(1)

    # action2 = [0.05, 0, 0, 0, 0, 0, 0]
    # obs3 = env.step(action2)
    # print(obs3["robot_state"])
    # image3 = Image.fromarray(obs3['image']["external_image"][1])
    # image3.show()
    # time.sleep(1)

    # fetch policy model
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    vla = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", 
        attn_implementation="flash_attention_2",  # [Optional] Requires `flash_attn`
        torch_dtype=torch.bfloat16, 
        low_cpu_mem_usage=True, 
        trust_remote_code=True
    ).to("cuda:0")

    while env.curr_path_length < roll_out_length:
        
        with torch.no_grad():
            prompt = "In: What action should the robot take to {<grasp the bottle and place it into the box>}?\nOut:"
            image = Image.fromarray(obs_dict['image']["external_image"])
            # image.show()
            # pdb.set_trace()

            # Predict Action (7-DoF; un-normalize for BridgeData V2)
            inputs = processor(prompt, image).to("cuda:0", dtype=torch.bfloat16)

            action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
            # action[6] = 1 if action[6] < 0.5 else 0


            print("predict action:", action)

            # pdb.set_trace()
            obs_dict = env.step(action)
            print(f"step: {env.curr_path_length}")
            time.sleep(0.1)






