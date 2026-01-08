#!/usr/bin/env python3
import cv2
import numpy as np
from collections import deque 
import imageio
import matplotlib.pyplot as plt
import pyrealsense2 as rs
from multiprocessing import Process, Pipe, Queue, Event
import time
from dataclasses import dataclass
import multiprocessing
multiprocessing.set_start_method('fork')

np.printoptions(3, suppress=True)

def get_realsense_id():
    ctx = rs.context()
    devices = ctx.query_devices()
    devices = [device.get_info(rs.camera_info.serial_number) for device in devices]
    devices.sort() # Make sure the order is correct
    print("Found {} devices: {}".format(len(devices), devices))
    return devices

def grid_sample_pcd(point_cloud, grid_size=0.005):
    """
    A simple grid sampling function for point clouds.

    Parameters:
    - point_cloud: A NumPy array of shape (N, 3) or (N, 6), where N is the number of points.
                   The first 3 columns represent the coordinates (x, y, z).
                   The next 3 columns (if present) can represent additional attributes like color or normals.
    - grid_size: Size of the grid for sampling.

    Returns:
    - A NumPy array of sampled points with the same shape as the input but with fewer rows.
    """
    coords = point_cloud[:, :3]  # Extract coordinates
    scaled_coords = coords / grid_size
    grid_coords = np.floor(scaled_coords).astype(int)
    
    # Create unique grid keys
    keys = grid_coords[:, 0] + grid_coords[:, 1] * 10000 + grid_coords[:, 2] * 100000000
    
    # Select unique points based on grid keys
    _, indices = np.unique(keys, return_index=True)
    
    # Return sampled points
    return point_cloud[indices]

def init_given_realsense(
    device,
    enable_rgb=True,
    enable_depth=False,
    enable_point_cloud=False,
    sync_mode=0,
):
    # use `rs-enumerate-devices` to check available resolutions
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(device)
    print("Initializing camera {}".format(device))

    if enable_depth:
        # Depth         1024x768      @ 30Hz     Z16
        # Depth         640x480       @ 30Hz     Z16
        # Depth         320x240       @ 30Hz     Z16
        # h, w = 768, 1024
        h, w = 480, 640
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        # h, w = 540, 960
        h, w = 480, 640
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        # print(f"-------{device}-------")
        depth_sensor = device.query_sensors()[0]

        # Set the inter-camera sync mode
        # Use 1 for master, 2 for slave, 0 for default (no sync)
        depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
        
        # set min distance
        depth_sensor.set_option(rs.option.min_distance, 0.05)
        
        # get depth scale
        depth_scale = profile.get_device().first_depth_sensor().get_depth_scale()
        align = rs.align(rs.stream.color)
        
        depth_profile = profile.get_stream(rs.stream.depth)
        intrinsics = depth_profile.as_video_stream_profile().get_intrinsics()
        camera_info = CameraInfo(intrinsics.width, intrinsics.height, intrinsics.fx, intrinsics.fy, intrinsics.ppx, intrinsics.ppy)
        
        print("camera {} init.".format(device))
        return pipeline, align, depth_scale, camera_info
    else:
        print("camera {} init.".format(device))
        return pipeline, None, None, None
    

@dataclass
class SingleCameraConfig:
    serial_number: str
    height: int
    width: int
    use_rgb: bool
    use_depth: bool
    use_pointcloud: bool
    use_grid_sampling: bool
    num_points: int
    sync_mode: int
    z_far: float
    z_near: float

@dataclass
class MultiCameraConfig:
    front_camera: SingleCameraConfig
    right_camera: SingleCameraConfig
    left_camera: SingleCameraConfig

class CameraInfo:
    """ Camera intrisics for point cloud creation. """
    def __init__(self, width, height, fx, fy, cx, cy, scale = 1) :
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

class SingleVisionProcess(Process):
    def __init__(self, 
                 queue: Queue,
                 cfg: SingleCameraConfig
    ) -> None:
        super(SingleVisionProcess, self).__init__()
        self.queue = queue
        self.device = cfg.serial_number

        self.enable_rgb = cfg.use_rgb
        self.enable_depth = cfg.use_depth
        self.enable_pointcloud = cfg.use_pointcloud
        self.sync_mode = cfg.sync_mode
            
        self.use_grid_sampling = cfg.use_grid_sampling

  
        self.resize = True
        # self.height, self.width = 512, 512
        self.height, self.width = cfg.height, cfg.width
        
        # point cloud params
        self.z_far = cfg.z_far
        self.z_near = cfg.z_near
        self.num_points = cfg.num_points
   
    def get_vision(self):
        frame = self.pipeline.wait_for_frames()
        timestamp = frame.get_timestamp()
        if self.enable_depth:
            aligned_frames = self.align.process(frame)
            # Get aligned frames
            color_frame = aligned_frames.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())

            depth_frame = aligned_frames.get_depth_frame()
            depth_frame = np.asanyarray(depth_frame.get_data())
            
            clip_lower =  0.01
            clip_high = 1.0
            depth_frame = depth_frame.astype(np.float32)
            depth_frame *= self.depth_scale
            depth_frame[depth_frame < clip_lower] = clip_lower
            depth_frame[depth_frame > clip_high] = clip_high
            
            if self.enable_pointcloud:
                # Nx6
                point_cloud_frame = self.create_colored_point_cloud(color_frame, depth_frame, 
                            far=self.z_far, near=self.z_near, num_points=self.num_points)
            else:
                point_cloud_frame = None
        else:
            color_frame = frame.get_color_frame()
            color_frame = np.asanyarray(color_frame.get_data())
            depth_frame = None
            point_cloud_frame = None

        # print("color:", color_frame.shape)
        # print("depth:", depth_frame.shape)
        
        if self.resize:
            if self.enable_rgb:
                color_frame = cv2.resize(color_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
            if self.enable_depth:
                depth_frame = cv2.resize(depth_frame, (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        return color_frame, depth_frame, point_cloud_frame, timestamp


    def run(self):
        self.pipeline, self.align, self.depth_scale, self.camera_info = init_given_realsense(self.device, 
                    enable_rgb=self.enable_rgb, enable_depth=self.enable_depth,
                    enable_point_cloud=self.enable_pointcloud,
                    sync_mode=self.sync_mode)

        debug = False
        while True:
            color_frame, depth_frame, point_cloud_frame = self.get_vision()
            self.queue.put([color_frame, depth_frame, point_cloud_frame])
            # time.sleep(0.05)

    def terminate(self) -> None:
        # self.pipeline.stop()
        return super().terminate()

   
    def create_colored_point_cloud(self, color, depth, far=1.0, near=0.1, num_points=10000):
        assert(depth.shape[0] == color.shape[0] and depth.shape[1] == color.shape[1])
    
        # Create meshgrid for pixel coordinates
        xmap = np.arange(color.shape[1])
        ymap = np.arange(color.shape[0])
        xmap, ymap = np.meshgrid(xmap, ymap)

        # Calculate 3D coordinates
        points_z = depth / self.camera_info.scale
        points_x = (xmap - self.camera_info.cx) * points_z / self.camera_info.fx
        points_y = (ymap - self.camera_info.cy) * points_z / self.camera_info.fy
        cloud = np.stack([points_x, points_y, points_z], axis=-1)
        cloud = cloud.reshape([-1, 3])
        
        # Clip points based on depth
        mask = (cloud[:, 2] < far) & (cloud[:, 2] > near)
        cloud = cloud[mask]
        color = color.reshape([-1, 3])
        color = color[mask]


        colored_cloud = np.hstack([cloud, color.astype(np.float32)])
        if self.use_grid_sampling:
            colored_cloud = grid_sample_pcd(colored_cloud, grid_size=0.005)
        
        if num_points > colored_cloud.shape[0]:
            num_pad = num_points - colored_cloud.shape[0]
            pad_points = np.zeros((num_pad, 6))
            colored_cloud = np.concatenate([colored_cloud, pad_points], axis=0)
        else: 
            # Randomly sample points
            selected_idx = np.random.choice(colored_cloud.shape[0], num_points, replace=True)
            colored_cloud = colored_cloud[selected_idx]
        
        # shuffle
        np.random.shuffle(colored_cloud)
        return colored_cloud
    

class MultiRealSense(object):
    def __init__(self, 
                 camera_cfg: MultiCameraConfig):

        # self.devices = get_realsense_id()
    
        # sync_mode: Use 1 for master, 2 for slave, 0 for default (no sync)
        if camera_cfg.front_camera is not None:
            self.front_queue = Queue(maxsize=10)
            self.front_process = SingleVisionProcess(self.front_queue, camera_cfg.front_camera)
            self.front_process.start()
            print("front camera start.")

        if camera_cfg.right_camera is not None:    
            self.right_queue = Queue(maxsize=10)
            self.right_process = SingleVisionProcess(self.right_queue, camera_cfg.right_camera)
            self.right_process.start()
            print("right camera start.")

        if camera_cfg.left_camera is not None:    
            self.left_queue = Queue(maxsize=10)
            self.left_process = SingleVisionProcess(self.left_queue, camera_cfg.left_camera)
            self.left_process.start()
            print("left camera start.")


        self.use_front_cam = True if camera_cfg.front_camera is not None else False
        self.use_right_cam = True if camera_cfg.right_camera is not None else False
        self.use_left_cam = True if camera_cfg.left_camera is not None else False
        
        
    def __call__(self):  
        cam_dict = {}
        if self.use_front_cam:  
            front_color, front_depth, front_point_cloud, timestamp = self.front_queue.get()
            cam_dict.update({'front_color': front_color, 'front_depth': front_depth, 
                             'front_point_cloud':front_point_cloud, 'front_timestamp': timestamp})
      
        if self.use_right_cam: 
            right_color, right_depth, right_point_cloud, right_timestamp = self.right_queue.get()
            cam_dict.update({'right_color': right_color, 'right_depth': right_depth, 
                             'right_point_cloud':right_point_cloud, "right_timestamp": right_timestamp})

        if self.use_left_cam:
            left_color, left_depth, left_point_cloud, left_timestamp = self.left_queue.get()
            cam_dict.update({'left_color': left_color, 'left_depth': left_depth, 
                             'left_point_cloud':left_point_cloud, "left_timestamp": left_timestamp})
        return cam_dict

    def finalize(self):
        if self.use_front_cam:
            self.front_process.terminate()
        if self.use_right_cam:
            self.right_process.terminate()
        if self.use_left_cam:
            self.left_process.terminate()

    def __del__(self):
        self.finalize()

if __name__ == "__main__":
    front_camera_cfg = SingleCameraConfig("", 480, 640, True, True, True, True, 1024, 0, 1.0, 0.01)
    right_camera_cfg = SingleCameraConfig("", 480, 640, True, True, True, True, 1024, 0, 1.0, 0.01)
    left_camera_cfg = SingleCameraConfig("", 480, 640, True, True, True, True, 1024, 0, 1.0, 0.01)

    multi_camera_cfg = MultiCameraConfig(front_camera_cfg, right_camera_cfg, left_camera_cfg)

    camera = MultiRealSense(multi_camera_cfg)
    