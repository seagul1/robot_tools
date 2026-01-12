#!/usr/bin/env python3
import cv2
import numpy as np
from collections import deque 
import imageio
import pyrealsense2 as rs
from multiprocessing import Process, Pipe, Queue, Event
from queue import Empty
import time
import multiprocessing
if not multiprocessing.get_start_method(allow_none=True):
    multiprocessing.set_start_method('spawn')
# import torch
# import pytorch3d.ops as torch3d_ops

np.printoptions(3, suppress=True)

def get_realsense_id():
    ctx = rs.context()
    devices = ctx.query_devices()
    devices = [devices[i].get_info(rs.camera_info.serial_number) for i in range(len(devices))]
    devices.sort() # Make sure the order is correct
    print("Found {} devices: {}".format(len(devices), devices))
    return devices

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
        h, w = 480, 640
        config.enable_stream(rs.stream.depth, w, h, rs.format.z16, 30)
    if enable_rgb:
        h, w = 480, 640
        config.enable_stream(rs.stream.color, w, h, rs.format.rgb8, 30)

    config.resolve(pipeline)
    profile = pipeline.start(config)


    if enable_depth:

        # Get the depth sensor (or any other sensor you want to configure)
        device = profile.get_device()
        depth_sensor = device.query_sensors()[0]

        pipeline.stop()

        # Set the inter-camera sync mode if supported
        # Use 1 for master, 2 for slave, 0 for default (no sync)
        try:
            depth_sensor.set_option(rs.option.inter_cam_sync_mode, sync_mode)
        except Exception as e:
            print(f'Warning: could not set inter_cam_sync_mode for device {device}: {e}')
        
        # set min distance
        # depth_sensor.set_option(rs.option.min_distance, 0.05)

        profile = pipeline.start(config)
        
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

def farthest_point_sampling(points, num_points=1024):
    K = [num_points]
    points = torch.from_numpy(points).cuda()
    sampled_points, indices = torch3d_ops.sample_farthest_points(points=points.unsqueeze(0), K=K)
    sampled_points = sampled_points.squeeze(0).cpu().numpy()

    return sampled_points, indices


class CameraInfo():
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
    def __init__(self, device, queue,
                enable_rgb=True,
                enable_depth=False,
                enable_pointcloud=False,
                sync_mode=0,
                num_points=2048,
                z_far=1.0,
                z_near=0.1,
                use_grid_sampling=True,
                img_size=224,
                resize=False) -> None:
        super(SingleVisionProcess, self).__init__()
        self.queue = queue
        self.device = device

        self.enable_rgb = enable_rgb
        self.enable_depth = enable_depth
        self.enable_pointcloud = enable_pointcloud
        self.sync_mode = sync_mode
            
        self.use_grid_sampling = use_grid_sampling

  
        self.resize = resize
        if self.resize:
            self.height, self.width = img_size, img_size
        else:
            self.height, self.width = 480, 640
        
        # point cloud params
        self.z_far = z_far
        self.z_near = z_near
        self.num_points = num_points
   
    def get_vision(self):
        frame = self.pipeline.wait_for_frames()
        timestamp = frame.get_timestamp()
        # 时间戳单位转换为秒
        timestamp = timestamp / 1000.0


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
        # publish camera intrinsics back to parent via queue (sentinel message)
        try:
            self.queue.put(('__camera_info__', self.camera_info))
        except Exception:
            pass
        
        debug = False
        while True:
            color_frame, depth_frame, point_cloud_frame, timestamp = self.get_vision()
            self.queue.put([color_frame, depth_frame, point_cloud_frame, timestamp])
            time.sleep(0.05)

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
        # extrinsics_matrix = np.array([
        # [0.64018354, -0.51073456, 0.57385995, -0.41398303],
        # [-0.76566781, -0.48505943, 0.42245728, -0.31420408],
        # [0.06259264, -0.70983629, -0.70158008, 0.64089477],
        # [0, 0, 0, 1]
        # ])
        # WORK_SPACE = [
        # [-0.30, 0.40],
        # [-0.40, 0.66],
        # [0.01, 0.35]
        # ]
        #     # 坐标变换
        # colored_cloud_xyz = colored_cloud[..., :3]
        # homogeneous = np.hstack((colored_cloud_xyz, np.ones((colored_cloud_xyz.shape[0], 1))))
        # transformed = (extrinsics_matrix @ homogeneous.T).T[:, :3]
        # colored_cloud[..., :3] = transformed

        #     # 裁剪
        # mask = (
        #     (colored_cloud[..., 0] > WORK_SPACE[0][0]) & 
        #     (colored_cloud[..., 0] < WORK_SPACE[0][1]) &
        #     (colored_cloud[..., 1] > WORK_SPACE[1][0]) & 
        #     (colored_cloud[..., 1] < WORK_SPACE[1][1]) &
        #     (colored_cloud[..., 2] > WORK_SPACE[2][0]) & 
        #     (colored_cloud[..., 2] < WORK_SPACE[2][1])
        # )
        # colored_cloud = colored_cloud[mask]

        # # 采样
        # colored_cloud_xyz, indices = farthest_point_sampling(colored_cloud[..., :3])
        # indices = indices[0].cpu().numpy()  # 转换为 NumPy 数组
        # colored_cloud_rgb = colored_cloud[indices, 3:]  # 正确索引 NumPy 数组
        # colored_cloud = np.hstack((colored_cloud_xyz, colored_cloud_rgb))  # 确保全部是 NumPy 数组

        return colored_cloud


    
class MultiRealSense(object):
    def __init__(self, 
                 use_head1_cam=True, use_head2_cam=False, use_right_cam=False, use_left_cam=False,
                 head1_cam_idx=0, head2_cam_idx=1, right_cam_idx=2, left_cam_idx=3,
                 head_num_points=4096, right_num_points=1024, left_num_points=1024,
                 head_z_far=1.0, head_z_near=0.1,
                 right_z_far=0.5, right_z_near=0.01,
                 left_z_far=0.5, left_z_near=0.01,
                 use_grid_sampling=True,
                 resize=False,
                 img_size=384):

        self.devices = get_realsense_id()

        # Pre-initialize usage flags and process placeholders to ensure __del__ is safe
        n_devs = len(self.devices)
        self.use_head1_cam = use_head1_cam and (0 <= head1_cam_idx < n_devs)
        if use_head1_cam and not self.use_head1_cam:
            print(f'Warning: head1_cam_idx {head1_cam_idx} out of range (found {n_devs} devices). Disabling head1_cam.')
        self.use_head2_cam = use_head2_cam and (0 <= head2_cam_idx < n_devs)
        if use_head2_cam and not self.use_head2_cam:
            print(f'Warning: head2_cam_idx {head2_cam_idx} out of range (found {n_devs} devices). Disabling head2_cam.')
        self.use_right_cam = use_right_cam and (0 <= right_cam_idx < n_devs)
        if use_right_cam and not self.use_right_cam:
            print(f'Warning: right_cam_idx {right_cam_idx} out of range (found {n_devs} devices). Disabling right_cam.')
        self.use_left_cam = use_left_cam and (0 <= left_cam_idx < n_devs)
        if use_left_cam and not self.use_left_cam:
            print(f'Warning: left_cam_idx {left_cam_idx} out of range (found {n_devs} devices). Disabling left_cam.')

        # queues for each potential camera
        self.head1_queue = Queue(maxsize=3)
        self.head2_queue = Queue(maxsize=3)
        self.right_queue = Queue(maxsize=3)
        self.left_queue = Queue(maxsize=3)

        # storage for camera intrinsics reported from child processes
        self._camera_info = {}

      
        # 0: f1380328, 1: f1422212

        # sync_mode: Use 1 for master, 2 for slave, 0 for default (no sync)

        if self.use_head1_cam:
            self.head1_process = SingleVisionProcess(self.devices[head1_cam_idx], self.head1_queue,
                            enable_rgb=True, enable_depth=True, enable_pointcloud=False, sync_mode=1,
                            num_points=head_num_points, z_far=head_z_far, z_near=head_z_near, 
                            use_grid_sampling=use_grid_sampling,  resize=resize, img_size=img_size)
            
        if self.use_head2_cam:
            self.head2_process = SingleVisionProcess(self.devices[head2_cam_idx], self.head2_queue,
                    enable_rgb=True, enable_depth=True, enable_pointcloud=False, sync_mode=2,
                        num_points=head_num_points, z_far=head_z_far, z_near=head_z_near, 
                        use_grid_sampling=use_grid_sampling, resize=resize, img_size=img_size)
            
        if self.use_right_cam:
            self.right_process = SingleVisionProcess(self.devices[right_cam_idx], self.right_queue,
                    enable_rgb=True, enable_depth=True, enable_pointcloud=False, sync_mode=2,
                        num_points=right_num_points, z_far=right_z_far, z_near=right_z_near, 
                        use_grid_sampling=use_grid_sampling, resize=resize,  img_size=img_size)
            
        if self.use_left_cam:
            self.left_process = SingleVisionProcess(self.devices[left_cam_idx], self.left_queue,
                    enable_rgb=True, enable_depth=True, enable_pointcloud=False, sync_mode=2,
                        num_points=left_num_points, z_far=left_z_far, z_near=left_z_near, 
                        use_grid_sampling=use_grid_sampling, resize=resize, img_size=img_size)


        if getattr(self, 'head1_process', None) is not None:
            self.head1_process.start()
            print("head_1 camera start.")

        if getattr(self, 'head2_process', None) is not None:
            self.head2_process.start()
            print("head_2 camera start.")

        if getattr(self, 'right_process', None) is not None:
            self.right_process.start()
            print("right camera start.")

        if getattr(self, 'left_process', None) is not None:
            self.left_process.start()
            print("left camera start.")

        

        # usage flags were set earlier based on available devices
        
    @property
    def camera_info(self):
        # return any camera_info reported by child processes (populated from queue sentinel messages)
        info_dict = {}
        if self.use_head1_cam and 'head1' in self._camera_info:
            info_dict['head1_camera_info'] = self._camera_info.get('head1')
        if self.use_head2_cam and 'head2' in self._camera_info:
            info_dict['head2_camera_info'] = self._camera_info.get('head2')
        if self.use_right_cam and 'right' in self._camera_info:
            info_dict['right_camera_info'] = self._camera_info.get('right')
        if self.use_left_cam and 'left' in self._camera_info:
            info_dict['left_camera_info'] = self._camera_info.get('left')
        return info_dict
    
    @property
    def enabled_cams(self):
        cams = []
        if self.use_head1_cam:
            cams.append('head1')
        if self.use_head2_cam:
            cams.append('head2')
        if self.use_right_cam:
            cams.append('right')
        if self.use_left_cam:
            cams.append('left')
        return cams
        
    def __call__(self):  
        cam_dict = {}
        # helper to read one frame, draining any camera_info sentinel messages first
        def _read_frame_from_queue(q, name):
            frame_color = frame_depth = frame_pcd = frame_ts = None
            try:
                while True:
                    item = q.get(timeout=0.5)
                    # sentinel: ('__camera_info__', camera_info)
                    if isinstance(item, (list, tuple)) and len(item) == 2 and item[0] == '__camera_info__':
                        self._camera_info[name] = item[1]
                        continue
                    if isinstance(item, (list, tuple)) and len(item) == 4:
                        frame_color, frame_depth, frame_pcd, frame_ts = item
                    break
            except Empty:
                pass
            return frame_color, frame_depth, frame_pcd, frame_ts

        if self.use_head1_cam:
            c, d, p, ts = _read_frame_from_queue(self.head1_queue, 'head1')
            cam_dict.update({'head1_color': c, 'head1_depth': d, 'head1_point_cloud': p, 'head1_timestamp': ts})

        if self.use_head2_cam:
            c, d, p, ts = _read_frame_from_queue(self.head2_queue, 'head2')
            cam_dict.update({'head2_color': c, 'head2_depth': d, 'head2_point_cloud': p, 'head2_timestamp': ts})

        if self.use_right_cam:
            c, d, p, ts = _read_frame_from_queue(self.right_queue, 'right')
            cam_dict.update({'right_color': c, 'right_depth': d, 'right_point_cloud': p, 'right_timestamp': ts})

        if self.use_left_cam:
            c, d, p, ts = _read_frame_from_queue(self.left_queue, 'left')
            cam_dict.update({'left_color': c, 'left_depth': d, 'left_point_cloud': p, 'left_timestamp': ts})

        return cam_dict

    def finalize(self):
        if self.use_head1_cam:
            self.head1_process.terminate()
        if self.use_head2_cam:
            self.head2_process.terminate()
        if self.use_right_cam:
            self.right_process.terminate()
        if self.use_left_cam:
            self.left_process.terminate()   


    def __del__(self):
        self.finalize()
        

if __name__ == "__main__":
    cam = MultiRealSense(use_right_cam=False, front_num_points=2000, use_grid_sampling=True)
    import matplotlib.pyplot as plt
    while True:
        out = cam()
        print(out.keys())
    
        imageio.imwrite(f'color.png', out['color'])
        print(out['color'].shape)
        # import visualizer
        # visualizer.visualize_pointcloud(out['point_cloud'])
        # np.save('point_cloud.npy', out['point_cloud'])
        cam.finalize()