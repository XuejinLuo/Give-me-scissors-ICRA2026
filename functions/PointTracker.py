import os
import cv2
import torch
import argparse
import rclpy # type: ignore
import imageio.v3 as iio
import numpy as np
import pyrealsense2 as rs
from cotracker.utils.visualizer import Visualizer
from cotracker.predictor import CoTrackerOnlinePredictor
from collections import deque
from datetime import datetime
from functions.utils import *
from functions.transform_utils import *

import sys
from graspnetAPI import GraspGroup
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(current_dir, 'anygrasp'))
sys.path.append(project_root)
from gsnet import AnyGrasp # type: ignore
import open3d as o3d

DEFAULT_DEVICE = (
    "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
)

class CameraInfo:
    def __init__(self, width, height, fx, fy, cx, cy, scale):
        self.width = width
        self.height = height
        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.scale = scale

def create_point_cloud_from_depth_image(depth, camera, organized=True):
    assert(depth.shape[0] == camera.height and depth.shape[1] == camera.width)
    xmap = np.arange(camera.width)
    ymap = np.arange(camera.height)
    xmap, ymap = np.meshgrid(xmap, ymap)
    points_z = depth / camera.scale
    points_x = (xmap - camera.cx) * points_z / camera.fx
    points_y = (ymap - camera.cy) * points_z / camera.fy
    points = np.stack([points_x, points_y, points_z], axis=-1)
    if not organized:
        points = points.reshape([-1, 3])
    return points

def capture_data(colors, depths):
    # set camera intrinsics
    width, height = depths.shape[1], depths.shape[0]
    fx, fy = 606.512, 606.219
    cx, cy = 321.978, 250.519
    scale = 1000.0
    camera = CameraInfo(width, height, fx, fy, cx, cy, scale)
    # get point cloud
    points = create_point_cloud_from_depth_image(depths, camera)
    mask = (points[:,:,2] > 0) & (points[:,:,2] < 1.5)
    points = points[mask]
    colors = colors[mask]
    return points, colors

class point_tracker:
    def __init__(self, candidate_pixels, save_path, global_config):
        #? --------------------CoTracker---------------------
        self.running_flag = True
        self.candidate_pixels = candidate_pixels
        self.save_path = save_path
        self.camera_base_robot = np.array(global_config['camera']['camera337_base_robot'])
        self.camera_calib_offset = np.array(global_config['camera']['camera_calib_offset'])
        self.grasp_depth = global_config['main']['grasp_depth']
        self.roi_x_min = global_config['camera']['roi_x_min']
        self.roi_x_max = global_config['camera']['roi_x_max']
        self.roi_y_min = global_config['camera']['roi_y_min']
        self.roi_y_max = global_config['camera']['roi_y_max']
        self.selected_pixels = self.candidate_pixels[:10]
        self.model = torch.hub.load("/home/luo/.cache/torch/hub/facebookresearch_co-tracker_main", "cotracker3_online", trust_repo=True, source='local').cuda()
        self.model = self.model.to(DEFAULT_DEVICE)

        self.max_window_size = self.model.step * 2
        self.window_frames = deque(maxlen=self.model.step * 2)
        self.all_frames = []
        self.keypoints_pixels = []
        self.keypoints_coordinates = deque(maxlen=len(self.selected_pixels))
        self.keypoints_grasp_transform_now = deque(maxlen=len(self.selected_pixels))
        self.keypoints_grasp_transform = deque(maxlen=len(self.selected_pixels))

        self.pipeline = self.setup_camera()

        self.queries = torch.tensor([
                [0., pixel[1], pixel[0]] for pixel in self.selected_pixels  # track from initial positions
            ])
        if torch.cuda.is_available():
            self.queries = self.queries.cuda()

        #? --------------------AnyGrasp----------------------
        parser = argparse.ArgumentParser()
        parser.add_argument('--checkpoint_path', default='/home/luo/ICRA/anygrasp/log/checkpoint_detection.tar', help='Model checkpoint path')
        parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
        parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
        parser.add_argument('--top_down_grasp', action='store_true', help='Output top-down grasps.')
        parser.add_argument('--debug', action='store_true', help='Enable debug mode')
        cfgs = parser.parse_args()
        cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
        cfgs.top_down_grasp = True
        cfgs.debug = True
        self.cfgs_debug = cfgs.debug

        self.anygrasp = AnyGrasp(cfgs)
        self.anygrasp.load_net()

    def setup_camera(self):
        pipeline = rs.pipeline()
        config = rs.config()
        pipeline_wrapper = rs.pipeline_wrapper(pipeline)
        pipeline_profile = config.resolve(pipeline_wrapper)
        camera = pipeline_profile.get_device()
        found_rgb = any(s.get_info(rs.camera_info.name) == 'RGB Camera' for s in camera.sensors)
        if not found_rgb:
            print("The demo requires Depth camera with Color sensor")
            exit(0)
        config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
        pipeline.start(config)
        self.pc = rs.pointcloud()
        self.points = rs.points()
        self.align_to = rs.stream.color 
        self.align = rs.align(self.align_to) 
        return pipeline

    def get_aligned_images(self):
        frames = self.pipeline.wait_for_frames()
        aligned_frames = self.align.process(frames)
        aligned_depth_frame = aligned_frames.get_depth_frame()
        aligned_color_frame = aligned_frames.get_color_frame()
        img_color = np.asanyarray(aligned_color_frame.get_data())
        img_depth = np.asanyarray(aligned_depth_frame.get_data())
        depth_mapped_image = cv2.applyColorMap(cv2.convertScaleAbs(img_depth, alpha=0.03), cv2.COLORMAP_JET)
        return img_color, img_depth, depth_mapped_image, aligned_color_frame, aligned_depth_frame

    def _process_step(self, window_frames, is_first_step, queries):
        video_chunk = (
            torch.tensor(
                np.stack(window_frames[-self.model.step * 2 :]), device=DEFAULT_DEVICE
            )
            .float()
            .permute(0, 3, 1, 2)[None]
        )  # (1, T, 3, H, W)
        return self.model(
            video_chunk,
            is_first_step=is_first_step,
            queries=queries[None]
        )
    
    def _get_3d_camera_coordinates(self, aligned_color_frame, aligned_depth_frame):
        self.pc.map_to(aligned_color_frame)
        self.points = self.pc.calculate(aligned_depth_frame)
        vtx = np.asanyarray(self.points.get_vertices())
        vtx = np.reshape(vtx, (480, 640, -1))
        return vtx

    

    def process_frames(self):
        counter = 0
        is_first_step = True
        theta = np.pi / 2 
        R_y = np.array([[np.cos(theta), 0, np.sin(theta), 0],
                        [0, 1, 0, 0],
                        [-np.sin(theta), 0, np.cos(theta), 0],
                        [0, 0, 0, 1]])
        R_z_180 = np.array([[-1, 0, 0],
                    [0, -1, 0],
                    [0, 0, 1]])
        print("Capturing...")
        print("Press 'q' to exit the capture.")
        vis = o3d.visualization.Visualizer()
        vis.create_window(height=self.roi_y_max - self.roi_y_min, width=self.roi_x_max - self.roi_x_min)
        xmin, xmax = -0.2, 0.2
        ymin, ymax = -0.2, 0.2
        zmin, zmax = 0.0, 1.0
        lims = [xmin, xmax, ymin, ymax, zmin, zmax]
        quaterion = np.array([1, 0, 0, 0])
        target_rotation_matrix = quat2mat(quaterion)
        target_z_axis = np.array([0, 0, -1])
        plt.ion()
        fig, ax = plt.subplots()
        def on_key(event):
            if event.key == 'q' or event.key == 'escape':
                plt.ioff()
                plt.close()
                self.running_flag = False
        fig.canvas.mpl_connect('key_press_event', on_key)
        while self.running_flag:
            img_color, img_depth, depth_mapped_image, aligned_color_frame, aligned_depth_frame = self.get_aligned_images()
            point_cloud = self._get_3d_camera_coordinates(aligned_color_frame, aligned_depth_frame)

            #? ----------------------CoTracker----------------------
            img_color2 = img_color.copy()
            self.all_frames.append(img_color2)
            # cv2.imshow("live", img_color2)
            ax.clear()
            ax.imshow(img_color2)
            keypoints_x = []
            keypoints_y = []
            for (x, y) in self.keypoints_pixels:
                if 0 <= x < 640 and 0 <= y < 480:
                    keypoints_x.append(x)
                    keypoints_y.append(y)
            ax.scatter(keypoints_x, keypoints_y, color='green', s=30, edgecolors='black', zorder=10)
            ax.axis('off')
            plt.draw()
            plt.pause(0.001)
            

            if counter % self.model.step == 0 and counter != 0:
                window_frames_list = list(self.window_frames)
                pred_tracks, pred_visibility = self._process_step(
                                window_frames_list,
                                is_first_step,
                                self.queries
                            )
                is_first_step = False
                if counter > self.model.step * 2:
                    tracks = pred_tracks[-1].cpu().numpy()
                    visibility = pred_visibility[-1].cpu().numpy()
                    T, N, _ = tracks.shape
                    self.keypoints_pixels.clear()
                    for object_index in range(N):
                        if visibility[T - 1, object_index]:
                            x, y = tracks[T - 1, object_index, 0], tracks[T - 1, object_index, 1]
                            self.keypoints_pixels.append((int(x), int(y)))

            for (x, y) in self.keypoints_pixels:
                if 0 <= x < 640 and 0 <= y < 480:
                    point = point_cloud[y, x]
                    point_coordinates = np.array([point['f0'], point['f1'], point['f2']], dtype=np.float32).flatten()
                    self.keypoints_coordinates.append(point_coordinates)
            # print("self.keypoints_coordinates: ", self.keypoints_coordinates)
            self.window_frames.append(img_color)

            #? ----------------------AnyGrasp----------------------
            if self.keypoints_pixels:        
                roi_color = img_color[self.roi_y_min:self.roi_y_max, self.roi_x_min:self.roi_x_max]
                colors = roi_color.astype(np.float32) / 255.0
                roi_point_cloud = point_cloud[self.roi_y_min:self.roi_y_max, self.roi_x_min:self.roi_x_max]
                points = np.zeros(((self.roi_x_max - self.roi_x_min) * (self.roi_y_max - self.roi_y_min), 3), dtype=np.float32)
                points[:, 0] = roi_point_cloud['f0'].flatten()
                points[:, 1] = roi_point_cloud['f1'].flatten()
                points[:, 2] = roi_point_cloud['f2'].flatten()
                points = points.reshape((self.roi_y_max - self.roi_y_min), (self.roi_x_max - self.roi_x_min), 3)
                mask = (points[:,:,2] > -1) & (points[:,:,2] < 2)
                points = points[mask]
                colors = colors[mask]
                filtered_points = points
                filtered_colors = colors
                gg, cloud = self.anygrasp.get_grasp(filtered_points, filtered_colors, lims=lims, apply_object_mask=True, dense_grasp=False, collision_detection=True)
                # print("gg", gg)
                if len(gg) == 0:
                    print('No Grasp detected after collision detection!')
                
                gg = gg.nms().sort_by_score()
                gg_pick = gg[0:20]
                high_score_translation = gg_pick.translations[0]
                high_score_rotation = gg_pick.rotation_matrices[0]
                high_score_transformation_matrix = np.eye(4)
                high_score_transformation_matrix[:3, :3] = high_score_rotation
                high_score_transformation_matrix[:3, 3] = high_score_translation
                high_score_transformation_matrix = np.dot(high_score_transformation_matrix, R_y) # rotate the grasp axis orientation to be along the z axis
                high_score_grasp_robot_matrix = np.dot(self.camera_base_robot, high_score_transformation_matrix)

                #- calcluate the point cloud corresponding to the keypoints pixels
                keypoints_pixels = self.keypoints_pixels
                keypoints_3d_pos = []
                for pixel in keypoints_pixels:
                    u, v = pixel
                    index = (v - self.roi_y_min) * (self.roi_x_max - self.roi_x_min) + (u - self.roi_x_min)
                    if 0 <= index < points.shape[0]:
                        point = points[index]
                        keypoints_3d_pos.append(point)
                keypoints_3d_pos = np.array(keypoints_3d_pos)
                if len(keypoints_3d_pos) < len(self.selected_pixels):
                    continue

                # -calcluate the closet point in the target grasp group
                closest_target_indices = []
                for keypoint in keypoints_3d_pos:
                    distances = np.linalg.norm(gg.translations - keypoint, axis=1)
                    closest_index = np.argmin(distances)
                    closest_target_indices.append(closest_index)
                
                keypoints_grasp_translation = gg.translations[closest_target_indices]
                keypoints_grasp_rotation = gg.rotation_matrices[closest_target_indices]
                for i in range(len(keypoints_grasp_translation)):
                    translation = keypoints_grasp_translation[i]
                    rotation = keypoints_grasp_rotation[i]
                    transformation_matrix = np.eye(4)
                    transformation_matrix[:3, :3] = rotation
                    transformation_matrix[:3, 3] = translation
                    transformation_matrix = np.dot(transformation_matrix, R_y) # rotate the grasp axis orientation to be along the z axis
                    grasp_robot_matrix = np.dot(self.camera_base_robot, transformation_matrix)
                    #? Adjust the grasp_robot_matrix to be in the right position
                    grasp_robot_matrix[:3, 3] += self.camera_calib_offset
                    self.keypoints_grasp_transform_now.append(grasp_robot_matrix)
                
                # -calcluate the optimal matrix
                for i in range(len(self.keypoints_grasp_transform_now)):
                    current_rotation_matrix = self.keypoints_grasp_transform_now[i][:3, :3]
                    current_x_axis = current_rotation_matrix[:, 0]
                    if np.dot(current_x_axis, np.array([1, 0, 0])) < 0:
                        current_rotation_matrix = np.dot(current_rotation_matrix, R_z_180)
                        self.keypoints_grasp_transform_now[i][:3, :3] = current_rotation_matrix
                    if not self.keypoints_grasp_transform:
                        self.keypoints_grasp_transform = deque(self.keypoints_grasp_transform_now)
                        break
                    optimal_rotation_matrix = self.keypoints_grasp_transform[i][:3, :3]
                    # current_rotation_error = angle_between_rotmat(current_rotation_matrix, target_rotation_matrix)
                    # optimal_rotation_error = angle_between_rotmat(optimal_rotation_matrix, target_rotation_matrix)
                    # if np.abs(optimal_rotation_error) > np.abs(current_rotation_error):
                    #     self.keypoints_grasp_transform[i] = self.keypoints_grasp_transform_now[i]
                    #     self.keypoints_grasp_transform[i][:3, 3] += self.keypoints_grasp_transform[i][:3, :3] @ np.array([0, 0, self.grasp_depth])
                    current_z_axis = current_rotation_matrix[:, 2]
                    optimal_z_axis = optimal_rotation_matrix[:, 2]
                    current_z_angle = np.arccos(np.dot(current_z_axis, target_z_axis) / (np.linalg.norm(current_z_axis) * np.linalg.norm(target_z_axis)))
                    optimal_z_angle = np.arccos(np.dot(optimal_z_axis, target_z_axis) / (np.linalg.norm(optimal_z_axis) * np.linalg.norm(target_z_axis)))
                    if current_z_angle < optimal_z_angle:
                        self.keypoints_grasp_transform[i] = self.keypoints_grasp_transform_now[i]
                        # self.keypoints_grasp_transform[i][:3, 3] += self.keypoints_grasp_transform[i][:3, :3] @ np.array([0, 0, self.grasp_depth])
                
                #! keypoints_grasp_transform is output
                # for idx, optimal_transformation in enumerate(self.keypoints_grasp_transform):
                #     print(f"Optimal Transformation for {idx}:")
                #     print(optimal_transformation)

                if self.cfgs_debug:
                    trans_mat = np.array([[1,0,0,0],[0,-1,0,0],[0,0,-1,0],[0,0,0,1]])
                    cloud.transform(trans_mat)
                    grippers = gg.to_open3d_geometry_list()
                    for gripper in grippers:
                        gripper.transform(trans_mat)
                    # # first frame
                    # o3d.visualization.draw_geometries([*grippers, cloud])
                    # o3d.visualization.draw_geometries([grippers[0], cloud])

                    # multi frames
                    vis.add_geometry(cloud)
                    for gripper in grippers:
                        vis.add_geometry(gripper)
                    vis.poll_events()
                    vis.remove_geometry(cloud)
                    for gripper in grippers:
                        vis.remove_geometry(gripper)
            counter += 1
            #? ----------------------Exit----------------------
            
        print("Exiting capture...")
        window_frames_list = list(self.window_frames)
        pred_tracks, pred_visibility = self._process_step(
            window_frames_list[-(counter % self.model.step) - self.model.step - 1 :],
            is_first_step,
            self.queries
        )
        video = torch.tensor(np.stack(self.all_frames), device=DEFAULT_DEVICE).permute(
            0, 3, 1, 2
        )[None]
        vis = Visualizer(save_dir=self.save_path, pad_value=120, linewidth=3)
        vis.visualize(
            video, pred_tracks, pred_visibility
        )

        self.pipeline.stop()
        print("Over")

############################################################################################

if __name__ == "__main__":
    f_path = "/home/luo/ICRA/pictures/live_record"
    now = datetime.datetime.now()
    date_folder = now.strftime("%Y-%m-%d")
    date_folder_path = os.path.join(f_path, date_folder)
    os.makedirs(date_folder_path, exist_ok=True)
    time_folder = now.strftime("%H-%M-%S")
    save_path = os.path.join(date_folder_path, time_folder)
    os.makedirs(save_path, exist_ok=True)
    ICRA_program_dir = "/home/luo/ICRA/vlm_query/test"
    candidate_pixels_file = os.path.join(ICRA_program_dir, 'candidate_pixels.npy')
    candidate_pixels = np.load(candidate_pixels_file)
    global_config = get_config(config_path="/home/luo/ICRA/configs/config.yaml")
    point_tracker = point_tracker(candidate_pixels, save_path, global_config)
    point_tracker.process_frames()  
