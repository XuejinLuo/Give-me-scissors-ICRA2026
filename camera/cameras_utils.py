#
# Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#

import os
import numpy as np
import pyrealsense2 as rs
import yaml

# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from collections import defaultdict
from camera.realsense_device_manager import DeviceManager, post_process_depth_frame
from camera.calibration_kabsch import PoseEstimation
from camera.measurement_task import visualise_images, calculate_pointcloud, transform_point_cloud
from camera.helper_functions import convert_depth_frame_to_pointcloud, get_clipped_pointcloud


class State:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.paused = False

state = State()

def get_config(config_path=None):
    if config_path is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_file_dir, 'configs/config.yaml')
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def setup_realsense_manager(global_config):
	# Define some constants
	resolution_width = 640 # pixels
	resolution_height = 480 # pixels
	frame_rate = 30  # fps

	dispose_frames_for_stablisation = 30  # frames

	chessboard_width = 8 # squares
	chessboard_height = 11 	# squares
	square_size = 0.015 # meters

	# Enable the streams from all the intel realsense devices
	rs_config = rs.config()
	rs_config.enable_stream(rs.stream.depth, resolution_width, resolution_height, rs.format.z16, frame_rate)
	rs_config.enable_stream(rs.stream.infrared, 1, resolution_width, resolution_height, rs.format.y8, frame_rate)
	rs_config.enable_stream(rs.stream.color, resolution_width, resolution_height, rs.format.bgr8, frame_rate)

	# Use the device manager class to enable the devices and get the frames
	device_manager = DeviceManager(rs.context(), rs_config)
	device_manager.enable_all_devices()
	# Allow some frames for the auto-exposure controller to stablise
	for frame in range(dispose_frames_for_stablisation):
		frames = device_manager.poll_frames()

	assert( len(device_manager._available_devices) > 0 )
	camera_num = len(device_manager._available_devices)
	print("Number of available devices: ", camera_num)
     
	calibration_file = 'output/cameras_calibration_data.npy'
	"""
	1: Calibration
	Calibrate all the available devices to the world co-ordinates.
	For this purpose, a chessboard printout for use with opencv based calibration process is needed.

	"""
	# Get the intrinsics of the realsense device
	intrinsics_devices = device_manager.get_device_intrinsics(frames)

	# # Set the chessboard parameters for calibration
	# chessboard_params = [chessboard_height, chessboard_width, square_size]

	# # Estimate the pose of the chessboard in the world coordinate using the Kabsch Method
	# calibrated_device_count = 0
	# pose_mats_dict = {}
	# Camera1baseCamera2 = np.eye(4)
	# if os.path.exists(calibration_file):
	# 	transformation_devices = np.load(calibration_file, allow_pickle=True).item()
	# 	print("Loaded calibration data from file.")
	# 	for device_info in device_manager._available_devices:
	# 		device = device_info[0]
	# 		print("Device: ", device)
	# 		pose_mats_dict[device] = transformation_devices[device].pose_mat
	# 	Camera1baseCamera2 = np.dot(np.linalg.inv(transformation_devices['244622073846'].pose_mat), transformation_devices['337122071679'].pose_mat)
	# else:
	# 	while calibrated_device_count < len(device_manager._available_devices):
	# 		frames = device_manager.poll_frames()
	# 		pose_estimator = PoseEstimation(frames, intrinsics_devices, chessboard_params)
	# 		transformation_result_kabsch = pose_estimator.perform_pose_estimation()
	# 		object_point = pose_estimator.get_chessboard_corners_in3d()
	# 		calibrated_device_count = 0
			
	# 		for device_info in device_manager._available_devices:
	# 			device = device_info[0]
	# 			if not transformation_result_kabsch[device][0]:
	# 				print("Place the chessboard on the plane where the object needs to be detected..")
	# 				print("calibrated_device_count:\n", calibrated_device_count)
	# 			else:
	# 				calibrated_device_count += 1

	# 	transformation_devices = {}
	# 	for device_info in device_manager._available_devices:
	# 		device = device_info[0]
	# 		transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
	# 		print("Device: ", device)
	# 		pose_mats_dict[device] = transformation_devices[device].pose_mat

	# 	np.save(calibration_file, transformation_devices)
	# 	print("Calibration data saved to file.")

	# if len(pose_mats_dict) >= 2:
	# 	Camera1baseCamera2 = np.dot(np.linalg.inv(pose_mats_dict['244622073846']), pose_mats_dict['337122071679'])
	# 	# Camera1baseCamera2 = np.dot(np.linalg.inv(pose_mats[1]), pose_mats[0])
	# 	print("Camera1 based Camera2 matrix: \n{}".format(Camera1baseCamera2)) # camera 337122071679 base on 244622073846
	# else:
	# 	Camera1baseCamera2 = Camera1baseCamera2
	# 	print("Not enough pose matrices to compute the final transformation.")
	# print("Calibration completed... \n")
     
	transformation_devices = {}
	for device_info in device_manager._available_devices:
		device = device_info[0]
		transformation_devices[device] = np.eye(4)

	"""
	2: Capturing and display
	Capturing data from multiple RealSense devices
	The information from Phase 1 will be used here

	"""

	# Enable the emitter of the devices
	device_manager.enable_emitter(True)

	# Load the JSON settings file in order to enable High Accuracy preset for the realsense
	# device_manager.load_settings_json("test/MultiCamera/HighResHighAccuracyPreset.json")

	# Get the extrinsics of the device to be used later
	extrinsics_devices = device_manager.get_depth_to_color_extrinsics(frames)

	# Get the calibration info as a dictionary to help with display of the measurements onto the color image instead of infra red image
	calibration_info_devices = defaultdict(list)
	for calibration_info in (transformation_devices, intrinsics_devices, extrinsics_devices):
		for key, value in calibration_info.items():
			calibration_info_devices[key].append(value)

	roi_x_min = global_config['camera']['roi_x_min']
	roi_x_max = global_config['camera']['roi_x_max']
	roi_y_min = global_config['camera']['roi_y_min']
	roi_y_max = global_config['camera']['roi_y_max']
	num_devices = len(device_manager._available_devices)
	realsense_manager = {
        "device_manager": device_manager,
        "num_devices": num_devices,
        "calibration_info_devices": calibration_info_devices,
        "roi_x_min": roi_x_min,
        "roi_x_max": roi_x_max,
        "roi_y_min": roi_y_min,
        "roi_y_max": roi_y_max,
        # "Camera1baseCamera2": Camera1baseCamera2,
    }

	return realsense_manager



def get_next_depth_and_color(realsense_manager):
    frames_devices = realsense_manager["device_manager"].poll_frames()
    depth_images = []
    color_images = []
    for idx, (device_info, frame) in enumerate(frames_devices.items()):
        device = device_info[0] #serial number
        color_image = np.asarray(frame[rs.stream.color].get_data())
        roi_color = color_image[realsense_manager["roi_y_min"]:realsense_manager["roi_y_max"], 
                                realsense_manager["roi_x_min"]:realsense_manager["roi_x_max"]]
        filtered_depth_frame = post_process_depth_frame(frame[rs.stream.depth], temporal_smooth_alpha=0.1, temporal_smooth_delta=80)	
        depth_image = np.asarray(filtered_depth_frame.get_data())
        depth_images.append(depth_image)
        color_images.append(roi_color)

    return depth_images, color_images


def get_intrinsics_matrix(camera_intrinsics):
    intrinsic_matrix = np.array(
        [
            [camera_intrinsics.fx, 0, camera_intrinsics.ppx],
            [0, camera_intrinsics.fy, camera_intrinsics.ppy],
            [0, 0, 1],
        ]
    )
    return intrinsic_matrix


def get_default_pose(serial, global_config):
    T_offset_camera_first = global_config['camera']['camera337_base_robot']
    T_offset_camera_second = global_config['camera']['camera244_base_robot']
    T_offset_camera_third = global_config['camera']['camera231_base_robot']
    camera_offset_matrix = np.eye(4)
    if serial == '337122071679':
        camera_offset_matrix = T_offset_camera_first
    elif serial == '244622073846':
        camera_offset_matrix = T_offset_camera_second
    elif serial == '231122073844':
        camera_offset_matrix = T_offset_camera_third
    return np.array(camera_offset_matrix)

def stop_realsense_manager(realsense_manager):
    realsense_manager["device_manager"].disable_streams()


import torch


class CamerasDataloader:
    def __init__(self, max_steps=1e6):
        self.max_steps = int(max_steps)
        self.step = 0
        self.global_config = get_config(config_path="/home/luo/ReKep/configs/config.yaml")
        self.realsense_manager = setup_realsense_manager(self.global_config)
        align_to = rs.stream.color
        self.align = rs.align(align_to)

    def __len__(self):
        return self.max_steps

    def __iter__(self):
        if self.step > 0:
            raise IndexError("CamerasDataloader can only be iterated once")
        self.step = 0
        return self

    def __next__(self):
        if self.step >= self.max_steps:
            raise StopIteration

        cameras_data = {}
        frames_devices = self.realsense_manager["device_manager"].wait_for_frames()
        for idx, (serial, device) in enumerate(self.realsense_manager["device_manager"]._enabled_devices.items()):
            dev_info = (serial, device.product_line)
            depth_sensor = device.pipeline_profile.get_device().first_depth_sensor()
            depth_scale = depth_sensor.get_depth_scale()
            frameset = frames_devices[dev_info]
            aligned_frames = self.align.process(frameset)
            aligned_depth_frame = (
                aligned_frames.get_depth_frame()
            )
            color_frame = aligned_frames.get_color_frame()
            rgb_np = np.asanyarray(color_frame.get_data())
            roi_color = rgb_np[self.realsense_manager["roi_y_min"]:self.realsense_manager["roi_y_max"], 
                                    self.realsense_manager["roi_x_min"]:self.realsense_manager["roi_x_max"]]
            rgb = torch.from_numpy(rgb_np).permute((2, 0, 1))
            rgba = torch.cat([rgb, torch.ones_like(rgb[0:1, :, :]) * 255], dim=0)
            
            filtered_depth_frame = post_process_depth_frame(aligned_depth_frame, temporal_smooth_alpha=0.1, temporal_smooth_delta=80)	
            depth_np = np.asanyarray( filtered_depth_frame.get_data())
            depth_np = depth_np.astype(np.float32) * depth_scale
            depth = torch.from_numpy(depth_np).float()
            depth[depth > 1.5] = 0
            
            pose_np = get_default_pose(serial, self.global_config)
            pose = torch.from_numpy(pose_np).float()

            intrinsics_np = get_intrinsics_matrix(self.realsense_manager["calibration_info_devices"][serial][1][rs.stream.color])
            intrinsics = torch.from_numpy(intrinsics_np).float()

            cameras_data[f"camera_{idx}"] = {
                "serial": serial,
                "rgb_np": rgb_np,
                "rgba": rgba,
                "depth_np": np.asanyarray(filtered_depth_frame.get_data()),
                "depth": depth,
                "pose": pose,
                "intrinsics": intrinsics
            }

        return cameras_data
    
    def stop(self):
        stop_realsense_manager(self.realsense_manager)
        self.step = self.max_steps


# Test loop
if __name__ == "__main__":
    import cv2
    try:
        cameras = CamerasDataloader()
        for frame_num, cameras_data in enumerate(cameras):
            print(f"Frame {frame_num} / {len(cameras)}")
            for camera_idx, camera_info in cameras_data.items():
                serial = camera_info["serial"]
                pose = camera_info["pose"]
                print(f"Camera {serial} pose: {pose}")

            print("\n")
    finally:
        cameras.stop()
