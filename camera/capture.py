"""
 _
| |     _   _   ___
| |    | | | | / _ \ 
| |___ | |_| || (_) |
|_____| \__,_| \___/

"""


# Import RealSense, OpenCV and NumPy
import pyrealsense2 as rs
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

# Import helper functions and classes written to wrap the RealSense, OpenCV and Kabsch Calibration usage
from collections import defaultdict
from camera.realsense_device_manager import DeviceManager, post_process_depth_frame
from camera.calibration_kabsch import PoseEstimation
from camera.measurement_task import visualise_images, calculate_pointcloud, transform_point_cloud
from camera.helper_functions import convert_depth_frame_to_pointcloud, get_clipped_pointcloud
from functions.utils import get_config

class State:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.paused = False

state = State()
def start_capture2(global_config, save_path):

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
	# pose_mats = []
	# if os.path.exists(calibration_file):
	# 	transformation_devices = np.load(calibration_file, allow_pickle=True).item()
	# 	print("Loaded calibration data from file.")
	# 	for device_info in device_manager._available_devices:
	# 		device = device_info[0]
	# 		pose_mats.append(transformation_devices[device].pose_mat)
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
	# 			else:
	# 				calibrated_device_count += 1

	# 	transformation_devices = {}
	# 	for device_info in device_manager._available_devices:
	# 		device = device_info[0]
	# 		transformation_devices[device] = transformation_result_kabsch[device][1].inverse()
	# 		pose_mats.append(transformation_devices[device].pose_mat)

	# 	np.save(calibration_file, transformation_devices)
	# 	print("Calibration data saved to file.")

	# if len(pose_mats) >= 2:
	# 	final_transformation = np.dot(np.linalg.inv(pose_mats[0]), pose_mats[1])
	# 	print("Camera2 based Camera1 matrix: \n{}".format(final_transformation)) # camera2 base camera1
	# else:
	# 	print("Not enough pose matrices to compute the final transformation.")
	# print("Calibration completed... \nPlace the box in the field of view of the devices...")

	transformation_devices = {}
	for device_info in device_manager._available_devices:
		device = device_info[0]
		transformation_devices[device] = np.eye(4)

	"""
	2: Capturing and display
	Capturing data from multiple RealSense devices
	The information from Phase 1 will be used here

	"""
	align_to = rs.stream.color
	align = rs.align(align_to)

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

	# Continue acquisition until terminated with Ctrl+C by the user
	roi_x_min = global_config['camera']['roi_x_min']
	roi_x_max = global_config['camera']['roi_x_max']
	roi_y_min = global_config['camera']['roi_y_min']
	roi_y_max = global_config['camera']['roi_y_max']
	roi_vertices = np.array([[roi_x_min, roi_y_min],
                         [roi_x_max, roi_y_min],
                         [roi_x_max, roi_y_max],
                         [roi_x_min, roi_y_max]])
	first_camera_image = []
	first_camera_roi_image = []
	height = 480
	width = 640
	center_x = width // 2
	center_y = height // 2
	index = center_y * width + center_x
	num_devices = len(device_manager._available_devices)
	plt.ion()
	fig, ax = plt.subplots(1, num_devices, figsize=(3 + 5 * num_devices, 5))
	def on_key(event):
		global saved_count, state
		if event.key == 'q' or event.key == 'escape':
			# 退出循环
			state.paused ^= True
			plt.ioff()
			plt.close()
	fig.canvas.mpl_connect('key_press_event', on_key)
	while not state.paused:
		point_cloud_cumulative = np.array([-1, -1, -1]).transpose()
		filtered_point_cloud = []
		frames_devices = device_manager.wait_for_frames()
		for idx, (serial, device) in enumerate(device_manager._enabled_devices.items()):
			dev_info = (serial, device.product_line)
			frameset = frames_devices[dev_info]
			aligned_frames = align.process(frameset)
			aligned_depth_frame = (
				aligned_frames.get_depth_frame()
			)
			color_frame = aligned_frames.get_color_frame()
			color_image = np.asanyarray(color_frame.get_data())
			roi_color = color_image[roi_y_min:roi_y_max, roi_x_min:roi_x_max]
			filtered_depth_frame = post_process_depth_frame(aligned_depth_frame, temporal_smooth_alpha=0.1, temporal_smooth_delta=80)	
			if num_devices == 1:
				ax.clear()  # 只有一个子图
				ax.imshow(color_image, aspect='auto')
				ax.add_patch(plt.Rectangle((roi_x_min, roi_y_min), roi_x_max - roi_x_min, roi_y_max - roi_y_min,
                                edgecolor='r', facecolor='none'))
				ax.axis('off')
			else:
				ax[idx].clear()  # 多个子图
				ax[idx].imshow(color_image, aspect='auto')
				ax[idx].add_patch(plt.Rectangle((roi_x_min, roi_y_min), roi_x_max - roi_x_min, roi_y_max - roi_y_min,
                                edgecolor='r', facecolor='none'))
				ax[idx].axis('off')
			point_cloud = convert_depth_frame_to_pointcloud( np.asarray( filtered_depth_frame.get_data()), calibration_info_devices[serial][1][rs.stream.color])
			point_cloud = np.asanyarray(point_cloud)
			point_cloud_cumulative = np.column_stack((point_cloud_cumulative, point_cloud))
			if serial == '231522073039':
				point_cloud_reshaped = point_cloud.reshape(3, 480, 640)
				first_camera_image = color_image
				first_camera_roi_image = roi_color
				filtered_point_cloud = point_cloud_reshaped[:, roi_y_min:roi_y_max, roi_x_min:roi_x_max]

			# print("intrinsics: ", calibration_info_devices[serial][1][rs.stream.color])
		
		point_cloud_cumulative = np.delete(point_cloud_cumulative, 0, 1)
		plt.pause(0.001)

		

	cv2.imwrite(os.path.join(save_path, "roi_color.png"), first_camera_roi_image)
	cv2.imwrite(os.path.join(save_path, "saved_image.png"), first_camera_image)
	device_manager.disable_streams()
	cv2.destroyAllWindows()
	return point_cloud_cumulative, filtered_point_cloud

if __name__ == "__main__":
	global_config = get_config(config_path="./configs/config.yaml")
	point_cloud_cumulative, point_cloud_first_camera = start_capture2(global_config,"output")
	print("Point cloud shape:", point_cloud_first_camera.shape)
	print(point_cloud_first_camera[:, 200, 200])
	point_cloud_cumulative_first_camera = point_cloud_cumulative[:, 0:307200]
	np.savetxt("output/point_cloud.txt", point_cloud_cumulative_first_camera.T, delimiter=' ', fmt='%.6f')

