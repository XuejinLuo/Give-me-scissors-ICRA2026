import time
import numpy as np
import cv2
import mediapipe as mp
import pyrealsense2 as rs
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from matplotlib.colors import Normalize
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.tasks.python.vision import HandLandmarkerOptions, HandLandmarkerResult
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
from camera.cameras_utils import CamerasDataloader

class State:
    def __init__(self, *args, **kwargs):
        self.WIN_NAME = 'RealSense'
        self.paused = False

class HandKeypointCapture:
    def __init__(self):
        self.hands_detector()#Call hand detection model
        self.setup_plot()
        self.rgb_dict = {}
        self.depth_dict = {}
        self.keypoint_pixel_positions = []
        self.camera_1_serial = "244622073846"
        self.camera_2_serial = "337122071679"
        self.camera_3_serial = "231522073039"
        self.previous_center = None
        self.hand_keypoint = None
        self.stable_time_threshold = 2 # senconds
        self.last_stable_time = 0
        self.camara_flag = False

    def setup_plot(self):
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        self.rgb_display = self.ax.imshow(np.zeros((480, 640, 3), dtype=np.uint8))
        self.ax.set_title('231522073039')
        self.ax.axis('off')
        plt.ion()
        plt.show()

    def draw_landmarks_on_image(self, rgb_image, detection_result, center_point=None, camera_intrinsics=None):#关键点关键线绘制
        hand_landmarks_list = detection_result.hand_landmarks
        handedness_list = detection_result.handedness
        annotated_image = np.copy(rgb_image)
        self.keypoint_pixel_positions = []
        for idx in range(len(hand_landmarks_list)):
            hand_landmarks = hand_landmarks_list[idx]
            self.keypoint_pixel_positions = self.get_landmark_pixels(rgb_image, hand_landmarks)
            handedness = handedness_list[idx]
 
            # Draw the hand landmarks.
            hand_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
            hand_landmarks_proto.landmark.extend([
                landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in hand_landmarks
            ])
            solutions.drawing_utils.draw_landmarks(
                annotated_image,
                hand_landmarks_proto,
                solutions.hands.HAND_CONNECTIONS,#关键点连线
                solutions.drawing_styles.get_default_hand_landmarks_style(),#关键点绘制形式
                solutions.drawing_styles.get_default_hand_connections_style())#连线绘制形式
            
        if center_point is not None and camera_intrinsics is not None:
            pixel_coords = self.convert_3d_to_pixel(center_point[0], camera_intrinsics)
            cv2.circle(annotated_image, pixel_coords, 5, (255, 0, 0), -1)

        return annotated_image
    
    def get_landmark_pixels(self, rgb_image, hand_landmarks):
        image_height, image_width, _ = rgb_image.shape
        keypoint_pixel_positions = []

        for landmark in hand_landmarks:
            pixel_x = int(landmark.x * image_width)
            pixel_y = int(landmark.y * image_height)
            keypoint_pixel_positions.append((pixel_x, pixel_y))

        return keypoint_pixel_positions
    
    def convert_pixels_to_pointcloud(self, pixel_positions, depth_image, camera_intrinsics):
        """
        Convert a list of pixel coordinates to their corresponding 3D point cloud coordinates.
        """
        
        points = []

        for (u, v) in pixel_positions:
            if 0 <= v < depth_image.shape[0] and 0 <= u < depth_image.shape[1]:
                # Get the depth value at the pixel location
                z = depth_image[v, u]

                # Calculate the 3D coordinates
                x = (u - camera_intrinsics.ppx) * z / camera_intrinsics.fx
                y = (v - camera_intrinsics.ppy) * z / camera_intrinsics.fy

                points.append((x, y, z))

        return np.array(points)
    
    def convert_3d_to_pixel(self, point, camera_intrinsics):
        x, y, z = point
        pixel_x = int((x * camera_intrinsics.fx / z) + camera_intrinsics.ppx)
        pixel_y = int((y * camera_intrinsics.fy / z) + camera_intrinsics.ppy)
        return (pixel_x, pixel_y)
    
    def cluster_point_cloud(self, point_cloud, n_clusters=1):
        """
        Perform KMeans clustering on the point cloud data.
        Returns the cluster center.
        """
        if len(point_cloud) == 0:
            return None
        point_cloud = point_cloud[np.any(point_cloud != 0, axis=1)]
        if len(point_cloud) == 0:
            return None
        kmeans = KMeans(n_clusters=n_clusters)
        kmeans.fit(point_cloud)
        
        return kmeans.cluster_centers_

    def keypoint_calculate(self, cluster_center):
        if self.previous_center is not None and cluster_center is not None:
            distance = np.linalg.norm(cluster_center - self.previous_center)

            if distance < 150:  # 定义稳定阈值
                if self.last_stable_time == 0:
                    self.last_stable_time = time.time()
                elif time.time() - self.last_stable_time > self.stable_time_threshold:
                    self.hand_keypoint = cluster_center
                    self.last_stable_time = 0  # 重置
                    self.camara_flag = False

            else:
                self.last_stable_time = 0  # 如果不稳定则重置
                self.previous_center = cluster_center  # 更新到新的中心点
        else:
            self.previous_center = cluster_center  # 初始化

    def hands_detector(self):#创建手部检测任务
        hand_options = HandLandmarkerOptions(#训练模型
            base_options=python.BaseOptions(model_asset_path='models/hand_landmarker.task'),#导入学习模型
            running_mode=vision.RunningMode.LIVE_STREAM,#任务运行模式 视频流模式
            result_callback=self.audio_monitor#结果监听器 用于返回值
        )
        self.hand_detector = vision.HandLandmarker.create_from_options(hand_options)#创建检测任务
        self.hand_result = None#初始化监听器返回值
        self.finger_index_to_angles = dict()#存储手部关键向量角度
        self.last_trigger_cmd_time = 0#记录触发时间

    def audio_monitor(self, result, output_image, timestamp_ms):#回调函数,用于处理检测结果。当检测到手部信息时,会将其存储在self.hand_result
        self.hand_result = result#检测到的手部信息
        # print(f'hand landmarker result: consume {self.get_current_time() - timestamp_ms} ms, {self.hand_result}')
    
    def get_current_time(self):#当前时间戳
        return int(time.time() * 1000)#当前时间转化为ms
    
    def run(self):#运行摄像头，接收视频流
        cameras = CamerasDataloader()
        self.cameras = cameras
        while True:
            if self.camara_flag == False:
                time.sleep(0.1)
                self.previous_center = np.array([0,0,0])
                self.last_stable_time = 0
            else:
                self.previous_center = np.array([0,0,0])
                self.keypoint_pixel_positions = []
                for frame_num, cameras_data in enumerate(cameras):
                    # print(f"Frame {frame_num} / {len(cameras)}")
                    if(frame_num < 50):
                        continue

                    if self.camara_flag == False:
                        break

                    start_time = time.time()
                    for camera_idx, camera_info in cameras_data.items():
                        serial = camera_info["serial"]
                        self.rgb_dict[serial] = camera_info["rgb_np"]
                        self.depth_dict[serial] = camera_info["depth_np"]

                        if serial == '231522073039':
                            img_color = self.rgb_dict[serial]
                            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)
                            
                            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data = img_color)
                            self.hand_detector.detect_async(mp_image, self.get_current_time())
                            point_cloud = self.convert_pixels_to_pointcloud(self.keypoint_pixel_positions, self.depth_dict[serial], cameras.realsense_manager["calibration_info_devices"][serial][1][rs.stream.color])
                            cluster_center = self.cluster_point_cloud(point_cloud, n_clusters=1)
                            self.keypoint_calculate(cluster_center)
                            # print("hand keypoints:", self.hand_keypoint)
                            if self.hand_result:
                                img_color = self.draw_landmarks_on_image(img_color, self.hand_result, self.hand_keypoint, cameras.realsense_manager["calibration_info_devices"][serial][1][rs.stream.color])
                            
                            self.rgb_display.set_data(img_color)
                    plt.pause(0.02)
                    end_time = time.time()
                    loop_time = end_time - start_time
                    # print(f"Loop time: {loop_time:.4f} seconds")

    