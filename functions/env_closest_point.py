import cv2
import time
import torch
import threading
import numpy as np
import pyrealsense2 as rs
import matplotlib.pyplot as plt
import roboticstoolbox as rtb
import sys
from numba import njit, prange
from sklearn.cluster import DBSCAN
sys.path.append('/home/luo/ICRA')
from camera.cameras_utils import CamerasDataloader
from camera.helper_functions import *
from curobo.types.math import Pose
from functions.utils import *
from network.env_collision_bp import BP
from curobo.types.camera import CameraObservation
from curobo.wrap.model.robot_segmenter import RobotSegmenter
from curobo.types.robot import RobotConfig
from curobo.types.state import JointState
from curobo.cuda_robot_model.cuda_robot_model import CudaRobotModel
from curobo.util_file import get_robot_configs_path, get_world_configs_path, join_path, load_yaml
import rclpy # type: ignore
sys.path.append('/home/luo/ICRA/pyros/pyros')
from env_closest_point_interface import EnvClosestPointROS2Interface
with open("output/env_closest_point.txt", "w") as f:
    f.write("")

def load_model_weights(model, model_file_path):
    if os.path.exists(model_file_path):
        if os.path.isfile(model_file_path):
            model.load_state_dict(torch.load(model_file_path))
            print(f'Model loaded from {model_file_path}')
        else:
                print(f'No model file found at {model_file_path}')
    else:
        print(f'Checkpoint path {model_file_path} does not exist.')


@njit(cache=True, parallel=True)
def transform_points_parallel(points, transform_matrix):
    num_points = points.shape[1]
    transformed = np.zeros_like(points)
    for i in prange(num_points):
        point_homogeneous = np.array([points[0, i], points[1, i], points[2, i], 1.0])
        transformed_point = transform_matrix @ point_homogeneous
        transformed[:, i] = transformed_point[:3]
    
    return transformed

@njit(cache=True, parallel=True)
def filter_points_nonzero(points):
    num_points = points.shape[1]
    valid_points_mask = np.zeros(num_points, dtype=np.bool_)
    for i in range(num_points):
        for j in range(points.shape[0]):
            if points[j, i] != 0:
                valid_points_mask[i] = True
                break
    filtered_points = points[:, valid_points_mask]
    return filtered_points

@njit(cache=True, parallel=True)
def combined_boundary_mask(transformed_points):
    num_points = transformed_points.shape[1]
    combined_mask = np.zeros(num_points, dtype=np.bool_)
    for i in range(num_points):
        z = transformed_points[2, i]
        y = transformed_points[1, i]
        x = transformed_points[0, i]
        if (0.1 < z < 1.0) and (-0.5 < y < 0.5) and (0 < x < 1.0):
            combined_mask[i] = True
    return combined_mask

def get_robot_point_cloud(points, camera_base_robot):
    points = filter_points_nonzero(points)
    transformed_points = transform_points_parallel(points, np.array(camera_base_robot))
    combined_mask = combined_boundary_mask(transformed_points)
    transformed_points = transformed_points[:, combined_mask]
    return transformed_points # base robot

class TensorDeviceType:
    device: torch.device = torch.device("cuda", 0)
    dtype: torch.dtype = torch.float32
    collision_geometry_dtype: torch.dtype = torch.float32
    collision_gradient_dtype: torch.dtype = torch.float32
    collision_distance_dtype: torch.dtype = torch.float32

    @staticmethod
    def from_basic(device: str, dev_id: int):
        return TensorDeviceType(torch.device(device, dev_id))

    def to_device(self, data_tensor):
        if isinstance(data_tensor, torch.Tensor):
            return data_tensor.to(device=self.device, dtype=self.dtype)
        else:
            return torch.as_tensor(np.array(data_tensor), device=self.device, dtype=self.dtype)

    def to_int8_device(self, data_tensor):
        return data_tensor.to(device=self.device, dtype=torch.int8)

    def cpu(self):
        return TensorDeviceType(device=torch.device("cpu"), dtype=self.dtype)

    def as_torch_dict(self):
        return {"device": self.device, "dtype": self.dtype}

def calculate_capsules(transforms, radius=0.25):
    capsule_p1s = []
    capsule_p2s = []
    capsule_radii = []
    for i in range(len(transforms)-1):
        T1 = transforms[i]
        T2 = transforms[i + 1]
        p1 = T1.t
        p2 = T2.t
        length = np.linalg.norm(p2 - p1)
        if length < 1e-6:
            continue
        capsule_p1s.append(p1)
        capsule_p2s.append(p2)
        capsule_radii.append(radius)
    return np.array(capsule_p1s), np.array(capsule_p2s), np.array(capsule_radii)

@njit(cache=True, parallel=True)
def points_in_capsules(points, capsule_p1s, capsule_p2s, capsule_radii):
    """
    Numba-accelerated point and capsule containment relationship calculation
    """
    n_points = len(points)
    n_capsules = len(capsule_p1s)
    in_any_capsule = np.zeros(n_points, dtype=np.bool_)
    
    for i in prange(n_points):  # Process each point in parallel
        point = points[i]
        for j in range(n_capsules):
            p1 = capsule_p1s[j]
            p2 = capsule_p2s[j]
            radius = capsule_radii[j]
            
            direction = p2 - p1
            length_squared = np.sum(direction**2)
            length = np.sqrt(length_squared)
            if length < 1e-6:
                continue
            direction = direction / length
            ap = point - p1
            t = np.dot(ap, direction)
            t_clamped = max(0, min(length, t))
            closest_point = p1 + t_clamped * direction
            distance_squared = np.sum((point - closest_point)**2)
            distance = np.sqrt(distance_squared)
            if distance <= radius:
                in_any_capsule[i] = True
                break
    return in_any_capsule

def get_closest_point(env_collision_model, joints_data, point_cloud):
    if point_cloud.shape[0] == 0:
        return None, None
    joints_data_repeated = np.tile(joints_data, (point_cloud.shape[0], 1))
    combined_input = np.concatenate((joints_data_repeated, point_cloud * 1000), axis=1)
    input_tensor = torch.tensor(combined_input, dtype=torch.float32, device='cuda')
    output_tensor = env_collision_model.forward(input_tensor)
    min_distance, min_idx = torch.min(output_tensor, dim=0)
    min_distance = min_distance.item()
    closest_point = point_cloud[min_idx, :].flatten()
    return closest_point * 1000, min_distance

def _Execute(env_closest_point_interface):
    begin_time = time.time()
    target_period = 0.05
    env_collision_model = BP(input_dim=10, nodes_per_layer=[256, 256, 256, 256]).to("cuda")
    env_collision_model_file = "models/bp_env_collision_model_bp.pt"
    load_model_weights(env_collision_model, env_collision_model_file)
    cameras = CamerasDataloader()
    global_config = get_config(config_path="configs/config.yaml")
    camera_337122071679_base_robot1_np = np.array(global_config['camera']['camera337_base_robot'])
    camera_244622073846_base_robot1_np = np.array(global_config['camera']['camera244_base_robot'])
    T_offset_robot2 = np.loadtxt("models/dualarm_robot2_offset.txt")
    T_offset_robot2_inv = np.linalg.inv(T_offset_robot2)
    camera_337122071679_base_robot2_np = T_offset_robot2_inv @ camera_337122071679_base_robot1_np
    camera_244622073846_base_robot2_np = T_offset_robot2_inv @ camera_244622073846_base_robot1_np

    camera_337122071679_base_robot1 = torch.from_numpy(camera_337122071679_base_robot1_np).float()
    camera_244622073846_base_robot1 = torch.from_numpy(camera_244622073846_base_robot1_np).float()
    camera_337122071679_base_robot2 = torch.from_numpy(camera_337122071679_base_robot2_np).float()
    camera_244622073846_base_robot2 = torch.from_numpy(camera_244622073846_base_robot2_np).float()

    rgb_dict = {}
    depth_dict = {}
    Init_joints_robot1 = np.array([-0.778596, -0.017534, -0.030657, -2.5462, 0.010462, 2.62165, 0.0654274])
    Init_joints_robot2 = np.array([0.364773, 0.0850793, -0.125461, -2.46158, -0.0868246, 2.54863, 0.937512])
    tensor_args = TensorDeviceType()
    robot_file="franka.yml"
    robot_dict = load_yaml(join_path(get_robot_configs_path(), robot_file))
    robot_dict["robot_cfg"]["kinematics"]["load_link_names_with_mesh"] = True
    robot_dict["robot_cfg"]["kinematics"]["load_meshes"] = True
    robot_cfg = RobotConfig.from_dict(robot_dict["robot_cfg"])
    kin_model = CudaRobotModel(robot_cfg.kinematics)
    curobo_segmenter = RobotSegmenter.from_robot_file(
        robot_file, collision_sphere_buffer=0.02, distance_threshold=0.07, use_cuda_graph=True
    )
    panda = rtb.models.Panda()
    pose_dict = {
        '337122071679': {
            'robot1': Pose.from_matrix(camera_337122071679_base_robot1.to(device=tensor_args.device)),
            'robot2': Pose.from_matrix(camera_337122071679_base_robot2.to(device=tensor_args.device)),
        },
        '244622073846': {
            'robot1': Pose.from_matrix(camera_244622073846_base_robot1.to(device=tensor_args.device)),
            'robot2': Pose.from_matrix(camera_244622073846_base_robot2.to(device=tensor_args.device)),
        },
    }
    pose_dict_np = {
        '337122071679': {
            'robot1': camera_337122071679_base_robot1_np,
            'robot2': camera_337122071679_base_robot2_np,
        },
        '244622073846': {
            'robot1': camera_244622073846_base_robot1_np,
            'robot2': camera_244622073846_base_robot2_np,
        },
    }
    for frame_num, cameras_data in enumerate(cameras):
        if(frame_num < 50):
            continue
        start_time = time.time()
        print(f"\n==========================================\n")
        print(T_offset_robot2_inv)
        # print("curr_joint: ", env_closest_point_interface.robot_1_curr_joint_pos)
        curr_joint_pos_robot1 = np.append(env_closest_point_interface.robot_1_curr_joint_pos, [0.04, 0.04])
        curr_joint_pos_robot2 = np.append(env_closest_point_interface.robot_2_curr_joint_pos, [0.04, 0.04])
        # curr_joint_pos_robot1 = np.append(Init_joints_robot1, [0.04, 0.04])
        # curr_joint_pos_robot2 = np.append(Init_joints_robot2, [0.04, 0.04])
        print("curr_joint_pos_robot1: ", curr_joint_pos_robot1)
        print("curr_joint_pos_robot2: ", curr_joint_pos_robot2)
        print(f"curr_joint_pos_robot1:\n{curr_joint_pos_robot1[:7] * 180 / np.pi}")
        print(f"curr_joint_pos_robot2:\n{curr_joint_pos_robot2[:7] * 180 / np.pi}")
        q_input_robot1 = torch.tensor(curr_joint_pos_robot1, dtype=torch.float32, device='cuda')
        q_input_robot2 = torch.tensor(curr_joint_pos_robot2, dtype=torch.float32, device='cuda')
        q_js_robot1 = JointState(position=q_input_robot1, joint_names=kin_model.joint_names)
        q_js_robot2 = JointState(position=q_input_robot2, joint_names=kin_model.joint_names)
    
        transforms_robot1 = [panda.fkine(curr_joint_pos_robot1[:7], end=link.name) for link in panda.links]
        transforms_robot2 = [panda.fkine(curr_joint_pos_robot2[:7], end=link.name) for link in panda.links]
        transforms_robot1.append(panda.fkine(curr_joint_pos_robot1[:7]))
        transforms_robot2.append(panda.fkine(curr_joint_pos_robot2[:7]))
        capsule_p1s_robot1, capsule_p2s_robot1, capsule_radii_robot1 = calculate_capsules(transforms_robot1)
        capsule_p1s_robot2, capsule_p2s_robot2, capsule_radii_robot2 = calculate_capsules(transforms_robot2)
        all_capsule_inside_points_robot1 = []
        all_capsule_inside_points_robot2 = []

        for camera_idx, camera_info in cameras_data.items():
            serial = camera_info["serial"]
            rgb_dict[serial] = camera_info["rgb_np"]
            depth_dict[serial] = camera_info["depth_np"]
            img_color = rgb_dict[serial]
            img_color = cv2.cvtColor(img_color, cv2.COLOR_BGR2RGB)

            camera_robot1_obs = CameraObservation(
                        depth_image=tensor_args.to_device(camera_info["depth"]) * 1000,
                        intrinsics=camera_info["intrinsics"].to(device=tensor_args.device),
                        pose=pose_dict[serial]['robot1'],
                    )
            camera_robot2_obs = CameraObservation(
                        depth_image=tensor_args.to_device(camera_info["depth"]) * 1000,
                        intrinsics=camera_info["intrinsics"].to(device=tensor_args.device),
                        pose=pose_dict[serial]['robot2'],
                    )
            cam_obs_robot1 = camera_robot1_obs.stack(camera_robot1_obs)
            cam_obs_robot2 = camera_robot2_obs.stack(camera_robot2_obs)
            depth_mask_robot1, _ = curobo_segmenter.get_robot_mask_from_active_js(
                cam_obs_robot1,
                q_js_robot1,
            )
            depth_mask_robot2, _ = curobo_segmenter.get_robot_mask_from_active_js(
                cam_obs_robot2,
                q_js_robot2,
            )
            point_cloud = np.asanyarray(
                convert_depth_frame_to_pointcloud(
                    depth_dict[serial], 
                    cameras.realsense_manager["calibration_info_devices"][serial][1][rs.stream.color]
                )
            ).reshape(3, 480, 640)
            point_cloud_flat = point_cloud.reshape(3, -1)

            depth_mask_robot1_flat = depth_mask_robot1[0].bool().flatten()
            depth_mask_robot2_flat = depth_mask_robot2[0].bool().flatten()
            combined_mask_flat = depth_mask_robot1_flat | depth_mask_robot2_flat
            non_robot_mask = ~combined_mask_flat
            non_robot_mask = non_robot_mask.cpu().numpy()
            point_cloud_flat = point_cloud_flat[:, non_robot_mask]

            env_points_robot1 = get_robot_point_cloud(point_cloud_flat, pose_dict_np[serial]['robot1']).T
            env_points_robot2 = get_robot_point_cloud(point_cloud_flat, pose_dict_np[serial]['robot2']).T
            print(f"{serial}: env_points_robot1.shape: {env_points_robot1.shape}")
            print(f"{serial}: env_points_robot2.shape: {env_points_robot2.shape}")

            in_capsule_robot1 = points_in_capsules(env_points_robot1, capsule_p1s_robot1, capsule_p2s_robot1, capsule_radii_robot1)
            in_capsule_robot2 = points_in_capsules(env_points_robot2, capsule_p1s_robot2, capsule_p2s_robot2, capsule_radii_robot2)
            all_capsule_inside_points_robot1.append(env_points_robot1[in_capsule_robot1])
            all_capsule_inside_points_robot2.append(env_points_robot2[in_capsule_robot2])
            if frame_num % 10 == 0:
                inside_points = env_points_robot1[in_capsule_robot1]
                with open(f'output/env_closest_point/points_inside_capsules_robot1_{serial}.txt', 'w') as f:
                    for point in inside_points:
                        f.write(f"{point[0]*1000:.3f} {point[1]*1000:.3f} {point[2]*1000:.3f}\n")
                outside_points = env_points_robot1[~in_capsule_robot1]
                with open(f'output/env_closest_point/points_outside_capsules_robot1_{serial}.txt', 'w') as f:
                    for point in outside_points:
                        f.write(f"{point[0]*1000:.3f} {point[1]*1000:.3f} {point[2]*1000:.3f}\n")

                inside_points = env_points_robot2[in_capsule_robot2]
                with open(f'output/env_closest_point/points_inside_capsules_robot2_{serial}.txt', 'w') as f:
                    for point in inside_points:
                        f.write(f"{point[0]*1000:.3f} {point[1]*1000:.3f} {point[2]*1000:.3f}\n")
                outside_points = env_points_robot2[~in_capsule_robot2]
                with open(f'output/env_closest_point/points_outside_capsules_robot2_{serial}.txt', 'w') as f:
                    for point in outside_points:
                        f.write(f"{point[0]*1000:.3f} {point[1]*1000:.3f} {point[2]*1000:.3f}\n")

        closest_point_robot1, min_distance_robot1 = get_closest_point(env_collision_model, curr_joint_pos_robot1[:7], np.vstack(all_capsule_inside_points_robot1))
        closest_point_robot2, min_distance_robot2 = get_closest_point(env_collision_model, curr_joint_pos_robot2[:7], np.vstack(all_capsule_inside_points_robot2))
        print(f"Closest point robot1: {closest_point_robot1}, Min distance robot1: {min_distance_robot1}")
        print(f"Closest point robot2: {closest_point_robot2}, Min distance robot2: {min_distance_robot2}")

        env_closest_point_interface.closest_point_robot1 = closest_point_robot1
        env_closest_point_interface.closest_point_robot2 = closest_point_robot2
        env_closest_point_interface.distance_robot1 = min_distance_robot1
        env_closest_point_interface.distance_robot2 = min_distance_robot2
        command = env_closest_point_interface._command2string2ROS()
        env_closest_point_interface._string2command(command)
        with open('output/env_closest_point.txt', 'a') as env_closest_point_file:
            env_closest_point_file.write(f"{closest_point_robot1} {min_distance_robot1} {closest_point_robot2} {min_distance_robot2}\n")
                
        end_time = time.time()
        loop_time = end_time - start_time
        sleep_time = target_period - loop_time
        if sleep_time > 0:
            time.sleep(sleep_time)
            actual_total_time = target_period
        else:
            actual_total_time = loop_time
        print(f"Loop time: {loop_time:.4f}s | Sleep: {max(sleep_time, 0):.4f}s | Total: {actual_total_time:.4f}s")

def start_ros2_node(node):
    while node._is_running and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)
    node.destroy_node()
    
if __name__ == "__main__":
    rclpy.init()
    env_closest_point_interface = EnvClosestPointROS2Interface()
    ros2_thread = threading.Thread(target=start_ros2_node, args=(env_closest_point_interface,))
    ros2_thread.daemon = True
    ros2_thread.start()
    _Execute(env_closest_point_interface)
    ros2_thread.join()