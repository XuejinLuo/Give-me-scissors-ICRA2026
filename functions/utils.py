# utils.py
import os
import time
import datetime
import numpy as np
import inspect
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from tqdm import tqdm
import yaml
from numba import njit
import functions.transform_utils as T
import scipy.interpolate as interpolate
from scipy.spatial.transform import Slerp
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import RotationSpline

debug_counter = 1

def debug_print(para=None):
    global debug_counter
    frame = inspect.currentframe()
    caller_frame = frame.f_back
    filename = caller_frame.f_code.co_filename
    line_number = caller_frame.f_lineno
    print(f'{bcolors.WARNING}***DEBUG*** {debug_counter} (File: {filename}, Line: {line_number}) (para: {para}){bcolors.ENDC}')
    debug_counter += 1

def get_config(config_path=None):
    if config_path is None:
        this_file_dir = os.path.dirname(os.path.abspath(__file__))
        config_path = os.path.join(this_file_dir, 'configs/config.yaml')
    assert config_path and os.path.exists(config_path), f'config file does not exist ({config_path})'
    with open(config_path, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    return config

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

def pca_transform(features, patch_h, patch_w):
    # 对特征进行PCA降维
    pca = PCA(n_components=3)
    pca.fit(features)
    pca_features = pca.transform(features)
    pca_features[:, 0] = (pca_features[:, 0] - pca_features[:, 0].min()) / (pca_features[:, 0].max() - pca_features[:, 0].min())
    pca_features_fg = pca_features[:, 0] > 0
    pca_features_bg = ~pca_features_fg
    b = np.where(pca_features_bg)

    pca.fit(features[pca_features_fg])
    pca_features_rem = pca.transform(features[pca_features_fg])
    for i in range(3):
        pca_features_rem[:, i] = (pca_features_rem[:, i] - pca_features_rem[:, i].min()) / (pca_features_rem[:, i].max() - pca_features_rem[:, i].min())

    pca_features_rgb = pca_features.copy()
    pca_features_rgb[pca_features_fg] = pca_features_rem
    pca_features_rgb[b] = 0

    pca_features_rgb = pca_features_rgb.reshape(4, patch_h, patch_w, 3)


    return pca_features_rgb, pca_features, pca

def save_depth_to_txt(point_cloud, rgb_image, filename):
    """
    将深度图保存到 txt 文件中，格式为 "x y depth"。
    :param point_cloud: 深度图数据，形状为 (height, width, 1)
    :param filename: 保存的文件名
    """
    height, width, _ = point_cloud.shape
    total_pixels = height * width  # 总像素数
    processed_pixels = 0  # 已处理的像素数

    with open(filename, "w") as output_file:
        for y in range(height):
            for x in range(width):
                depth = point_cloud[y, x, 0]  # 获取深度值
                r, g, b = rgb_image[y, x]  # 获取 RGB 值
                output_file.write(f"{x} {y} {depth} {r} {g} {b}\n")  # 保存为 "r g b x y depth"

                # 更新进度
                processed_pixels += 1
                progress = (processed_pixels / total_pixels) * 100  # 计算进度百分比
                print(f"\rSaving Progress: {progress:.2f}%", end="")  # 显示进度

    print(f"\nDepth map saved to {filename}")

def save_image(img, path):
    """
    Save the projected image with a timestamped filename.

    Parameters:
    - projected: The image data to save.
    - base_path: The base directory where the image will be saved.
    """
    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    save_path = f"{path}/projected_image_{timestamp}.png"

    # 保存图像
    plt.imsave(save_path, img)

def unnormalize_vars(normalized_vars, og_bounds):
    """
    Given 1D variables in [-1, 1] and original bounds, denormalize the variables to the original range.
    """
    vars = np.empty_like(normalized_vars)
    for i, (b_min, b_max) in enumerate(og_bounds):
        vars[i] = (normalized_vars[i] + 1) / 2 * (b_max - b_min) + b_min
    return vars

@njit(cache=True, fastmath=True)
def angle_between_rotmat(P, Q):
    R = np.dot(P, Q.T)
    cos_theta = (np.trace(R)-1)/2
    if cos_theta > 1:
        cos_theta = 1
    elif cos_theta < -1:
        cos_theta = -1
    return np.arccos(cos_theta)

@njit(cache=True, fastmath=True)
def consistency(poses_a, poses_b, rot_weight=0.5):
    assert poses_a.shape[1:] == (4, 4) and poses_b.shape[1:] == (4, 4), 'poses must be of shape (N, 4, 4)'
    min_distances = np.zeros(len(poses_a), dtype=np.float64)
    for i in range(len(poses_a)):
        min_distance = 9999999
        a = poses_a[i]
        for j in range(len(poses_b)):
            b = poses_b[j]
            pos_distance = np.linalg.norm(a[:3, 3] - b[:3, 3])
            rot_distance = angle_between_rotmat(a[:3, :3], b[:3, :3])
            distance = pos_distance + rot_distance * rot_weight
            min_distance = min(min_distance, distance)
        min_distances[i] = min_distance
    return np.mean(min_distances)

def transform_keypoints_old(transform, keypoints, movable_mask):
    assert transform.shape == (4, 4)
    transformed_keypoints = keypoints.copy()
    if movable_mask.sum() > 0:
        transformed_keypoints[movable_mask] = np.dot(keypoints[movable_mask], transform[:3, :3].T) + transform[:3, 3]
    return transformed_keypoints

def transform_keypoints(transform, keypoints):
    assert transform.shape == (4, 4)
    transformed_keypoints = np.dot(keypoints, transform[:3, :3].T) + transform[:3, 3]
    return transformed_keypoints

def normalize_vars(vars, og_bounds):
    """
    Given 1D variables and bounds, normalize the variables to [-1, 1] range.
    """
    normalized_vars = np.empty_like(vars)
    for i, (b_min, b_max) in enumerate(og_bounds):
        normalized_vars[i] = (vars[i] - b_min) / (b_max - b_min) * 2 - 1
    return normalized_vars

@njit(cache=True, fastmath=True)
def get_samples_jitted(control_points_homo, control_points_quat, opt_interpolate_pos_step_size, opt_interpolate_rot_step_size):
    assert control_points_homo.shape[1:] == (4, 4)
    # calculate number of samples per segment
    num_samples_per_segment = np.empty(len(control_points_homo) - 1, dtype=np.int64)
    for i in range(len(control_points_homo) - 1):
        start_pos = control_points_homo[i, :3, 3]
        start_rotmat = control_points_homo[i, :3, :3]
        end_pos = control_points_homo[i+1, :3, 3]
        end_rotmat = control_points_homo[i+1, :3, :3]
        pos_diff = np.linalg.norm(start_pos - end_pos)
        rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
        pos_num_steps = np.ceil(pos_diff / opt_interpolate_pos_step_size)
        rot_num_steps = np.ceil(rot_diff / opt_interpolate_rot_step_size)
        num_path_poses = int(max(pos_num_steps, rot_num_steps))
        num_path_poses = max(num_path_poses, 2)  # at least 2 poses, start and end
        num_samples_per_segment[i] = num_path_poses
    # fill in samples
    num_samples = num_samples_per_segment.sum()
    samples_7 = np.empty((num_samples, 7))
    sample_idx = 0
    for i in range(len(control_points_quat) - 1):
        start_pos, start_xyzw = control_points_quat[i, :3], control_points_quat[i, 3:]
        end_pos, end_xyzw = control_points_quat[i+1, :3], control_points_quat[i+1, 3:]
        # using proper quaternion slerp interpolation
        poses_7 = np.empty((num_samples_per_segment[i], 7))
        for j in range(num_samples_per_segment[i]):
            alpha = j / (num_samples_per_segment[i] - 1)
            pos = start_pos * (1 - alpha) + end_pos * alpha
            blended_xyzw = T.quat_slerp_jitted(start_xyzw, end_xyzw, alpha)
            pose_7 = np.empty(7)
            pose_7[:3] = pos
            pose_7[3:] = blended_xyzw
            poses_7[j] = pose_7
        samples_7[sample_idx:sample_idx+num_samples_per_segment[i]] = poses_7
        sample_idx += num_samples_per_segment[i]
    # assert num_samples >= 2, f'num_samples: {num_samples}'
    return samples_7, num_samples

@njit(cache=True, fastmath=True)
def get_samples_jitted_dual(control_points_homo_robot1, control_points_homo_robot2,
                            control_points_quat_robot1, control_points_quat_robot2,
                            opt_interpolate_pos_step_size, opt_interpolate_rot_step_size):
    # assert control_points_homo.shape[1:] == (4, 4)  # 可选的断言，确保输入维度正确
    assert control_points_homo_robot1.shape[1:] == (4, 4)
    assert control_points_homo_robot2.shape[1:] == (4, 4)

    # 计算每个机械臂的样本数量
    num_samples_per_segment_robot1 = np.empty(len(control_points_homo_robot1) - 1, dtype=np.int64)
    num_samples_per_segment_robot2 = np.empty(len(control_points_homo_robot2) - 1, dtype=np.int64)

    # 计算每个机械臂每个段落的样本数
    for i in range(len(control_points_homo_robot1) - 1):
        start_pos_robot1 = control_points_homo_robot1[i, :3, 3]
        start_rotmat_robot1 = control_points_homo_robot1[i, :3, :3]
        end_pos_robot1 = control_points_homo_robot1[i+1, :3, 3]
        end_rotmat_robot1 = control_points_homo_robot1[i+1, :3, :3]
        
        pos_diff_robot1 = np.linalg.norm(start_pos_robot1 - end_pos_robot1)
        rot_diff_robot1 = angle_between_rotmat(start_rotmat_robot1, end_rotmat_robot1)
        
        pos_num_steps_robot1 = np.ceil(pos_diff_robot1 / opt_interpolate_pos_step_size)
        rot_num_steps_robot1 = np.ceil(rot_diff_robot1 / opt_interpolate_rot_step_size)
        num_path_poses_robot1 = int(max(pos_num_steps_robot1, rot_num_steps_robot1))
        num_path_poses_robot1 = max(num_path_poses_robot1, 2)
        num_samples_per_segment_robot1[i] = num_path_poses_robot1

    for i in range(len(control_points_homo_robot2) - 1):
        start_pos_robot2 = control_points_homo_robot2[i, :3, 3]
        start_rotmat_robot2 = control_points_homo_robot2[i, :3, :3]
        end_pos_robot2 = control_points_homo_robot2[i+1, :3, 3]
        end_rotmat_robot2 = control_points_homo_robot2[i+1, :3, :3]
        
        pos_diff_robot2 = np.linalg.norm(start_pos_robot2 - end_pos_robot2)
        rot_diff_robot2 = angle_between_rotmat(start_rotmat_robot2, end_rotmat_robot2)
        
        pos_num_steps_robot2 = np.ceil(pos_diff_robot2 / opt_interpolate_pos_step_size)
        rot_num_steps_robot2 = np.ceil(rot_diff_robot2 / opt_interpolate_rot_step_size)
        num_path_poses_robot2 = int(max(pos_num_steps_robot2, rot_num_steps_robot2))
        num_path_poses_robot2 = max(num_path_poses_robot2, 2)
        num_samples_per_segment_robot2[i] = num_path_poses_robot2

    # 计算总样本数
    num_samples_robot1 = num_samples_per_segment_robot1.sum()
    num_samples_robot2 = num_samples_per_segment_robot2.sum()
    num_samples = max(num_samples_robot1, num_samples_robot2)  # 保证两个机械臂的样本数一致
    max_samples_per_segment = np.maximum(num_samples_per_segment_robot1, num_samples_per_segment_robot2)

    # 初始化样本数组
    samples_7_robot1 = np.empty((num_samples, 7))
    samples_7_robot2 = np.empty((num_samples, 7))

    sample_idx_robot1 = 0
    sample_idx_robot2 = 0

    # 插值两个机械臂的样本
    for i in range(len(control_points_quat_robot1) - 1):
        start_pos_robot1, start_xyzw_robot1 = control_points_quat_robot1[i, :3], control_points_quat_robot1[i, 3:]
        end_pos_robot1, end_xyzw_robot1 = control_points_quat_robot1[i+1, :3], control_points_quat_robot1[i+1, 3:]
        poses_7_robot1 = np.empty((max_samples_per_segment[i], 7))
        for j in range(max_samples_per_segment[i]):
            alpha = j / (max_samples_per_segment[i] - 1)
            pos_robot1 = start_pos_robot1 * (1 - alpha) + end_pos_robot1 * alpha
            blended_xyzw_robot1 = T.quat_slerp_jitted(start_xyzw_robot1, end_xyzw_robot1, alpha)
            pose_7_robot1 = np.empty(7)
            pose_7_robot1[:3] = pos_robot1
            pose_7_robot1[3:] = blended_xyzw_robot1
            poses_7_robot1[j] = pose_7_robot1
        samples_7_robot1[sample_idx_robot1:sample_idx_robot1 + max_samples_per_segment[i]] = poses_7_robot1
        sample_idx_robot1 += max_samples_per_segment[i]

    for i in range(len(control_points_quat_robot2) - 1):
        start_pos_robot2, start_xyzw_robot2 = control_points_quat_robot2[i, :3], control_points_quat_robot2[i, 3:]
        end_pos_robot2, end_xyzw_robot2 = control_points_quat_robot2[i+1, :3], control_points_quat_robot2[i+1, 3:]
        poses_7_robot2 = np.empty((max_samples_per_segment[i], 7))
        for j in range(max_samples_per_segment[i]):
            alpha = j / (max_samples_per_segment[i] - 1)
            pos_robot2 = start_pos_robot2 * (1 - alpha) + end_pos_robot2 * alpha
            blended_xyzw_robot2 = T.quat_slerp_jitted(start_xyzw_robot2, end_xyzw_robot2, alpha)
            pose_7_robot2 = np.empty(7)
            pose_7_robot2[:3] = pos_robot2
            pose_7_robot2[3:] = blended_xyzw_robot2
            poses_7_robot2[j] = pose_7_robot2
        samples_7_robot2[sample_idx_robot2:sample_idx_robot2 + max_samples_per_segment[i]] = poses_7_robot2
        sample_idx_robot2 += max_samples_per_segment[i]

    return samples_7_robot1, samples_7_robot2, num_samples


@njit(cache=True, fastmath=True)
def path_length(samples_homo):
    assert samples_homo.shape[1:] == (4, 4), 'samples_homo must be of shape (N, 4, 4)'
    pos_length = 0
    rot_length = 0
    for i in range(len(samples_homo) - 1):
        pos_length += np.linalg.norm(samples_homo[i, :3, 3] - samples_homo[i+1, :3, 3])
        rot_length += angle_between_rotmat(samples_homo[i, :3, :3], samples_homo[i+1, :3, :3])
    return pos_length, rot_length


def get_linear_interpolation_steps(start_pose, end_pose, pos_step_size, rot_step_size):
    """
    Given start and end pose, calculate the number of steps to interpolate between them.
    Args:
        start_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        end_pose: [6] position + euler or [4, 4] pose or [7] position + quat
        pos_step_size: position step size
        rot_step_size: rotation step size
    Returns:
        num_path_poses: number of poses to interpolate
    """
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = T.euler2mat(start_euler)
        end_rotmat = T.euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = T.quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]
        end_rotmat = T.quat2mat(end_quat)
    else:
        raise ValueError('start_pose and end_pose not recognized')
    pos_diff = np.linalg.norm(start_pos - end_pos)
    rot_diff = angle_between_rotmat(start_rotmat, end_rotmat)
    pos_num_steps = np.ceil(pos_diff / pos_step_size)
    rot_num_steps = np.ceil(rot_diff / rot_step_size)
    num_path_poses = int(max(pos_num_steps, rot_num_steps))
    num_path_poses = max(num_path_poses, 2)  # at least start and end poses
    return num_path_poses


def linear_interpolate_poses(start_pose, end_pose, num_poses):
    """
    Interpolate between start and end pose.
    """
    assert num_poses >= 2, 'num_poses must be at least 2'
    if start_pose.shape == (6,) and end_pose.shape == (6,):
        start_pos, start_euler = start_pose[:3], start_pose[3:]
        end_pos, end_euler = end_pose[:3], end_pose[3:]
        start_rotmat = T.euler2mat(start_euler)
        end_rotmat = T.euler2mat(end_euler)
    elif start_pose.shape == (4, 4) and end_pose.shape == (4, 4):
        start_pos = start_pose[:3, 3]
        start_rotmat = start_pose[:3, :3]
        end_pos = end_pose[:3, 3]
        end_rotmat = end_pose[:3, :3]
    elif start_pose.shape == (7,) and end_pose.shape == (7,):
        start_pos, start_quat = start_pose[:3], start_pose[3:]
        start_rotmat = T.quat2mat(start_quat)
        end_pos, end_quat = end_pose[:3], end_pose[3:]
        end_rotmat = T.quat2mat(end_quat)
    else:
        raise ValueError('start_pose and end_pose not recognized')
    slerp = Slerp([0, 1], R.from_matrix([start_rotmat, end_rotmat]))
    poses = []
    for i in range(num_poses):
        alpha = i / (num_poses - 1)
        pos = start_pos * (1 - alpha) + end_pos * alpha
        rotmat = slerp(alpha).as_matrix()
        if start_pose.shape == (6,):
            euler = T.mat2euler(rotmat)
            poses.append(np.concatenate([pos, euler]))
        elif start_pose.shape == (4, 4):
            pose = np.eye(4)
            pose[:3, :3] = rotmat
            pose[:3, 3] = pos
            poses.append(pose)
        elif start_pose.shape == (7,):
            quat = T.mat2quat(rotmat)
            pose = np.concatenate([pos, quat])
            poses.append(pose)
    return np.array(poses)

def fit_b_spline(control_points):
    # determine appropriate k
    k = min(3, control_points.shape[0]-1)
    spline = interpolate.splprep(control_points.T, s=0, k=k)
    return spline

def sample_from_spline(spline, num_samples):
    sample_points = np.linspace(0, 1, num_samples)
    if isinstance(spline, RotationSpline):
        samples = spline(sample_points).as_matrix()  # [num_samples, 3, 3]
    else:
        assert isinstance(spline, tuple) and len(spline) == 2, 'spline must be a tuple of (tck, u)'
        tck, u = spline
        samples = interpolate.splev(np.linspace(0, 1, num_samples), tck)  # [spline_dim, num_samples]
        samples = np.array(samples).T  # [num_samples, spline_dim]
    return samples

def spline_interpolate_poses(control_points, num_steps):
    """
    Interpolate between through the control points using spline interpolation.
    1. Fit a b-spline through the positional terms of the control points.
    2. Fit a RotationSpline through the rotational terms of the control points.
    3. Sample the b-spline and RotationSpline at num_steps.

    Args:
        control_points: [N, 6] position + euler or [N, 4, 4] pose or [N, 7] position + quat
        num_steps: number of poses to interpolate
    Returns:
        poses: [num_steps, 6] position + euler or [num_steps, 4, 4] pose or [num_steps, 7] position + quat
    """
    assert num_steps >= 2, 'num_steps must be at least 2'
    if isinstance(control_points, list):
        control_points = np.array(control_points)
    if control_points.shape[1] == 6:
        control_points_pos = control_points[:, :3]  # [N, 3]
        control_points_euler = control_points[:, 3:]  # [N, 3]
        control_points_rotmat = []
        for control_point_euler in control_points_euler:
            control_points_rotmat.append(T.euler2mat(control_point_euler))
        control_points_rotmat = np.array(control_points_rotmat)  # [N, 3, 3]
    elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
        control_points_pos = control_points[:, :3, 3]  # [N, 3]
        control_points_rotmat = control_points[:, :3, :3]  # [N, 3, 3]
    elif control_points.shape[1] == 7:
        control_points_pos = control_points[:, :3]
        control_points_rotmat = []
        for control_point_quat in control_points[:, 3:]:
            control_points_rotmat.append(T.quat2mat(control_point_quat))
        control_points_rotmat = np.array(control_points_rotmat)
    else:
        raise ValueError('control_points not recognized')
    # remove the duplicate points (threshold 1e-3)
    diff = np.linalg.norm(np.diff(control_points_pos, axis=0), axis=1)
    mask = diff > 1e-3
    # always keep the first and last points
    mask = np.concatenate([[True], mask[:-1], [True]])
    control_points_pos = control_points_pos[mask]
    control_points_rotmat = control_points_rotmat[mask]
    # fit b-spline through positional terms control points
    pos_spline = fit_b_spline(control_points_pos)
    # fit RotationSpline through rotational terms control points
    times = pos_spline[1]
    rotations = R.from_matrix(control_points_rotmat)
    rot_spline = RotationSpline(times, rotations)
    # sample from the splines
    pos_samples = sample_from_spline(pos_spline, num_steps)  # [num_steps, 3]
    rot_samples = sample_from_spline(rot_spline, num_steps)  # [num_steps, 3, 3]
    if control_points.shape[1] == 6:
        poses = []
        for i in range(num_steps):
            pose = np.concatenate([pos_samples[i], T.mat2euler(rot_samples[i])])
            poses.append(pose)
        poses = np.array(poses)
    elif control_points.shape[1] == 4 and control_points.shape[2] == 4:
        poses = np.empty((num_steps, 4, 4))
        poses[:, :3, :3] = rot_samples
        poses[:, :3, 3] = pos_samples
        poses[:, 3, 3] = 1
    elif control_points.shape[1] == 7:
        poses = np.empty((num_steps, 7))
        for i in range(num_steps):
            quat = T.mat2quat(rot_samples[i])
            pose = np.concatenate([pos_samples[i], quat])
            poses[i] = pose
    return poses

def print_opt_debug_dict(debug_dict):
    print('\n' + '#' * 40)
    print(f'# Optimization debug info:')
    max_key_length = max(len(str(k)) for k in debug_dict.keys())
    for k, v in debug_dict.items():
        if isinstance(v, int) or isinstance(v, float):
            print(f'# {k:<{max_key_length}}: {v:.05f}')
        elif isinstance(v, list) and all(isinstance(x, int) or isinstance(x, float) for x in v):
            print(f'# {k:<{max_key_length}}: {np.array(v).round(5)}')
        else:
            print(f'# {k:<{max_key_length}}: {v}')
    print('#' * 40 + '\n')


def process_path(global_config, curr_ee_pose, path):
    # spline interpolate the path from the current ee pose
    full_control_points = np.concatenate([
        curr_ee_pose.reshape(1, -1),
        path,
    ], axis=0)
    print(f'full_control_points: {full_control_points}')
    num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1],
                                                # global_config['interpolate_pos_step_size'],
                                                # global_config['interpolate_rot_step_size'],
                                                0.01,
                                                0.05
                                                )
    print(f'num_steps: {num_steps}')
    dense_path = spline_interpolate_poses(full_control_points, num_steps)
    # add gripper action
    ee_action_seq = np.zeros((dense_path.shape[0], 8))
    ee_action_seq[:, :7] = dense_path
    ee_action_seq[:, 7] = 0.0
    return ee_action_seq

def load_functions_from_txt(txt_path, get_grasping_cost_fn):
    if txt_path is None:
        return []
    # load txt file
    with open(txt_path, 'r') as f:
        functions_text = f.read()
    # execute functions
    gvars_dict = {
        'np': np,
        'get_grasping_cost_by_keypoint_idx': get_grasping_cost_fn,
    }  # external library APIs
    lvars_dict = dict()
    exec_safe(functions_text, gvars=gvars_dict, lvars=lvars_dict)
    return list(lvars_dict.values())

def exec_safe(code_str, gvars=None, lvars=None):
    """
    Execute code string safely.
    """
    banned_phrases = ['import', '__']
    for phrase in banned_phrases:
        assert phrase not in code_str
  
    if gvars is None:
        gvars = {}
    if lvars is None:
        lvars = {}
    empty_fn = lambda *args, **kwargs: None
    custom_gvars = merge_dicts([
        gvars,
        {'exec': empty_fn, 'eval': empty_fn}
    ])
    try:
        exec(code_str, custom_gvars, lvars)
    except Exception as e:
        print(f'Error executing code:\n{code_str}')
        raise e
    
def merge_dicts(dicts):
    """
    Merge multiple dictionaries into one.
    """
    return {
        k : v 
        for d in dicts
        for k, v in d.items()
    }

def get_callable_grasping_cost_fn(grab):
    def get_grasping_cost(grab):
        return grab  # return 0 if grasping an object, 1 if not grasping any object
    return get_grasping_cost

def encode_robot_command(joint_positions, is_grabbing_command):
    """
    编码目标关节位置和抓取状态为字符串
    :param joint_positions: 列表，包含目标关节位置
    :param is_grabbing_command: 布尔值，表示是否抓取
    :return: 字符串，编码后的指令
    """
    positions_str = ','.join(map(str, joint_positions))
    grabbing_str = '1' if is_grabbing_command else '0'
    command_str = f'joints: [{positions_str}], grab: {grabbing_str}'
    return command_str

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def get_clock_time(milliseconds=False):
    curr_time = datetime.datetime.now()
    if milliseconds:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}.{curr_time.microsecond // 1000}'
    else:
        return f'{curr_time.hour}:{curr_time.minute}:{curr_time.second}'
    
def angle_between_quats(q1, q2):
    """Angle between two quaternions"""
    return 2 * np.arccos(np.clip(np.abs(np.dot(q1, q2)), -1, 1))

