import numpy as np
import torch
import time
import copy
from scipy.optimize import dual_annealing, minimize
from scipy.interpolate import RegularGridInterpolator
from functions.utils import *
import functions.transform_utils as T

# ====================================
# = objective function
# ====================================
def objective(opt_vars, #[pos, euler]
                og_bounds_robot1,
                og_bounds_robot2,
                start_pose_robot1,
                start_pose_robot2,
                end_pose_robot1,
                end_pose_robot2,
                keypoints_centered,
                opt_interpolate_pos_step_size,
                opt_interpolate_rot_step_size,
                T_offset_robot2,
                return_debug_dict=False):
    debug_dict = {}
    num_control_points = len(opt_vars) // 12
    debug_dict['num_control_points'] = num_control_points

    # unnormalize variables and do conversion
    unnormalized_opt_vars_robot1 = unnormalize_vars(opt_vars[:6 * num_control_points], og_bounds_robot1)
    unnormalized_opt_vars_robot2 = unnormalize_vars(opt_vars[6 * num_control_points:], og_bounds_robot2)
    control_points_robot1_euler = np.concatenate([start_pose_robot1[None], unnormalized_opt_vars_robot1.reshape(-1, 6), end_pose_robot1[None]], axis=0) # [num_control_points, 6]
    control_points_robot2_euler = np.concatenate([start_pose_robot2[None], unnormalized_opt_vars_robot2.reshape(-1, 6), end_pose_robot2[None]], axis=0)
    control_points_robot1_homo = T.convert_pose_euler2mat(control_points_robot1_euler) # [num_control_points, 4, 4]
    control_points_robot2_homo = T.convert_pose_euler2mat(control_points_robot2_euler)
    control_points_robot1_quat = T.convert_pose_mat2quat(control_points_robot1_homo) # [num_control_points, 7]
    control_points_robot2_quat = T.convert_pose_mat2quat(control_points_robot2_homo)

    # get dense samples. eg: 50 points -> 100 points
    # print(f"control_points_robot1_quat: {control_points_robot1_quat}\ncontrol_points_robot2_quat: {control_points_robot2_quat}\n")
    poses_robot1_quat, poses_robot2_quat, num_samples = get_samples_jitted_dual(
                                                        control_points_robot1_homo, control_points_robot2_homo,
                                                        control_points_robot1_quat, control_points_robot2_quat,
                                                        opt_interpolate_pos_step_size, opt_interpolate_rot_step_size
                                                    )
    # print(f"poses_robot1_quat: {poses_robot1_quat}")
    # print(f"poses_robot2_quat: {poses_robot2_quat}")
    poses_robot1_homo = T.convert_pose_quat2mat(poses_robot1_quat)
    poses_robot2_homo = T.convert_pose_quat2mat(poses_robot2_quat)

    debug_dict['num_poses'] = num_samples
    start_idx, end_idx = 1, num_samples - 1  # exclude start and goal
    cost= 0
    # collision cost
    # if collision_points_centered is not None:
    #     collision_cost = 0.5 * calculate_collision_cost(poses_homo[start_idx:end_idx], sdf_func, collision_points_centered, 0.20)
    #     debug_dict['collision_cost'] = collision_cost
    #     cost += collision_cost

    # penalize path length
    pos_length_robot1, rot_length_robot1 = path_length(poses_robot1_homo)
    pos_length_robot2, rot_length_robot2 = path_length(poses_robot2_homo)
    approx_length_robot1 = pos_length_robot1 + rot_length_robot1 * 1.0
    approx_length_robot2 = pos_length_robot2 + rot_length_robot2 * 1.0
    path_length_cost_robot1 = 4.0 * approx_length_robot1
    path_length_cost_robot2 = 4.0 * approx_length_robot2
    debug_dict['path_length_cost_robot1'] = path_length_cost_robot1
    debug_dict['path_length_cost_robot2'] = path_length_cost_robot2
    # cost += path_length_cost_robot1
    # cost += path_length_cost_robot2

    debug_dict['total_cost'] = cost
    if return_debug_dict:
        return cost, debug_dict
    return cost


class PathSolver:
    """
    Given a goal pose and a start pose, solve for a sequence of intermediate poses for the end effector to follow.
    
    Optimization variables:
    - sequence of intermediate control points
    """

    def __init__(self, args, config, reset_joint_pos, warmup=True):
        self.args = args
        self.config = config
        self.reset_joint_pos = reset_joint_pos
        self.last_opt_result = None
        # warmup
        self._warmup() if warmup else None

    def _warmup(self):
        start_pose = np.array([0.0, 0.0, 0.3, 0, 0, 0, 1,
                        0.0, 0.0, 0.3, 0, 0, 0, 1])
        end_pose = np.array([0.0, 0.0, 0.0, 0, 0, 0, 1,
                        0.0, 0.0, 0.0, 0, 0, 0, 1])
        keypoints = np.random.rand(10, 3)
        # keypoint_movable_mask = np.random.rand(10) > 0.5
        path_constraints = []
        sdf_voxels = np.zeros((10, 10, 10))
        collision_points = np.random.rand(100, 3)
        initial_joint_pos = self.reset_joint_pos
        # self.solve(start_pose, end_pose, keypoints, keypoint_movable_mask, path_constraints, sdf_voxels, collision_points, initial_joint_pos, from_scratch=True)
        self.solve(self.args, start_pose, end_pose, keypoints, initial_joint_pos, np.eye(4))
        print(self.last_opt_result)
        self.last_opt_result = None

    def _setup_sdf(self, sdf_voxels):
        # create callable sdf function with interpolation
        x = np.linspace(self.config['bounds_min'][0], self.config['bounds_max'][0], sdf_voxels.shape[0])
        y = np.linspace(self.config['bounds_min'][1], self.config['bounds_max'][1], sdf_voxels.shape[1])
        z = np.linspace(self.config['bounds_min'][2], self.config['bounds_max'][2], sdf_voxels.shape[2])
        sdf_func = RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=0)
        return sdf_func

    def _check_opt_result(self, opt_result, path_quat, debug_dict, og_bounds):
        # accept the opt_result if it's only terminated due to iteration limit
        if (not opt_result.success and ('maximum' in opt_result.message.lower() or 'iteration' in opt_result.message.lower() or 'not necessarily' in opt_result.message.lower())):
            opt_result.success = True
        elif not opt_result.success:
            opt_result.message += '; invalid solution'

        return opt_result

    def _center_collision_points_and_keypoints_old(self, ee_pose, collision_points, keypoints, keypoint_movable_mask):
        ee_pose_homo = T.pose2mat([ee_pose[:3], T.euler2quat(ee_pose[3:])])
        centering_transform = np.linalg.inv(ee_pose_homo)
        collision_points_centered = np.dot(collision_points, centering_transform[:3, :3].T) + centering_transform[:3, 3]
        keypoints_centered = transform_keypoints(centering_transform, keypoints, keypoint_movable_mask)
        return collision_points_centered, keypoints_centered
    
    def _center_collision_points_and_keypoints(self, ee_pose, keypoints):
        ee_pose_homo = T.pose2mat([ee_pose[:3], T.euler2quat(ee_pose[3:])])
        centering_transform = np.linalg.inv(ee_pose_homo)
        keypoints_centered = transform_keypoints(centering_transform, keypoints)
        return keypoints_centered
    
    def solve(self,
            args,
            start_pose,
            end_pose,
            keypoints,
            initial_joint_pos,
            T_offset_robot2,
        ):
        """
        Args:
            - start_pose (np.ndarray): [7], [x, y, z, qx, qy, qz, qw]
            - end_pose (np.ndarray): [7], [x, y, z, qx, qy, qz, qw]
            - keypoints (np.ndarray): [num_keypoints, 3]
            - keypoint_movable_mask (bool): whether the keypoints are on the object being grasped
            - path_constraints (List[Callable]): path constraints
            - sdf_voxels (np.ndarray): [H, W, D]
            - collision_points (np.ndarray): [num_points, 3], point cloud of the object being grasped
            - initial_joint_pos (np.ndarray): [N] initial joint positions of the robot.
            - from_scratch (bool): whether to start from scratch

        Returns:
            - opt_result (scipy.optimize.OptimizeResult): optimization opt_result
            - debug_dict (dict): debug information
        """
        # downsample collision points
        # if collision_points is not None and collision_points.shape[0] > self.config['max_collision_points']:
        #     collision_points = farthest_point_sampling(collision_points, self.config['max_collision_points'])
        # sdf_func = self._setup_sdf(sdf_voxels)

        # ====================================
        # = setup bounds
        # ====================================
        # calculate an appropriate number of control points, including start and goal
        num_control_points_robot1 = get_linear_interpolation_steps(start_pose[:7], end_pose[:7], self.config['opt_pos_step_size'], self.config['opt_rot_step_size'])
        num_control_points_robot2 = get_linear_interpolation_steps(start_pose[7:], end_pose[7:], self.config['opt_pos_step_size'], self.config['opt_rot_step_size'])
        num_control_points_robot1 = np.clip(num_control_points_robot1, 3, 6)
        num_control_points_robot2 = np.clip(num_control_points_robot2, 3, 6)
        num_control_points = max(num_control_points_robot1, num_control_points_robot2)
        # transform to euler representation
        start_pose_robot1 = np.concatenate([start_pose[:3], T.quat2euler(start_pose[3:7])])
        end_pose_robot1 = np.concatenate([end_pose[:3], T.quat2euler(end_pose[3:7])])
        start_pose_robot2 = np.concatenate([start_pose[7:10], T.quat2euler(start_pose[10:])])
        end_pose_robot2 = np.concatenate([end_pose[7:10], T.quat2euler(end_pose[10:])])

        # bounds for decision variables
        og_bounds_robot1 = [(b_min, b_max) for b_min, b_max in zip(self.config['bounds_min'], self.config['bounds_max'])] + \
                        [(-np.pi, np.pi) for _ in range(3)]  # 3 for robot1
        og_bounds_robot1 *= (num_control_points - 2)
        og_bounds_robot2 = [(b_min, b_max) for b_min, b_max in zip(self.config['bounds_min'], self.config['bounds_max'])] + \
                        [(-np.pi, np.pi) for _ in range(3)]  # 3 for robot2
        og_bounds_robot2 *= (num_control_points - 2)
        og_bounds = np.array(og_bounds_robot1 + og_bounds_robot2, dtype=np.float64)

        bounds = [(-1, 1)] * len(og_bounds)
        num_vars = len(bounds)
        # ====================================
        # = setup initial guess
        # ====================================
        interp_poses_robot1 = linear_interpolate_poses(start_pose_robot1, end_pose_robot1, num_control_points) #! important!
        interp_poses_robot2 = linear_interpolate_poses(start_pose_robot2, end_pose_robot2, num_control_points)
        init_sol_robot1 = interp_poses_robot1[1:-1].flatten()
        init_sol_robot2 = interp_poses_robot2[1:-1].flatten()
        init_sol = np.concatenate([init_sol_robot1, init_sol_robot2]) #! a lot of points not one point. [robot1.1, robot1.2, robot2.1, robot2.2]
        init_sol = normalize_vars(init_sol, og_bounds)

        # clip the initial guess to be within bounds
        for i, (b_min, b_max) in enumerate(bounds):
            init_sol[i] = np.clip(init_sol[i], b_min, b_max)

        # ====================================
        # = other setup
        # ====================================
        aux_args = (og_bounds_robot1,
                    og_bounds_robot2,
                    start_pose_robot1,
                    start_pose_robot2,
                    end_pose_robot1,
                    end_pose_robot2,
                    keypoints,
                    self.config['opt_interpolate_pos_step_size'],
                    self.config['opt_interpolate_rot_step_size'],
                    T_offset_robot2,
                )

        # ====================================
        # = solve optimization
        # ====================================
        start = time.time()
        opt_result = minimize(
            fun=objective,
            x0=init_sol,
            args=aux_args,
            bounds=bounds,
            method='SLSQP',
            options=self.config['minimizer_options'],
        )
        solve_time = time.time() - start
        # ====================================
        # = post-process opt_result
        # ====================================
        if isinstance(opt_result.message, list):
            opt_result.message = opt_result.message[0]
       # rerun to get debug info
        _, debug_dict = objective(opt_result.x, *aux_args, return_debug_dict=True)
        debug_dict['sol'] = opt_result.x.reshape(-1, 12)
        debug_dict['msg'] = opt_result.message
        debug_dict['solve_time'] = solve_time
        debug_dict['type'] = 'path_solver'

        # unnormailze
        sol_robot1 = unnormalize_vars(opt_result.x[:len(opt_result.x)//2], og_bounds_robot1)
        sol_robot2 = unnormalize_vars(opt_result.x[len(opt_result.x)//2:], og_bounds_robot2)

        # add end pose
        poses_euler_robot1 = np.concatenate([sol_robot1.reshape(-1, 6), end_pose_robot1[None]], axis=0)
        poses_euler_robot2 = np.concatenate([sol_robot2.reshape(-1, 6), end_pose_robot2[None]], axis=0)

        poses_quat_robot1 = T.convert_pose_euler2quat(poses_euler_robot1)  # [num_control_points, 7]
        poses_quat_robot2 = T.convert_pose_euler2quat(poses_euler_robot2)
        concatenated_poses = np.concatenate([poses_quat_robot1, poses_quat_robot2], axis=0)

        opt_result = self._check_opt_result(opt_result, concatenated_poses, debug_dict, og_bounds)
        # cache opt_result for future use if successful
        if opt_result.success:
            self.last_opt_result = copy.deepcopy(opt_result)
        return poses_quat_robot1, poses_quat_robot2, debug_dict