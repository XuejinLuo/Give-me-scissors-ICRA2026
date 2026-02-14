import numpy as np
import torch
import time
import copy
from scipy.optimize import dual_annealing, minimize
from scipy.interpolate import RegularGridInterpolator
from functions.utils import *
import functions.transform_utils as T
from collections import deque

# ====================================
# = objective function
# ====================================
def objective(opt_vars, #[pos, euler]
              og_bounds_robot1,
              og_bounds_robot2,
              keypoints,
              human_hand_keypoint,
              subgoal_constraints_robot1,
              subgoal_constraints_robot2,
              is_release_stage_robot1,
              is_release_stage_robot2,
              human_pick,
              T_offset_robot2,
              final_stage_robot1=False,
              final_stage_robot2=False,
              return_debug_dict=False):

    debug_dict = {}
    # unnormalize variables and do conversion
    opt_pose_robot1 = unnormalize_vars(opt_vars[:len(og_bounds_robot1)], og_bounds_robot1)
    opt_pose_homo_robot1 = T.pose2mat([opt_pose_robot1[:3], T.euler2quat(opt_pose_robot1[3:])])
    
    opt_pose_robot2 = unnormalize_vars(opt_vars[len(og_bounds_robot1):], og_bounds_robot2)
    opt_pose_homo_robot2 = T.pose2mat([opt_pose_robot2[:3], T.euler2quat(opt_pose_robot2[3:])])
    cost = 0

    # goal constraint violation cost
    debug_dict['subgoal_constraint_cost_robot1'] = None
    debug_dict['subgoal_constraint_cost_robot2'] = None
    debug_dict['subgoal_violation_robot1'] = None
    debug_dict['subgoal_violation_robot2'] = None
    subgoal_constraint_cost_robot1 = 0
    subgoal_constraint_cost_robot2 = 0
    subgoal_violation_robot1 = []
    subgoal_violation_robot2 = []
    pos_robot1 = opt_pose_robot1[:3]
    pos_robot2 = opt_pose_robot2[:3]

    if human_pick and is_release_stage_robot1:
        if human_hand_keypoint is None or final_stage_robot1:
            human_hand_keypoint = pos_robot1
        robot_human_hand_violation = np.linalg.norm(pos_robot1 - human_hand_keypoint)
        subgoal_violation_robot1.append(robot_human_hand_violation)
        subgoal_constraint_cost_robot1 += np.clip(robot_human_hand_violation, 0, np.inf)
    elif subgoal_constraints_robot1 is not None and callable(subgoal_constraints_robot1):
        violation = subgoal_constraints_robot1(pos_robot1, keypoints[2:])
        subgoal_violation_robot1.append(violation)
        subgoal_constraint_cost_robot1 += np.clip(violation, 0, np.inf)

    if human_pick and is_release_stage_robot2:
        if human_hand_keypoint is None or final_stage_robot2:
            human_hand_keypoint = pos_robot2
        robot_human_hand_violation = np.linalg.norm(pos_robot2 - human_hand_keypoint)
        subgoal_violation_robot2.append(robot_human_hand_violation)
        subgoal_constraint_cost_robot2 += np.clip(robot_human_hand_violation, 0, np.inf)
    elif subgoal_constraints_robot2 is not None and callable(subgoal_constraints_robot2):
        violation = subgoal_constraints_robot2(pos_robot2, keypoints[2:])
        subgoal_violation_robot2.append(violation)
        subgoal_constraint_cost_robot2 += np.clip(violation, 0, np.inf)

    subgoal_constraint_cost = 200.0*(subgoal_constraint_cost_robot1 + subgoal_constraint_cost_robot2)
    debug_dict['subgoal_constraint_cost'] = subgoal_constraint_cost
    debug_dict['subgoal_violation_robot1'] = subgoal_violation_robot1
    debug_dict['subgoal_violation_robot2'] = subgoal_violation_robot2
    cost += subgoal_constraint_cost

    if return_debug_dict:
        return cost, debug_dict

    return cost


class SubgoalSolver:
    def __init__(self, args, config, reset_joint_pos, warmup=True):
        self.args = args
        self.config = config
        self.reset_joint_pos = reset_joint_pos
        self.last_opt_result = None
        # warmup
        self._warmup() if warmup else None

    def _warmup(self):
        ee_pose = np.array([0.0, 0.0, 0.0, 0, 0, 0, 1,
                        0.0, 0.0, 0.0, 0, 0, 0, 1])
        keypoints = np.random.rand(10, 3)
        for _ in range(3):
            transform_matrix = np.random.rand(4, 4)
        goal_constraints_robot1 = []
        goal_constraints_robot2 = []
        collision_points = np.random.rand(100, 3)
        initial_joint_pos = self.reset_joint_pos
        self.solve(self.args, ee_pose, keypoints, keypoints, goal_constraints_robot1, goal_constraints_robot2, True, True, initial_joint_pos, np.eye(4), False, False)
        print(self.last_opt_result)
        self.last_opt_result = None

    def _setup_sdf(self, sdf_voxels):
        # create callable sdf function with interpolation
        x = np.linspace(self.config['bounds_min'][0], self.config['bounds_max'][0], sdf_voxels.shape[0])
        y = np.linspace(self.config['bounds_min'][1], self.config['bounds_max'][1], sdf_voxels.shape[1])
        z = np.linspace(self.config['bounds_min'][2], self.config['bounds_max'][2], sdf_voxels.shape[2])
        sdf_func = RegularGridInterpolator((x, y, z), sdf_voxels, bounds_error=False, fill_value=0)
        return sdf_func

    def _check_opt_result(self, opt_result, debug_dict):
        # accept the opt_result if it's only terminated due to iteration limit
        if (not opt_result.success and ('maximum' in opt_result.message.lower() or 'iteration' in opt_result.message.lower() or 'not necessarily' in opt_result.message.lower())):
            opt_result.success = True
        elif not opt_result.success:
            opt_result.message += '; invalid solution'
        # check whether goal constraints are satisfied
        if debug_dict['subgoal_violation_robot1'] is not None:
            goal_constraints_results_robot1 = np.array(debug_dict['subgoal_violation_robot1'])
            opt_result.message += f'; goal_constraints_results_robot1: {goal_constraints_results_robot1} (higher is worse)'
            goal_constraints_satisfied = all([violation <= self.config['constraint_tolerance'] for violation in goal_constraints_results_robot1])
            if not goal_constraints_satisfied:
                opt_result.success = False
                opt_result.message += f'; robot 1 goal not satisfied'

        if debug_dict['subgoal_violation_robot2'] is not None:
            goal_constraints_results_robot2 = np.array(debug_dict['subgoal_violation_robot2'])
            opt_result.message += f'; goal_constraints_results_robot2: {goal_constraints_results_robot2} (higher is worse)'
            goal_constraints_satisfied = all([violation <= self.config['constraint_tolerance'] for violation in goal_constraints_results_robot2])
            if not goal_constraints_satisfied:
                opt_result.success = False
                opt_result.message += f'; robot 2 goal not satisfied'

        return opt_result

    def solve(self,
            args,
            ee_pose,
            keypoints,
            human_hand_keypoint,
            subgoal_constraints_robot1,
            subgoal_constraints_robot2,
            is_release_stage_robot1,
            is_release_stage_robot2,
            initial_joint_pos,
            T_offset_robot2,
            final_stage_robot1,
            final_stage_robot2,
            ):
        """
        Args:
            - ee_pose (np.ndarray): [7], [x, y, z, qx, qy, qz, qw] end effector pose.
            - keypoints (np.ndarray): [M, 3] keypoint positions.
            - keypoint_movable_mask (bool): [M] boolean array indicating whether the keypoint is on the grasped object.
            - goal_constraints (List[Callable]): subgoal constraint functions.
            - path_constraints (List[Callable]): path constraint functions.
            - sdf_voxels (np.ndarray): [X, Y, Z] signed distance field of the environment.
            - collision_points (np.ndarray): [N, 3] point cloud of the object.
            - is_grasp_stage (bool): whether the current stage is a grasp stage.
            - initial_joint_pos (np.ndarray): [N] initial joint positions of the robot.
            - from_scratch (bool): whether to start from scratch.
        Returns:
            - result (scipy.optimize.OptimizeResult): optimization result.
            - debug_dict (dict): debug information.
        """
        # downsample collision points
        # if collision_points is not None and collision_points.shape[0] > self.config['max_collision_points']:
        #     collision_points = farthest_point_sampling(collision_points, self.config['max_collision_points'])
        # sdf_func = self._setup_sdf(sdf_voxels)
        # ====================================
        # = setup bounds and initial guess
        # ====================================
        ee_pose_robot1 = ee_pose[:7]
        ee_pose_robot2 = ee_pose[7:]
        ee_pose_robot1 = ee_pose_robot1.astype(np.float64)
        ee_pose_robot2 = ee_pose_robot2.astype(np.float64)

        ee_pose_homo_robot1 = T.pose2mat([ee_pose_robot1[:3], ee_pose_robot1[3:]])
        ee_pose_homo_robot2 = T.pose2mat([ee_pose_robot2[:3], ee_pose_robot2[3:]])

        ee_pose_euler_robot1 = np.concatenate([ee_pose_robot1[:3], T.quat2euler(ee_pose_robot1[3:])])
        ee_pose_euler_robot2 = np.concatenate([ee_pose_robot2[:3], T.quat2euler(ee_pose_robot2[3:])])
        # normalize opt variables to [0, 1]
        pos_bounds_min = self.config['bounds_min'] # [-1, -1, -1]
        pos_bounds_max = self.config['bounds_max'] # [1, 1, 1]
        rot_bounds_min = np.array([-2*np.pi, -2*np.pi, -2*np.pi])  # euler angles
        rot_bounds_max = np.array([2*np.pi, 2*np.pi, 2*np.pi])  # euler angles

        og_bounds_robot1 = [(b_min, b_max) for b_min, b_max in 
                    zip(np.concatenate([pos_bounds_min, rot_bounds_min]), 
                        np.concatenate([pos_bounds_max, rot_bounds_max]))]
        og_bounds_robot2 = og_bounds_robot1
        bounds = [(-1, 1)] * (len(og_bounds_robot1) + len(og_bounds_robot2))

        init_sol_robot1 = normalize_vars(ee_pose_euler_robot1, og_bounds_robot1)
        init_sol_robot2 = normalize_vars(ee_pose_euler_robot2, og_bounds_robot2)
        init_sol = np.concatenate([init_sol_robot1, init_sol_robot2])

        # ====================================
        # = other setup
        # ====================================
        aux_args = (og_bounds_robot1,
                    og_bounds_robot2,
                    keypoints,
                    human_hand_keypoint,
                    subgoal_constraints_robot1,
                    subgoal_constraints_robot2,
                    is_release_stage_robot1,
                    is_release_stage_robot2,
                    args.human_pick,
                    T_offset_robot2,
                    final_stage_robot1,
                    final_stage_robot2
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
        debug_dict['sol'] = opt_result.x
        debug_dict['msg'] = opt_result.message
        debug_dict['solve_time'] = solve_time
        debug_dict['type'] = 'subgoal_solver'
        # unnormailze
        sol_robot1 = unnormalize_vars(opt_result.x[:len(og_bounds_robot1)], og_bounds_robot1)
        sol_robot2 = unnormalize_vars(opt_result.x[len(og_bounds_robot1):], og_bounds_robot2)
        # sol = opt_result.x
        sol = np.concatenate([sol_robot1[:3], T.euler2quat(sol_robot1[3:]), 
                      sol_robot2[:3], T.euler2quat(sol_robot2[3:])])
        opt_result = self._check_opt_result(opt_result, debug_dict)
        # cache opt_result for future use if successful
        if opt_result.success:
            self.last_opt_result = copy.deepcopy(opt_result)

        print("sol_robot1: ", sol_robot1)

        return sol, debug_dict