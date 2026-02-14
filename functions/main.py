import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import json
import argparse
from datetime import datetime
from functions.RGBDcapture import start_capture
from functions.hand_keypoint_capture import HandKeypointCapture
from functions.mask_generator import MaskGenerator
from functions.keypoints_extractor import KeypointsExtractor
from functions.constraint_generation import ConstraintGenerator
from functions.motion_planner import MotionPlanner
from functions.dualarm_simulation_manager import RobotArmSimulation
from functions.subgoal_solver import SubgoalSolver
from functions.path_solver import PathSolver
from functions.utils import *
from functions.transform_utils import *
import rclpy # type: ignore
import sys
import os
from pyros.pyros.robot_interface_dualarm import RobotInterface
from camera.capture import start_capture2
import threading
import warnings
import roboticstoolbox as rtb
warnings.filterwarnings("ignore", category=UserWarning)

def start_ros2_node(node):
    while node._is_running and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)
    node.destroy_node()
    print("ros2 node over")

class Main:
    def __init__(self, args):
        self.args = args
        self.robot = rtb.models.Panda()
        global_config = get_config(config_path="configs/config.yaml")
        self.global_config = global_config
        self.config = global_config['main']
        self.interpolate_pos_step_size = self.config['interpolate_pos_step_size']
        self.interpolate_rot_step_size = self.config['interpolate_rot_step_size']
        f_path = "/home/luo/ICRA/pictures/live_record"
        now = datetime.datetime.now()
        date_folder = now.strftime("%Y-%m-%d")
        date_folder_path = os.path.join(f_path, date_folder)
        os.makedirs(date_folder_path, exist_ok=True)
        time_folder = now.strftime("%H-%M-%S")
        save_path = os.path.join(date_folder_path, time_folder)
        os.makedirs(save_path, exist_ok=True)
        origin_image_path = os.path.join(save_path, "saved_image.png")
        roi_image_path = os.path.join(save_path, "roi_color.png")
        self.save_path = save_path
        self.candidate_pixels = []
        
        rclpy.init()
        if self.args.sim: 
            self.robot_simulation = RobotArmSimulation() 
        else: self.robot_simulation = None
        self.robot_interface = RobotInterface()
        reset_joint_pos = [0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]
        self.subgoal_solver = SubgoalSolver(self.args, global_config['subgoal_solver'], reset_joint_pos, warmup=True)
        self.path_solver = PathSolver(self.args, global_config['path_solver'], reset_joint_pos, warmup=True)
        self.camera_base_robot = np.array(global_config['camera']['camera231_base_robot'])
        self.camera_calib_offset = np.array(global_config['camera']['camera_calib_offset'])
        self.robot_base_camera = np.linalg.inv(self.camera_base_robot)

        self.verbose = True
        #------------------------------------
        self.mask_generator = MaskGenerator(self.global_config, roi_image_path)
        self.keypoints_extractor = KeypointsExtractor(self.global_config, origin_image_path, roi_image_path, save_path)
        self.motion_planner = MotionPlanner(self.args, self.robot, self.global_config, self.robot_interface, self.robot_simulation)


    def _perform_task(self, instruction, ICRA_program_dir, ask_gpt):
        if ICRA_program_dir is None:
            # point_cloud = start_capture(self.save_path)
            point_cloud_cumulative, point_cloud = start_capture2(self.global_config, self.save_path)
            masks = self.mask_generator._Generate_masks(self.save_path)
            projected, self.candidate_keypoints, self.candidate_pixels, self.candidate_rigid_group_ids = self.keypoints_extractor.Get_Keypoints(point_cloud, masks)
            ICRA_program_dir = self._Ask_GPT(instruction, projected) if ask_gpt else None
            self._Save_candidate_pixels(ICRA_program_dir, candidate_pixels_path = 'candidate_pixels.npy')
            
        torch.cuda.empty_cache()
        self._Load_candidate_pixels(ICRA_program_dir, candidate_pixels_path = 'candidate_pixels.npy')
        if self.args.human_pick: self.human_hand_keypoint_capture = HandKeypointCapture()
        execute_thread = threading.Thread(target=self._Execute, args=(ICRA_program_dir,))
        execute_thread.daemon = True
        ros2_thread = threading.Thread(target=start_ros2_node, args=(self.robot_interface,))
        ros2_thread.daemon = True
        robot_state_cal_thread = threading.Thread(target=self._calculate_robot_state)
        robot_state_cal_thread.daemon = True
        if self.args.sim: 
            simulation_thread = threading.Thread(target=self.robot_simulation.run)
            simulation_thread.daemon = True
        ros2_thread.start()  
        execute_thread.start()
        robot_state_cal_thread.start()
        if self.args.sim: simulation_thread.start()
        if self.args.human_pick: self.human_hand_keypoint_capture.run()
        execute_thread.join()
        ros2_thread.join()
        robot_state_cal_thread.join()
        if self.args.sim: simulation_thread.join()

    def _calculate_robot_state(self):
        self.T_offset_robot2 = np.loadtxt("models/dualarm_robot2_offset.txt")
        while self.robot_interface._is_running:
            start_time = time.time()
            if rclpy.ok():
                self.curr_joint_pos, self.grab_state = self._get_robot_data() if not self.args.sim else self._sim_get_robot_data()
                if self.curr_joint_pos is not None:
                    # robot1
                    T = self.robot.fkine(self.curr_joint_pos[:7])
                    self.ee_matrix_base_robot1 = T.A
                    ee_position_robot1 = self.ee_matrix_base_robot1[:3, 3]
                    ee_rotation_matrix_robot1 = self.ee_matrix_base_robot1[:3, :3]
                    ee_orientation_robot1 = mat2quat(ee_rotation_matrix_robot1)
                    ee_pose_robot1 = np.concatenate([ee_position_robot1, ee_orientation_robot1])
                    self.curr_ee_pose_robot1 = ee_pose_robot1
                    # robot2
                    T = self.robot.fkine(self.curr_joint_pos[-7:])
                    self.ee_matrix_base_robot2 = T.A
                    self.ee_matrix_base_robot2 = self.T_offset_robot2 @ self.ee_matrix_base_robot2
                    ee_position_robot2 = self.ee_matrix_base_robot2[:3, 3]
                    ee_rotation_matrix_robot2 = self.ee_matrix_base_robot2[:3, :3]
                    ee_orientation_robot2 = mat2quat(ee_rotation_matrix_robot2)
                    ee_pose_robot2 = np.concatenate([ee_position_robot2, ee_orientation_robot2])
                    self.curr_ee_pose_robot2 = ee_pose_robot2

                    self.curr_ee_pose = np.concatenate([self.curr_ee_pose_robot1, self.curr_ee_pose_robot2]) # Robot1 coordinate
                    self.motion_planner._update_current_ee_pose(self.curr_ee_pose)

                    if self.args.sim: 
                        self.motion_planner.env_nearest_point_robot1 = self.motion_planner.env_nearest_sphere1
                        self.motion_planner.env_nearest_point_robot2 = np.linalg.inv(self.T_offset_robot2) @ np.append(self.motion_planner.env_nearest_sphere2 * 0.001, 1)
                        self.motion_planner.env_nearest_point_robot2 = self.motion_planner.env_nearest_point_robot2[:3] * 1000
                        # print(f"env_nearest_point_robot1: {self.motion_planner.env_nearest_point_robot1}\n env_nearest_point_robot2: {self.motion_planner.env_nearest_point_robot2}")
                    else:
                        self.motion_planner.env_nearest_point_robot1 = self.robot_interface.env_closest_point_robot1
                        self.motion_planner.env_nearest_point_robot2 = self.robot_interface.env_closest_point_robot2

            finish_time = time.time()
            cycle_time = finish_time - start_time
            if cycle_time < self.config['time_period']:
                time.sleep(self.config['time_period'] - cycle_time)
        print("_calculate_robot_state over")

    def _Execute(self, ICRA_program_dir):
        print("Executing...")
        time.sleep(0.1)
        self.curr_joint_pos, self.grab_state = self._get_robot_data() if not self.args.sim else self._sim_get_robot_data()
        # load metadata
        with open(os.path.join(ICRA_program_dir, 'metadata.json'), 'r') as f:
            self.program_info = json.load(f)
        print(self.program_info)
        self.motion_planner.program_info = self.program_info
        # load constraints
        self.constraint_fns = dict()
        for stage in range(1, self.program_info['num_stages'] + 1):  # stage starts with 1
            stage_dict = dict()
            load_path = os.path.join(ICRA_program_dir, f'stage{stage}_subgoal_constraint.txt')
            get_grasping_cost_fn = get_callable_grasping_cost_fn(self.grab_state)  # special grasping function for VLM to call
            stage_dict['subgoal'] = load_functions_from_txt(load_path, get_grasping_cost_fn) if os.path.exists(load_path) else []
            self.constraint_fns[stage] = stage_dict
        self.motion_planner._update_stage(stage_robot1 = 1, stage_robot2 = 1)

        countdown = 2
        print(f"Starting countdown: {countdown} seconds...")
        while countdown > 0:
            print(f"{countdown} seconds remaining...")
            time.sleep(1)
            countdown -= 1
        all_start_time = time.time()
        next_action_robot1 = None
        next_action_robot2 = None
        # main loop
        while True:
            if rclpy.ok():
                if self.curr_joint_pos is not None:
                    print(f"Current joint positions: {self.curr_joint_pos}")
                    camera_keypoints = self.program_info['init_keypoint_positions']
                    keypoints_base_robot = transform_points(camera_keypoints, self.camera_base_robot, self.camera_calib_offset) # Adjust y-coordinate to match the camera coordinate system
                    self.keypoints = np.concatenate([self.curr_ee_pose_robot1[:3].reshape(1, 3), self.curr_ee_pose_robot2[:3].reshape(1, 3), keypoints_base_robot], axis=0)
                    print("keypoints", self.keypoints)
                    if self.args.sim and self.motion_planner.first_iter:
                        self.robot_simulation.load_sphere_small(np.array(keypoints_base_robot))

                    # ====================================
                    # = get stage trajactory
                    # ====================================
                    next_subgoal = self._get_next_subgoal()
                    next_path_robot1, next_path_robot2 = self._get_next_path(next_subgoal)
                    self.action_queue_robot1 = next_path_robot1.tolist() if not self.motion_planner.stay_stage_robot1 else []
                    self.action_queue_robot2 = next_path_robot2.tolist() if not self.motion_planner.stay_stage_robot2 else []
                    self.motion_planner.first_iter = False
                    if not self.motion_planner.stay_stage_robot1: 
                        self.motion_planner.back_action_robot1 = self.action_queue_robot1[-1]
                    if not self.motion_planner.stay_stage_robot2:
                        self.motion_planner.back_action_robot2 = self.action_queue_robot2[-1]
                    # print(self.action_queue_robot1)
                    self.motion_planner.stay_stage_robot1 = self.motion_planner.stay_stage_robot2 = False

                    # ====================================
                    # = execute
                    # ====================================
                    count = self.motion_planner.stay_count_robot1 = self.motion_planner.stay_count_robot2 = 0
                    while (len(self.action_queue_robot1) > 0 or len(self.action_queue_robot2) > 0) and count < self.config['action_steps_per_iter'] and not self.motion_planner.dualarm_collision_detection:
                        start_time = time.time()
                        if (len(self.action_queue_robot1) > 0):
                            next_action_robot1 = self.action_queue_robot1.pop(0)
                        if (len(self.action_queue_robot2) > 0):
                            next_action_robot2 = self.action_queue_robot2.pop(0)
                        self.motion_planner._execute_action_dualarm(next_action_robot1, next_action_robot2)
                        dualarm_ee_long_stay = self.motion_planner._check_ee_stay_status()
                        print(f"{bcolors.WARNING}Stay Status: {dualarm_ee_long_stay} \n{bcolors.ENDC}")
                        if dualarm_ee_long_stay:
                            break
                        
                        count += 1
                        print("count: ", count)
                        print(f"len(self.action_queue): {len(self.action_queue_robot1)}, {len(self.action_queue_robot2)}")
                        finish_time = time.time()
                        cycle_time = finish_time - start_time
                        if cycle_time < self.config['time_period']:
                            time.sleep(self.config['time_period'] - cycle_time)
                        end_time = time.time()
                        loop_time = end_time - start_time
                        print(f'loop {count} took {1000 * loop_time:.4f} miliseconds')
                        print(f"{bcolors.OKGREEN}Robot1 stage: {self.motion_planner.stage_robot1} Robot2 stage: {self.motion_planner.stage_robot2}\n{bcolors.ENDC}")
                        print(f"{bcolors.OKGREEN}Avoidance_operation: {self.motion_planner.OptimizeSolver.robot_avoidance_operation}\n{bcolors.ENDC}")
                        print(f"{bcolors.FAIL}Stay Stage: {self.motion_planner.stay_stage_robot1}  {self.motion_planner.stay_stage_robot2}\n{bcolors.ENDC}")

                    if len(self.action_queue_robot1) == 0 or len(self.action_queue_robot2) == 0:
                        self.verbose and print(f'{bcolors.FAIL}[main.py | {get_clock_time()}] execute grasp/release action{bcolors.ENDC}')
                        self.motion_planner._execute_gripper_action()
                        print(f"waiting for gripper...")
                        time.sleep(3)
                        if self.motion_planner.stage_robot1 == self.program_info['num_stages']:
                            self.motion_planner.final_stage_robot1 = True
                        if self.motion_planner.stage_robot2 == self.program_info['num_stages']:
                            self.motion_planner.final_stage_robot2 = True
                        # if completed, save video and return
                        if self.motion_planner.stage_robot1 == self.program_info['num_stages'] and self.motion_planner.stage_robot2 == self.program_info['num_stages']: 
                            print(f"{bcolors.OKGREEN}Program completed\n\n{bcolors.ENDC}")
                            all_end_time = time.time()
                            total_time = all_end_time - all_start_time
                            print(f"{bcolors.OKGREEN}Program time: {total_time}\n\n{bcolors.ENDC}")
                            self.robot_interface._is_running = False
                            return
                    # progress to next stage
                    self.verbose and print(f'{bcolors.FAIL}[main.py | {get_clock_time()}] update stage{bcolors.ENDC}')
                    robot1_new_stage = min(self.motion_planner.stage_robot1 + 1, self.program_info['num_stages'])
                    robot2_new_stage = min(self.motion_planner.stage_robot2 + 1, self.program_info['num_stages'])
                    self.motion_planner._update_stage(stage_robot1 = robot1_new_stage, stage_robot2 = robot2_new_stage)

                    # ====================================
                    # = Backtrack Stage
                    # ====================================
                    if self.motion_planner.collision_detection:
                        robot1_new_stage = self.motion_planner.stage_robot1
                        robot2_new_stage = self.motion_planner.stage_robot2
                        robot1_error, robot2_error = self.motion_planner._check_reached_ee()
                        if robot1_error > 0.02 and self.motion_planner.stage_robot1 < self.program_info['num_stages']:
                            robot1_new_stage = self.motion_planner.stage_robot1 - 1
                        if robot2_error > 0.02 and self.motion_planner.stage_robot2 < self.program_info['num_stages']:
                            robot2_new_stage = self.motion_planner.stage_robot2 - 1
                        self.motion_planner._update_stage(stage_robot1 = robot1_new_stage, stage_robot2 = robot2_new_stage)
                        self.motion_planner._update_avoidance_robot(robot1_error, robot2_error)

                    self.motion_planner.collision_detection = False
                    self.motion_planner.dualarm_collision_detection = False

                else:
                    print(f'{bcolors.WARNING}No ROS2 received data, exiting...{bcolors.ENDC}')
                    self.robot_interface._is_running = False
                    break
            else:
                break

    def _Ask_GPT(self, instruction, projected):
        print("Asking GPT...")
        metadata = {'init_keypoint_positions': self.candidate_keypoints, 'num_keypoints': len(self.candidate_keypoints)}
        constraint_generator = ConstraintGenerator(self.global_config['constraint_generator'])
        ICRA_program_dir = constraint_generator.generate(projected, instruction, metadata)
        return ICRA_program_dir
    
    def _Save_candidate_pixels(self, ICRA_program_dir, candidate_pixels_path):
        candidate_pixels_file = os.path.join(ICRA_program_dir, candidate_pixels_path)
        np.save(candidate_pixels_file, self.candidate_pixels)

    def _Load_candidate_pixels(self, ICRA_program_dir, candidate_pixels_path):
        candidate_pixels_file = os.path.join(ICRA_program_dir, candidate_pixels_path)
        self.candidate_pixels = np.load(candidate_pixels_file)

    def _get_next_subgoal(self):
        subgoal_constraints_robot1 = self.constraint_fns[self.motion_planner.stage_robot1]['subgoal'][0]
        subgoal_constraints_robot2 = self.constraint_fns[self.motion_planner.stage_robot2]['subgoal'][1]
        self.collision_points = []
        self.human_hand_keypoint = None
        if self.args.human_pick: 
            if (self.motion_planner.is_release_stage_robot1 and not self.motion_planner.final_stage_robot1) or (self.motion_planner.is_release_stage_robot2 and not self.motion_planner.final_stage_robot2):
                self.human_hand_keypoint_capture.camara_flag = True
                print('Waiting for human hand keypoint')
                while self.human_hand_keypoint is None:
                    self.human_hand_keypoint = self.human_hand_keypoint_capture.hand_keypoint
                    # change the human hand keypoint to the robot base coordinate
                    time.sleep(0.1)
                T_offset_camera_first = self.global_config['camera']['camera231_base_robot']
                self.human_hand_keypoint = transform_points(self.human_hand_keypoint / 1000.0, T_offset_camera_first, self.camera_calib_offset)
                self.human_hand_keypoint[0][2] += 0.08
                print(self.human_hand_keypoint)
                
        subgoal_pose, debug_dict = self.subgoal_solver.solve(self.args,
                                                            self.curr_ee_pose,
                                                            self.keypoints,
                                                            self.human_hand_keypoint,
                                                            subgoal_constraints_robot1,
                                                            subgoal_constraints_robot2,
                                                            self.motion_planner.is_release_stage_robot1,
                                                            self.motion_planner.is_release_stage_robot2,
                                                            self.curr_joint_pos,
                                                            self.T_offset_robot2,
                                                            self.motion_planner.final_stage_robot1,
                                                            self.motion_planner.final_stage_robot2,
                                                            )
        subgoal_pose_homo_robot1 = T.convert_pose_quat2mat(subgoal_pose[:7])
        subgoal_pose_homo_robot2 = T.convert_pose_quat2mat(subgoal_pose[7:])
        # if grasp stage, back up a bit to leave room for grasping
        if self.motion_planner.is_grasp_stage_robot1:
            subgoal_pose[0:3] += subgoal_pose_homo_robot1[:3, :3] @ np.array([0, 0, -self.config['grasp_depth']])
            subgoal_pose[3:7] = [1, 0, 0, 0]

        if self.motion_planner.is_grasp_stage_robot2:
            subgoal_pose[7:10] += subgoal_pose_homo_robot2[:3, :3] @ np.array([0, 0, -self.config['grasp_depth']])
            subgoal_pose[10:14] = [1, 0, 0, 0]

        debug_dict['stage_robot1'] = self.motion_planner.stage_robot1
        debug_dict['stage_robot2'] = self.motion_planner.stage_robot2
        print_opt_debug_dict(debug_dict)
        if self.args.human_pick:
            self.human_hand_keypoint = None
            self.human_hand_keypoint_capture.hand_keypoint = None
        return subgoal_pose
    
    def _get_next_path(self, next_subgoal):
        robot1_path, robot2_path, debug_dict = self.path_solver.solve(self.args,
                                                    self.curr_ee_pose,
                                                    next_subgoal,
                                                    self.keypoints,
                                                    self.curr_joint_pos,
                                                    self.T_offset_robot2,
                                                    )
        print_opt_debug_dict(debug_dict)
        robot1_processed_path = self._process_path(self.curr_ee_pose[:7], robot1_path)
        robot2_processed_path = self._process_path(self.curr_ee_pose[7:], robot2_path)
        return robot1_processed_path, robot2_processed_path

    def _process_path(self, curr_ee_pose, path):
        # spline interpolate the path from the current ee pose
        full_control_points = np.concatenate([
            curr_ee_pose.reshape(1, -1),
            path,
        ], axis=0)
        num_steps = get_linear_interpolation_steps(full_control_points[0], full_control_points[-1], # the first one and the last one
                                                    self.config['interpolate_pos_step_size'],
                                                    self.config['interpolate_rot_step_size'])
        print(num_steps)
        if( num_steps < 3) or curr_ee_pose[2] < self.config['bounds_min'][2]:
            dense_path = full_control_points
        else:
            dense_path = spline_interpolate_poses(full_control_points, num_steps)
        # add gripper action
        ee_action_seq = np.zeros((dense_path.shape[0], 8))
        ee_action_seq[:, :7] = dense_path
        ee_action_seq[:, 7] = 0 #! keep the gripper state (do not change it)
        return ee_action_seq

    # ====================================
    # = Robot Interface
    # ====================================

    def _get_robot_data(self):
        robot_1_curr_joint_pos = self.robot_interface.robot_1_curr_joint_pos
        robot_2_curr_joint_pos = self.robot_interface.robot_2_curr_joint_pos
        curr_joint_pos = np.concatenate([robot_1_curr_joint_pos, robot_2_curr_joint_pos])
        robot_1_grab_state = self.robot_interface.robot_1_grab_state
        robot_2_grab_state = self.robot_interface.robot_2_grab_state
        grab_state = np.where((robot_1_grab_state == 0) | (robot_2_grab_state == 0), 0, 1)
        return curr_joint_pos, grab_state
    
    def _sim_get_robot_data(self):
        robot_1_curr_joint_pos = self.robot_interface.joints_command_robot1
        robot_2_curr_joint_pos = self.robot_interface.joints_command_robot2
        curr_joint_pos = np.concatenate([robot_1_curr_joint_pos, robot_2_curr_joint_pos])
        robot_1_grab_state = self.robot_interface.grab_command_robot1
        robot_2_grab_state = self.robot_interface.grab_command_robot2
        grab_state = np.where((robot_1_grab_state == 0) | (robot_2_grab_state == 0), 0, 1)
        self.motion_planner.env_nearest_sphere1 = np.squeeze(self.robot_simulation.sphere1_position * 1000) 
        self.motion_planner.env_nearest_sphere2 = np.squeeze(self.robot_simulation.sphere2_position * 1000) 
        return curr_joint_pos, grab_state


    