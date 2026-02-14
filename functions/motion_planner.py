import numpy as np
import time
import torch
import roboticstoolbox as rtb
from spatialmath import SE3, UnitQuaternion
from functions.utils import *
from functions.collision_optimizer import CollisionOptimizeSolver
from network.cross_attn import Cross_attn
from network.env_collision_bp import BP
from network.env_collision_attn import EnvAttn
from collections import deque
with open("output/origin_joint_angles.txt", "w") as f:
    f.write("")
with open("output/final_joint_angles.txt", "w") as f:
    f.write("")
with open("output/target_joints_robot1.txt", "w") as f:
    f.write("")
with open("output/target_joints_robot2.txt", "w") as f:
    f.write("")
with open("output/pos_delta.txt", "w") as f:
    f.write("")

class MotionPlanner:
    def __init__(self, args, robot, config, robot_interface, robot_simulation):
        self.args = args
        self.robot = robot
        self.robot_simulation = robot_simulation
        self.config = config
        self.action_queue = []
        self.stage_robot1 = 1
        self.stage_robot2 = 1
        self.stay_stage_robot1 = False
        self.stay_stage_robot2 = False
        self.final_stage_robot1 = False
        self.final_stage_robot2 = False
        self.is_grasp_stage_robot1 = False
        self.is_grasp_stage_robot2 = False
        self.is_release_stage_robot1 = False
        self.is_release_stage_robot2 = False
        self.first_iter = True
        self.backtrack = False
        self.collision_detection = False
        self.dualarm_collision_detection = False
        self.last_action = None
        self.verbose = True
        self.program_info = None
        self.robot_interface = robot_interface
        self.bounds_min = np.array(self.config['main']['bounds_min'])
        self.bounds_max = np.array(self.config['main']['bounds_max'])
        self.camera_base_robot = np.array(self.config['camera']['camera231_base_robot'])
        self.robot_base_camera = np.linalg.inv(self.camera_base_robot)
        self.T_offset_robot2 = np.loadtxt("models/dualarm_robot2_offset.txt")
        self.robot1_base_robot2 = np.linalg.inv(self.T_offset_robot2)
        self.last_og_gripper_action_robot1 = -1
        self.last_og_gripper_action_robot2 = -1
        self.last_joints_command_robot1 = np.array([-0.5, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]) # Initial grasp position
        self.last_joints_command_robot2 = np.array([0.0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]) # Initial grasp position
        self.joint_max_position = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.joint_min_position = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.delta_joint_limit = self.config['main']['interpolate_pos_step_size'] * 10
        self.self_collision_safety_threshold = 50.0
        self.back_action_robot1 = None
        self.back_action_robot2 = None
        self.self_collision_continuous_counter = 0
        self.required_continuous_count = 100
        self.first_avoidance = True
        self.stay_count_robot1 = self.stay_count_robot2 = 0

        self.history_joint_state = deque(maxlen=4)
        self_collision_model = Cross_attn(
            d_model=256,
            n_heads=4,
            d_ff=512,
            num_layers=1,
            n_features=14
        ).to("cuda")
        # self_collision_model_file = "models/best_cross_attn_model_S6_5_256_512_256_4_1_8.6_real.pt"
        self_collision_model_file = "models/best_cross_attn_model_real_with_gripper_7.81.pt"
        self._load_model_weights(self_collision_model, self_collision_model_file)
        env_collision_model = BP(input_dim=10, nodes_per_layer=[256, 256, 256, 256]).to("cuda")
        env_collision_model_file = "models/bp_env_collision_model_bp.pt"
        # env_collision_model = EnvAttn(d_model=256, n_heads=4, d_ff=512, num_layers=1, n_features=10, n_frames=5).to("cuda")
        # env_collision_model_file = "models/best_env_attn_model_4.74.pt"
        self._load_model_weights(env_collision_model, env_collision_model_file)
        self.OptimizeSolver = CollisionOptimizeSolver(self.args, self_collision_model, env_collision_model)
        self.env_nearest_sphere1 = np.array([1000, 1000, 1500])
        self.env_nearest_sphere2 = np.array([1000, 1000, 1500])
        self.env_nearest_point_robot1 = np.array([1000, 1000, 1500])
        self.env_nearest_point_robot2 = np.array([1000, 1000, 1500])
        self.self_collision_distance = 1000.0

    def _load_model_weights(self, model, model_file_path):
        if os.path.exists(model_file_path):
            if os.path.isfile(model_file_path):
                model.load_state_dict(torch.load(model_file_path))
                print(f'Model loaded from {model_file_path}')
            else:
                    print(f'No model file found at {model_file_path}')
        else:
            print(f'Checkpoint path {model_file_path} does not exist.')

    def _ik_calculate(self, target_position, target_orientation, initial_joint_pos=np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])): # x,y,z,w
        target_orientation = target_orientation[[3, 0, 1, 2]] # Quaternion format: w, x, y, z
        target_quaternion = UnitQuaternion(target_orientation)
        rotation_matrix = target_quaternion.R
        Tep = SE3.Trans(target_position) * SE3.Rt(rotation_matrix)
        sol = self.robot.ik_LM(Tep, q0=initial_joint_pos)
        return sol[0]

    def _update_stage(self, stage_robot1, stage_robot2):
        # update stage
        self.stage_robot1 = stage_robot1
        self.stage_robot2 = stage_robot2
        self.is_grasp_stage_robot1 = self.program_info['grasp_keypoints_robot1'][self.stage_robot1 - 1] != -1
        self.is_grasp_stage_robot2 = self.program_info['grasp_keypoints_robot2'][self.stage_robot2 - 1] != -1
        self.is_release_stage_robot1 = self.program_info['release_keypoints_robot1'][self.stage_robot1 - 1] != -1
        self.is_release_stage_robot2 = self.program_info['release_keypoints_robot2'][self.stage_robot2 - 1] != -1

        # can only be grasp stage or release stage or none
        assert self.is_grasp_stage_robot1 + self.is_release_stage_robot1 <= 1, "Cannot be both grasp and release stage"
        assert self.is_grasp_stage_robot2 + self.is_release_stage_robot2 <= 1, "Cannot be both grasp and release stage"

        if self.first_iter:  # ensure gripper is open for first time
            print(f"grasp_stage_robot1: {self.is_grasp_stage_robot1}")
            absolute_pos_robot1 = np.array([0.2693, -0.14713, 0.486882]) # Initial grasp position
            absolute_quat_robot1 = np.array([0.968912,-0.2474, 0.0, 0.0]) # Quaternion: x, y, z, w
            joints_command_robot1 = self._ik_calculate(absolute_pos_robot1, absolute_quat_robot1, self.last_joints_command_robot1)
            self.last_joints_command_robot1 = joints_command_robot1
            grab_command_robot1 = 0 # -1 for open, 1 for close, 0 for no change
            self._send_robot_command_robot1(joints_command_robot1, grab_command_robot1)
            
            print(f"grasp_stage_robot2: {self.is_grasp_stage_robot2}")
            absolute_pos_robot2 = np.array([0.30689, 0.0, 0.486882]) # Initial grasp position
            absolute_quat_robot2 = np.array([1, 0.0, 0.0, 0.0])
            joints_command_robot2 = self._ik_calculate(absolute_pos_robot2, absolute_quat_robot2, self.last_joints_command_robot2)
            self.last_joints_command_robot2 = joints_command_robot2
            grab_command_robot2 = 0 # -1 for open, 1 for close, 0 for no change
            self._send_robot_command_robot2(joints_command_robot2, grab_command_robot2)

        if self.is_grasp_stage_robot1:
            self._send_robot_command_robot1(self.last_joints_command_robot1, 0)

        if self.is_grasp_stage_robot2:
            self._send_robot_command_robot2(self.last_joints_command_robot2, 0)

        self._update_command()

        # clear action queue
        self.action_queue = []

    def _execute_action_dualarm(
            self,
            store_action_robot1,
            store_action_robot2
        ):
            """
            Moves the robot gripper to a target pose by specifying the absolute pose in the world frame and executes gripper action.

            Args:
                action (x, y, z, qx, qy, qz, qw, gripper_action): absolute target pose in the world frame + gripper action.
                precise (bool): whether to use small position and rotation thresholds for precise movement (robot would move slower).
            Returns:
                tuple: A tuple containing the position and rotation errors after reaching the target pose.
            """
            action_robot1 = np.array(store_action_robot1).copy()
            action_robot2 = np.array(store_action_robot2).copy()
            assert action_robot1.shape == (8,)
            assert action_robot2.shape == (8,)
            target_pose_robot1 = action_robot1[:7]
            target_pose_robot2 = action_robot2[:7]
            gripper_action_robot1 = action_robot1[7]
            gripper_action_robot2 = action_robot2[7]
            # print(f"action_robot1: {action_robot1}\naction_robot2: {action_robot2}")

            # ======================================
            # = status and safety check
            # ======================================
            if np.any(target_pose_robot1[:3] < self.bounds_min) \
                 or np.any(target_pose_robot1[:3] > self.bounds_max):
                self.verbose and print(f'{bcolors.WARNING}[main.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}')
                target_pose_robot1[:3] = np.clip(target_pose_robot1[:3], self.bounds_min, self.bounds_max)

            if np.any(target_pose_robot2[:3] < self.bounds_min) \
                 or np.any(target_pose_robot2[:3] > self.bounds_max):
                self.verbose and print(f'{bcolors.WARNING}[main.py | {get_clock_time()}] Target position is out of bounds, clipping to workspace bounds{bcolors.ENDC}')
                target_pose_robot2[:3] = np.clip(target_pose_robot2[:3], self.bounds_min, self.bounds_max)

            # ======================================
            # = move to target pose
            # ======================================
            self._move_to_waypoint_dualarm(target_pose_robot1, target_pose_robot2) 

            # ======================================
            # = apply gripper action
            # ======================================
            if gripper_action_robot1 == -1:# -1 for open, 1 for close, 0 for no change
                    self._open_gripper_robot1()
            elif gripper_action_robot1 == 1:
                    self._close_gripper_robot1()
            elif gripper_action_robot1 == 0:
                pass
            else:
                raise ValueError(f"Invalid robot1 gripper action: {gripper_action_robot1}")
            
            if gripper_action_robot2 == -1:# -1 for open, 1 for close, 0 for no change
                    self._open_gripper_robot2()
            elif gripper_action_robot2 == 1:
                    self._close_gripper_robot2()
            elif gripper_action_robot2 == 0:
                pass
            else:
                raise ValueError(f"Invalid robot2 gripper action: {gripper_action_robot2}")
            
            self._update_command()

    def _execute_gripper_action(self):
        robot1_error, robot2_error = self._check_reached_ee()
        pregrasp_pose_robot1 = self.last_action_robot1
        grasp_pose_robot1 = pregrasp_pose_robot1.copy() # Cannot set to 0 here, which would prevent reaching the specified position during execution, force 10 cycles.
        grasp_action_queue_robot1 = [grasp_pose_robot1.tolist()]
        action_robot1 = np.concatenate([grasp_pose_robot1, [0]])
        if self.is_grasp_stage_robot1 and robot1_error < self.config['main']['grasp_depth'] * 1.5:
            grasp_pose_robot1[:3] += T.quat2mat(pregrasp_pose_robot1[3:]) @ np.array([0, 0, self.config['main']['grasp_depth']])
            num_control_points_robot1 = get_linear_interpolation_steps(pregrasp_pose_robot1[:7], grasp_pose_robot1[:7], self.config['main']['interpolate_pos_step_size'], self.config['main']['interpolate_rot_step_size'])
            interp_poses_robot1 = linear_interpolate_poses(pregrasp_pose_robot1, grasp_pose_robot1, num_control_points_robot1)
            grasp_action_queue_robot1 = interp_poses_robot1.tolist()

        if self.is_release_stage_robot1:
            # self.final_stage_robot1 = True
            grasp_action_queue_robot1 = [grasp_pose_robot1.tolist()]
        self.gripper_action_robot1 = grasp_pose_robot1[:7]

        pregrasp_pose_robot2 = self.last_action_robot2
        grasp_pose_robot2 = pregrasp_pose_robot2.copy()
        grasp_action_queue_robot2 = [grasp_pose_robot2.tolist()]
        action_robot2 = np.concatenate([grasp_pose_robot2, [0]])
        if self.is_grasp_stage_robot2 and robot2_error < self.config['main']['grasp_depth'] * 1.5:
            grasp_pose_robot2[:3] += T.quat2mat(pregrasp_pose_robot2[3:]) @ np.array([0, 0, self.config['main']['grasp_depth']])
            num_control_points_robot2 = get_linear_interpolation_steps(pregrasp_pose_robot2[:7], grasp_pose_robot2[:7], self.config['main']['interpolate_pos_step_size'], self.config['main']['interpolate_rot_step_size'])
            interp_poses_robot2 = linear_interpolate_poses(pregrasp_pose_robot2, grasp_pose_robot2, num_control_points_robot2)
            grasp_action_queue_robot2 = interp_poses_robot2.tolist()

        if self.is_release_stage_robot2:
            # self.final_stage_robot2 = True
            grasp_action_queue_robot2 = [grasp_pose_robot2.tolist()]
        self.gripper_action_robot2 = grasp_pose_robot2[:7]

        while (len(grasp_action_queue_robot1) > 0 or len(grasp_action_queue_robot2) > 0):
            if (len(grasp_action_queue_robot1) > 0):
                next_action_robot1 = grasp_action_queue_robot1.pop(0)
            if (len(grasp_action_queue_robot2) > 0):
                next_action_robot2 = grasp_action_queue_robot2.pop(0)

            if self.is_grasp_stage_robot1:
                action_robot1 = np.concatenate([next_action_robot1, [1]])
            if self.is_grasp_stage_robot2:
                action_robot2 = np.concatenate([next_action_robot2, [1]])
            if self.is_release_stage_robot1:
                action_robot1 = np.concatenate([next_action_robot1, [-1]])
            if self.is_release_stage_robot2:
                action_robot2 = np.concatenate([next_action_robot2, [-1]])

            robot1_error, robot2_error = self._check_reached_ee()
            if robot1_error > self.config['main']['interpolate_pos_step_size'] * 20:
                action_robot1[-1] = 0
            if robot2_error > self.config['main']['interpolate_pos_step_size'] * 20:
                action_robot2[-1] = 0
            self._execute_action_dualarm(action_robot1, action_robot2)

    def _close_gripper_robot1(self):
        """
        Exposed interface: 1.0 for closed, -1.0 for open, 0.0 for no change
        """
        if self.last_og_gripper_action_robot1 == 1.0:
            return
        self.last_og_gripper_action_robot1 = 1.0
        absolute_pos = self.gripper_action_robot1[:3]
        absolute_quat = self.gripper_action_robot1[3:]
        joints_command_robot1 = self._ik_calculate(absolute_pos, absolute_quat, self.last_joints_command_robot1)
        grab_command = 1 # -1 for open, 1 for close, 0 for no change
        self._send_robot_command_robot1(joints_command_robot1, grab_command)
        
    def _close_gripper_robot2(self):
        """
        Exposed interface: 1.0 for closed, -1.0 for open, 0.0 for no change
        """
        if self.last_og_gripper_action_robot2 == 1.0:
            return
        self.last_og_gripper_action_robot2 = 1.0
        target_matrix_robot2 = T.convert_pose_quat2mat(self.gripper_action_robot2[:7])
        target_matrix_robot2 = self.robot1_base_robot2 @ target_matrix_robot2 # 4 x 4
        target_pose_robot2 = T.convert_pose_mat2quat(target_matrix_robot2) # 7 x 1
        joints_command_robot2 = self._ik_calculate(target_pose_robot2[:3], target_pose_robot2[3:], self.last_joints_command_robot2)
        grab_command = 1 # -1 for open, 1 for close, 0 for no change
        self._send_robot_command_robot2(joints_command_robot2, grab_command)

    def _open_gripper_robot1(self):
        if self.last_og_gripper_action_robot1 == -1.0:
            return
        self.last_og_gripper_action_robot1 = -1.0
        absolute_pos = self.gripper_action_robot1[:3]
        absolute_quat = self.gripper_action_robot1[3:]
        joints_command_robot1 = self._ik_calculate(absolute_pos, absolute_quat, self.last_joints_command_robot1)
        grab_command = -1 # -1 for open, 1 for close, 0 for no change
        self._send_robot_command_robot1(joints_command_robot1, grab_command)

    def _open_gripper_robot2(self):
        if self.last_og_gripper_action_robot2 == -1.0:
            return
        self.last_og_gripper_action_robot2 = -1.0
        target_matrix_robot2 = T.convert_pose_quat2mat(self.gripper_action_robot2[:7])
        target_matrix_robot2 = self.robot1_base_robot2 @ target_matrix_robot2 # 4 x 4
        target_pose_robot2 = T.convert_pose_mat2quat(target_matrix_robot2) # 7 x 1
        joints_command_robot2 = self._ik_calculate(target_pose_robot2[:3], target_pose_robot2[3:], self.last_joints_command_robot2)
        grab_command = -1 # -1 for open, 1 for close, 0 for no change
        self._send_robot_command_robot2(joints_command_robot2, grab_command)

    def _move_to_waypoint_dualarm(self, target_pose_world_robot1, target_pose_world_robot2):
        #! World coordinate system is base coordinate system
        self.world2robot_homo = self.camera_base_robot
        begin_time = time.time()

        target_matrix_robot2 = T.convert_pose_quat2mat(target_pose_world_robot2)
        target_matrix_robot2 = self.robot1_base_robot2 @ target_matrix_robot2 # 4 x 4
        target_pose_robot2 = T.convert_pose_mat2quat(target_matrix_robot2) # 7 x 1
        # target_pose_world_robot1[2] = max(target_pose_world_robot1[2], 0.05)
        # target_pose_robot2[2] = max(target_pose_robot2[2], 0.08)
        
        joints_command_robot1 = self._ik_calculate(target_pose_world_robot1[:3], target_pose_world_robot1[3:], self.last_joints_command_robot1)
        joints_command_robot2 = self._ik_calculate(target_pose_robot2[:3], target_pose_robot2[3:], self.last_joints_command_robot2)
        Current_joints_command = np.concatenate([joints_command_robot1, joints_command_robot2])
        with open('output/origin_joint_angles.txt', 'a') as origin_joint_angles:
                origin_joint_angles.write(', '.join(map(str, Current_joints_command)) + "\n\n")

        #- Optimize solve
        if len(self.history_joint_state) >= 4:
            history_joint_state_np = np.array(self.history_joint_state)
            last_joints_command_np = np.concatenate((self.last_joints_command_robot1, self.last_joints_command_robot2))
            Optimize_input_data = np.concatenate((history_joint_state_np, last_joints_command_np.reshape(1, -1)), axis=0)
            self._choose_robot_avoidance(last_joints_command_np, Current_joints_command)
            print(f"{bcolors.WARNING}robot_avoidance_operation: {self.OptimizeSolver.robot_avoidance_operation}{bcolors.ENDC}")
            optimized_last_step, self_collision_distance, env_collision_distance = self.OptimizeSolver.solve(Optimize_input_data, Current_joints_command, self.env_nearest_point_robot1, self.env_nearest_point_robot2)
            self.self_collision_distance = self_collision_distance
            self.env_collision_distance = env_collision_distance
            with open('output/final_joint_angles.txt', 'a') as final_joint_angles:
                final_joint_angles.write(' '.join(map(str, optimized_last_step[:7])) + " " + " ".join(map(str, self.env_nearest_point_robot1)) + "\n")
            joints_command_robot1 = optimized_last_step[:7]
            joints_command_robot2 = optimized_last_step[7:]

        robot1_pos_now = self.robot.fkine(self.last_joints_command_robot1).t
        robot2_pos_now = self.robot.fkine(self.last_joints_command_robot2).t
        pos_error_robot1 = np.linalg.norm(robot1_pos_now - target_pose_world_robot1[:3]) 
        pos_error_robot2 = np.linalg.norm(robot2_pos_now - target_pose_robot2[:3])
        with open('output/pos_delta.txt', 'a') as pos_delta_file:
            pos_delta_file.write(f"{pos_error_robot1} {pos_error_robot2}\n")
        
        for i in range(len(joints_command_robot1)):
            if abs(joints_command_robot1[i] - self.last_joints_command_robot1[i]) <= self.delta_joint_limit:
                joints_command_robot1[i] = joints_command_robot1[i]
            else:
                if joints_command_robot1[i] > self.last_joints_command_robot1[i]:
                    joints_command_robot1[i] = self.last_joints_command_robot1[i] + self.delta_joint_limit
                else:
                    joints_command_robot1[i] = self.last_joints_command_robot1[i] - self.delta_joint_limit

        for i in range(len(joints_command_robot2)):
            if abs(joints_command_robot2[i] - self.last_joints_command_robot2[i]) <= self.delta_joint_limit:
                joints_command_robot2[i] = joints_command_robot2[i]
            else:
                if joints_command_robot2[i] > self.last_joints_command_robot2[i]:
                    joints_command_robot2[i] = self.last_joints_command_robot2[i] + self.delta_joint_limit
                else:
                    joints_command_robot2[i] = self.last_joints_command_robot2[i] - self.delta_joint_limit

        with open('output/target_joints_robot1.txt', 'a') as target_joints_robot1:
                target_joints_robot1.write(', '.join(map(str, joints_command_robot1)) + "\n\n")
        with open('output/target_joints_robot2.txt', 'a') as target_joints_robot2:
                target_joints_robot2.write(', '.join(map(str, joints_command_robot2)) + "\n\n")
                    
        self.last_joints_command_robot1 = joints_command_robot1
        self.last_joints_command_robot2 = joints_command_robot2
        last_joints_command_np = np.concatenate((self.last_joints_command_robot1, self.last_joints_command_robot2))

        T_ee_robot1 = self.robot.fkine(joints_command_robot1).A
        T_ee_robot2 = self.robot.fkine(joints_command_robot2).A
        T_ee_robot2 = self.T_offset_robot2 @ T_ee_robot2
        self.last_action_robot1 = T.convert_pose_mat2quat(T_ee_robot1)
        self.last_action_robot2 = T.convert_pose_mat2quat(T_ee_robot2)
        with open('output/dualarm_ee_pose.txt', 'a') as dualarm_ee_pose:
            pose1_str = ', '.join(f"{x:.5f}" for x in self.last_action_robot1[:3])
            pose2_str = ', '.join(f"{x:.5f}" for x in self.last_action_robot2[:3])
            dualarm_ee_pose.write(f"{pose1_str}, {pose2_str}\n")
            
        # step the action
        self._send_robot_command_robot1(joints_command_robot1, self.last_og_gripper_action_robot1)
        self._send_robot_command_robot2(joints_command_robot2, self.last_og_gripper_action_robot2)
        self._update_command()
        self._update_history_joint_state(last_joints_command_np)

        finish_time = time.time()
        while_time = finish_time - begin_time
        # print(f'move to way point took {while_time:.4f} seconds\n')

    def _send_robot_command_robot1(self, joints_command_robot1, grab_command_robot1):
        # print(f"send command: {joints_command_robot1}")
        self.robot_interface.joints_command_robot1 = joints_command_robot1
        self.robot_interface.grab_command_robot1 = grab_command_robot1

    def _send_robot_command_robot2(self, joints_command_robot2, grab_command_robot2):
        self.robot_interface.joints_command_robot2 = joints_command_robot2
        self.robot_interface.grab_command_robot2 = grab_command_robot2
        
    def _update_command(self):
        # print(f"update command: {self.robot_interface.joints_command_robot1}")
        command_string = self.robot_interface._command2string2ROS()
        self.robot_interface._string2command(command_string)
        # print(f'relative_pos: {self.robot_interface.relative_pos}, relative_quat: {self.robot_interface.relative_quat}, grab_command: {self.robot_interface.grab_command}')
        if self.args.sim:
            self.robot_simulation.update_arm_positions(self.robot_interface.joints_command_robot1, 
                                                       self.robot_interface.joints_command_robot2, 
                                                       self.robot_interface.grab_command_robot1, 
                                                       self.robot_interface.grab_command_robot2)

    def _update_current_ee_pose(self, curr_ee_pose):
        self.curr_ee_pose = curr_ee_pose
        self.curr_ee_pose_robot1 = curr_ee_pose[:7]
        self.curr_ee_pose_robot2 = curr_ee_pose[7:]

    def _update_history_joint_state(self, history_joints_data):
        self.history_joint_state.append(history_joints_data)

    def _choose_robot_avoidance(self, last_joints_command_np, Current_joints_command):
        if self.self_collision_distance < self.self_collision_safety_threshold + 20:
            self.self_collision_continuous_counter += 1
            if self.self_collision_continuous_counter >= self.required_continuous_count:
                self.collision_detection = True
                if self.OptimizeSolver.robot_avoidance_operation == 0:
                    self.dualarm_collision_detection = True # Dual-arm collision avoidance, end current phase directly
                elif self.OptimizeSolver.robot_avoidance_operation == 1:
                    Current_joints_command[:7] = last_joints_command_np[:7]
                elif self.OptimizeSolver.robot_avoidance_operation == 2:
                    Current_joints_command[7:] = last_joints_command_np[7:]
        else:
            self.self_collision_continuous_counter = 0

    def _check_reached_ee(self):
        last_action_robot1 = self.back_action_robot1.copy()
        if self.program_info['grasp_keypoints_robot1'][max(self.stage_robot1 - 2, 0)] != -1:
            last_action_robot1[:3] += T.quat2mat(last_action_robot1[3:7]) @ np.array([0, 0, self.config['main']['grasp_depth']])
        pos_error_robot1 = np.linalg.norm(self.curr_ee_pose_robot1[:3] - last_action_robot1[:3])

        last_action_robot2 = self.back_action_robot2.copy()
        if self.program_info['grasp_keypoints_robot2'][max(self.stage_robot2 - 2, 0)] != -1:
            last_action_robot2[:3] += T.quat2mat(last_action_robot2[3:7]) @ np.array([0, 0, self.config['main']['grasp_depth']])
        pos_error_robot2 = np.linalg.norm(self.curr_ee_pose_robot2[:3] - last_action_robot2[:3])

        return pos_error_robot1, pos_error_robot2

    def _update_avoidance_robot(self, robot1_error, robot2_error):
        # First collision avoidance
        if self.first_avoidance:
            if robot1_error < robot2_error: # robot 1 is closer to the back action position
                self.OptimizeSolver.robot_avoidance_operation = 2 # robot 2 avoidance
                self.stay_stage_robot2 = True
            else:
                self.OptimizeSolver.robot_avoidance_operation = 1
                self.stay_stage_robot1 = True
            self.first_avoidance = False
            return

        # One manipulator reaches the final point and the other collision avoidance
        print(f"final stage:{self.final_stage_robot1}, {self.final_stage_robot2}")
        if self.final_stage_robot2 and self.stage_robot1 < self.program_info['num_stages']:
            self.OptimizeSolver.robot_avoidance_operation = 2
            return
        if self.final_stage_robot1 and self.stage_robot2 < self.program_info['num_stages']:
            self.OptimizeSolver.robot_avoidance_operation = 1
            return

    def _check_ee_stay_status(self):
        pos_diff_robot1 = np.linalg.norm(self.curr_ee_pose_robot1[:3] - self.last_action_robot1[:3])
        pos_diff_robot2 = np.linalg.norm(self.curr_ee_pose_robot2[:3] - self.last_action_robot2[:3])
        if pos_diff_robot1 < 0.0002:
            self.stay_count_robot1 += 1
        else:
            self.stay_count_robot1 = 0

        if self.stay_count_robot1 > 20:
            robot1_stay_exceeded = True
        else:
            robot1_stay_exceeded = False

        if pos_diff_robot2 < 0.0002:
            self.stay_count_robot2 += 1
        else:
            self.stay_count_robot2 = 0

        if self.stay_count_robot2 > 20:
            robot2_stay_exceeded = True
        else:
            robot2_stay_exceeded = False

        return robot1_stay_exceeded and robot2_stay_exceeded
