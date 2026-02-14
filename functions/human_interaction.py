import numpy as np
import time
import torch
import torch.nn as nn
import threading
import rclpy # type: ignore
import os
import copy
from functions.utils import *
from functions.collision_optimizer import CollisionOptimizeSolver
from functions.dualarm_simulation_manager import RobotArmSimulation
from network.cross_attn import Cross_attn
from network.env_collision_bp import BP
from network.env_collision_attn import EnvAttn
from functions.transform_utils import *
from pyros.pyros.robot_interface_dualarm import RobotInterface
from collections import deque
import roboticstoolbox as rtb
from spatialmath import SE3, UnitQuaternion

with open("output/origin_joint_angles.txt", "w") as f:
    f.write("")
with open("output/final_joint_angles.txt", "w") as f:
    f.write("")
with open("output/dualarm_ee_pose.txt", "w") as f:
    f.write("")
with open("output/opt_solve_time.txt", "w") as f:
    f.write("")
with open("output/joint_delta.txt", "w") as f:
    f.write("")
with open("output/pos_delta.txt", "w") as f:
    f.write("")

def start_ros2_node(node):
    while node._is_running and rclpy.ok():
        rclpy.spin_once(node, timeout_sec=0.01)
    node.destroy_node()

class HumanInteraction:
    def __init__(self, args):
        self.args = args
        global_config = get_config(config_path="configs/config.yaml")
        self.global_config = global_config
        self.config = global_config['main']
        self.robot = rtb.models.Panda()
        rclpy.init()
        self.robot_interface = RobotInterface()
        self.Init_joint_robot1 = np.array([0.00864473, -0.45876415,  0.03032723, -2.38522483,  0.01432525, 1.9266098,  0.81624648])
        self.Init_joint_robot2 = np.array([0, -0.7854, 0, -2.3562, 0, 1.57, 0.7854])
        self.robot_interface.robot_1_curr_joint_pos = self.Init_joint_robot1
        self.robot_interface.robot_2_curr_joint_pos = self.Init_joint_robot2
        radius = 0.1
        omega = 0.005

        Tep_robot1 = SE3.Trans(0.356 - radius * np.sin(omega * 2 * 1000), -0.15 - 2 * radius * np.cos(omega * 2 * 1000), 0.386) * SE3.OA([0, -1, 0], [0, 0, -1]) # OA: x, z
        # Tep_robot1 = SE3.Trans(0.456 + radius * np.sin(0), 0.0 - 2 * radius * np.sin(0), 0.386) * SE3.OA([-0.7071, 0.7071, 0], [0, 0, -1]) # OA: x, z
        Tep_robot2 = SE3.Trans(0.456 + radius * np.sin(omega * 1000), 0.1 + 2 * radius * np.cos(omega * 1000), 0.386) * SE3.OA([0, -1, 0], [0, 0, -1]) # OA: x, z

        sol_robot1 = self.robot.ik_LM(Tep_robot1, q0=self.Init_joint_robot1)
        sol_robot2 = self.robot.ik_LM(Tep_robot2, q0=self.Init_joint_robot2)
        print("sol_robot1: ", sol_robot1)
        self.Init_joint_robot1 = sol_robot1[0]
        self.Init_joint_robot2 = sol_robot2[0]

        self.Target_joint_combine = np.concatenate((self.Init_joint_robot1, self.Init_joint_robot2))
        if self.args.sim:
            self.robot_interface.joints_command_robot1 = self.Init_joint_robot1
            self.robot_interface.joints_command_robot2 = self.Init_joint_robot2

        self.joint_max_position = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
        self.joint_min_position = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
        self.history_joint_state = deque(maxlen=4)
        self.delta_joint_limit = self.global_config['main']['interpolate_pos_step_size'] * 10

        # self_collision_model = BP(input_dim=14, nodes_per_layer=[128, 64, 32, 16]).to("cuda")
        # self_collision_model_file = "models/best_bp_model_ral.pt"
        # self_collision_model = BP(input_dim=14, nodes_per_layer=[128, 128, 128, 128]).to("cuda")
        # self_collision_model_file = "models/best_bp_model_tro.pt"
        # self_collision_model = BP(input_dim=14, nodes_per_layer=[256, 256, 256, 256, 256]).to("cuda")
        # self_collision_model_file = "models/best_bp_model_jsdf.pt"

        self_collision_model = Cross_attn(
            d_model=256,
            n_heads=4,
            d_ff=512,
            num_layers=1,
            n_features=14
        ).to("cuda")
        self_collision_model_file = "models/best_cross_attn_model_S6_5_256_512_256_4_1_7.7.pt"
        # self_collision_model_file = "models/best_cross_attn_model_real_with_gripper_7.81.pt"
        self._load_model_weights(self_collision_model, self_collision_model_file)

        # env_collision_model = BP(input_dim=10, nodes_per_layer=[256, 256, 256, 256]).to("cuda")
        # env_collision_model_file = "models/bp_env_collision_model_bp.pt"
        env_collision_model = EnvAttn(d_model=256, n_heads=4, d_ff=512, num_layers=1, n_features=10, n_frames=5).to("cuda")
        env_collision_model_file = "models/best_env_attn_model_4.74.pt"
        self._load_model_weights(env_collision_model, env_collision_model_file)
        self.OptimizeSolver = CollisionOptimizeSolver(self.args, self_collision_model, env_collision_model)
        self.env_nearest_point_robot1 = np.array([1000, 1000, 1500])
        self.env_nearest_point_robot2 = np.array([1000, 1000, 1500])
        self.env_nearest_point = np.array([1000, 1000, 1500])

        if self.args.sim: 
            self.robot_simulation = RobotArmSimulation() 
            self.robot_simulation.draw_trajectories()
        else: self.robot_simulation = None

        self.max_joint_delta = 0
        self.max_pos_delta = 0
        self.avg_pos_delta = 0
        self.avg_opt_time = 0

    def _thread_start(self):
        robot_state_cal_thread = threading.Thread(target=self._calculate_robot_state)
        robot_state_cal_thread.daemon = True
        execute_thread = threading.Thread(target=self._Execute)
        execute_thread.daemon = True
        ros2_thread = threading.Thread(target=start_ros2_node, args=(self.robot_interface,))
        ros2_thread.daemon = True
        if self.args.sim: 
            simulation_thread = threading.Thread(target=self.robot_simulation.run)
            simulation_thread.daemon = True
        ros2_thread.start()  
        robot_state_cal_thread.start()
        execute_thread.start()
        if self.args.sim: simulation_thread.start()
        execute_thread.join()
        ros2_thread.join()
        robot_state_cal_thread.join()
        if self.args.sim: simulation_thread.join()

    def _Execute(self):
        print("Executing...")
        time.sleep(0.1)
        self.curr_joint_pos, self.grab_state = self._get_robot_data() if not self.args.sim else self._sim_get_robot_data()
        self.last_joints_command_robot1 = self.curr_joint_pos[:7]
        self.last_joints_command_robot2 = self.curr_joint_pos[-7:]
        self.last_joints_command_combine = np.concatenate((self.last_joints_command_robot1, self.last_joints_command_robot2))
        self.Target_joint_combine = np.concatenate((self.last_joints_command_robot1, self.last_joints_command_robot2))
        self.robot1_T_now = self.robot.fkine(self.last_joints_command_robot1)
        self.robot1_pos_now = self.robot1_T_now.t
        self.robot1_pos_target = self.robot1_T_now.t
        self.robot2_T_now = self.robot.fkine(self.last_joints_command_robot2)
        self.robot2_pos_now = self.robot2_T_now.t
        self.robot2_pos_target = self.robot2_T_now.t

        countdown = 2
        print(f"Starting countdown: {countdown} seconds...")
        while countdown > 0:
            print(f"{countdown} seconds remaining...")
            time.sleep(1)  # decrement by 1 each second
            countdown -= 1
        cnt = 1000
        radius = 0.1
        omega = 0.005
        while cnt < 1500:
        # while True:
            start_time = time.time()
            if len(self.history_joint_state) >= 4:
                cnt +=1
                print(f"\ncnt: {cnt}\n")
                if self.args.sim:
                    Tep_robot1 = SE3.Trans(0.356 - radius * np.sin(omega * 2 * cnt), -0.15 - 2 * radius * np.cos(omega * 2 * cnt), 0.386) * SE3.OA([0, -1, 0], [0, 0, -1]) # OA: x, z
                    # Tep_robot1 = SE3.Trans(0.456 + radius * np.sin(omega * 2 * cnt), -0.0 + 2 * radius * np.sin(omega * 2 * cnt), 0.386) * SE3.OA([-0.7071, 0.7071, 0], [0, 0, -1]) # OA: x, z
                    Tep_robot2 = SE3.Trans(0.456 + radius * np.sin(omega * cnt), 0.1 + 2 * radius * np.cos(omega * cnt), 0.386) * SE3.OA([0, -1, 0], [0, 0, -1]) # OA: x, z
                    sol_robot1 = self.robot.ik_LM(Tep_robot1, q0=self.last_joints_command_robot1)
                    sol_robot2 = self.robot.ik_LM(Tep_robot2, q0=self.last_joints_command_robot2)
                    self.robot1_pos_target = Tep_robot1.t
                    self.robot2_pos_target = Tep_robot2.t

                    with open('output/origin_joint_angles.txt', 'a') as origin_joint_angles:
                        origin_joint_angles.write(f"{sol_robot1}\n")
                    self.Target_joint_combine[:7] = sol_robot1[0]
                    self.Target_joint_combine[7:] = sol_robot2[0]
                else:
                    with open('output/origin_joint_angles.txt', 'a') as origin_joint_angles:
                        origin_joint_angles.write(f"{self.curr_joint_pos}\n")
                    Tep_robot1 = SE3.Trans(0.356 - radius * np.sin(omega * 2 * cnt), -0.15 - 2 * radius * np.cos(omega * 2 * cnt), 0.386) * SE3.OA([0, -1, 0], [0, 0, -1]) # OA: x, z
                    sol_robot1 = self.robot.ik_LM(Tep_robot1, q0=self.last_joints_command_robot1)
                    self.Target_joint_combine[:7] = sol_robot1[0]
                    self.Target_joint_combine[7:] = self.curr_joint_pos[7:]

                history_joint_state_np = np.array(self.history_joint_state)
                last_joints_command_np = np.concatenate((self.last_joints_command_robot1, self.last_joints_command_robot2))
                Optimize_input_data = np.concatenate((history_joint_state_np, last_joints_command_np.reshape(1, -1)), axis=0)
                if not self.args.sim:
                    Optimize_input_data[-1,7:] = self.curr_joint_pos[7:]

                opt_fstart_time = time.time()

                optimized_last_step, self_collision_distance, env_collision_distance = self.OptimizeSolver.solve(Optimize_input_data, self.Target_joint_combine, self.env_nearest_point_robot1, self.env_nearest_point_robot2)
                joints_command_robot1 = optimized_last_step[:7]
                joints_command_robot2 = optimized_last_step[7:]

                opt_finish_time = time.time()
                opt_cost_time = opt_finish_time - opt_fstart_time
                self.avg_opt_time += opt_cost_time

                delta_joint = np.concatenate((joints_command_robot1, joints_command_robot2)) - self.Target_joint_combine
                # delta_joint = joints_command_robot1 - self.Target_joint_combine[:7]
                delta_joint_item = np.dot(delta_joint.T, delta_joint)
                if delta_joint_item > self.max_joint_delta:
                    self.max_joint_delta = delta_joint_item
                
                self.robot1_T_now = self.robot.fkine(joints_command_robot1)
                self.robot1_pos_now = self.robot1_T_now.t
                self.robot2_T_now = self.robot.fkine(joints_command_robot2)
                self.robot2_pos_now = self.robot2_T_now.t
                pos_error = np.linalg.norm(self.robot1_pos_now - self.robot1_pos_target) + np.linalg.norm(self.robot2_pos_now - self.robot2_pos_target)
                if pos_error > self.max_pos_delta:
                    self.max_pos_delta = pos_error
                self.avg_pos_delta += pos_error
                print(f"\nmax joint delta: {self.max_joint_delta}\n")
                print(f"max pos_error: {self.max_pos_delta}\n")
                print(f"avg pos_error: {self.avg_pos_delta}\n")
                print(f"avg opt_time: {self.avg_opt_time}\n")
                with open('output/joint_delta.txt', 'a') as joint_delta_file:
                    joint_delta_file.write(f"{delta_joint_item}\n")
                with open('output/pos_delta.txt', 'a') as pos_delta_file:
                    pos_delta_file.write(f"{pos_error}\n")
                with open('output/opt_solve_time.txt', 'a') as opt_solve_time_file:
                    opt_solve_time_file.write(f"{opt_cost_time}\n")

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

                self.last_joints_command_robot1 = joints_command_robot1
                self.last_joints_command_robot2 = joints_command_robot2
                self.last_joints_command_combine = np.concatenate((joints_command_robot1, joints_command_robot2))
                print("self.last_joints_command_combine: ", self.last_joints_command_combine)
                print(f"{bcolors.FAIL}======================================================{bcolors.ENDC}")
                self._send_robot_command_robot1(joints_command_robot1, 0)
                self._send_robot_command_robot2(joints_command_robot2, 0)
                self._update_command()

            # self.history_joint_state.append(self.last_joints_command_combine)
            self._update_history_joint_state(self.curr_joint_pos)
            finish_time = time.time()
            cycle_time = finish_time - start_time
            if cycle_time < self.config['time_period']:
                time.sleep(self.config['time_period'] - cycle_time)

    def _calculate_robot_state(self):
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
                    self.T_offset_robot2 = np.loadtxt("models/dualarm_robot2_offset.txt")
                    self.ee_matrix_base_robot2 = self.T_offset_robot2 @ self.ee_matrix_base_robot2 # TODO: validate the value
                    ee_position_robot2 = self.ee_matrix_base_robot2[:3, 3]
                    ee_rotation_matrix_robot2 = self.ee_matrix_base_robot2[:3, :3]
                    ee_orientation_robot2 = mat2quat(ee_rotation_matrix_robot2)
                    ee_pose_robot2 = np.concatenate([ee_position_robot2, ee_orientation_robot2])
                    self.curr_ee_pose_robot2 = ee_pose_robot2
                    self.curr_ee_pose = np.concatenate([self.curr_ee_pose_robot1, self.curr_ee_pose_robot2]) # Robot1 coordinate

                    # self.env_nearest_point_robot1 = self.robot_interface.env_closest_point_robot1
                    # self.env_nearest_point_robot2 = self.robot_interface.env_closest_point_robot2
                    self.env_nearest_point_robot1 = self.env_nearest_point
                    self.env_nearest_point_robot2 = np.linalg.inv(self.T_offset_robot2) @ np.append(self.env_nearest_point_robot1 * 0.001, 1)
                    self.env_nearest_point_robot2 = self.env_nearest_point_robot2[:3] * 1000

                    with open('output/dualarm_ee_pose.txt', 'a') as dualarm_ee_pose:
                        pose1_str = ', '.join(f"{x:.5f}" for x in ee_pose_robot1)
                        pose2_str = ', '.join(f"{x:.5f}" for x in ee_pose_robot2)
                        dualarm_ee_pose.write(f"{pose1_str}, {pose2_str}\n")

            finish_time = time.time()
            cycle_time = finish_time - start_time
            if cycle_time < self.config['time_period']:
                time.sleep(self.config['time_period'] - cycle_time)
        print("_calculate_robot_state over")

    def _get_robot_data(self):
        self.robot_1_curr_joint_pos = self.robot_interface.robot_1_curr_joint_pos
        self.robot_2_curr_joint_pos = self.robot_interface.robot_2_curr_joint_pos
        curr_joint_pos = np.concatenate([self.robot_1_curr_joint_pos, self.robot_2_curr_joint_pos])
        robot_1_grab_state = self.robot_interface.robot_1_grab_state
        robot_2_grab_state = self.robot_interface.robot_2_grab_state
        grab_state = np.where((robot_1_grab_state == 0) | (robot_2_grab_state == 0), 0, 1)
        return curr_joint_pos, grab_state
    
    def _update_history_joint_state(self, history_joints_data):
        self.history_joint_state.append(history_joints_data)

    def _sim_get_robot_data(self):
        robot_1_curr_joint_pos = self.robot_interface.joints_command_robot1
        robot_2_curr_joint_pos = self.robot_interface.joints_command_robot2
        curr_joint_pos = np.concatenate([robot_1_curr_joint_pos, robot_2_curr_joint_pos])
        robot_1_grab_state = self.robot_interface.grab_command_robot1
        robot_2_grab_state = self.robot_interface.grab_command_robot2
        grab_state = np.where((robot_1_grab_state == 0) | (robot_2_grab_state == 0), 0, 1)
        self.env_nearest_point = self.robot_simulation.sphere_position * 1000
        return curr_joint_pos, grab_state
    
    def _send_robot_command_robot1(self, joints_command_robot1, grab_command_robot1):
        self.robot_interface.joints_command_robot1 = joints_command_robot1
        self.robot_interface.grab_command_robot1 = grab_command_robot1

    def _send_robot_command_robot2(self, joints_command_robot2, grab_command_robot2):
        self.robot_interface.joints_command_robot2 = joints_command_robot2
        self.robot_interface.grab_command_robot2 = grab_command_robot2

    def _update_command(self):
        # print(f"update command: {self.robot_interface.joints_command_robot1}")
        command_string = self.robot_interface._command2string2ROS()
        self.robot_interface._string2command(command_string)
        if self.args.sim:
            self.robot_simulation.update_arm_positions(self.robot_interface.joints_command_robot1, 
                                                       self.robot_interface.joints_command_robot2, 
                                                       self.robot_interface.grab_command_robot1, 
                                                       self.robot_interface.grab_command_robot2)

    def _load_model_weights(self, model, model_file_path):
        if os.path.exists(model_file_path):
            if os.path.isfile(model_file_path):
                model.load_state_dict(torch.load(model_file_path))
                print(f'Model loaded from {model_file_path}')
            else:
                    print(f'No model file found at {model_file_path}')
        else:
            print(f'Checkpoint path {model_file_path} does not exist.')