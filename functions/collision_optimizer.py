import numpy as np
import torch
import time
from functions.utils import *
from functions.transform_utils import *
from scipy.optimize import minimize
import roboticstoolbox as rtb
from spatialmath import SE3, UnitQuaternion

with open("output/self_distance_output.txt", "w") as f:
    f.write("")
with open("output/env_distance_output.txt", "w") as f:
    f.write("")
with open("output/optimize_joint_angles.txt", "w") as f:
    f.write("")
with open("output/joint_angles.txt", "w") as f:
    f.write("")

class BoundsCalculator:
    def __init__(self, cycletime):
        """
        Initialize BoundsCalculator class.

        :param cycletime: cycle time
        """
        self.joint_max_position = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float64)
        self.joint_min_position = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float64)
        self.jerk_limit = np.array([7500, 3750, 5000, 6250, 7500, 10000, 10000], dtype=np.float64)
        self.acc_limit = np.array([15, 7.5, 10, 12.5, 15, 20, 20], dtype=np.float64)
        self.vel_limit = np.array([2.1750, 2.1750, 2.1750, 2.1750, 2.6100, 2.6100, 2.6100], dtype=np.float64)
        self.cycletime = cycletime

        # Adjust limits based on cycle time
        self.vel_limit *= cycletime * 0.001  # rad/20ms
        self.acc_limit *= cycletime * cycletime * 0.001 * 0.001  # rad/20ms^2
        self.jerk_limit *= cycletime * cycletime * cycletime * 0.001 * 0.001 * 0.001  # rad/20ms^3

    def get_bounded_value(self, value, bound):
        """
        Ensure value is within given bounds.

        :param value: input value
        :param bound: bound value
        :return: value bounded within limits
        """
        if value > bound:
            return bound
        if value < -bound:
            return -bound
        return value

    def calculate_bounds_kinematics_deltaQ(self, prev_vel_robot=0, prev_acc_robot=0):
        """
        Calculate and return bounds.
        :param prev_vel_robot: robot's previous velocity
        :param prev_acc_robot: robot's previous acceleration
        :return: calculated bounds
        """
        robot_bounds = []
        for i in range(len(self.joint_max_position)):
            limited_vel_robot = self.get_bounded_value(prev_vel_robot, self.vel_limit[i])
            limited_acc_robot = self.get_bounded_value(prev_acc_robot, self.acc_limit[i])

            min_step_robot = limited_vel_robot + limited_acc_robot - self.jerk_limit[i]
            max_step_robot = limited_vel_robot + limited_acc_robot + self.jerk_limit[i]

            robot_bounds.append((min_step_robot, max_step_robot))

        return robot_bounds

class CollisionOptimizeSolver(object):
    def __init__(self, args, self_collision_model, env_collision_model):
        self.args = args
        self.num_frames = 5
        self.robot_avoidance_operation = 0
        self.self_collision_distance_threshold = 10.0
        self.self_collision_model = self_collision_model
        self.self_collision_model.eval()
        self.env_collision_distance_threshold = 100.0
        self.env_collision_model = env_collision_model
        self.env_collision_model.eval()
        self.robot = rtb.models.Panda()
        self.online_flag_self = True
        self.online_flag_env = False
        self.bounds_calculator = BoundsCalculator(cycletime=20)
        self.joint_max_position = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973], dtype=np.float32)
        self.joint_min_position = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973], dtype=np.float32)
        self.T_offset_robot2 = np.loadtxt("models/dualarm_robot2_offset.txt")
        self.T_offset_robot2_inv = np.linalg.inv(self.T_offset_robot2)
        self.self_distance = None
        self.robot1_decrease_ref = self.robot2_decrease_ref = 0

        self.joint_grad_norm = 0.1
        self.self_collision_joint_grad_scale = 30
        self.env_collision_joint_grad_scale = 10
        self.cbf_joint_grad_scale = 300
        self.maxiter = 2
        self.bound_scale = 10
        # self.minimize_method = 'L-BFGS-B'
        self.minimize_method = 'SLSQP'
        self.objective_function_verbose = False
        self.self_minimum_distance_verbose = False
        self.env_minimum_distance_verbose = False
        self.kinematics_verbose = False

        self.cnt = 0

    def normalize_input(self, input):
        input_robot1 = input[:,:,:7]
        input_robot2 = input[:,:,-7:]
        input_robot1 = (input_robot1 - self.joint_min_position) / (self.joint_max_position - self.joint_min_position)
        input_robot2 = (input_robot2 - self.joint_min_position) / (self.joint_max_position - self.joint_min_position)
        return np.concatenate((input_robot1, input_robot2), axis=2)

    def objective_function(self, x):
        delta_joint_robot1 = x[:7]
        delta_joint_robot2 = x[7:]
        delta_joint_item = np.dot(x.T, x)

        reference_joint_error = self.Current_state + x - self.reference_joint
        reference_joint_item = np.dot(reference_joint_error.T, reference_joint_error)
        
        joint_error_robot1 = self.desired_joint_velocity_robot1 - delta_joint_robot1
        joint_error_robot2 = self.desired_joint_velocity_robot2 - delta_joint_robot2
        joint_error_robot1 = np.dot(joint_error_robot1.T, joint_error_robot1)
        joint_error_robot2 = np.dot(joint_error_robot2.T, joint_error_robot2)

        #######################################
        #! Dawn IK objective
        # collision_cost = self.Dawn_IK_objective(x)

        #! Collision IK objective
        # collision_cost = 0.001 * self.Collision_IK_objective(x)

        # print("\n\ncollision_cost: ", collision_cost)
        # print("\n\n")

        #######################################

        return joint_error_robot1 + joint_error_robot2 + delta_joint_item + reference_joint_item * 5
        # return delta_joint_item + reference_joint_item * 5 + collision_cost

    def Dawn_IK_objective(self, x):
        weight = 100
        #! self-collision avoidance
        optimized_input = np.concatenate([self.History_state, (self.Current_state + x).reshape(1, 14)], axis=0).reshape(1, self.num_frames, 14)
        optimized_input = self.normalize_input(optimized_input)
        if self.online_flag_self:
            input_tensor = torch.tensor(optimized_input, dtype=torch.float32, device='cuda', requires_grad=True)
        else:
            input_tensor = torch.tensor(optimized_input[0, -1, :], dtype=torch.float32, device='cuda', requires_grad=True)
        start_time = time.time()
        output_tensor_self = self.self_collision_model.forward(input_tensor)
        output_tensor_self = output_tensor_self.detach().cpu().numpy()
        if output_tensor_self < self.self_collision_distance_threshold:
            r_self = weight / output_tensor_self** 2
        else:
            r_self = 0

        #! env collision avoidance robot1
        #----------------------------------------------------------#
        if self.online_flag_env:
            #! env_attn model
            env_point_expanded = np.repeat(self.env_point_robot1.reshape(1, 3), repeats=4, axis=0)
            history_combined = np.concatenate([self.History_state[:,:7], env_point_expanded], axis=1)
            current_joints = (self.Current_state[:7] + x[:7]).reshape(1, 7)
            current_state = np.concatenate([current_joints, self.env_point_robot1.reshape(1, 3)], axis=1)
            input_data = np.concatenate([history_combined, current_state], axis=0).reshape(1, self.num_frames, 10)
        else:
            #! env_bp model
            input_data = np.concatenate([x[:7] + self.Current_state_robot1.squeeze().squeeze(), self.env_point_robot1])
        #----------------------------------------------------------#
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device='cuda', requires_grad=True)
        output_tensor_env_robot1 = self.env_collision_model.forward(input_tensor)
        output_tensor_env_robot1 = output_tensor_env_robot1.detach().cpu().numpy()
        if output_tensor_env_robot1 < self.env_collision_distance_threshold:
            r_env_robot1 = weight / output_tensor_env_robot1 ** 2
        else:
            r_env_robot1 = 0

        #! env collision avoidance robot2
        #----------------------------------------------------------#
        if self.online_flag_env:
            #! env_attn model
            env_point_expanded = np.repeat(self.env_point_robot2.reshape(1, 3), repeats=4, axis=0)
            history_combined = np.concatenate([self.History_state[:,7:], env_point_expanded], axis=1)
            current_joints = (self.Current_state[7:] + x[7:]).reshape(1, 7)
            current_state = np.concatenate([current_joints, self.env_point_robot2.reshape(1, 3)], axis=1)
            input_data = np.concatenate([history_combined, current_state], axis=0).reshape(1, self.num_frames, 10)
        else:
            #! env_bp model
            input_data = np.concatenate([x[7:]+ self.Current_state_robot2.squeeze().squeeze(), self.env_point_robot2])
        #----------------------------------------------------------#
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device='cuda', requires_grad=True)
        output_tensor_env_robot2 = self.env_collision_model.forward(input_tensor)
        output_tensor_env_robot2 = output_tensor_env_robot2.detach().cpu().numpy()
        if output_tensor_env_robot2 < self.env_collision_distance_threshold:
            r_env_robot2 = weight / output_tensor_env_robot2** 2
        else:
            r_env_robot1 = 0

        return r_self + r_env_robot1 + r_env_robot1


    def Collision_IK_objective(self, x):
        weight = 100
        #! self-collision avoidance
        optimized_input = np.concatenate([self.History_state, (self.Current_state + x).reshape(1, 14)], axis=0).reshape(1, self.num_frames, 14)
        optimized_input = self.normalize_input(optimized_input)
        if self.online_flag_self:
            input_tensor = torch.tensor(optimized_input, dtype=torch.float32, device='cuda', requires_grad=True)
        else:
            input_tensor = torch.tensor(optimized_input[0, -1, :], dtype=torch.float32, device='cuda', requires_grad=True)
        start_time = time.time()
        output_tensor_self = self.self_collision_model.forward(input_tensor)
        output_tensor_self = output_tensor_self.detach().cpu().numpy()
        if output_tensor_self < self.self_collision_distance_threshold:
            Xc_self = (5 * self.self_collision_distance_threshold) **2 / (output_tensor_self)**2
            f_self = -1 * np.exp( -(Xc_self- 0)**2 / (2 * 2.5**2) ) + 0.01 * (Xc_self- 0) **4
        else:
            f_self = 0

        #! env collision avoidance robot1
        #----------------------------------------------------------#
        if self.online_flag_env:
            #! env_attn model
            env_point_expanded = np.repeat(self.env_point_robot1.reshape(1, 3), repeats=4, axis=0)
            history_combined = np.concatenate([self.History_state[:,:7], env_point_expanded], axis=1)
            current_joints = (self.Current_state[:7] + x[:7]).reshape(1, 7)
            current_state = np.concatenate([current_joints, self.env_point_robot1.reshape(1, 3)], axis=1)
            input_data = np.concatenate([history_combined, current_state], axis=0).reshape(1, self.num_frames, 10)
        else:
            #! env_bp model
            input_data = np.concatenate([x[:7] + self.Current_state_robot1.squeeze().squeeze(), self.env_point_robot1])
        #----------------------------------------------------------#
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device='cuda', requires_grad=True)
        output_tensor_env_robot1 = self.env_collision_model.forward(input_tensor)
        output_tensor_env_robot1 = output_tensor_env_robot1.detach().cpu().numpy()
        if output_tensor_env_robot1 < self.env_collision_distance_threshold:
            Xc_env_robot1 = (5 * self.env_collision_distance_threshold) **2 / (output_tensor_env_robot1)**2
            f_env_robot1 = -1 * np.exp( -(Xc_env_robot1- 0)**2 / (2 * 2.5**2) ) + 0.01 * (Xc_env_robot1- 0) **4
        else:
            f_env_robot1 = 0

        #! env collision avoidance robot2
        #----------------------------------------------------------#
        if self.online_flag_env:
            #! env_attn model
            env_point_expanded = np.repeat(self.env_point_robot2.reshape(1, 3), repeats=4, axis=0)
            history_combined = np.concatenate([self.History_state[:,7:], env_point_expanded], axis=1)
            current_joints = (self.Current_state[7:] + x[7:]).reshape(1, 7)
            current_state = np.concatenate([current_joints, self.env_point_robot2.reshape(1, 3)], axis=1)
            input_data = np.concatenate([history_combined, current_state], axis=0).reshape(1, self.num_frames, 10)
        else:
            #! env_bp model
            input_data = np.concatenate([x[7:]+ self.Current_state_robot2.squeeze().squeeze(), self.env_point_robot2])
        #----------------------------------------------------------#
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device='cuda', requires_grad=True)
        output_tensor_env_robot2 = self.env_collision_model.forward(input_tensor)
        output_tensor_env_robot2 = output_tensor_env_robot2.detach().cpu().numpy()
        if output_tensor_env_robot2 < self.env_collision_distance_threshold:
            Xc_env_robot2 = (5 * self.env_collision_distance_threshold) **2 / (output_tensor_env_robot2)**2
            f_env_robot2 = -1 * np.exp( -(Xc_env_robot2- 0)**2 / (2 * 2.5**2) ) + 0.01 * (Xc_env_robot2- 0) **4
        else:
            f_env_robot2 = 0

        return f_self + f_env_robot1 + f_env_robot2


    def self_minimun_distance_constraint(self, x):
        start_time = time.time()
        optimized_input = np.concatenate([self.History_state, (self.Current_state + x).reshape(1, 14)], axis=0).reshape(1, self.num_frames, 14)
        optimized_input = self.normalize_input(optimized_input)
        if self.online_flag_self:
            input_tensor = torch.tensor(optimized_input, dtype=torch.float32, device='cuda', requires_grad=True)
        else:
            input_tensor = torch.tensor(optimized_input[0, -1, :], dtype=torch.float32, device='cuda', requires_grad=True)
        start_time = time.time()
        output_tensor = self.self_collision_model.forward(input_tensor)
        end_time = time.time()
        zero_distance = torch.zeros_like(output_tensor, dtype=torch.float32).to("cuda")
        Safety_threshold = torch.full_like(output_tensor, self.self_collision_distance_threshold, dtype=torch.float32).to("cuda")
        distance = torch.nn.functional.mse_loss(output_tensor, zero_distance)
        distance.backward(retain_graph=True)
        if self.online_flag_self:
            joint_grad = input_tensor.grad[0, self.num_frames-1, :].cpu().numpy()
        else:
            joint_grad = input_tensor.grad.cpu().numpy()

        joint_grad_norm = np.linalg.norm(joint_grad)
        if joint_grad_norm > self.joint_grad_norm:
            joint_grad = (joint_grad / joint_grad_norm) * self.joint_grad_norm
        deltaQ_direction = x @ joint_grad.T
        ratio = output_tensor / Safety_threshold
        ln_value = torch.log(torch.clamp(ratio, min=1e-6)).detach().cpu().numpy().squeeze()
        minimun_distance_constraint_ineq = ln_value + deltaQ_direction

        constraint_grad = np.zeros_like(x)
        if ln_value < 0:
            self.self_minimum_distance_verbose = False
            constraint_grad = joint_grad * self.self_collision_joint_grad_scale
            if self.robot_avoidance_operation == 1:
                constraint_grad[7:] = 0
            elif self.robot_avoidance_operation == 2:
                constraint_grad[:7] = 0
            # print(f"{bcolors.FAIL}self output_tensor:\n{output_tensor}{bcolors.ENDC}")
            # print(f"{bcolors.WARNING}constraint_grad:\n{constraint_grad}{bcolors.ENDC}") 
        else:
            self.self_minimum_distance_verbose = False

        end_time = time.time()
        self.self_minimum_distance_verbose and print(f"{bcolors.WARNING}-----------------------------------{bcolors.ENDC}")
        self.self_minimum_distance_verbose and print(f"{bcolors.FAIL}self output_tensor:\n{output_tensor}{bcolors.ENDC}")
        self.self_minimum_distance_verbose and print(f"{bcolors.WARNING}delta_joints:\n{x}{bcolors.ENDC}")
        self.self_minimum_distance_verbose and print(f"{bcolors.WARNING}input_grad:\n{joint_grad}{bcolors.ENDC}")
        self.self_minimum_distance_verbose and print(f"{bcolors.WARNING}ln_value:\n{ln_value}{bcolors.ENDC}")
        self.self_minimum_distance_verbose and print(f"{bcolors.WARNING}deltaQ_direction:\n{deltaQ_direction}{bcolors.ENDC}")
        self.self_minimum_distance_verbose and print(f"{bcolors.WARNING}constraint_grad:\n{constraint_grad}{bcolors.ENDC}")        
        return minimun_distance_constraint_ineq, constraint_grad

    def self_minimun_distance_constraint_cbf(self, x):
        """
        construct dualarm CBF constraint: h(x_{t+1}) >= gamma·h(x_t)
        """
        #- compute current state safety function h(x_t)
        current_input = np.concatenate([self.History_state, (self.Current_state).reshape(1, 14)], axis=0).reshape(1, self.num_frames, 14)
        current_input = self.normalize_input(current_input)
        current_h = self._compute_self_safety_function(current_input)  # h(x_t)
        
        #- compute next state safety function h(x_{t+1})
        next_input = np.concatenate([self.History_state, (self.Current_state + x).reshape(1, 14)], axis=0).reshape(1, self.num_frames, 14)
        next_input = self.normalize_input(next_input)
        next_h = self._compute_self_safety_function(next_input)  # h(x_{t+1})
        
        #- compute CBF constraint: h(x_{t+1}) - γ·h(x_t) >= 0
        self.cbf_gamma = 0.9
        cbf_constraint = next_h + self.cbf_gamma * current_h
        
        #- compute gradient of CBF constraint
        constraint_grad = np.zeros_like(x)
        input_tensor = torch.tensor(current_input, dtype=torch.float32, device='cuda', requires_grad=True)
        output_tensor = self.self_collision_model.forward(input_tensor)
        zero_distance = torch.zeros_like(output_tensor, dtype=torch.float32).to("cuda")
        distance = torch.nn.functional.mse_loss(output_tensor, zero_distance)
        distance.backward(retain_graph=True)
        joint_grad = input_tensor.grad[0, self.num_frames-1, :].cpu().numpy()
        joint_grad_norm = np.linalg.norm(joint_grad)
        if joint_grad_norm > self.joint_grad_norm:
            joint_grad = (joint_grad / joint_grad_norm) * self.joint_grad_norm
        constraint_grad = joint_grad * self.cbf_joint_grad_scale
        
        return cbf_constraint, constraint_grad

    def _compute_self_safety_function(self, input_data):
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device='cuda')
        with torch.no_grad():
            distance = self.self_collision_model(input_tensor)
        return distance.item() - self.self_collision_distance_threshold

    def env_robot1_minimun_distance_constraint(self, x):
        start_time = time.time()
        #----------------------------------------------------------#
        if self.online_flag_env:
            #! env_attn model
            env_point_expanded = np.repeat(self.env_point_robot1.reshape(1, 3), repeats=4, axis=0)
            history_combined = np.concatenate([self.History_state[:,:7], env_point_expanded], axis=1)
            current_joints = (self.Current_state[:7] + x[:7]).reshape(1, 7)
            current_state = np.concatenate([current_joints, self.env_point_robot1.reshape(1, 3)], axis=1)
            input_data = np.concatenate([history_combined, current_state], axis=0).reshape(1, self.num_frames, 10)
        else:
            #! env_bp model
            input_data = np.concatenate([x[:7] + self.Current_state_robot1.squeeze().squeeze(), self.env_point_robot1])
        #----------------------------------------------------------#
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device='cuda', requires_grad=True)
        output_tensor = self.env_collision_model.forward(input_tensor)
        zero_distance = torch.zeros_like(output_tensor, dtype=torch.float32).to("cuda")
        Safety_threshold = torch.full_like(output_tensor, self.env_collision_distance_threshold, dtype=torch.float32).to("cuda")
        distance = torch.nn.functional.mse_loss(output_tensor, zero_distance)
        distance.backward(retain_graph=True)
        if self.online_flag_env:
            #! env_attn model
            joint_grad = input_tensor.grad[0, self.num_frames-1, :7].cpu().numpy().squeeze()
        else:
            #! env_bp model
            input_grad = input_tensor.grad.cpu().numpy()
            joint_grad = input_grad[:7]
        joint_grad_norm = np.linalg.norm(joint_grad)
        if joint_grad_norm > self.joint_grad_norm:
            joint_grad = (joint_grad / joint_grad_norm) * self.joint_grad_norm
        deltaQ_direction = x[:7] @ joint_grad.T
        ratio = output_tensor / Safety_threshold
        ln_value = torch.log(torch.clamp(ratio, min=1e-6)).detach().cpu().numpy().squeeze()
        minimun_distance_constraint_ineq = ln_value + deltaQ_direction

        constraint_grad = np.zeros_like(x)
        constraint_grad[:7] = joint_grad * self.env_collision_joint_grad_scale

        if minimun_distance_constraint_ineq < 0:
            self.env_minimum_distance_verbose = True
        else:
            self.env_minimum_distance_verbose = False

        end_time = time.time()
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}-----------------------------------{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.FAIL}robot1 output_tensor:\n{output_tensor}{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}delta_joints:\n{x}{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}input_grad:\n{joint_grad}{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}ln_value:\n{ln_value}{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}deltaQ_direction:\n{deltaQ_direction}{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}constraint_grad:\n{constraint_grad}{bcolors.ENDC}")        
        return minimun_distance_constraint_ineq, constraint_grad

    def env_robot2_minimun_distance_constraint(self, x):
        start_time = time.time()        
        #----------------------------------------------------------#
        if self.online_flag_env:
            #! env_attn model
            env_point_expanded = np.repeat(self.env_point_robot2.reshape(1, 3), repeats=4, axis=0)
            history_combined = np.concatenate([self.History_state[:,7:], env_point_expanded], axis=1)
            current_joints = (self.Current_state[7:] + x[7:]).reshape(1, 7)
            current_state = np.concatenate([current_joints, self.env_point_robot2.reshape(1, 3)], axis=1)
            input_data = np.concatenate([history_combined, current_state], axis=0).reshape(1, self.num_frames, 10)
        else:
            #! env_bp model
            input_data = np.concatenate([x[7:]+ self.Current_state_robot2.squeeze().squeeze(), self.env_point_robot2])
        #----------------------------------------------------------#
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device='cuda', requires_grad=True)
        output_tensor = self.env_collision_model.forward(input_tensor)
        zero_distance = torch.zeros_like(output_tensor, dtype=torch.float32).to("cuda")
        Safety_threshold = torch.full_like(output_tensor, self.env_collision_distance_threshold, dtype=torch.float32).to("cuda")
        distance = torch.nn.functional.mse_loss(output_tensor, zero_distance)
        distance.backward(retain_graph=True)
        if self.online_flag_env:
            joint_grad = input_tensor.grad[0, self.num_frames-1, :7].cpu().numpy().squeeze()
        else:
            input_grad = input_tensor.grad.cpu().numpy()
            joint_grad = input_grad[:7]
        joint_grad_norm = np.linalg.norm(joint_grad)
        if joint_grad_norm > self.joint_grad_norm:
            joint_grad = (joint_grad / joint_grad_norm) * self.joint_grad_norm
        deltaQ_direction = x[7:] @ joint_grad.T
        ratio = output_tensor / Safety_threshold
        ln_value = torch.log(torch.clamp(ratio, min=1e-6)).detach().cpu().numpy().squeeze()
        minimun_distance_constraint_ineq = ln_value + deltaQ_direction

        constraint_grad = np.zeros_like(x)
        constraint_grad[7:] = joint_grad * self.env_collision_joint_grad_scale
        if minimun_distance_constraint_ineq < 0:
            self.env_minimum_distance_verbose = True
        else:
            self.env_minimum_distance_verbose = False

        end_time = time.time()
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}-----------------------------------{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.FAIL}robot2 output_tensor:\n{output_tensor}{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}delta_joints:\n{x}{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}input_grad:\n{joint_grad}{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}ln_value:\n{ln_value}{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}deltaQ_direction:\n{deltaQ_direction}{bcolors.ENDC}")
        self.env_minimum_distance_verbose and print(f"{bcolors.WARNING}constraint_grad:\n{constraint_grad}{bcolors.ENDC}")        
        return minimun_distance_constraint_ineq, constraint_grad

    def env_robot1_minimun_distance_constraint_cbf(self, x):
        """
        construct robot1 CBF constraint: h(x_{t+1}) >= gamma·h(x_t)
        """
        #- compute current state safety function h(x_t)
        if self.online_flag_env:
            env_point_expanded = np.repeat(self.env_point_robot1.reshape(1, 3), repeats=4, axis=0)
            history_combined = np.concatenate([self.History_state[:,:7], env_point_expanded], axis=1)
            current_state = np.concatenate([self.Current_state[:7].reshape(1, 7), self.env_point_robot1.reshape(1, 3)], axis=1)
            current_input = np.concatenate([history_combined, current_state], axis=0).reshape(1, self.num_frames, 10)
            current_h = self._compute_env_safety_function(current_input)  # h(x_t)
        else:
            current_input = np.concatenate([self.Current_state_robot1.squeeze().squeeze(), self.env_point_robot1])
            current_h = self._compute_env_safety_function(current_input)
        
        #- compute next state safety function h(x_{t+1})
        if self.online_flag_env:
            next_state = self.Current_state_robot1.squeeze() + x[:7]
            next_state = np.concatenate([next_state.reshape(1, 7), self.env_point_robot1.reshape(1, 3)], axis=1)
            next_input = np.concatenate([history_combined, next_state], axis=0).reshape(1, self.num_frames, 10)
            next_h = self._compute_env_safety_function(next_input)  # h(x_{t+1})
        else:
            next_input = np.concatenate([x[:7]+ self.Current_state_robot1.squeeze().squeeze(), self.env_point_robot1])
            next_h = self._compute_env_safety_function(next_input)
            
        #- compute CBF constraint: h(x_{t+1}) - γ·h(x_t) >= 0
        self.cbf_gamma = 0.9
        cbf_constraint = next_h + self.cbf_gamma * current_h
        
        #- compute gradient of CBF constraint
        constraint_grad = np.zeros_like(x)
        input_tensor = torch.tensor(current_input, dtype=torch.float32, device='cuda', requires_grad=True)
        output_tensor = self.env_collision_model.forward(input_tensor)
        zero_distance = torch.zeros_like(output_tensor, dtype=torch.float32).to("cuda")
        distance = torch.nn.functional.mse_loss(output_tensor, zero_distance)
        distance.backward(retain_graph=True)
        if self.online_flag_env:
            joint_grad = input_tensor.grad[0, self.num_frames-1, :7].cpu().numpy()
        else:
            input_grad = input_tensor.grad.cpu().numpy()
            joint_grad = input_grad[:7]
        joint_grad_norm = np.linalg.norm(joint_grad)
        if joint_grad_norm > self.joint_grad_norm:
            joint_grad = (joint_grad / joint_grad_norm) * self.joint_grad_norm
        constraint_grad[:7] = joint_grad * self.cbf_joint_grad_scale
        
        if self.env_minimum_distance_verbose:
            print(f"CBF Constraint: {cbf_constraint:.6f} (y={self.cbf_gamma})")
            print(f"h(x_t): {current_h:.6f}, h(x_{{t+1}}): {next_h:.6f}")
        
        return cbf_constraint, constraint_grad

    def env_robot2_minimun_distance_constraint_cbf(self, x):
        """
        construct robot2 CBF constraint: h(x_{t+1}) >= gamma·h(x_t)
        """
        #- compute current state safety function h(x_t)
        if self.online_flag_env:
            env_point_expanded = np.repeat(self.env_point_robot2.reshape(1, 3), repeats=4, axis=0)
            history_combined = np.concatenate([self.History_state[:,7:], env_point_expanded], axis=1)
            current_state = np.concatenate([self.Current_state[7:].reshape(1, 7), self.env_point_robot2.reshape(1, 3)], axis=1)
            current_input = np.concatenate([history_combined, current_state], axis=0).reshape(1, self.num_frames, 10)
            current_h = self._compute_env_safety_function(current_input)  # h(x_t)
        else:
            current_input = np.concatenate([self.Current_state_robot2.squeeze().squeeze(), self.env_point_robot2])
            current_h = self._compute_env_safety_function(current_input)
        
        #- compute next state safety function h(x_{t+1})
        if self.online_flag_env:
            next_state = self.Current_state_robot2.squeeze() + x[7:]
            next_state = np.concatenate([next_state.reshape(1, 7), self.env_point_robot2.reshape(1, 3)], axis=1)
            next_input = np.concatenate([history_combined, next_state], axis=0).reshape(1, self.num_frames, 10)
            next_h = self._compute_env_safety_function(next_input)  # h(x_{t+1})
        else:
            next_input = np.concatenate([x[:7]+ self.Current_state_robot2.squeeze().squeeze(), self.env_point_robot2])
            next_h = self._compute_env_safety_function(next_input)
        
        #- compute CBF constraint: h(x_{t+1}) - γ·h(x_t) >= 0
        self.cbf_gamma = 0.9
        cbf_constraint = next_h + self.cbf_gamma * current_h
        
        #- compute gradient of CBF constraint
        constraint_grad = np.zeros_like(x)
        input_tensor = torch.tensor(current_input, dtype=torch.float32, device='cuda', requires_grad=True)
        output_tensor = self.env_collision_model.forward(input_tensor)
        zero_distance = torch.zeros_like(output_tensor, dtype=torch.float32).to("cuda")
        distance = torch.nn.functional.mse_loss(output_tensor, zero_distance)
        distance.backward(retain_graph=True)
        if self.online_flag_env:
            joint_grad = input_tensor.grad[0, self.num_frames-1, :7].cpu().numpy()
        else:
            input_grad = input_tensor.grad.cpu().numpy()
            joint_grad = input_grad[:7]
        joint_grad_norm = np.linalg.norm(joint_grad)
        if joint_grad_norm > self.joint_grad_norm:
            joint_grad = (joint_grad / joint_grad_norm) * self.joint_grad_norm
        constraint_grad[7:] = joint_grad * self.cbf_joint_grad_scale
        
        if self.env_minimum_distance_verbose:
            print(f"CBF Constraint: {cbf_constraint:.6f} (y={self.cbf_gamma})")
            print(f"h(x_t): {current_h:.6f}, h(x_{{t+1}}): {next_h:.6f}")
        
        return cbf_constraint, constraint_grad

    def _compute_env_safety_function(self, input_data):
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device='cuda')
        with torch.no_grad():
            distance = self.env_collision_model(input_tensor)
        return distance.item() - self.env_collision_distance_threshold

    def setup_constraints(self):
        constraints = []
        if self.args.enable_self_collision_avoidance:
            #! ours
            constraints.append(
                {
                    'type': 'ineq', 
                    'fun': lambda x: self.self_minimun_distance_constraint(x)[0],
                    'jac': lambda x: self.self_minimun_distance_constraint(x)[1]
                }
            )
            #! cbf
            # constraints.append(
            #     {
            #         'type': 'ineq', 
            #         'fun': lambda x: self.self_minimun_distance_constraint_cbf(x)[0],
            #         'jac': lambda x: self.self_minimun_distance_constraint_cbf(x)[1]
            #     }
            # )

        if self.args.enable_env_collision_avoidance:
            #! ours
            constraints.append(
                {
                    'type': 'ineq', 
                    'fun': lambda x: self.env_robot1_minimun_distance_constraint(x)[0],
                    'jac': lambda x: self.env_robot1_minimun_distance_constraint(x)[1]
                }
            )
            constraints.append(
                {
                    'type': 'ineq', 
                    'fun': lambda x: self.env_robot2_minimun_distance_constraint(x)[0],
                    'jac': lambda x: self.env_robot2_minimun_distance_constraint(x)[1]
                }
            )
            #! CBF
            # constraints.append(
            #     {
            #         'type': 'ineq', 
            #         'fun': lambda x: self.env_robot1_minimun_distance_constraint_cbf(x)[0],
            #         'jac': lambda x: self.env_robot1_minimun_distance_constraint_cbf(x)[1]
            #     }
            # )
            # constraints.append(
            #     {
            #         'type': 'ineq', 
            #         'fun': lambda x: self.env_robot2_minimun_distance_constraint_cbf(x)[0],
            #         'jac': lambda x: self.env_robot2_minimun_distance_constraint_cbf(x)[1]
            #     }
            # )

        return constraints

    def solve(self, init_joints_data, Target_joints_data, env_point_robot1, env_point_robot2):
        self.cnt +=1
        if self.online_flag_env:
            self.env_point_robot1 = env_point_robot1 * 0.001 # base robot1
            self.env_point_robot2 = env_point_robot2 * 0.001 # base robot2
        else:
            self.env_point_robot1 = env_point_robot1
            self.env_point_robot2 = env_point_robot2
        init_joints_data = init_joints_data.reshape(1, self.num_frames, 14)
        self.History_state = init_joints_data[0, :self.num_frames-1, :]
        self.Current_state = init_joints_data[0, self.num_frames-1, :]
        self.Current_state_robot1 = init_joints_data[0, self.num_frames-1, :7]
        self.Current_state_robot2 = init_joints_data[0, self.num_frames-1, 7:]
        self.Current_Te_robot1 = self.robot.fkine(self.Current_state_robot1)
        self.Current_Te_robot2 = self.robot.fkine(self.Current_state_robot2)
        self.Current_pose_robot1 = np.concatenate([self.Current_Te_robot1.t, self.Current_Te_robot1.eul()]) # x, y, z, ex, ey, ez
        self.Current_pose_robot2 = np.concatenate([self.Current_Te_robot2.t, self.Current_Te_robot2.eul()]) # x, y, z, ex, ey, ez
        self.Current_R_robot1 = self.Current_Te_robot1.R
        self.Current_R_robot2 = self.Current_Te_robot2.R

        init_joints_data_normalize = self.normalize_input(init_joints_data)
        if self.online_flag_self:
            final_input_tensor = torch.tensor(init_joints_data_normalize, dtype=torch.float32).to("cuda")
        else:
            final_input_tensor = torch.tensor(init_joints_data_normalize[0,-1,:], dtype=torch.float32).to("cuda")
        self_collision_output_tensor = self.self_collision_model(final_input_tensor)
        self_collision_output = self_collision_output_tensor.detach().cpu().numpy()

        if self.self_distance is not None and self_collision_output < (self.self_collision_distance_threshold + 20):
            print(f"self_collision_output: {self_collision_output} last self distance: {self.self_distance} ")
            if self_collision_output > self.self_collision_distance_threshold and self.self_collision_distance_threshold > self.self_distance:
                if self.robot_avoidance_operation == 1:
                    self.robot1_decrease_ref = 1
                elif self.robot_avoidance_operation == 2:
                    self.robot2_decrease_ref = 1
                else:
                    self.robot1_decrease_ref = self.robot2_decrease_ref = 1
        self.self_distance = self_collision_output
        print(f"{bcolors.WARNING}robot1_decrease_ref: {self.robot1_decrease_ref} robot2_decrease_ref: {self.robot2_decrease_ref}{bcolors.ENDC}")
        self.reference_joint = Target_joints_data.copy()
        self.reference_joint[:7] = Target_joints_data[:7] + self.robot1_decrease_ref * (self.Current_state_robot1 - Target_joints_data[:7])
        self.reference_joint[7:] = Target_joints_data[7:] + self.robot2_decrease_ref * (self.Current_state_robot2 - Target_joints_data[7:])
        if self.robot1_decrease_ref > 0:
            self.robot1_decrease_ref -= 0.05
        if self.robot2_decrease_ref > 0:
            self.robot2_decrease_ref -= 0.05
        self.robot1_decrease_ref = max(0, self.robot1_decrease_ref)
        self.robot2_decrease_ref = max(0, self.robot2_decrease_ref)

        self.Target_Te_robot1 = self.robot.fkine(Target_joints_data[:7])
        self.Target_Te_robot2 = self.robot.fkine(Target_joints_data[7:])
        self.Target_pose_robot1 = np.concatenate([self.Target_Te_robot1.t, self.Target_Te_robot1.eul()]) # x, y, z, ex, ey, ez
        self.Target_pose_robot2 = np.concatenate([self.Target_Te_robot2.t, self.Target_Te_robot2.eul()]) # x, y, z, ex, ey, ez
        self.Target_R_robot1 = self.Target_Te_robot1.R
        self.Target_R_robot2 = self.Target_Te_robot2.R

        self.Jacobian_robot1 = self.robot.jacob0(self.Current_state_robot1)
        self.Jacobian_robot2 = self.robot.jacob0(self.Current_state_robot2)
        self.Jacobian_robot1_pinv = np.linalg.pinv(self.Jacobian_robot1)
        self.Jacobian_robot2_pinv = np.linalg.pinv(self.Jacobian_robot2)

        R_error_robot1 = self.Current_R_robot1 @ self.Target_R_robot1.T
        R_error_robot2 = self.Current_R_robot2 @ self.Target_R_robot2.T
        eul_error_robot1 = mat2euler(R_error_robot1)
        eul_error_robot2 = mat2euler(R_error_robot2)

        pose_error_robot1 = np.concatenate([self.Target_Te_robot1.t - self.Current_Te_robot1.t, eul_error_robot1])
        pose_error_robot2 = np.concatenate([self.Target_Te_robot2.t - self.Current_Te_robot2.t, eul_error_robot2])
        self.desired_joint_velocity_robot1 = self.Jacobian_robot1_pinv.dot(pose_error_robot1)
        self.desired_joint_velocity_robot2 = self.Jacobian_robot2_pinv.dot(pose_error_robot2)
        norm_robot1 = np.linalg.norm(self.desired_joint_velocity_robot1)
        norm_robot2 = np.linalg.norm(self.desired_joint_velocity_robot2)
        if norm_robot1 > 0.01:
            self.desired_joint_velocity_robot1 = (self.desired_joint_velocity_robot1 / norm_robot1) * 0.01
        if norm_robot2 > 0.01:
            self.desired_joint_velocity_robot2 = (self.desired_joint_velocity_robot2 / norm_robot2) * 0.01

        deltaQ_bounds = self.bounds_calculator.calculate_bounds_kinematics_deltaQ()
        deltaQ_bounds = [(low * self.bound_scale, high * self.bound_scale) for low, high in deltaQ_bounds]
        bounds = np.concatenate([deltaQ_bounds, deltaQ_bounds], axis=0)
        
        start_time = time.time()
        constraints = self.setup_constraints()
        init_sol = np.zeros(14)
        result = minimize(
            self.objective_function, 
            init_sol, 
            method=self.minimize_method, 
            bounds=bounds,
            constraints=constraints,
            tol=1e-6,
            options={
                'maxiter': self.maxiter,
                'initial_tr_radius': 0.01,
            }
        )
        end_time = time.time()
        elapsed_time = end_time - start_time
        optimized_solution = result.x
        
        print(f"{bcolors.OKBLUE}Optimize Solution:\n{optimized_solution}{bcolors.ENDC}")
        print(f"{bcolors.WARNING}Time Taken for Optimization: {elapsed_time:.4f} seconds{bcolors.ENDC}")
        if np.all(np.abs(optimized_solution) < 1e-6):
            print("Optimization result is close to zero, no significant movement detected.")
        else:
            init_joints_data[0, self.num_frames-1, :] += optimized_solution
        init_joints_data_normalize = self.normalize_input(init_joints_data)

        if self.online_flag_self:
            final_input_tensor = torch.tensor(init_joints_data_normalize, dtype=torch.float32).to("cuda")
        else:
            final_input_tensor = torch.tensor(init_joints_data_normalize[0,-1,:], dtype=torch.float32).to("cuda")
        self_collision_output_tensor = self.self_collision_model(final_input_tensor)
        self_collision_output = self_collision_output_tensor.detach().cpu().numpy()
        print(f"Self Collision Model Output After Optimization for Sample: {bcolors.FAIL}{self_collision_output}{bcolors.ENDC}\n")

        #----------------------------------------------------------#
        if self.online_flag_env:
            #! env_attn model
            env_point_expanded = np.repeat(self.env_point_robot1.reshape(1, 3), repeats=4, axis=0)
            history_combined = np.concatenate([self.History_state[:,:7], env_point_expanded], axis=1)
            current_joints = init_joints_data[0, self.num_frames-1, :7].reshape(1, 7)
            current_state = np.concatenate([current_joints, self.env_point_robot1.reshape(1, 3)], axis=1)
            input_data = np.concatenate([history_combined, current_state], axis=0).reshape(1, self.num_frames, 10)
        else:
            #! env_bp model
            input_data = np.concatenate([init_joints_data[0, self.num_frames-1, :7], self.env_point_robot1])
        #----------------------------------------------------------#
        input_tensor = torch.tensor(input_data, dtype=torch.float32, device='cuda', requires_grad=True)
        env_collision_output_tensor = self.env_collision_model.forward(input_tensor)
        env_collision_output = env_collision_output_tensor.detach().cpu().numpy()
        print(f"Env Collision Model Output After Optimization for Sample: {env_collision_output}\n")
        
        print(f"----------------------------------------------------------------")
        
        with open('output/self_distance_output.txt', 'a') as self_distance_output_file:
            self_distance_output_file.write(f"{self.self_distance} {elapsed_time:.6f} \n")
        with open('output/env_distance_output.txt', 'a') as env_distance_output_file:
            env_distance_output_file.write(f"{env_collision_output} {elapsed_time:.6f} \n")
        with open('output/optimize_joint_angles.txt', 'a') as joint_angles_file:
            joint_angles_file.write(f"{init_joints_data}\n{init_joints_data[0, self.num_frames-1, :]}\n\n")
        with open('output/joint_angles.txt', 'a') as joint_angles_file:
            optimized_input = self.normalize_input(init_joints_data)
            robot_joint = ', '.join(f"{x:.5f}" for x in optimized_input[0, 0, :])
            joint_angles_file.write(f"{robot_joint}\n")    
        print(f"cnt: ", self.cnt)

        return init_joints_data[0, self.num_frames-1, :], self_collision_output, env_collision_output
