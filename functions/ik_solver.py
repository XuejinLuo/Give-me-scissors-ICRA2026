import pybullet as p
import time
import random
import numpy as np
from scipy.spatial.transform import Rotation as R
from functions.transform_utils import *
from functions.utils import *

class IKSolver:
    def __init__(self, urdf_path, MaxNumIterations=100, Threshold=1e-6):
        p.connect(p.DIRECT)
        self.robot_id = p.loadURDF(urdf_path)
        self.end_effector_link_index = 7
        self.upper_limits = [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]
        self.lower_limits = [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]
        self.MaxNumIterations = MaxNumIterations
        self.Threshold = Threshold
        self.dh_table = np.array([
            [0, 0.333, 0, 0],
            [0, 0, 0, -np.pi/2],
            [0, 0.316, 0, np.pi/2],
            [0, 0, 0.0825, np.pi/2],
            [0, 0.384, -0.0825, -np.pi/2],
            [0, 0, 0, np.pi/2],
            [0, 0, 0.088, np.pi/2],
        ])
        self.T_flange = np.array([[1, 0, 0, 0],
                                  [0, 1, 0, 0],
                                  [0, 0, 1, 0.107],
                                  [0, 0, 0, 1]])
        self.T_finger = np.array([[0.7071068, 0.7071068, 0, 0],
                                  [-0.7071068, 0.7071068, 0, 0],
                                  [0, 0, 1, 0.1034],
                                  [0, 0, 0, 1]])
        self.T_flange_inv = np.linalg.inv(self.T_flange)
        self.T_finger_inv = np.linalg.inv(self.T_finger)
        # print(self.T_flange @ self.T_flange_inv)
        # print(self.T_finger @ self.T_finger_inv)

    def calculate_ik(self, target_position, target_orientation, initial_joint_pos=np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4]), with_finger=True): # x,y,z,w
        rotation = R.from_quat(target_orientation).as_matrix()
        translation = np.array(target_position).reshape(3, 1)
        transform_matrix = np.eye(4)
        transform_matrix[:3, :3] = rotation
        transform_matrix[:3, 3:] = translation
        if with_finger:
            T_flange = transform_matrix @ self.T_finger_inv
            T_end = T_flange @ self.T_flange_inv
            end_position = T_end[:3, 3]
            end_rotation_matrix = T_end[:3, :3]
            end_orientation = mat2quat(end_rotation_matrix) # x,y,z,w
        else:
            T_flange = transform_matrix @ self.T_flange_inv
            T_end = T_flange
            end_position = T_end[:3, 3]
            end_rotation_matrix = T_end[:3, :3]
            end_orientation = mat2quat(end_rotation_matrix) # x,y,z,w
        # print(end_position, end_orientation)
        initial_joint_pos = initial_joint_pos.tolist()
        joint_angles = p.calculateInverseKinematics(
            self.robot_id,
            self.end_effector_link_index,
            end_position,
            end_orientation,
            lowerLimits=self.lower_limits,
            upperLimits=self.upper_limits,
            currentPositions=initial_joint_pos,
            maxNumIterations=self.MaxNumIterations,
            residualThreshold=self.Threshold
        )
        return np.array(joint_angles)

    def test_ik(self, base_position, base_orientation, iterations=10):
        for i in range(iterations):
            noise_position = [random.uniform(-0.01, 0.01) for _ in range(3)]
            noise_orientation = [random.uniform(-0.01, 0.01) for _ in range(3)]
            target_position = [base_position[j] + noise_position[j] for j in range(3)]
            target_orientation = [
                base_orientation[0] + noise_orientation[0],
                base_orientation[1] + noise_orientation[1],
                base_orientation[2] + noise_orientation[2],
                base_orientation[3]
            ]
            start_time = time.time()
            joint_angles = self.calculate_ik(target_position, target_orientation)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Test {i + 1}: Joint Angles: {joint_angles}, Time taken: {elapsed_time:.6f} seconds")

    def compute_fk(self, joint_angles, with_finger=False):
        T_result = np.eye(4)
        for i in range(self.dh_table.shape[0]):
            alpha = self.dh_table[i, 3]
            a = self.dh_table[i, 2]
            d = self.dh_table[i, 1]
            theta = theta = joint_angles[i]
            ct = np.cos(theta)
            st = np.sin(theta)
            cpha = np.cos(alpha)
            spha = np.sin(alpha)
            T = np.array([[ct, -st, 0, a],
                        [st * cpha, ct * cpha, -spha, -d * spha],
                        [st * spha, ct * spha, cpha, d * cpha],
                        [0, 0, 0, 1]])
            T_result = T_result @ T  # Use @ for matrix multiplication
        T_result = T_result @ self.T_flange
        if with_finger:
            T_result = T_result @ self.T_finger
        return T_result
        # print(T_result)

    def compute_baoluo_points(self, joint_angles):
        baoluo_positions = []
        T_result = np.eye(4)
        for i in range(self.dh_table.shape[0]):
            alpha = self.dh_table[i, 3]
            a = self.dh_table[i, 2]
            d = self.dh_table[i, 1]
            theta = theta = joint_angles[i]
            ct = np.cos(theta)
            st = np.sin(theta)
            cpha = np.cos(alpha)
            spha = np.sin(alpha)
            T = np.array([[ct, -st, 0, a],
                        [st * cpha, ct * cpha, -spha, -d * spha],
                        [st * spha, ct * spha, cpha, d * cpha],
                        [0, 0, 0, 1]])
            T_result = T_result @ T
        T_result = T_result @ self.T_flange
        T_TCP = T_result @ self.T_finger
        num_points = 10
        y_values = np.linspace(100, -100, num_points)
        interpolated_points1 = np.zeros((num_points, 4))
        interpolated_points2 = np.zeros((num_points, 4))
        for i in range(num_points):
            interpolated_points1[i, :] = [25, y_values[i], 90, 0]
            interpolated_points2[i, :] = [-25, y_values[i], 90, 0]
        points = np.array([[0, -100, 103.4, 0], 
                        [0, 100, 103.4, 0],
                        [-25, 100, 90, 0],
                        [-25, -100, 90, 0]])
        points_expanded = np.tile(points, (num_points // points.shape[0], 1))
        points_expanded = points_expanded[:num_points, :]
        points = np.vstack((points_expanded, interpolated_points1, interpolated_points2))
        points = points / 1000
        for point in points:
            point = point.reshape(4, 1)
            point_pos = np.dot(self.T_finger, point)
            T_point = np.eye(4)
            T_point[:3, :3] = T_TCP[:3, :3]
            T_point[:3, 3:] = point_pos[:3]
            T_baoluo = T_result @ T_point
            # print(f"Point {T_baoluo[:3, 3].flatten()*1000}")
            baoluo_positions.append(T_baoluo[:3, 3].flatten())
        baoluo_positions = np.array(baoluo_positions)
        return baoluo_positions
    
    def test(self):
        initial_joint_pos = [0 + 0.01, -np.pi/4, 0 + 0.01, -3*np.pi/4, 0 - 0.01, np.pi/2, np.pi/4+0.01]
        target_position = [0.30689, 0.0, 0.486882]
        target_orientation = [1, 0.0, 0.0, 0.0] # x,y,z,w      
        random_range = 0.01
        for _ in range(10):
            random_offsets = np.random.uniform(-random_range, random_range, size=len(initial_joint_pos))
            modified_joint_pos = np.array(initial_joint_pos) + random_offsets
            start_time = time.time()
            results = self.calculate_ik(target_position, target_orientation, modified_joint_pos, with_finger = True)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"Time taken: {elapsed_time:.6f} seconds")
            print(results)
            T_result = self.compute_fk(results, with_finger=True)
            print(T_result)

if __name__ == "__main__":

    urdf_path = "/home/luo/ICRA/models/franka_urdf/fr3.urdf"
    ik_solver = IKSolver(urdf_path, 100, 1e-4)
    # initial_joint_pos = [0, -np.pi/8, 0, -2*np.pi/4, 0, np.pi/4, np.pi/8]
    initial_joint_pos = [0 + 0.01, -np.pi/4, 0 + 0.01, -3*np.pi/4, 0 - 0.01, np.pi/2, np.pi/4+0.01]

    # #? Position of last link center of mass
    # # target_position = [0.30689, 0.0, 0.697282]
    # # target_orientation = [0.923879532511287, -0.382683432365090, 0.0, 0.0] # x,y,z,w
    # #? Flange position
    # # target_position = [0.30689, 0.0, 0.590282]
    # # target_orientation = [0.923879532511287, -0.382683432365090, 0.0, 0.0] # x,y,z,w
    # #? Gripper position
    target_position = [0.30689, 0.0, 0.486882]
    target_orientation = [-1, 0.0, 0.0, 0.0] # x,y,z,w
    results = ik_solver.calculate_ik(target_position, target_orientation, initial_joint_pos, with_finger = True)
    print(results)
    ik_solver.compute_fk(results, with_finger=True)
    print(results)

    # angles = [0.087018 , 0.379789 , 0.220162 , -2.164007 , -0.401296 , 2.070653 , 0.315281]
    # baoluo_positions = ik_solver.compute_baoluo_points(angles)
