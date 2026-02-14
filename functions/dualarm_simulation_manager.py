import pybullet as p
import pybullet_data as pd
import math
import time
import threading
import numpy as np
from scipy.spatial.transform import Rotation as R
import pybullet_robots.panda.panda_sim as panda_sim

##############################################################
def mat2quat(rmat):
    return R.from_matrix(rmat).as_quat()  # (x, y, z, w)

def extract_transform(matrix):
    position = matrix[:3, 3]
    rotation = matrix[:3, :3]
    q = mat2quat(rotation)
    return position, q

def transform_point(point, transform_matrix):
    homogeneous_point = np.array([point[0], point[1], point[2], 1.0])
    transformed_homogeneous = transform_matrix @ homogeneous_point
    return transformed_homogeneous[:3]

class PandaSim(object):
    def __init__(self, bullet_client, transform_matrix):
        self.bullet_client = bullet_client
        position, orientation = extract_transform(transform_matrix)
        self.jointPositions=[0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04]
        
        flags = self.bullet_client.URDF_ENABLE_CACHED_GRAPHICS_SHAPES
        self.panda = self.bullet_client.loadURDF("franka_panda/panda.urdf", position, orientation, useFixedBase=True, flags=flags)
        index = 0
        for j in range(self.bullet_client.getNumJoints(self.panda)):
            self.bullet_client.changeDynamics(self.panda, j, linearDamping=0, angularDamping=0) # Set joint dynamics properties to eliminate damping in simulation
            info = self.bullet_client.getJointInfo(self.panda, j)
            # print(f"Joint {j}: name={info[1].decode()}, type={info[2]}")

            jointType = info[2]
            if (jointType == self.bullet_client.JOINT_PRISMATIC):
                self.bullet_client.resetJointState(self.panda, j, self.jointPositions[index]) 
                index=index+1
            if (jointType == self.bullet_client.JOINT_REVOLUTE):
                self.bullet_client.resetJointState(self.panda, j, self.jointPositions[index]) 
                index=index+1

    def step(self, joint_angles):
        self.jointPositions = joint_angles
        for i in range(7):
            self.bullet_client.setJointMotorControl2(self.panda, i, self.bullet_client.POSITION_CONTROL, joint_angles[i], force=5 * 240.)
        # control the gripper
        self.bullet_client.setJointMotorControl2(self.panda, 9, self.bullet_client.POSITION_CONTROL, joint_angles[7], force=20)
        self.bullet_client.setJointMotorControl2(self.panda, 10, self.bullet_client.POSITION_CONTROL, joint_angles[8], force=20)

class RobotArmSimulation:
    def __init__(self):
        p.connect(p.GUI)
        p.setAdditionalSearchPath(pd.getDataPath())
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)  # Close GUI
        # planeId = p.loadURDF("plane.urdf")
        cameraDistance = 1.02
        cameraYaw = 60.80
        cameraPitch = 3.80
        cameraTarget = [0.03, -0.19, 0.32]
        p.resetDebugVisualizerCamera(cameraDistance, cameraYaw, cameraPitch, cameraTarget)

        # p.setGravity(0, -9.8, 0)
        p.setTimeStep(1./1000.)
        self.p = p
        self.angles1 = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04]
        self.angles2 = [0, -np.pi / 4, 0, -3 * np.pi / 4, 0, np.pi / 2, np.pi / 4, 0.04, 0.04]
        self.self_collision_distance = 0

        self.T_offset_robot1 = np.eye(4)
        self.T_offset_robot2 = np.loadtxt("models/dualarm_robot2_offset.txt")
        self.panda1 = PandaSim(p, self.T_offset_robot1)
        self.panda2 = PandaSim(p, self.T_offset_robot2)
        self.sphere1_position = np.array([[0.6,0,0.6]])
        self.sphere2_position = np.array([[0.6,0.1,0.6]])
        self.sphere_id = []
        self.debug_ids = []
        self.obstacle_ids = {}
        # self.add_table(center_position=[0.5, -0.3, 0.0], length=0.5, width=1, table_height=0.007)
        self.load_sphere_small(np.array([[0.6, 0, 0.6], [0.6, 0.1, 0.6]]), radius=0.03)

        # #- test update angles
        # self.running = True
        # self.update_thread = threading.Thread(target=self.update_angles_test)
        # self.update_thread.start()

    def draw_trajectories(self):
        # Trajectory parameter settings
        radius = 0.1  # trajectory radius
        omega = 1.0   # angular velocity parameter
        num_points = 100  # number of trajectory points (more = smoother)

        # Generate first trajectory points
        traj1 = []
        for cnt in np.linspace(0, 2*np.pi, num_points):
            x = 0.386 - radius * np.sin(omega * 2 * cnt)
            y = -0.2 - 2 * radius * np.cos(omega * 2 * cnt)
            z = 0.200
            traj1.append([x, y, z])

        # Generate second trajectory points
        traj2 = []
        for cnt in np.linspace(0, 2*np.pi, num_points):
            x = 0.506  + radius * np.sin(omega * cnt)
            y = 0.15 + 2 * radius * np.cos(omega * cnt)
            z = 0.200
            original_point = [x, y, z]
            transformed_point = transform_point(original_point, self.T_offset_robot2)
            traj2.append(transformed_point)

        sphere1_traj = []
        for cnt in np.linspace(0, 2 * np.pi, num_points):
            center_x = 0.386
            center_y = 0.1
            center_z = 0.200
            b = 0.1  # y-axis semi-major axis length
            c = 0.2  # z-axis semi-major axis length
            theta = omega * cnt  # Calculate current angle

            x = center_x
            y = center_y + b * np.cos(theta)
            z = center_z + c * np.sin(theta)
            sphere1_traj.append([x, y, z])

        sphere2_traj = []
        for cnt in np.linspace(0, 2 * np.pi, num_points):
            center_x = 0.2
            center_y = -0.8
            center_z = 0.2
            b = 0.1  # y-axis semi-major axis length
            c = 0.2  # z-axis semi-major axis length
            theta = omega * cnt  # Calculate current angle

            x = center_x
            y = center_y + b * np.cos(theta)
            z = center_z + c * np.sin(theta)
            sphere2_traj.append([x, y, z])

        # # Draw first trajectory
        # for i in range(len(traj1)-1):
        #     id = self.p.addUserDebugLine(
        #         lineFromXYZ=traj1[i],
        #         lineToXYZ=traj1[i+1],
        #         lineColorRGB=[0.4, 0.0, 0.4],
        #         lineWidth=3,
        #         lifeTime=0
        #     )
        #     self.debug_ids.append(id)

        # # Draw second trajectory
        # for i in range(len(traj2)-1):
        #     id = self.p.addUserDebugLine(
        #         lineFromXYZ=traj2[i],
        #         lineToXYZ=traj2[i+1],
        #         lineColorRGB=[0.2667, 0.5176, 0.6941],
        #         lineWidth=3,
        #         lifeTime=0
        #     )
        #     self.debug_ids.append(id)

        # for i in range(len(sphere1_traj) - 1):
        #     id = self.p.addUserDebugLine(
        #         lineFromXYZ=sphere1_traj[i],
        #         lineToXYZ=sphere1_traj[i + 1],
        #         lineColorRGB=[0.0, 0.0, 0.5],  # Blue trajectory
        #         lineWidth=3,
        #         lifeTime=0
        #     )
        #     self.debug_ids.append(id)

        # for i in range(len(sphere2_traj) - 1):
        #     id = self.p.addUserDebugLine(
        #         lineFromXYZ=sphere2_traj[i],
        #         lineToXYZ=sphere2_traj[i + 1],
        #         lineColorRGB=[0.0, 0.5, 0.0],  # Green trajectory
        #         lineWidth=3,
        #         lifeTime=0
        #     )
        #     self.debug_ids.append(id)

    def add_table(self, center_position, length, width, table_height):
        # Create a plane (table)
        table_shape = self.p.createCollisionShape(self.p.GEOM_BOX, halfExtents=[length / 2, width / 2, table_height])
        self.table_id = self.p.createMultiBody(baseCollisionShapeIndex=table_shape, basePosition=center_position)

    def check_collisions(self):
        pass
        # closest_points = p.getClosestPoints(bodyA=self.panda1.panda, bodyB=self.table_id, distance=0.01)
        # for closest_point in closest_points:
            # if closest_point[8] < 0:
            #     print("Collision detected!")

    def update_angles_test(self):
        #- test update angles
        t = 0
        while self.running:
            grip_angle = 0.02 + 0.02 * np.sin(t)
            self.angles1[7] = self.angles1[8] = self.angles2[7] = self.angles2[8] = grip_angle
            t += 0.1
            time.sleep(0.1)

    def load_sphere_small(self, keypoints_base_robot, radius = 0.005):
        sphere_shape = self.p.createCollisionShape(self.p.GEOM_SPHERE, radius=radius)
        for keypoint in keypoints_base_robot:
            ball_id = self.p.createMultiBody(baseCollisionShapeIndex=sphere_shape, basePosition=keypoint)
            self.sphere_id.append(ball_id)

    def update_arm_positions(self, angles1, angles2, grab1, grab2):
        self.angles1[:7] = angles1
        self.angles2[:7] = angles2
        if grab1 == 1:
            self.angles1[7] = self.angles1[8] = 0.0
        elif grab1 == -1:
            self.angles1[7] = self.angles1[8] = 0.04
        if grab2 == 1:
            self.angles2[7] = self.angles2[8] = 0.0
        elif grab2 == -1:
            self.angles2[7] = self.angles2[8] = 0.04

    def update_sphere_position(self, sphere_position):
        key_dict = p.getKeyboardEvents()
        sphere_position = np.array(sphere_position)
        self.sphere1_position = sphere_position
        if p.B3G_UP_ARROW in key_dict and key_dict[p.B3G_UP_ARROW] & p.KEY_IS_DOWN:  # Up
            sphere_position[2] += 0.0005
        elif p.B3G_DOWN_ARROW in key_dict and key_dict[p.B3G_DOWN_ARROW] & p.KEY_IS_DOWN:  # Down
            sphere_position[2] -= 0.0005
        elif p.B3G_LEFT_ARROW in key_dict and key_dict[p.B3G_LEFT_ARROW] & p.KEY_IS_DOWN:  # Left
            sphere_position[1] -= 0.0005
        elif p.B3G_RIGHT_ARROW in key_dict and key_dict[p.B3G_RIGHT_ARROW] & p.KEY_IS_DOWN:  # Right
            sphere_position[1] += 0.0005
        if ord('1') in key_dict and key_dict[ord('1')] & p.KEY_IS_DOWN:  # Forward
            sphere_position[0] -= 0.0005
        elif ord('2') in key_dict and key_dict[ord('2')] & p.KEY_IS_DOWN:  # Backward
            sphere_position[0] += 0.0005

        return sphere_position
    
    def auto_sphere1_position(self, cnt):
        """
        Make sphere move in ellipse around x-axis (in y-z plane), ellipse center at (0.5, 0.3, 0.4)
        Ellipse parametric equations:
        - x = center x coordinate
        - y = center y coordinate + b*cos(θ)  (y-axis semi-major axis length b=0.1)
        - z = center z coordinate + c*sin(θ)  (z-axis semi-major axis length c=0.2)
        where θ = ω*t, ω is angular velocity, t is time
        """
        sphere_position, _ = p.getBasePositionAndOrientation(self.sphere_id[0])
        # Ellipse center position (specified as 0.5, 0.3, 0.4)
        center_x = 0.386
        center_y = 0.3
        center_z = 0.200
        
        # Ellipse motion parameter settings
        b = 0.1        # y-axis semi-major axis length
        c = 0.2        # z-axis semi-major axis length
        omega = 0.2    # angular velocity (controls motion speed)
        
        # Calculate current time and motion angle
        theta = omega * cnt * 0.06 # angle changes linearly with time
        
        # Calculate position based on ellipse parametric equations (rotating around x-axis with specified center)
        n = 12
        x = center_x  # x coordinate fixed as center x value
        y = center_y + b * np.cos(theta + n * np.pi/12)  # motion relative to center in y-direction
        z = center_z - c * np.sin(theta + n * np.pi/12)  # motion relative to center in z-direction
        
        # Update sphere position
        sphere_position = np.array([x, y, z])
        self.sphere1_position = sphere_position
        
        p.resetBasePositionAndOrientation(self.sphere_id[0], sphere_position, [0, 0, 0, 1])

        return sphere_position
    
    def auto_sphere2_position(self, cnt):
        """
        Make sphere move in ellipse around x-axis (in y-z plane), ellipse center at (0.5, 0.3, 0.4)
        Ellipse parametric equations:
        - x = center x coordinate
        - y = center y coordinate + b*cos(θ)  (y-axis semi-major axis length b=0.1)
        - z = center z coordinate + c*sin(θ)  (z-axis semi-major axis length c=0.2)
        where θ = ω*t, ω is angular velocity, t is time
        """
        sphere_position, _ = p.getBasePositionAndOrientation(self.sphere_id[1])
        # Ellipse center position (specified as 0.5, 0.3, 0.4)
        center_x = 0.2
        center_y = -0.9
        center_z = 0.2
        
        # Ellipse motion parameter settings
        b = 0.1        # y-axis semi-major axis length
        c = 0.2        # z-axis semi-major axis length
        omega = 0.2   # angular velocity (controls motion speed)
        
        theta = omega * cnt * 0.06  # angle changes linearly with time
        
        # Calculate position based on ellipse parametric equations (rotating around x-axis with specified center)
        n = -3
        x = center_x  # x coordinate fixed as center x value
        y = center_y + b * np.cos(theta + n * np.pi/6)  # motion relative to center in y-direction
        z = center_z + c * np.sin(theta + n * np.pi/6)  # motion relative to center in z-direction
        
        # Update sphere position
        sphere_position = np.array([x, y, z])
        self.sphere2_position = sphere_position
        p.resetBasePositionAndOrientation(self.sphere_id[1], sphere_position, [0, 0, 0, 1])

        return sphere_position
    
    
    def load_world(self, world_config):
        # Load obstacles in world
        if "cuboid" in world_config:
            for name, cuboid in world_config["cuboid"].items():
                dims = cuboid["dims"]
                pose = cuboid["pose"]
                position = pose[:3]
                orientation = pose[3:]
                
                visual_shape_id = p.createVisualShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2]
                )
                collision_shape_id = p.createCollisionShape(
                    shapeType=p.GEOM_BOX,
                    halfExtents=[dims[0]/2, dims[1]/2, dims[2]/2]
                )
                
                obj_id = p.createMultiBody(
                    baseMass=0,  # static object
                    baseCollisionShapeIndex=collision_shape_id,
                    baseVisualShapeIndex=visual_shape_id,
                    basePosition=position,
                    baseOrientation=orientation
                )
                self.obstacle_ids[name] = obj_id

    def clear_debug_drawings(self):
        if not self.debug_ids:
            return
        for debug_id in self.debug_ids:
            try:
                self.p.removeUserDebugItem(debug_id)
            except Exception as e:
                print(f"Error clearing debug graphics: {e}")
        self.debug_ids = []

    def draw_rollouts(self, rollouts):
        if rollouts.ndim == 2:
            rollouts = rollouts[np.newaxis, ...]
        self.clear_debug_drawings()
        all_points = []
        all_colors = []
        for i, trajectory in enumerate(rollouts):
            hue = i / len(rollouts)
            rgb = self.hsv_to_rgb(hue, 0.8, 0.8)
            all_points.extend(trajectory.tolist())
            all_colors.extend([rgb] * len(trajectory))
        if all_points:
            point_ids = self.p.addUserDebugPoints(
                pointPositions=all_points,
                pointColorsRGB=all_colors,
                pointSize=5,
                lifeTime=0
            )
            if not isinstance(point_ids, list):
                point_ids = [point_ids]
            self.debug_ids.extend(point_ids)
    @staticmethod
    def hsv_to_rgb(h, s, v):
        import colorsys
        return colorsys.hsv_to_rgb(h, s, v)

    
    def run(self):
        try:
            self.update_interval = 20
            self.step_counter = 0
            self.cycle_text_id = None
            while True:
                self.panda1.step(self.angles1)
                self.panda2.step(self.angles2)
                p.stepSimulation()
                # sphere_position, _ = p.getBasePositionAndOrientation(self.sphere_id[0])
                # sphere_position = self.update_sphere_position(sphere_position) # 键盘操控
                # sphere_position = self.auto_sphere1_position(sphere_position) # 自动

                # sphere_position, _ = p.getBasePositionAndOrientation(self.sphere_id[1])
                # sphere_position = self.auto_sphere2_position(sphere_position)

                self.joint_states_sim_robot1 = [self.p.getJointState(self.panda1.panda, i)[0] for i in range(7)]
                self.joint_states_sim_robot2 = [self.p.getJointState(self.panda2.panda, i)[0] for i in range(7)]
                self.check_collisions()

                # self.step_counter += 1
                # if self.step_counter % self.update_interval == 0:
                #     if self.cycle_text_id is not None:
                #         p.removeUserDebugItem(self.cycle_text_id)
                #     self.cycle_text_id = p.addUserDebugText(
                #         text=f"Self Distance: {self.self_collision_distance}",  # 显示的文本内容
                #         textPosition=[0.0, 0.0, 0.1],  # 文本在3D空间中的位置（可调整）
                #         textColorRGB=[1, 0, 0],  # 文本颜色（红色）
                #         textSize=1.0,  # 文本大小
                #         lifeTime=0.1  # 文本持续时间（0表示永久）
                #     )

                time.sleep(1./1000.)
        except KeyboardInterrupt:
            pass
        finally:
            print("Simulation over.")

if __name__ == "__main__":
    simulation = RobotArmSimulation()
    simulation.run()