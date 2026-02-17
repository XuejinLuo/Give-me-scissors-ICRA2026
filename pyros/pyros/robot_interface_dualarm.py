import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from datetime import datetime
import threading
import time
import numpy as np
import re
import roboticstoolbox as rtb
from scipy.spatial.transform import Rotation as R
from spatialmath import SE3, UnitQuaternion

class RobotInterface(Node):
    def __init__(self):
        super().__init__('robot_interface_node')
        self.get_logger().info("Hello, I am robot_interface_node!")
        # Publishers
        self.command_publisher_ = self.create_publisher(String, "dual_robot_command", 10)
        self.timer = self.create_timer(0.02, self._timer_callback)
        self.command = "robot1_joints: [0,0,0,0,0,0,0] robot1_grab: 0; robot2_joints: [0,0,0,0,0,0,0] robot2_grab: 0"
        self.grab_command_robot1 = 0
        self.grab_command_robot2 = 0
        self.joints_command_robot1 = np.zeros(7)  # Robot 1 joints position
        self.joints_command_robot2 = np.zeros(7) # Robot 2 joints position

        # Subscribers
        self.robot_1_curr_joint_pos = np.array([-0.5, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        # self.robot_2_curr_joint_pos = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, -np.pi/2])
        self.robot_2_curr_joint_pos = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        self.robot_1_grab_state = np.zeros(1)
        self.robot_2_grab_state = np.zeros(1)
        self.env_closest_point_robot1 = np.array([1000, 1000, 1000])
        self.env_closest_point_robot2 = np.array([1000, 1000, 1000])
        self.min_distance_robot1 = np.array([1000])
        self.min_distance_robot2 = np.array([1000])

        self._is_running = True
        self.env_closest_point_subscribe_ = self.create_subscription(
            String,
            "env_closest_point",
            self._env_closest_point_listener_callback,
            10)
        self.robot1_state_subscribe_ = self.create_subscription(
            String,
            "robot1_state",
            self._robot1_listener_callback,
            10)
        self.robot2_state_subscribe_ = self.create_subscription(
            String,
            "robot2_state",
            self._robot2_listener_callback,
            10)

        self.last_time = time.time()
        
    #######? Command Publisher ?########
    def _string2command(self, command_string):
        """Update the command"""
        self.command = command_string

    def _command2string2ROS(self):
        joints_command_robot1 = self.joints_command_robot1
        grab_command_robot1 = self.grab_command_robot1
        robot1_joints_ptr = np.array2string(joints_command_robot1, separator=',', formatter={'all': lambda x: f'{x:.6f}'})

        joints_command_robot2 = self.joints_command_robot2
        grab_command_robot2 = self.grab_command_robot2
        robot2_joints_ptr = np.array2string(joints_command_robot2, separator=',', formatter={'all': lambda x: f'{x:.6f}'})

        robot_command = (
            f"robot1_joints: {robot1_joints_ptr}, robot1_grab: {grab_command_robot1}; "
            f"robot2_joints: {robot2_joints_ptr}, robot2_grab: {grab_command_robot2}"
        )
        # print("robot_command: ", robot_command)
        return robot_command
    
    def _timer_callback(self):
        """Timer callback function to publish the command"""
        msg = String()
        msg.data = self.command
        self.command_publisher_.publish(msg) 
        # print(f"Published command: {msg.data}")
        # self.get_logger().info(f'Published command: {msg.data}')
        # current_time = time.time()  # Get current time
        # loop_duration = current_time - self.last_time  # Calculate duration since last loop
        # self.last_time = current_time  # Update last time
        # print(f"ros time: {loop_duration}")
        

    #######? State Subscriber ?########
    def _robot1_listener_callback(self, msg):
        #! 打印输出
        print_flag = False
        # self.get_logger().info(f'Received state: {msg.data}') if print_flag else None
        curr_joint_pos, grab_state = self._extract_joints_and_grab(msg.data)
        if curr_joint_pos is not None:
            self.robot_1_curr_joint_pos = curr_joint_pos
            self.get_logger().info(f'Robot1 curr_joint_pos: {self.robot_1_curr_joint_pos}') if print_flag else None
        else:
            self.get_logger().warn('No valid curr_joint_pos data found in the state.')
        if grab_state is not None:
            self.robot_1_grab_state = np.array([grab_state])
            self.get_logger().info(f'Robot1 grab: {self.robot_1_grab_state}') if print_flag else None
        else:
            self.get_logger().warn('No valid grab data found in the state.')

    def _robot2_listener_callback(self, msg):
        #! 打印输出
        print_flag = False
        # self.get_logger().info(f'Received state: {msg.data}') if print_flag else None
        curr_joint_pos, grab_state = self._extract_joints_and_grab(msg.data)
        if curr_joint_pos is not None:
            self.robot_2_curr_joint_pos = curr_joint_pos
            self.get_logger().info(f'Robot2 curr_joint_pos: {self.robot_2_curr_joint_pos}') if print_flag else None
        else:
            self.get_logger().warn('No valid curr_joint_pos data found in the state.')
        if grab_state is not None:
            self.robot_2_grab_state = np.array([grab_state])
            self.get_logger().info(f'Robot2 grab: {self.robot_2_grab_state}\n') if print_flag else None
        else:
            self.get_logger().warn('No valid grab data found in the state.')

    def _env_closest_point_listener_callback(self, msg):
        #! 打印输出
        print_flag = False
        self.get_logger().info(f'Received state: {msg.data}') if print_flag else None
        closest_point_robot1, distance_robot1, closest_point_robot2, distance_robot2 = self._extract_closest_point_and_distance(msg.data)

        if closest_point_robot1 is not None:
            self.env_closest_point_robot1 = closest_point_robot1
            self.get_logger().info(f'env_closest_point_robot1: {self.env_closest_point_robot1}') if print_flag else None
        else:
            self.env_closest_point_robot1 = np.array([1000, 1000, 1000])
            self.get_logger().warn('No valid closest_point data found in the state.')
        if distance_robot1 is not None:
            self.min_distance_robot1 = np.array([distance_robot1])
            self.get_logger().info(f'minimum distance robot1: {self.min_distance_robot1}\n') if print_flag else None
        else:
            self.get_logger().warn('No valid distance data found in the state.')

        if closest_point_robot2 is not None:
            self.env_closest_point_robot2 = closest_point_robot2
            self.get_logger().info(f'env_closest_point_robot2: {self.env_closest_point_robot2}') if print_flag else None
        else:
            self.env_closest_point_robot2 = np.array([1000, 1000, 1000])
            self.get_logger().warn('No valid closest_point data found in the state.')
        if distance_robot2 is not None:
            self.min_distance_robot2 = np.array([distance_robot2])
            self.get_logger().info(f'minimum distance robot2: {self.min_distance_robot2}\n') if print_flag else None
        else:
            self.get_logger().warn('No valid distance data found in the state.')
            
    def _extract_joints_and_grab(self, state_str):
        joints_match = re.search(r'joints: \[([^\]]+)\]', state_str)
        grab_match = re.search(r'grab: (-?\d+)', state_str)
        joints = None
        grab = None
        if joints_match:
            joints_str = joints_match.group(1)
            joints_list = [float(x.strip()) for x in joints_str.split(',')]
            joints = np.array(joints_list)
        if grab_match:
            grab = int(grab_match.group(1))
        
        return joints, grab
    
    def _extract_closest_point_and_distance(self, state_str):
        closest_point_robot1_match = re.search(r'closest_point_robot1: \[([^\]]+)\]', state_str)
        distance_robot1_match = re.search(r'distance_robot1: (-?\d+)', state_str)
        closest_point_robot2_match = re.search(r'closest_point_robot2: \[([^\]]+)\]', state_str)
        distance_robot2_match = re.search(r'distance_robot2: (-?\d+)', state_str)

        closest_point_robot1 = None
        distance_robot1 = None
        closest_point_robot2 = None
        distance_robot2 = None

        if closest_point_robot1_match:
            closest_point_str = closest_point_robot1_match.group(1)
            closest_point_robot1_list = [float(x.strip()) for x in closest_point_str.split(',')]
            closest_point_robot1 = np.array(closest_point_robot1_list)
        if distance_robot1_match:
            distance_robot1 = float(distance_robot1_match.group(1))

        if closest_point_robot2_match:
            closest_point_str = closest_point_robot2_match.group(1)
            closest_point_robot2_list = [float(x.strip()) for x in closest_point_str.split(',')]
            closest_point_robot2 = np.array(closest_point_robot2_list)
        if distance_robot2_match:
            distance_robot2 = float(distance_robot2_match.group(1))

        return closest_point_robot1, distance_robot1, closest_point_robot2, distance_robot2

def command_generator(node):
    """Simulate an external process that updates the node"""
    file_path = "./pyros/test/test_points.txt"
    try:
        file = open(file_path, 'r')
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到! 使用随机数据替代")
        file = None
    
    cnt = 0
    line_number = 0
    
    while rclpy.ok():
        if file:
            line = file.readline()
            if not line:  # 文件结束，重置文件指针
                print("文件读取完毕，从头开始")
                file.seek(0)
                line = file.readline()
                line_number = 1
            else:
                line_number += 1
            line = line.strip()
        
            if line:
                robot_data = parse_robot_data(line)
                print(robot_data)
                
                robot1_absolute_pos = robot_data.get('robot1_absolute_pos', [0, 0, 0])
                robot1_absolute_quat = robot_data.get('robot1_absolute_quat', [1, 0, 0, 0])
                robot1_grab = robot_data.get('robot1_grab', 0)
                
                robot2_absolute_pos = robot_data.get('robot2_absolute_pos', [0, 0, 0])
                robot2_absolute_quat = robot_data.get('robot2_absolute_quat', [1, 0, 0, 0])
                robot2_grab = robot_data.get('robot2_grab', 0)
                
                print(f"处理第 {line_number} 行数据:")
                print(f"Robot 1 - Position: {robot1_absolute_pos}, Orientation: {robot1_absolute_quat}, Grab: {robot1_grab}")
                print(f"Robot 2 - Position: {robot2_absolute_pos}, Orientation: {robot2_absolute_quat}, Grab: {robot2_grab}")
                print("-" * 50)
            else:
                print(f"第 {line_number} 行为空行，使用随机数据")
                robot1_absolute_pos = np.random.rand(3)
                robot1_absolute_quat = np.random.rand(4)
                robot1_absolute_quat /= np.linalg.norm(robot1_absolute_quat)
                robot1_grab = 1
                
                robot2_absolute_pos = np.random.rand(3)
                robot2_absolute_quat = np.random.rand(4)
                robot2_absolute_quat /= np.linalg.norm(robot2_absolute_quat)
                robot2_grab = 1
        else:
            robot1_absolute_pos = np.random.rand(3)
            robot1_absolute_quat = np.random.rand(4)
            robot1_absolute_quat /= np.linalg.norm(robot1_absolute_quat)
            robot1_grab = 1
            
            robot2_absolute_pos = np.random.rand(3)
            robot2_absolute_quat = np.random.rand(4)
            robot2_absolute_quat /= np.linalg.norm(robot2_absolute_quat)
            robot2_grab = 1
        
        joints_command_robot1 = ik_calculate(robot1_absolute_pos, robot1_absolute_quat)
        node.joints_command_robot1 = joints_command_robot1
        node.grab_command_robot1 = np.array(robot1_grab)
        joints_command_robot2 = ik_calculate(robot2_absolute_pos, robot2_absolute_quat)
        node.joints_command_robot2 = joints_command_robot2
        node.grab_command_robot2 = np.array(robot2_grab)
        print(joints_command_robot1)

        command = node._command2string2ROS()
        node._string2command(command)
        
        cnt += 1
        time.sleep(5)
    
    if file:
        file.close() 

def parse_robot_data(line):
    """解析单行机器人数据"""
    # 定义正则表达式模式
    pos_pattern = r'robot(\d+)_absolute_pos: \[([\d.-]+), ([\d.-]+), ([\d.-]+)\]'
    quat_pattern = r'robot(\d+)_absolute_quat: \[([\d.-]+), ([\d.-]+), ([\d.-]+), ([\d.-]+)\]'
    grab_pattern = r'robot(\d+)_grab: ([+-]?\d+)'
    
    # 初始化变量
    robot_data = {}
    
    # 提取位置信息
    for match in re.finditer(pos_pattern, line):
        robot_id = int(match.group(1))
        x = float(match.group(2))
        y = float(match.group(3))
        z = float(match.group(4))
        robot_data[f'robot{robot_id}_absolute_pos'] = [x, y, z]
    
    # 提取四元数信息
    for match in re.finditer(quat_pattern, line):
        robot_id = int(match.group(1))
        qx = float(match.group(2))
        qy = float(match.group(3))
        qz = float(match.group(4))
        qw = float(match.group(5))
        robot_data[f'robot{robot_id}_absolute_quat'] = [qx, qy, qz, qw]
    
    # 提取抓取状态
    for match in re.finditer(grab_pattern, line):
        robot_id = int(match.group(1))
        grab = int(match.group(2))
        robot_data[f'robot{robot_id}_grab'] = grab
    
    return robot_data

def ik_calculate(target_position, target_orientation, initial_joint_pos=np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])): # x,y,z,w
    robot = rtb.models.Panda()
    target_orientation = np.array(target_orientation)
    target_orientation = target_orientation[[3, 0, 1, 2]] # 四元数, w, x, y, z
    target_quaternion = UnitQuaternion(target_orientation)
    rotation_matrix = target_quaternion.R
    Tep = SE3.Trans(target_position) * SE3.Rt(rotation_matrix)
    sol = robot.ik_LM(Tep, q0=initial_joint_pos)
    return sol[0]

def mat2quat(rmat):
    return R.from_matrix(rmat).as_quat()

def main(args=None):
    rclpy.init(args=args)
    robot_interface = RobotInterface()

    command_thread = threading.Thread(target=command_generator, args=(robot_interface,))
    command_thread.daemon = True  # Ensure thread exits when main exits
    command_thread.start()

    rclpy.spin(robot_interface)
    robot_interface.destroy_node()
    rclpy.shutdown()
    command_thread.join()


if __name__ == '__main__':
    main()