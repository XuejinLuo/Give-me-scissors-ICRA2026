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

class EnvClosestPointROS2Interface(Node):
    def __init__(self):
        super().__init__('env_closest_point_node')
        self.get_logger().info("Hello, I am EnvClosestPointROS2Interface node!")
        # Publishers
        self.command_publisher_ = self.create_publisher(String, "env_closest_point", 10)
        self.timer = self.create_timer(0.02, self._timer_callback)
        self.command = "closest_point: [1000,1000,1000], distance: 1000"
        self.closest_point_robot1 = None
        self.closest_point_robot2 = None
        self.distance_robot1 = 1000
        self.distance_robot2 = 1000

        # Subscribers
        self.robot_1_curr_joint_pos = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        self.robot_2_curr_joint_pos = np.array([0, -np.pi/4, 0, -3*np.pi/4, 0, np.pi/2, np.pi/4])
        self._is_running = True
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
        closest_point_robot1 = self.closest_point_robot1
        closest_point_robot2 = self.closest_point_robot2
        distance_robot1 = self.distance_robot1
        distance_robot2 = self.distance_robot2
        if closest_point_robot1 is None:
            closest_point_robot1 = np.array([1000, 1000, 1000])
        if closest_point_robot2 is None:
            closest_point_robot2 = np.array([1000, 1000, 1000])
        if distance_robot1 is None:
            distance_robot1 = 1000
        if distance_robot2 is None:
            distance_robot2 = 1000

        closest_point_robot1_ptr = np.array2string(
                closest_point_robot1, separator=',', formatter={'all': lambda x: f'{x:.2f}'}
            )
        closest_point_robot2_ptr = np.array2string(
                closest_point_robot2, separator=',', formatter={'all': lambda x: f'{x:.2f}'}
            )

        robot_command = (
            f"closest_point_robot1: {closest_point_robot1_ptr}, distance_robot1: {distance_robot1}; "
            f"closest_point_robot2: {closest_point_robot2_ptr}, distance_robot2: {distance_robot2}; "
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

def command_generator(node):
    """Simulate an external process that updates the node"""
    
    cnt = 0
    line_number = 0
    
    while rclpy.ok():
        now = datetime.now()
        minutes = now.minute
        seconds = now.second
        milliseconds = now.microsecond // 1000
        closest_point = np.array([minutes, seconds, milliseconds])
        distance = cnt

        node.closest_point = closest_point
        node.distance = distance
        command = node._command2string2ROS()
        node._string2command(command)
        
        cnt += 1
        time.sleep(0.02)


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
    env_closest_point_interface = EnvClosestPointROS2Interface()

    command_thread = threading.Thread(target=command_generator, args=(env_closest_point_interface,))
    command_thread.daemon = True  # Ensure thread exits when main exits
    command_thread.start()

    rclpy.spin(env_closest_point_interface)
    env_closest_point_interface.destroy_node()
    rclpy.shutdown()
    command_thread.join()


if __name__ == '__main__':
    main()