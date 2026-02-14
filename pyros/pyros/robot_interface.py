import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from datetime import datetime
import threading
import time
import numpy as np
import re

class RobotInterface(Node):
    def __init__(self):
        super().__init__('robot_interface_node')
        self.get_logger().info("Hello, I am robot_interface_node!")
        # Publishers
        self.command_publisher_ = self.create_publisher(String, "robot_command", 10)
        self.timer = self.create_timer(0.01, self._timer_callback)
        self.command = "relative_pos: [0,0,0] relative_quat: [0,0,0,0] grab: 0"
        self.grab_command = 0
        self.relative_pos = ""
        self.relative_quat = ""  #x,y,z,w

        # Subscribers
        self.curr_joint_pos = None
        self.grab_state = None
        self._is_running = True
        self.command_subscribe_ = self.create_subscription(
            String,
            "robot_state",
            self._listener_callback,
            10)
        
    #######? Command Publisher ?########
    def _string2command(self, command_string):
        """Update the command"""
        self.command = command_string

    def _command2string2ROS(self):
        relative_pos = self.relative_pos
        relative_quat = self.relative_quat
        robot_grab_command = self.grab_command
        pos_ptr = np.array2string(relative_pos, separator=',', formatter={'all': lambda x: f'{x:.6f}'})
        quat_ptr = np.array2string(relative_quat, separator=',', formatter={'all': lambda x: f'{x:.6f}'})
        robot_command = f"relative_pos: {pos_ptr} relative_quat: {quat_ptr} grab: {robot_grab_command}"
        # print("robot_command: ", robot_command)
        return robot_command
    
    def _timer_callback(self):
        """Timer callback function to publish the command"""
        msg = String()
        msg.data = self.command
        self.command_publisher_.publish(msg) 
        # print(f"Published command: {msg.data}")
        # self.get_logger().info(f'Published command: {msg.data}')

    #######? State Subscriber ?########
    def _listener_callback(self, msg):
        #! 打印输出
        print_flag = True
        # self.get_logger().info(f'Received state: {msg.data}') if print_flag else None
        curr_joint_pos, grab_state = self._extract_joints_and_grab(msg.data)
        if curr_joint_pos is not None:
            self.curr_joint_pos = curr_joint_pos
            # self.get_logger().info(f'Parsed curr_joint_pos: {curr_joint_pos}') if print_flag else None
        else:
            self.get_logger().warn('No valid curr_joint_pos data found in the state.')
        if grab_state is not None:
            self.grab_state = grab_state
            # self.get_logger().info(f'Parsed grab: {grab_state}') if print_flag else None
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

def get_robot_command():
    """Simulate an external process that gets the robot command"""
    robot_joints_command = np.random.rand(7)
    robot_grab_command = -1
    ## 保证关节角输出格式不出bug
    joints_str = np.array2string(robot_joints_command, separator=',', formatter={'all': lambda x: f'{x:.6f}'})
    robot_command = f"joints: {joints_str} grab: {robot_grab_command}"
    return robot_command

def command_generator(node):
    """Simulate an external process that updates the node"""
    txt_data = np.loadtxt('/home/luo/ICRA/pictures/test_points.txt')
    cnt = 0
    while rclpy.ok():
        if cnt < txt_data.shape[0]:
            node.relative_pos = txt_data[cnt, :3]
            node.relative_quat = np.zeros(4)
            node.grab_command = 1
            node.grab_command = txt_data[cnt, 3]
            command = node._command2string2ROS()
            node._string2command(command)
            print(command)
            cnt += 1
        time.sleep(5)

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