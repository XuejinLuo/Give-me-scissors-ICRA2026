import rclpy
from rclpy.node import Node
from std_msgs.msg import String
import numpy as np
import re

class RobotCommandSubscriber(Node):
    def __init__(self):
        super().__init__('robot_command_subscriber')
        self.get_logger().info('Robot command subscriber is starting...')
        self.curr_joint_pos = None
        self.grab = None
        self.command_subscription = self.create_subscription(
            String,
            'robot_state',
            self.listener_callback,
            10
        )

    def listener_callback(self, msg):
        print_flag = True
        self.get_logger().info(f'Received command: {msg.data}') if print_flag else None
        joints, grab = self.extract_joints_and_grab(msg.data)
        if joints is not None:
            self.curr_joint_pos = joints
            self.get_logger().info(f'Parsed joints: {joints}') if print_flag else None
        else:
            self.get_logger().warn('No valid joints data found in the command.')
        if grab is not None:
            self.grab = grab
            self.get_logger().info(f'Parsed grab: {grab}') if print_flag else None
        else:
            self.get_logger().warn('No valid grab data found in the command.')

    def extract_joints_and_grab(self, command_str):
        joints_match = re.search(r'joints: \[([^\]]+)\]', command_str)
        grab_match = re.search(r'grab: (\d+)', command_str)
        joints = None
        grab = None
        if joints_match:
            joints_str = joints_match.group(1)
            joints_list = [float(x.strip()) for x in joints_str.split(',')]
            joints = np.array(joints_list)
        if grab_match:
            grab = int(grab_match.group(1))
        
        return joints, grab


def main(args=None):
    rclpy.init(args=args)
    node = RobotCommandSubscriber()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
