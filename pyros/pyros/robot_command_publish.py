import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from datetime import datetime
import threading
import time
import numpy as np

class RobotCommandPublisher(Node):
    def __init__(self, name):
        super().__init__(name)
        self.get_logger().info("Hello, I am %s!" % name)
        self.command_publisher_ = self.create_publisher(String, "robot_command", 10) 
        self.timer = self.create_timer(0.01, self.timer_callback)  # Timer callback at 0.1s interval
        self.command = ""

    def set_command(self, command):
        """Update the command"""
        self.command = command

    def timer_callback(self):
        """Timer callback function to publish the command"""
        msg = String()
        msg.data = self.command
        self.command_publisher_.publish(msg) 
        self.get_logger().info(f'Published command: {msg.data}')

def get_current_time_command():
    """Return the current time formatted as a string"""
    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    time_command = f"Updated time: {current_time}"
    return time_command

def get_robot_command():
    """Simulate an external process that gets the robot command"""
    robot_joints_command = np.random.rand(7)
    robot_grab_command = 1
    ## 保证关节角输出格式不出bug
    joints_str = np.array2string(robot_joints_command, separator=',', formatter={'all': lambda x: f'{x:.6f}'})
    robot_command = f"joints: {joints_str} grab: {robot_grab_command}"
    return robot_command

def command_updater(node):
    """Simulate an external process that updates the command"""
    while rclpy.ok():
        # command = get_current_time_command()
        command = get_robot_command()
        node.set_command(command)
        time.sleep(0.01)

def main(args=None):
    rclpy.init(args=args)
    node = RobotCommandPublisher("robot_command_publish")

    # Start a separate thread to update the command
    updater_thread = threading.Thread(target=command_updater, args=(node,))
    updater_thread.start()

    rclpy.spin(node)
    rclpy.shutdown()
    updater_thread.join()

if __name__ == '__main__':
    main()