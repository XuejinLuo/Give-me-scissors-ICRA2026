from setuptools import find_packages, setup

package_name = 'pyros'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='luo',
    maintainer_email='luo@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            "topic_publisher_02 = pyros.topic_publisher_02:main",
            "topic_subscribe_02 = pyros.topic_subscribe_02:main",
            "robot_command_publish = pyros.robot_command_publish:main",
            "robot_state_subscibe = pyros.robot_state_subscibe:main",
            "robot_interface = pyros.robot_interface:main",
        ],
    },
)
