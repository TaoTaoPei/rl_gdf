import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from launch.event_handlers import OnProcessExit
import xacro

def generate_launch_description():
    package_name = 'm1_description'
    package_path = os.path.join(get_package_share_directory(package_name))

    # 使用xacro
    # xacro_file = os.path.join(package_path, 'xacro', 'robot.xacro')
    # robot_description = xacro.process_file(xacro_file).toxml()

     # 使用URDF
    urdf_file = os.path.join(package_path, 'urdf', 'xyuan_description_right.urdf')
    with open(urdf_file, 'r') as infp:
        robot_description = infp.read()

    robot_state_publisher_node = Node(
        package="robot_state_publisher",
        executable="robot_state_publisher",
        name='robot_state_publisher',
        output="screen",
        parameters=[{"robot_description": robot_description}],
    )

    joint_state_publisher_gui_node = Node(
        package='joint_state_publisher_gui',
        executable='joint_state_publisher_gui',
        name='joint_state_publisher_gui',
        output='screen',
    )

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory('gazebo_ros'), 'launch', 'gazebo.launch.py')
        ),
        launch_arguments={'verbose': 'true'}.items()
    )

    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', '/robot_description', '-entity', 'a1_gazebo'],
        output='screen',
    )

    joint_state_broadcaster_node = Node(
        package="controller_manager",
        executable='spawner.py' if os.environ.get('ROS_DISTRO', '') == 'foxy' else 'spawner',
        arguments=["joint_state_broadcaster"],
        output='screen',
    )

    robot_joint_controller_node = Node(
        package="controller_manager",
        executable='spawner.py' if os.environ.get('ROS_DISTRO', '') == 'foxy' else 'spawner',
        arguments=["robot_joint_controller"],
        output='screen',
    )

    return LaunchDescription([
        robot_state_publisher_node,
        joint_state_publisher_gui_node,
        gazebo,
        spawn_entity,
        joint_state_broadcaster_node,
        robot_joint_controller_node,
    ])
