import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    package_share = get_package_share_directory('gap_finder')
    config_file = os.path.join(package_share, 'configs', 'default.yaml')

    use_sim_time = LaunchConfiguration('use_sim_time', default='false')

    return LaunchDescription(
        [
            DeclareLaunchArgument('use_sim_time', default_value='false'),
            Node(
                package='gap_finder',
                executable='modified_gap_finder_visual_node.py',
                name='modified_gap_finder_visual_node',
                parameters=[config_file, {'use_sim_time': use_sim_time}],
                output='screen',
            ),
        ]
    )
