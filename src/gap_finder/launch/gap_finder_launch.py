import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node


def generate_launch_description() -> LaunchDescription:
    config_file = os.path.join(
        get_package_share_directory('gap_finder'),
        'configs',
        'default.yaml',
    )

    return LaunchDescription(
        [
            Node(
                package='gap_finder',
                executable='gap_finder_node.py',
                name='gap_finder',
                output='screen',
                parameters=[config_file],
            ),
            Node(
                package='gap_finder',
                executable='gap_finder_visual_node.py',
                name='gap_finder_visual_node',
                output='screen',
                parameters=[config_file],
            ),
            Node(
                package='gap_finder',
                executable='corridor_controller.py',
                name='corridor_controller',
                output='screen',
                parameters=[config_file],
            ),
        ]
    )
