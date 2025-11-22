#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Tuple, Dict

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Twist

from gap_finder_processing import GapFinderConfig, GapFinderProcessor, ProcessResult

class GapFinderNode(Node):
    """ROS2 node that extracts left/right cones inside the configured FOV and tracks clusters."""
    def __init__(self) -> None:    
        super().__init__('gap_finder_node')

        # Declare parameters (will be used as fallback if not provided by the YAML)
        self.declare_parameter('fov_min', -30.0)  # Left boundary of FOV in degrees
        self.declare_parameter('fov_max', 30.0)   # Right boundary of FOV in degrees
        self.declare_parameter('max_considered_range', 8.0)  # Max range

        # Declare other parameters (cluster_eps, desired_gap_width, etc.)
        self.declare_parameter('cluster_eps', 0.05)
        self.declare_parameter('cluster_min_points', 3)
        self.declare_parameter('desired_gap_width', 0.5)
        self.declare_parameter('gap_width_tolerance', 0.1)
        self.declare_parameter('corridor_median_threshold', 0.5)

        # Retrieve parameters from the parameter server
        self.fov_min = self.get_parameter('fov_min').get_parameter_value().double_value
        self.fov_max = self.get_parameter('fov_max').get_parameter_value().double_value
        self.max_distance = self.get_parameter('max_considered_range').get_parameter_value().double_value

        # Clustering Parameters
        cluster_eps = self.get_parameter('cluster_eps').get_parameter_value().double_value
        cluster_min_points = self.get_parameter('cluster_min_points').get_parameter_value().integer_value
        desired_gap_width = self.get_parameter('desired_gap_width').get_parameter_value().double_value
        gap_width_tolerance = self.get_parameter('gap_width_tolerance').get_parameter_value().double_value

        # Retrieve corridor_median_threshold from parameters
        corridor_threshold = self.get_parameter('corridor_median_threshold').get_parameter_value().double_value

        # Pass fov_min and fov_max to GapFinderConfig
        config = GapFinderConfig(
            fov_min=self.fov_min,  # Pass fov_min to the config
            fov_max=self.fov_max,  # Pass fov_max to the config
            max_range=self.max_distance,
            cluster_eps=cluster_eps,
            cluster_min_points=cluster_min_points,
            corridor_threshold=corridor_threshold,
            desired_gap_width=desired_gap_width,
            gap_width_tolerance=gap_width_tolerance,
        )

        self.processor = GapFinderProcessor(config, self.get_logger())
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, '/gap_finder/scan_markers', 10)
        self.corridor_enable_pub = self.create_publisher(Bool, '/corridor/enable', 10)
        self._last_enable_state = False

        self.get_logger().info('gap_finder_node will now only detect cone gaps.')

        # Track closest clusters for red and blue
        self.left_closest_cluster = None
        self.right_closest_cluster = None

    # ------------------------------------------------------------------
    def scan_callback(self, scan: LaserScan) -> None:
        result = self.processor.process_scan(scan)
        frame_id = scan.header.frame_id or 'base_link'
        stamp = scan.header.stamp
        self._publish_corridor_flag(result)
        self._track_clusters(scan)
        self._publish_closest_clusters(frame_id, stamp)  # Publish the closest clusters
     
        #self.get_logger().info(f"Left Closest Cluster: {self.left_closest_cluster}")
        #self.get_logger().info(f"Right Closest Cluster: {self.right_closest_cluster}")

    def _publish_corridor_flag(self, result: ProcessResult) -> None:
        # 항상 enabled 상태로 설정
        enabled = True
        
        if enabled != self._last_enable_state:
            self._last_enable_state = enabled
            state = 'enabled'  # 항상 enabled로 설정
            self.get_logger().info(
                f'Corridor controller {state} (gap tol={result.gap_within_tolerance}).'
            )
        
        # 항상 enabled 상태로 퍼블리시
        msg = Bool()
        msg.data = enabled
        self.corridor_enable_pub.publish(msg)

    def _track_clusters(self, scan: LaserScan) -> None:
        """Track the closest clusters within the left and right FOVs."""
        
        # Convert fov_min and fov_max from degrees to radians
        fov_min = math.radians(self.fov_min)
        fov_max = math.radians(self.fov_max)
        
        angle_min = scan.angle_min
        angle_max = scan.angle_max
        angle_increment = scan.angle_increment
        
        # Calculate indices for fov_min and fov_max in radians (ROI)
        min_index = int((fov_min - angle_min) / angle_increment)  # Left boundary of ROI
        max_index = int((fov_max - angle_min) / angle_increment)  # Right boundary of ROI

        # Ensure that indices are within valid range
        min_index = max(min_index, 0)
        max_index = min(max_index, len(scan.ranges) - 1)

        # Filter points within the ROI
        filtered_points = []
        for i in range(min_index, max_index + 1):  # Ensure we're accessing valid indices
            range_value = scan.ranges[i]
            angle = angle_min + i * angle_increment

            # Convert angle from radians to degrees
            angle_deg = math.degrees(angle)  # Convert radians to degrees

            # Skip clusters within ±5 degrees
            if -20 <= angle_deg <= 20:
                continue  # Skip points within this range

            if math.isfinite(range_value):  # Only consider valid points
                filtered_points.append({'x': range_value * math.cos(angle),
                                        'y': range_value * math.sin(angle),
                                        'range': range_value,
                                        'angle': angle_deg})  # Store angle in degrees

        # Separate left and right clusters (left: negative angles, right: positive angles)
        left_clusters = [p for p in filtered_points if p['angle'] < 0]  # Left FOV
        right_clusters = [p for p in filtered_points if p['angle'] > 0]  # Right FOV

        # Find closest cluster in left FOV
        if left_clusters:
            self.left_closest_cluster = min(left_clusters, key=lambda c: c['range'] if math.isfinite(c['range']) else float('inf'))

        # Find closest cluster in right FOV
        if right_clusters:
            self.right_closest_cluster = min(right_clusters, key=lambda c: c['range'] if math.isfinite(c['range']) else float('inf'))

        # Log the clusters for debugging
        #self.get_logger().info(f"Left Closest Cluster: {self.left_closest_cluster}")
        #self.get_logger().info(f"Right Closest Cluster: {self.right_closest_cluster}")


    def _publish_closest_clusters(self, frame_id: str, stamp) -> None:
        """Publish markers for the closest left (red) and right (blue) clusters."""
        marker_array = MarkerArray()

        # Ensure we are correctly using valid data
        if self.left_closest_cluster:
            marker_array.markers.append(
                self._create_cluster_marker(
                    frame_id, stamp, self.left_closest_cluster, (1.0, 0.0, 0.0), 'left_closest'
                )
            )

        if self.right_closest_cluster:
            marker_array.markers.append(
                self._create_cluster_marker(
                    frame_id, stamp, self.right_closest_cluster, (0.0, 0.0, 1.0), 'right_closest'
                )
            )

        # Only publish if markers exist
        if marker_array.markers:
            #self.get_logger().info("Markers added, publishing...")
            self.marker_pub.publish(marker_array)
        else:
            self.get_logger().info("No markers to publish.")

    def _create_cluster_marker(self, frame_id: str, stamp, cluster: Dict[str, float], color: Tuple[float, float, float], ns: str) -> Marker:
        """Create a marker for a cluster (left or right)."""
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = ns
        marker.id = 0  # Use the same ID for now
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = cluster['x']
        marker.pose.position.y = cluster['y']
        marker.pose.position.z = 0.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.15  # Size of the marker
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 1.0  # Full opacity
        return marker

    def _declare_and_get_double(self, name: str, default: float) -> float:
        self.declare_parameter(name, default)
        return self.get_parameter(name).get_parameter_value().double_value

    def _declare_and_get_int(self, name: str, default: int) -> int:
        self.declare_parameter(name, default)
        return self.get_parameter(name).get_parameter_value().integer_value

    def _declare_and_get_string(self, name: str, default: str) -> str:
        self.declare_parameter(name, default)
        return self.get_parameter(name).get_parameter_value().string_value


def main() -> None:
    rclpy.init()
    node = GapFinderNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
