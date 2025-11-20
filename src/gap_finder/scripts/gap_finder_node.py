#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Tuple

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from visualization_msgs.msg import Marker, MarkerArray

from gap_finder_processing import GapFinderConfig, GapFinderProcessor, ProcessResult


class GapFinderNode(Node):
    """ROS2 node that extracts left/right cones inside the configured FOV."""

    def __init__(self) -> None:
        super().__init__('gap_finder_node')

        scan_topic = self._declare_and_get_string('scan_topic', '/scan')
        marker_topic = self._declare_and_get_string('marker_topic', '/gap_finder/triangle_markers')
        corridor_enable_topic = self._declare_and_get_string(
            'corridor_enable_topic', '/corridor/enable'
        )

        half_fov = math.radians(self._declare_and_get_double('half_fov_deg', 40.0))
        max_considered_range = self._declare_and_get_double('max_considered_range', 8.0)
        cluster_eps = self._declare_and_get_double('cluster_eps', 0.05)
        cluster_min_points = self._declare_and_get_int('cluster_min_points', 3)
        desired_gap_width = self._declare_and_get_double('desired_gap_width', 0.5)
        gap_width_tolerance = self._declare_and_get_double('gap_width_tolerance', 0.1)

        config = GapFinderConfig(
            half_fov=half_fov,
            max_range=max_considered_range,
            cluster_eps=cluster_eps,
            cluster_min_points=cluster_min_points,
            corridor_threshold=self._declare_and_get_double('corridor_median_threshold', 0.5),
            desired_gap_width=desired_gap_width,
            gap_width_tolerance=gap_width_tolerance,
        )

        self.processor = GapFinderProcessor(config, self.get_logger())
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.corridor_enable_pub = self.create_publisher(Bool, corridor_enable_topic, 10)
        self._last_enable_state = False

        self.get_logger().info('gap_finder_node will now only detect cone gaps.')

    # ------------------------------------------------------------------
    def scan_callback(self, scan: LaserScan) -> None:
        result = self.processor.process_scan(scan)
        frame_id = scan.header.frame_id or 'base_link'
        stamp = scan.header.stamp
        self._publish_corridor_flag(result)
        self._publish_markers(result, frame_id, stamp)

    def _publish_corridor_flag(self, result: ProcessResult) -> None:
        enabled = bool(result.left_point and result.right_point)
        if enabled != self._last_enable_state:
            self._last_enable_state = enabled
            state = 'enabled' if enabled else 'disabled'
            self.get_logger().info(
                f'Corridor controller {state} (gap tol={result.gap_within_tolerance}).'
            )
        msg = Bool()
        msg.data = enabled
        self.corridor_enable_pub.publish(msg)

    def _publish_markers(
        self,
        result: ProcessResult,
        frame_id: str,
        stamp,
    ) -> None:
        marker_array = MarkerArray()
        marker_id = 0

        if result.left_point is not None:
            marker_array.markers.append(
                self._make_triangle_marker(
                    frame_id,
                    stamp,
                    marker_id,
                    result.left_point,
                    (1.0, 0.3, 0.3),
                )
            )
            marker_id += 1
        if result.right_point is not None:
            marker_array.markers.append(
                self._make_triangle_marker(
                    frame_id,
                    stamp,
                    marker_id,
                    result.right_point,
                    (0.3, 0.6, 1.0),
                )
            )
            marker_id += 1

        for cluster in result.clusters:
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = 'gap_clusters'
            marker.id = marker_id
            marker_id += 1
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = cluster[0]
            marker.pose.position.y = cluster[1]
            marker.pose.position.z = 0.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.08
            angle = math.atan2(cluster[1], cluster[0])
            rng = math.hypot(cluster[0], cluster[1])
            if abs(angle) <= math.radians(30.0) and rng <= 3.0:
                marker.color.r, marker.color.g, marker.color.b = (0.8, 0.2, 0.9)
            else:
                marker.color.r, marker.color.g, marker.color.b = (1.0, 0.9, 0.1)
            marker.color.a = 0.85
            marker.lifetime.sec = 0
            marker.lifetime.nanosec = int(0.5 * 1e9)
            marker_array.markers.append(marker)

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def _make_triangle_marker(
        self,
        frame_id: str,
        stamp,
        marker_id: int,
        point: Tuple[float, float],
        color: Tuple[float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'gap_triangle'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.12
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 0.95
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
