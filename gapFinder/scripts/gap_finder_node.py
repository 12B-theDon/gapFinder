#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Callable, List, Optional, Tuple

import rclpy
from rclpy.node import Node
from rclpy.time import Time
from geometry_msgs.msg import Twist
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from gap_finder_processing import (
    GapFinderConfig,
    GapFinderProcessor,
    ProcessResult,
    XYTransform,
    xy_transform_from_tf,
)


class GapFinderNode(Node):
    """ROS2 node that extracts left/right cones inside the configured FOV."""

    def __init__(self) -> None:
        super().__init__('gap_finder_node')

        scan_topic = self._declare_and_get_string('scan_topic', '/scan')
        marker_topic = self._declare_and_get_string(
            'marker_topic', '/gap_finder/triangle_markers'
        )
        corridor_enable_topic = self._declare_and_get_string(
            'corridor_enable_topic', '/corridor/enable'
        )

        half_fov = math.radians(self._declare_and_get_double('half_fov_deg', 40.0))
        max_considered_range = self._declare_and_get_double('max_considered_range', 8.0)
        cluster_eps = self._declare_and_get_double('cluster_eps', 0.05)
        cluster_min_points = self._declare_and_get_int('cluster_min_points', 3)
        desired_gap_width = self._declare_and_get_double('desired_gap_width', 0.5)
        gap_width_tolerance = self._declare_and_get_double('gap_width_tolerance', 0.1)
        self.target_frame = self._declare_and_get_string('base_frame', 'base_link')
        self._median_termination_threshold = self._declare_and_get_double(
            'termionate_left_right_scan_data_median_distnace', 0.12
        )
        approach_cmd_topic = self._declare_and_get_string('approach_cmd_topic', '/cmd_vel')
        self._approach_forward_speed = self._declare_and_get_double(
            'approach_forward_speed', 0.1
        )
        self._approach_turn_speed = self._declare_and_get_double(
            'approach_turn_speed', 0.4
        )
        self._approach_turn_deadband = self._declare_and_get_double(
            'approach_turn_deadband', 0.02
        )
        self._steering_margin = self._declare_and_get_double(
            'left_right_cluster_distance_margin_for_steering', 0.1
        )

        self._half_fov = half_fov

        config = GapFinderConfig(
            half_fov=half_fov,
            max_range=max_considered_range,
            cluster_eps=cluster_eps,
            cluster_min_points=cluster_min_points,
            corridor_threshold=self._declare_and_get_double(
                'corridor_median_threshold', 0.5
            ),
            desired_gap_width=desired_gap_width,
            gap_width_tolerance=gap_width_tolerance,
        )

        self.processor = GapFinderProcessor(config, self.get_logger())

        # TF buffer/listener for scan_frame -> target_frame (e.g. base_link)
        self.tf_buffer: Buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._tf_warning_logged = False
        self._tf_in_use_logged = False  # log once when we actually use TF

        # ROS I/O
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, marker_topic, 10)
        self.corridor_enable_pub = self.create_publisher(Bool, corridor_enable_topic, 10)
        self.cmd_pub = self.create_publisher(Twist, approach_cmd_topic, 10)
        self._last_enable_state = False
        self._approach_active = False
        self._shutdown_requested = False
        self._shutdown_timer = None
        self._center_logged = False

        self.get_logger().info('gap_finder_node will now only detect cone gaps.')

    # ------------------------------------------------------------------
    def scan_callback(self, scan: LaserScan) -> None:
        # Get transform from scan frame to target_frame (base_frame)
        transform = self._lookup_transform(scan)

        if transform is not None and not self._tf_in_use_logged:
            self._tf_in_use_logged = True
            self.get_logger().info(
                f'Applying TF {scan.header.frame_id} -> {self.target_frame} '
                f'for gap detection and RViz markers.'
            )

        # GapFinderProcessor is responsible for applying XYTransform to scan points
        result: ProcessResult = self.processor.process_scan(scan, transform)

        # If TF is applied, result coordinates are in target_frame
        if transform is not None:
            frame_id = self.target_frame
        else:
            frame_id = scan.header.frame_id or self.target_frame

        stamp = scan.header.stamp
        num_ranges = len(scan.ranges)
        if num_ranges > 0:
            center_angle = scan.angle_min + 0.5 * (num_ranges - 1) * scan.angle_increment
        else:
            center_angle = 0.0

        clusters_in_fov = self._filter_clusters_within_fov(result.clusters, center_angle)
        left_cluster = self._find_closest_cluster(
            clusters_in_fov,
            lambda pt: pt[0] > 0.0 and pt[1] >= 0.0,
        )
        right_cluster = self._find_closest_cluster(
            clusters_in_fov,
            lambda pt: pt[0] > 0.0 and pt[1] < 0.0,
        )

        left_dist = math.hypot(left_cluster[0], left_cluster[1]) if left_cluster else None
        right_dist = (
            math.hypot(right_cluster[0], right_cluster[1]) if right_cluster else None
        )
        tol_flag = result.gap_within_tolerance

        left_median = self._median_sector_distance(scan, 90.0)
        right_median = self._median_sector_distance(scan, -90.0)
        left_med_str = f'{left_median:.3f}m' if left_median is not None else 'N/A'
        right_med_str = f'{right_median:.3f}m' if right_median is not None else 'N/A'
        self.get_logger().info(
            '[SideScan] left=%s right=%s' % (left_med_str, right_med_str)
        )

        if (
            left_median is not None
            and right_median is not None
            and left_median <= self._median_termination_threshold
            and right_median <= self._median_termination_threshold
        ):
            self.get_logger().info(
                'Side scan medians within %.3fm threshold; enabling corridor controller.'
                % self._median_termination_threshold
            )
            self._publish_corridor_flag(True, tol_flag)
            self._publish_markers(
                frame_id,
                stamp,
                center_angle,
                clusters_in_fov,
                left_cluster,
                right_cluster,
            )
            self._request_shutdown_after_enable()
            return

        if left_dist is None and right_dist is None:
            self._approach_active = False
            self._publish_corridor_flag(False, tol_flag)
        else:
            self._publish_corridor_flag(False, tol_flag)
            if left_dist is not None and right_dist is not None:
                self._approach_active = True
                self._publish_approach_cmd(left_dist, right_dist)
            else:
                self._approach_active = False

        self._publish_markers(
            frame_id,
            stamp,
            center_angle,
            clusters_in_fov,
            left_cluster,
            right_cluster,
        )

    # ------------------------------------------------------------------
    def _lookup_transform(self, scan: LaserScan) -> XYTransform | None:
        source_frame = scan.header.frame_id
        if not source_frame:
            return None

        # If scan is already in target frame, no transform needed
        if source_frame == self.target_frame:
            return None

        try:
            tf_msg = self.tf_buffer.lookup_transform(
                self.target_frame,
                source_frame,
                Time.from_msg(scan.header.stamp),
            )
        except TransformException as exc:
            if not self._tf_warning_logged:
                self._tf_warning_logged = True
                self.get_logger().warn(
                    'Falling back to raw scan frame (missing TF %s -> %s): %s'
                    % (source_frame, self.target_frame, exc)
                )
            return None

        # If we previously warned and now succeed, log recovery once
        if self._tf_warning_logged:
            self._tf_warning_logged = False
            self.get_logger().info(
                'Recovered TF %s -> %s' % (source_frame, self.target_frame)
            )

        return xy_transform_from_tf(
            tf_msg.transform.translation,
            tf_msg.transform.rotation,
        )

    # ------------------------------------------------------------------
    def _publish_corridor_flag(self, enabled: bool, tol_flag: bool) -> None:
        if enabled != self._last_enable_state:
            self._last_enable_state = enabled
            if enabled:
                self.get_logger().info(
                    'Gap finder enabling corridor controller (gap tol=%s); expect corridor cmd_vel output.'
                    % tol_flag
                )
            else:
                self.get_logger().info(
                    'Gap finder disabling corridor controller (gap tol=%s); corridor cmd_vel output halted.'
                    % tol_flag
                )
        msg = Bool()
        msg.data = enabled
        self.corridor_enable_pub.publish(msg)

    def _publish_markers(
        self,
        frame_id: str,
        stamp,
        center_angle: float,
        clusters_in_fov: list[Tuple[float, float]],
        left_cluster: Optional[Tuple[float, float]],
        right_cluster: Optional[Tuple[float, float]],
    ) -> None:
        """
        Publish markers in the same frame as the processed coordinates.
        If TF was applied, `frame_id` == target_frame and all points
        are assumed to be in that frame.
        """
        marker_array = MarkerArray()
        marker_id = 0
        center_angle_deg = math.degrees(center_angle)
        if not self._center_logged:
            left_center_deg = center_angle_deg + 90.0
            right_center_deg = center_angle_deg - 90.0
            self.get_logger().info(
                'Scan center angle=%.1fdeg | left ref=%.1fdeg | right ref=%.1fdeg'
                % (center_angle_deg, left_center_deg, right_center_deg)
            )
            self._center_logged = True

        if left_cluster is not None:
            rng = math.hypot(left_cluster[0], left_cluster[1])
            angle_deg = math.degrees(math.atan2(left_cluster[1], left_cluster[0]))
            offset_deg = angle_deg - center_angle_deg
            #self.get_logger().info(
            #    'Closest LEFT cluster distance=%.2fm angle=%.1fdeg (offset=%.1fdeg vs center %.1fdeg)'
            #    % (rng, angle_deg, offset_deg, center_angle_deg)
            #)
            marker_array.markers.append(
                self._make_cluster_marker(
                    frame_id,
                    stamp,
                    marker_id,
                    left_cluster,
                    (0.2, 0.9, 0.4),
                    'closest_left_cluster',
                )
            )
            marker_id += 1
            marker_array.markers.append(
                self._make_angle_text_marker(
                    frame_id,
                    stamp,
                    marker_id,
                    left_cluster,
                    angle_deg,
                    'left_cluster_angle',
                )
            )
            marker_id += 1
        else:
            self.get_logger().info('No LEFT cluster detected in front hemisphere.')

        if right_cluster is not None:
            rng = math.hypot(right_cluster[0], right_cluster[1])
            angle_deg = math.degrees(math.atan2(right_cluster[1], right_cluster[0]))
            offset_deg = angle_deg - center_angle_deg
            #self.get_logger().info(
            #    'Closest RIGHT cluster distance=%.2fm angle=%.1fdeg (offset=%.1fdeg vs center %.1fdeg)'
            #    % (rng, angle_deg, offset_deg, center_angle_deg)
            #)
            marker_array.markers.append(
                self._make_cluster_marker(
                    frame_id,
                    stamp,
                    marker_id,
                    right_cluster,
                    (0.2, 0.5, 0.9),
                    'closest_right_cluster',
                )
            )
            marker_id += 1
            marker_array.markers.append(
                self._make_angle_text_marker(
                    frame_id,
                    stamp,
                    marker_id,
                    right_cluster,
                    angle_deg,
                    'right_cluster_angle',
                )
            )
            marker_id += 1
        else:
            self.get_logger().info('No RIGHT cluster detected in front hemisphere.')

        if marker_array.markers:
            self.marker_pub.publish(marker_array)

    def _make_cluster_marker(
        self,
        frame_id: str,
        stamp,
        marker_id: int,
        point: Tuple[float, float],
        color: Tuple[float, float, float],
        namespace: str,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.12
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 0.95
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = int(0.5 * 1e9)
        return marker

    def _make_angle_text_marker(
        self,
        frame_id: str,
        stamp,
        marker_id: int,
        point: Tuple[float, float],
        angle_deg: float,
        namespace: str,
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = namespace
        marker.id = marker_id
        marker.type = Marker.TEXT_VIEW_FACING
        marker.action = Marker.ADD
        marker.pose.position.x = point[0]
        marker.pose.position.y = point[1]
        marker.pose.position.z = 0.15
        marker.scale.z = 0.08
        marker.color.r = marker.color.g = marker.color.b = 1.0
        marker.color.a = 0.9
        marker.text = f'{angle_deg:.1f} deg'
        marker.lifetime.sec = 0
        marker.lifetime.nanosec = int(0.5 * 1e9)
        return marker

    def _find_closest_cluster(
        self,
        clusters: list[Tuple[float, float]],
        predicate: Callable[[Tuple[float, float]], bool],
    ) -> Tuple[float, float] | None:
        candidates = [pt for pt in clusters if predicate(pt)]
        if not candidates:
            return None
        return min(candidates, key=lambda pt: math.hypot(pt[0], pt[1]))

    def _request_shutdown_after_enable(self) -> None:
        if self._shutdown_requested:
            return
        self._shutdown_requested = True
        self.get_logger().info('Corridor enable published; shutting down gap_finder_node.')
        self._shutdown_timer = self.create_timer(0.1, self._perform_shutdown)

    def _perform_shutdown(self) -> None:
        if self._shutdown_timer is not None:
            self._shutdown_timer.cancel()
            self._shutdown_timer = None
        self.get_logger().info('gap_finder_node shutdown complete.')
        rclpy.shutdown()

    def _publish_approach_cmd(self, left_dist: float, right_dist: float) -> None:
        if self._last_enable_state:
            return
        twist = Twist()
        twist.linear.x = self._approach_forward_speed
        diff = left_dist - right_dist
        if abs(diff) > max(self._approach_turn_deadband, self._steering_margin):
            twist.angular.z = (
                self._approach_turn_speed if diff > 0.0 else -self._approach_turn_speed
            )
            # Positive angular.z is CCW (left turn). Log actual steering command.
            direction = 'left' if twist.angular.z > 0.0 else 'right'
            self.get_logger().info(
                '[Approach] Turning %s (left-right diff=%.3f m, margin=%.3f m, L=%.3f m, R=%.3f m)'
                % (direction, diff, self._steering_margin, left_dist, right_dist)
            )
        else:
            self.get_logger().info(
                '[Approach] Holding straight (left-right diff=%.3f m, margin=%.3f m)'
                % (diff, self._steering_margin)
            )
        self.cmd_pub.publish(twist)

    def _median_sector_distance(
        self,
        scan: LaserScan,
        center_deg: float,
        window_deg: float = 5.0,
    ) -> float | None:
        target = math.radians(center_deg)
        half_window = math.radians(window_deg)
        current_angle = scan.angle_min
        samples: List[float] = []
        for rng in scan.ranges:
            if math.isfinite(rng) and (target - half_window) <= current_angle <= (
                target + half_window
            ):
                samples.append(rng)
            current_angle += scan.angle_increment
        if not samples:
            return None
        samples.sort()
        mid = len(samples) // 2
        if len(samples) % 2 == 0:
            return 0.5 * (samples[mid - 1] + samples[mid])
        return samples[mid]

    def _log_edge_scan_distances(self, scan: LaserScan) -> None:
        left_med = self._median_sector_distance(scan, 90.0)
        right_med = self._median_sector_distance(scan, -90.0)
        left_str = f'{left_med:.2f}m' if left_med is not None else 'N/A'
        right_str = f'{right_med:.2f}m' if right_med is not None else 'N/A'
        self.get_logger().info('[SideScan] left=%s right=%s' % (left_str, right_str))

    def _filter_clusters_within_fov(
        self,
        clusters: list[Tuple[float, float]],
        center_angle: float,
    ) -> list[Tuple[float, float]]:
        filtered: list[Tuple[float, float]] = []
        for pt in clusters:
            angle = math.atan2(pt[1], pt[0])
            diff = math.atan2(math.sin(angle - center_angle), math.cos(angle - center_angle))
            if abs(diff) <= self._half_fov:
                filtered.append(pt)
        return filtered

    # ------------------------------------------------------------------
    # Parameter helpers
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