#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import List, Tuple

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from tf2_ros import Buffer, TransformException, TransformListener


class TunnelFollowingNode(Node):
    """Follow a tunnel by keeping similar distance to the walls while monitoring the front."""

    def __init__(self) -> None:
        super().__init__('tunnel_following')

        scan_topic = self.declare_parameter('scan_topic', '/scan').value
        cmd_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel').value

        # Parameters
        self._half_fov = max(
            0.05, math.radians(self.declare_parameter('half_fov_deg', 40.0).value)
        )
        self._max_range = self.declare_parameter('max_considered_range', 1.0).value
        self._linear_speed = self.declare_parameter('linear_speed', 0.1).value
        self._angular_speed = self.declare_parameter('angular_speed', 0.5).value

        self._front_window = min(self._half_fov * 0.25, math.radians(10.0))
        self._stop_ratio = 0.3
        self._angles: List[float] = []

        # TF setup (scan frame -> base_frame)
        self.target_frame = (
            self.declare_parameter('base_frame', 'base_link')
            .get_parameter_value()
            .string_value
        )
        self.tf_buffer: Buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._tf_warning_logged = False

        # ROS I/O
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

    def scan_callback(self, scan: LaserScan) -> None:
        # Precompute angles if scan layout changed
        if len(self._angles) != len(scan.ranges):
            self._angles = [
                scan.angle_min + i * scan.angle_increment for i in range(len(scan.ranges))
            ]

        # Look up transform from scan frame -> target_frame
        transform = self._lookup_transform(scan)

        left_samples: List[float] = []
        right_samples: List[float] = []
        front_samples: List[float] = []

        # ---- TF FIRST, THEN FILTERING ----
        for idx, rng in enumerate(scan.ranges):
            if not math.isfinite(rng):
                continue

            angle = self._angles[idx]
            distance = rng

            # (Optional) clamp raw distance to avoid crazy long rays
            distance = min(distance, self._max_range)

            if transform is not None:
                # Convert polar (distance, angle) in scan frame -> Cartesian
                x = distance * math.cos(angle)
                y = distance * math.sin(angle)

                # Apply TF into target_frame (e.g. base_link)
                x, y = self._apply_transform(x, y, transform)

                # Recompute distance and angle in target_frame
                distance = math.hypot(x, y)
                angle = math.atan2(y, x)

            # Now do all classification in target_frame coordinates
            if angle < -self._half_fov or angle > self._half_fov:
                continue

            value = min(distance, self._max_range)
            if abs(angle) <= self._front_window:
                front_samples.append(value)
            elif angle > 0.0:
                left_samples.append(value)
            else:
                right_samples.append(value)

        # No useful data: stop
        if not left_samples and not right_samples and not front_samples:
            self._publish_cmd(0.0, 0.0)
            return

        # Use medians for left/right, min for front
        left = self._median(left_samples) if left_samples else self._max_range
        right = self._median(right_samples) if right_samples else self._max_range
        front = min(front_samples) if front_samples else self._max_range
        self.get_logger().info(
            'Tunnel distances - left: %.2fm, right: %.2fm' % (left, right)
        )

        # Angular control: try to equalize left & right distances
        diff = left - right
        normalized = max(-1.0, min(1.0, diff / max(self._max_range, 1e-3)))
        angular = normalized * self._angular_speed

        # Linear control: stop if front too close
        stop_distance = self._stop_ratio * self._max_range
        linear = self._linear_speed if front > stop_distance else 0.0

        self.get_logger().info(
            'Tunnel distances - left: %.2fm, right: %.2fm, steer: %.2frad'
            % (left, right, angular)
        )
        self._publish_cmd(linear, angular)

    def _median(self, values: List[float]) -> float:
        sorted_vals = sorted(values)
        mid = len(sorted_vals) // 2
        if len(sorted_vals) % 2 == 0:
            return 0.5 * (sorted_vals[mid - 1] + sorted_vals[mid])
        return sorted_vals[mid]

    def _publish_cmd(self, linear: float, angular: float) -> None:
        twist = Twist()
        twist.linear.x = float(linear)
        twist.angular.z = float(angular)
        self.cmd_pub.publish(twist)

    def _apply_transform(
        self,
        x: float,
        y: float,
        transform: Tuple[float, float, float],
    ) -> Tuple[float, float]:
        tx, ty, yaw = transform
        cos_yaw = math.cos(yaw)
        sin_yaw = math.sin(yaw)
        x_t = cos_yaw * x - sin_yaw * y + tx
        y_t = sin_yaw * x + cos_yaw * y + ty
        return x_t, y_t

    def _lookup_transform(self, scan: LaserScan) -> Tuple[float, float, float] | None:
        source_frame = scan.header.frame_id
        if not source_frame:
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
                    f'Falling back to raw scan frame (missing TF {source_frame} -> {self.target_frame}): {exc}'
                )
            return None

        self._tf_warning_logged = False
        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        # yaw from quaternion
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return (t.x, t.y, yaw)


def main() -> None:
    rclpy.init()
    node = TunnelFollowingNode()
    try:
        rclpy.spin(node)
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()