#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import List

from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class TunnelFollowingNode(Node):
    """Follow a tunnel by keeping similar distance to the walls while monitoring the front."""

    def __init__(self) -> None:
        super().__init__('tunnel_following')

        scan_topic = self.declare_parameter('scan_topic', '/scan').value
        cmd_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel').value
        self._half_fov = max(
            0.05, math.radians(self.declare_parameter('half_fov_deg', 40.0).value)
        )
        self._max_range = self.declare_parameter('max_considered_range', 1.0).value
        self._linear_speed = self.declare_parameter('linear_speed', 0.1).value
        self._angular_speed = self.declare_parameter('angular_speed', 0.5).value

        self._front_window = min(self._half_fov * 0.25, math.radians(10.0))
        self._stop_ratio = 0.3
        self._angles: List[float] = []

        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)

    def scan_callback(self, scan: LaserScan) -> None:
        if len(self._angles) != len(scan.ranges):
            self._angles = [
                scan.angle_min + i * scan.angle_increment for i in range(len(scan.ranges))
            ]

        left_samples: List[float] = []
        right_samples: List[float] = []
        front_samples: List[float] = []
        for angle, rng in zip(self._angles, scan.ranges):
            if angle < -self._half_fov or angle > self._half_fov:
                continue
            if not math.isfinite(rng):
                continue
            value = min(rng, self._max_range)
            if abs(angle) <= self._front_window:
                front_samples.append(value)
            elif angle > 0.0:
                left_samples.append(value)
            else:
                right_samples.append(value)

        if not left_samples and not right_samples and not front_samples:
            self._publish_cmd(0.0, 0.0)
            return

        left = self._median(left_samples) if left_samples else self._max_range
        right = self._median(right_samples) if right_samples else self._max_range
        front = min(front_samples) if front_samples else self._max_range

        diff = left - right
        normalized = max(-1.0, min(1.0, diff / max(self._max_range, 1e-3)))
        angular = normalized * self._angular_speed

        stop_distance = self._stop_ratio * self._max_range
        linear = self._linear_speed if front > stop_distance else 0.0

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