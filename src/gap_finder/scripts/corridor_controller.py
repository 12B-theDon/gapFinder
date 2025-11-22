#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool

DIRECTIONS: List[Tuple[str, float, float]] = [
    ('front', math.radians(-22.5), math.radians(22.5)),
    ('front_left', math.radians(22.5), math.radians(67.5)),
    ('left', math.radians(67.5), math.radians(112.5)),
    ('back_left', math.radians(112.5), math.radians(157.5)),
    ('back', math.radians(157.5), math.radians(-157.5)),
    ('back_right', math.radians(-157.5), math.radians(-112.5)),
    ('right', math.radians(-112.5), math.radians(-67.5)),
    ('front_right', math.radians(-67.5), math.radians(-22.5)),
]


class CorridorController(Node):
    """Simple controller that keeps a safe distance from walls inside a cone corridor."""

    def __init__(self) -> None:
        super().__init__('corridor_controller')

        scan_topic = (
            self.declare_parameter('scan_topic', '/scan').get_parameter_value().string_value
        )
        cmd_topic = (
            self.declare_parameter('cmd_vel_topic', '/cmd_vel')
            .get_parameter_value()
            .string_value
        )
        enable_topic = (
            self.declare_parameter('enable_topic', '/corridor/enable')
            .get_parameter_value()
            .string_value
        )
        active_topic = (
            self.declare_parameter('active_topic', '/corridor/active')
            .get_parameter_value()
            .string_value
        )

        self.max_range = (
            self.declare_parameter('max_range', 2.0).get_parameter_value().double_value
        )
        self.front_threshold = (
            self.declare_parameter('front_threshold', 0.5).get_parameter_value().double_value
        )
        self.side_threshold = (
            self.declare_parameter('side_threshold', 0.4).get_parameter_value().double_value
        )
        self.critical_front = (
            self.declare_parameter('critical_front', 0.35).get_parameter_value().double_value
        )
        self.wheel_base = (
            self.declare_parameter('wheel_base', 0.33).get_parameter_value().double_value
        )
        self.forward_speed = (
            self.declare_parameter('forward_speed', 0.1).get_parameter_value().double_value
        )
        self.turn_speed = (
            self.declare_parameter('turn_speed', 0.6).get_parameter_value().double_value
        )
        self.turn_deadband = (
            self.declare_parameter('turn_deadband', 0.05).get_parameter_value().double_value
        )
        self.side_gain = (
            self.declare_parameter('side_gain', 1.0).get_parameter_value().double_value
        )
        self.curve_gain = (
            self.declare_parameter('curve_gain', 1.0).get_parameter_value().double_value
        )
        self.smoothing_gain = (
            self.declare_parameter('smoothing_gain', 0.2).get_parameter_value().double_value
        )
        self.log_interval = Duration(
            seconds=self.declare_parameter('log_interval_sec', 1.0)
            .get_parameter_value()
            .double_value
        )

        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.create_subscription(Bool, enable_topic, self.enable_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self.active_pub = self.create_publisher(Bool, active_topic, 10)

        self._last_log_time = self.get_clock().now()
        self._current_linear = 0.0
        self._current_angular = 0.0
        self._corridor_enabled = False

    def scan_callback(self, scan: LaserScan) -> None:
        now = self.get_clock().now()
        distances = self._compute_sector_distances(scan)
        left_front_adjusted = max(0.0, distances['front_left'] - 0.5 * self.wheel_base)
        right_front_adjusted = max(0.0, distances['front_right'] - 0.5 * self.wheel_base)
        left = min(distances['left'], left_front_adjusted)
        right = min(distances['right'], right_front_adjusted)
        front = distances['front']

        active_msg = Bool()
        active_msg.data = self._corridor_enabled
        self.active_pub.publish(active_msg)
        if not self._corridor_enabled:
            self._publish_idle_cmd()
            return

        target_linear = 0.0
        target_angular = 0.0
        target_linear = 0.0 if front < self.critical_front else self.forward_speed

        side_term = self.side_gain * (right - left)
        curve_term = self.curve_gain * (distances['front_right'] - distances['front_left'])
        combined = side_term + curve_term
        if abs(combined) > self.turn_deadband:
            ratio = max(-1.0, min(1.0, combined / self.side_threshold))
            target_angular = -self.turn_speed * ratio

        gain = max(0.0, min(1.0, self.smoothing_gain))
        self._current_linear += gain * (target_linear - self._current_linear)
        self._current_angular += gain * (target_angular - self._current_angular)

        twist = Twist()
        twist.linear.x = self._current_linear
        twist.angular.z = self._current_angular
        self.cmd_pub.publish(twist)

    def enable_callback(self, msg: Bool) -> None:
        previous = self._corridor_enabled
        self._corridor_enabled = bool(msg.data)
        if not self._corridor_enabled:
            self._publish_idle_cmd()
        if self._corridor_enabled != previous:
            state = 'enabled' if self._corridor_enabled else 'idle'
            self.get_logger().info(
                f'Corridor controller {state} by gap_finder flag.'
            )

    def _publish_idle_cmd(self) -> None:
        self._current_linear = 0.0
        self._current_angular = 0.0
        twist = Twist()
        twist.linear.x = 0.0
        twist.angular.z = 0.0
        self.cmd_pub.publish(twist)

    def _compute_sector_distances(self, scan: LaserScan) -> Dict[str, float]:
        readings: Dict[str, float] = {}
        angles = [
            scan.angle_min + i * scan.angle_increment for i in range(len(scan.ranges))
        ]
        zipped = list(zip(angles, scan.ranges))
        for name, start_angle, end_angle in DIRECTIONS:
            values = []
            for angle, rng in zipped:
                if not math.isfinite(rng):
                    continue
                if self._angle_in_interval(angle, start_angle, end_angle):
                    values.append(min(rng, self.max_range))
            readings[name] = min(values) if values else self.max_range
        return readings

    def _angle_in_interval(self, angle: float, start: float, end: float) -> bool:
        if start <= end:
            return start <= angle <= end
        return angle >= start or angle <= end


def main() -> None:
    rclpy.init()
    node = CorridorController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
