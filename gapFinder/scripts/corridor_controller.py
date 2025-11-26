#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Bool
from tf2_ros import Buffer, TransformException, TransformListener

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
            self.declare_parameter('cmd_vel_topic', '/control/gap_cmd')
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
        self.target_frame = (
            self.declare_parameter('base_frame', 'base_link')
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
        # Publish to dedicated gap control channel; FSM will arbitrate and republish to /cmd_vel.
        self.cmd_pub = self.create_publisher(Twist, cmd_topic, 10)
        self._cmd_topic = cmd_topic
        self.active_pub = self.create_publisher(Bool, active_topic, 10)

        self.tf_buffer: Buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)
        self._tf_warning_logged = False  # <-- define it once, here

        self._last_log_time = self.get_clock().now()
        self._current_linear = 0.0
        self._current_angular = 0.0
        self._corridor_enabled = False

    def scan_callback(self, scan: LaserScan) -> None:
        now = self.get_clock().now()
        transform = self._lookup_transform(scan)
        distances = self._compute_sector_distances(scan, transform)

        left_front_adjusted = max(0.0, distances['front_left'] - 0.5 * self.wheel_base)
        right_front_adjusted = max(0.0, distances['front_right'] - 0.5 * self.wheel_base)
        left = min(distances['left'], left_front_adjusted)
        right = min(distances['right'], right_front_adjusted)
        front = distances['front']
        back = distances['back']
        self.get_logger().info(
            'Corridor distances - left: %.2fm, right: %.2fm, front: %.2fm, back: %.2fm'
            % (left, right, front, back)
        )

        active_msg = Bool()
        active_msg.data = self._corridor_enabled
        self.active_pub.publish(active_msg)
        if not self._corridor_enabled:
            self._publish_idle_cmd()
            return

        # Forward speed (stop if too close)
        target_linear = 0.0 if front < self.critical_front else self.forward_speed

        # Turning based on left/right and front_left/front_right
        side_term = self.side_gain * (right - left)
        curve_term = self.curve_gain * (distances['front_right'] - distances['front_left'])
        combined = side_term + curve_term
        target_angular = 0.0
        if abs(combined) > self.turn_deadband:
            ratio = max(-1.0, min(1.0, combined / self.side_threshold))
            target_angular = -self.turn_speed * ratio

        # First-order smoothing
        gain = max(0.0, min(1.0, self.smoothing_gain))
        self._current_linear += gain * (target_linear - self._current_linear)
        self._current_angular += gain * (target_angular - self._current_angular)

        twist = Twist()
        twist.linear.x = self._current_linear
        twist.angular.z = self._current_angular
        self.cmd_pub.publish(twist)
        self.get_logger().info(
            'Corridor cmd_vel (%s): linear=%.2fm/s angular=%.2frad/s'
            % (self._cmd_topic, self._current_linear, self._current_angular)
        )

        if self._corridor_enabled and (now - self._last_log_time) >= self.log_interval:
            self._last_log_time = now
            msg = ' '.join(
                f'{name}: {distances[name]:.2f}m'
                for name in ['left', 'front_left', 'front', 'front_right',
                             'right', 'back_right', 'back', 'back_left']
            )
            self.get_logger().info(msg)

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

    def _compute_sector_distances(
        self,
        scan: LaserScan,
        transform: Tuple[float, float, float] | None,
    ) -> Dict[str, float]:
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
                # Convert to Cartesian in scan frame
                x = rng * math.cos(angle)
                y = rng * math.sin(angle)
                # Optional transform into target_frame
                if transform is not None:
                    x, y = self._apply_transform(x, y, transform)
                # Recompute range and angle in (possibly) transformed frame
                rng_t = math.hypot(x, y)
                angle_t = math.atan2(y, x)
                if self._angle_in_interval(angle_t, start_angle, end_angle):
                    values.append(min(rng_t, self.max_range))
            readings[name] = min(values) if values else self.max_range
        return readings

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

        # Optional: if scan is already in target frame, skip TF
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

        # If we previously warned and now succeeded, log recovery once
        if self._tf_warning_logged:
            self._tf_warning_logged = False
            self.get_logger().info(
                'Recovered TF %s -> %s' % (source_frame, self.target_frame)
            )

        t = tf_msg.transform.translation
        q = tf_msg.transform.rotation
        yaw = math.atan2(
            2.0 * (q.w * q.z + q.x * q.y),
            1.0 - 2.0 * (q.y * q.y + q.z * q.z),
        )
        return (t.x, t.y, yaw)

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
