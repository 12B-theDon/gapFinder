#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import List

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan


class ScanMedianNode(Node):
    def __init__(self) -> None:
        super().__init__('scan_median_node')
        scan_topic = self.declare_parameter('scan_topic', '/scan').get_parameter_value().string_value
        self.declare_parameter('print_interval', 1.0)
        self._print_interval = self.get_parameter('print_interval').get_parameter_value().double_value
        self._last_print_time = self.get_clock().now()
        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)

    def scan_callback(self, scan: LaserScan) -> None:
        now = self.get_clock().now()
        if (now - self._last_print_time).nanoseconds < self._print_interval * 1e9:
            return
        self._last_print_time = now

        values: List[float] = []
        angle = scan.angle_min
        for rng in scan.ranges:
            if math.isfinite(rng):
                values.append(rng)
            angle += scan.angle_increment
        if not values:
            self.get_logger().info('scan median: nan (no data)')
            return
        values.sort()
        mid = len(values) // 2
        if len(values) % 2 == 0:
            median = 0.5 * (values[mid - 1] + values[mid])
        else:
            median = values[mid]
        self.get_logger().info(f'scan median: {median:.2f} m')


def main() -> None:
    rclpy.init()
    node = ScanMedianNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
