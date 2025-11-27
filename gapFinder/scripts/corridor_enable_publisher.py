#!/usr/bin/env python3

from __future__ import annotations

import rclpy
from rclpy.node import Node
from std_msgs.msg import Bool


class CorridorEnablePublisher(Node):
    """Periodically publishes `True` on the /corridor/enable topic to keep the controller active."""

    def __init__(self) -> None:
        super().__init__('corridor_enable_publisher')

        enable_topic = (
            self.declare_parameter('enable_topic', '/corridor/enable')
            .get_parameter_value()
            .string_value
        )
        publish_period_sec = (
            self.declare_parameter('publish_period_sec', 0.5)
            .get_parameter_value()
            .double_value
        )

        self._enable_msg = Bool()
        self._enable_msg.data = True
        self._publish_period = max(0.05, publish_period_sec)

        self._publisher = self.create_publisher(Bool, enable_topic, 10)
        self._timer = self.create_timer(self._publish_period, self._publish_enable_flag)

        self.get_logger().info(
            'Corridor enable publisher started on %s (period: %.2f s)'
            % (enable_topic, self._publish_period)
        )

    def _publish_enable_flag(self) -> None:
        self._publisher.publish(self._enable_msg)


def main() -> None:
    rclpy.init()
    node = CorridorEnablePublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()