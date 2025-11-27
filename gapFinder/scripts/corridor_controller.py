#!/usr/bin/env python3

from __future__ import annotations

import math
from typing import Dict, Iterable, Tuple

import rclpy
from geometry_msgs.msg import Twist
from rclpy.node import Node
from sensor_msgs.msg import LaserScan

SectorDef = Tuple[str, float, float]


def _sector(name: str, start_deg: float, end_deg: float) -> SectorDef:
    return (name, math.radians(start_deg), math.radians(end_deg))



class CorridorController(Node):
    """Skeleton corridor controller waiting for the new cone-avoidance logic."""

    def __init__(self) -> None:
        super().__init__('corridor_controller')

        self.scan_topic = self.declare_parameter('scan_topic', '/scan').value
        self.cmd_vel_topic = self.declare_parameter('cmd_vel_topic', '/cmd_vel').value

        # Hyper-parameters for future cone avoidance logic
        self.free_considered_range = self.declare_parameter('free_considered_range', 0.35).value
        self.obstacle_detect_distance = self.declare_parameter('obstacle_detect_distance', 0.33).value
        self.default_speed = self.declare_parameter('default_speed', 0.12).value
        self.turning_speed = self.declare_parameter('turning_speed', 0.1).value
        self.front_LR_steering_deadband = self.declare_parameter(
            'front_LR_steering_deadband', 0.01
        ).value
        self.F_FL_FR_similar_threshold = self.declare_parameter(
            'F_FL_FR_similar_threshold', 0.05
        ).value
        self.half_front_degree_range = self.declare_parameter(
            'half_front_degree_range', 30.0
        ).value
        self.last_LR_mean_steering_deadband = self.declare_parameter(
            'last_LR_mean_steering_deadband', 0.05
        ).value
        self.steering_smoothing_gain = self.declare_parameter(
            'steering_smoothing_gain', 0.2
        ).value

        self._configure_sectors()
        self._smoothed_angular = 0.0

        self.create_subscription(LaserScan, self.scan_topic, self.scan_callback, 10)
        self.cmd_pub = self.create_publisher(Twist, self.cmd_vel_topic, 10)

    def scan_callback(self, scan: LaserScan) -> None:
        major_distances = self._compute_sector_distances(scan, self.major_sectors)
        near_distances = self._compute_sector_distances(scan, self.near_sectors)

        median_left_distance = self._compute_sector_median(scan, self.sector_lookup['left'])
        median_right_distance = self._compute_sector_median(scan, self.sector_lookup['right'])
        mean_left_distance = self._compute_sector_mean(scan, self.sector_lookup['left'])
        mean_right_distance = self._compute_sector_mean(scan, self.sector_lookup['right'])
        mean_back_left = self._compute_sector_mean(scan, self.sector_lookup['back_left'])
        mean_back_right = self._compute_sector_mean(scan, self.sector_lookup['back_right'])

        front_similar = self._front_sectors_similar(
            near_distances['front_left'],
            major_distances['front'],
            near_distances['front_right'],
        )
        if front_similar:
            left_score = median_left_distance
            right_score = median_right_distance
        else:
            left_score = near_distances['front_left']
            right_score = near_distances['front_right']

        twist = Twist()
        twist.linear.x = self.default_speed

        diff = left_score - right_score
        deadband = self.front_LR_steering_deadband
        steer_direction = 'straight'
        target_angular = 0.0
        mean_mode = 'N/A'
        mean_diff = mean_left_distance - mean_right_distance

        if diff > deadband:
            target_angular = self.turning_speed
            steer_direction = 'left'
        elif diff < -deadband:
            target_angular = -self.turning_speed
            steer_direction = 'right'
        else:
            median_diff = median_left_distance - median_right_distance
            if median_diff > deadband:
                target_angular = self.turning_speed
                steer_direction = 'left'
            elif median_diff < -deadband:
                target_angular = -self.turning_speed
                steer_direction = 'right'
            else:
                mean_diff = mean_left_distance - mean_right_distance
                mean_deadband = self.last_LR_mean_steering_deadband
                mean_mode = 'FALSE'
                if mean_diff > mean_deadband:
                    target_angular = self.turning_speed
                    steer_direction = 'left'
                    mean_mode = 'TRUE'
                elif mean_diff < -mean_deadband:
                    target_angular = -self.turning_speed
                    steer_direction = 'right'
                    mean_mode = 'TRUE'

        twist.angular.z = self._smooth_angular(target_angular)

        self.cmd_pub.publish(twist)

        sector_lines = self._format_sector_lines(
            major_distances,
            near_distances,
            median_left_distance,
            median_right_distance,
            mean_back_left,
            mean_back_right,
            mean_left_distance,
            mean_right_distance,
        )
        log = '\n'.join(
            [
                'Sector distances',
                *sector_lines,
                '===========================',
                'Steer: %s (diff=%.3f m)' % (steer_direction, diff),
                'FL_F_FR: %s' % ('TRUE' if front_similar else 'FALSE'),
                'LR_mean_mode: %s (diff=%.3f m)' % (mean_mode, mean_diff),
                'linear: %.2f || angular: %.2f' % (twist.linear.x, twist.angular.z),
                '=======*****************======',
            ]
        )
        self.get_logger().info(log)

    def _compute_sector_distances(
        self,
        scan: LaserScan,
        sectors: Iterable[SectorDef],
    ) -> Dict[str, float]:
        distances: Dict[str, float] = {
            name: math.inf for name, _, _ in sectors
        }

        angle = scan.angle_min
        for rng in scan.ranges:
            if math.isfinite(rng):
                for name, start, end in sectors:
                    if self._angle_in_interval(angle, start, end):
                        distances[name] = min(distances[name], rng)
            angle += scan.angle_increment

        return {
            name: (self.free_considered_range if math.isinf(value) else value)
            for name, value in distances.items()
        }

    @staticmethod
    def _angle_in_interval(angle: float, start: float, end: float) -> bool:
        if start <= end:
            return start <= angle <= end
        return angle >= start or angle <= end

    def _compute_sector_median(self, scan: LaserScan, sector: SectorDef) -> float:
        values = self._collect_sector_values(scan, sector)
        if not values:
            return self.free_considered_range
        values.sort()
        mid = len(values) // 2
        if len(values) % 2 == 1:
            return values[mid]
        return 0.5 * (values[mid - 1] + values[mid])

    def _compute_sector_mean(self, scan: LaserScan, sector: SectorDef) -> float:
        values = self._collect_sector_values(scan, sector)
        if not values:
            return self.free_considered_range
        return sum(values) / len(values)

    def _collect_sector_values(self, scan: LaserScan, sector: SectorDef) -> list[float]:
        name, start, end = sector
        values: list[float] = []
        angle = scan.angle_min
        for rng in scan.ranges:
            if math.isfinite(rng) and self._angle_in_interval(angle, start, end):
                values.append(rng)
            angle += scan.angle_increment
        return values

    def _front_sectors_similar(self, front_left: float, front: float, front_right: float) -> bool:
        threshold = self.F_FL_FR_similar_threshold
        return (
            abs(front_left - front) <= threshold
            and abs(front_right - front) <= threshold
            and abs(front_left - front_right) <= threshold
        )

    def _smooth_angular(self, target: float) -> float:
        gain = max(0.0, min(1.0, self.steering_smoothing_gain))
        self._smoothed_angular += gain * (target - self._smoothed_angular)
        return self._smoothed_angular

    def _configure_sectors(self) -> None:
        half = float(self.half_front_degree_range)
        half = max(1.0, min(89.0, half))
        far_half = 180.0 - half

        self.major_sectors = (
            _sector('front', -half, half),
            _sector('left', half, far_half),
            _sector('back', far_half, -far_half),
            _sector('right', -far_half, -half),
        )

        back_left_end = 180.0 - half
        back_right_start = -180.0 + half

        self.near_sectors = (
            _sector('front_left', half, 90.0),
            _sector('front_right', -90.0, -half),
            _sector('back_left', 90.0, back_left_end),
            _sector('back_right', back_right_start, -90.0),
        )

        self.sector_lookup = {
            sector[0]: sector for sector in (*self.major_sectors, *self.near_sectors)
        }

    def _format_sector_lines(
        self,
        major: Dict[str, float],
        near: Dict[str, float],
        median_left: float,
        median_right: float,
        mean_back_left: float,
        mean_back_right: float,
        mean_left: float,
        mean_right: float,
    ) -> Iterable[str]:
        def fmt(value: float) -> str:
            return 'free' if value > self.free_considered_range else f'{value:.2f}'

        return (
            'front_left: %s || front: %s || front_right: %s'
            % (fmt(near['front_left']), fmt(major['front']), fmt(near['front_right'])),
            'back_left: %s || back: %s || back_right: %s'
            % (fmt(near['back_left']), fmt(major['back']), fmt(near['back_right'])),
            'median_left: %s || median_right: %s'
            % (fmt(median_left), fmt(median_right)),
            'mean_back_left: %s || mean_back_right: %s'
            % (fmt(mean_back_left), fmt(mean_back_right)),
            'mean_left: %s || mean_right: %s' % (fmt(mean_left), fmt(mean_right)),
        )


def main() -> None:
    rclpy.init()
    node = CorridorController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()