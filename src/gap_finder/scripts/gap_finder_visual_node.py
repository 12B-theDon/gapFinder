#!/usr/bin/env python3

from __future__ import annotations

import math
from collections import deque
from typing import Dict, List, Optional

import rclpy
from geometry_msgs.msg import PointStamped, Twist
from rclpy.duration import Duration
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker, MarkerArray


class GapFinderVisualNode(Node):
    """Replay-only gap finder that publishes markers but keeps cmd_vel at zero."""

    def __init__(self) -> None:
        super().__init__('gap_finder_visual_node')

        self.declare_parameter('scan_topic', '/scan')
        self.declare_parameter('marker_topic', '/gap_finder/markers')
        self.declare_parameter('target_topic', '/gap_finder/target')
        self.declare_parameter('half_fov_deg', 40.0)
        self.declare_parameter('extended_half_fov_deg', 90.0)
        self.declare_parameter('fov_extend_distance', 0.2)
        self.declare_parameter('fov_extend_count', 10)
        self.declare_parameter('midpoint_blend_weight', 0.5)
        self.declare_parameter('max_considered_range', 8.0)
        self.declare_parameter('gap_jump_threshold', 0.5)
        self.declare_parameter('cluster_eps', 0.05)
        self.declare_parameter('cluster_min_points', 3)
        self.declare_parameter('status_log_interval_sec', 0.5)

        scan_topic = self.get_parameter('scan_topic').get_parameter_value().string_value
        self.marker_topic = self.get_parameter('marker_topic').get_parameter_value().string_value
        self.target_topic = self.get_parameter('target_topic').get_parameter_value().string_value
        self.half_fov = math.radians(
            self.get_parameter('half_fov_deg').get_parameter_value().double_value
        )
        self.extended_half_fov = math.radians(
            self.get_parameter('extended_half_fov_deg').get_parameter_value().double_value
        )
        self.fov_extend_distance = (
            self.get_parameter('fov_extend_distance').get_parameter_value().double_value
        )
        self.fov_extend_count = (
            self.get_parameter('fov_extend_count').get_parameter_value().integer_value
        )
        self.midpoint_blend_weight = max(
            0.0,
            min(
                1.0,
                self.get_parameter('midpoint_blend_weight')
                .get_parameter_value()
                .double_value,
            ),
        )
        self.max_considered_range = (
            self.get_parameter('max_considered_range').get_parameter_value().double_value
        )
        self.gap_jump_threshold = (
            self.get_parameter('gap_jump_threshold').get_parameter_value().double_value
        )
        self.cluster_eps = self.get_parameter('cluster_eps').get_parameter_value().double_value
        self.cluster_min_points = (
            self.get_parameter('cluster_min_points').get_parameter_value().integer_value
        )
        self.status_log_interval = (
            self.get_parameter('status_log_interval_sec').get_parameter_value().double_value
        )

        self.create_subscription(LaserScan, scan_topic, self.scan_callback, 10)
        self.marker_pub = self.create_publisher(MarkerArray, self.marker_topic, 10)
        self.target_pub = self.create_publisher(PointStamped, self.target_topic, 10)
        self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)

        self._last_log_time: Time | None = None
        self._extended_fov = False
        self.get_logger().info('Visualization-only gap finder ready.')

    # ------------------------------------------------------------------
    def scan_callback(self, scan: LaserScan) -> None:
        extend_fov = self._should_extend_fov(scan)
        active_half_fov = self.extended_half_fov if extend_fov else self.half_fov

        points = self._collect_points(scan, half_fov=active_half_fov)
        full_points = self._collect_points(scan, half_fov=math.pi)
        filtered_points, left_sequence, right_sequence = self._filter_center_out(
            points, override_half_fov=active_half_fov
        )
        left_point, right_point = self._detect_gate(left_sequence, right_sequence)
        clusters = self._cluster_filtered_points(filtered_points)
        markers = self._build_markers(
            left_point,
            right_point,
            frame_id=scan.header.frame_id or 'base_link',
            stamp=scan.header.stamp,
        )
        if markers.markers:
            self.marker_pub.publish(markers)
        full_clusters = self._cluster_filtered_points(full_points)
        self._publish_filtered_clusters(
            clusters,
            frame_id=scan.header.frame_id or 'base_link',
            stamp=scan.header.stamp,
            all_clusters=full_clusters,
        )
        self._publish_zero_cmd()
        self._maybe_log(filtered_points, left_point, right_point)

    def _collect_points(self, scan: LaserScan, *, half_fov: float) -> List[Dict[str, float]]:
        angle = scan.angle_min
        points: List[Dict[str, float]] = []
        for rng in scan.ranges:
            if not math.isfinite(rng):
                angle += scan.angle_increment
                continue
            rng = min(rng, self.max_considered_range)
            if abs(angle) <= half_fov:
                x = rng * math.cos(angle)
                y = rng * math.sin(angle)
                points.append({'angle': angle, 'range': rng, 'x': x, 'y': y})
            angle += scan.angle_increment
        return points

    def _should_extend_fov(self, scan: LaserScan) -> bool:
        left_points = 0
        right_points = 0
        angle = scan.angle_min
        for rng in scan.ranges:
            if not math.isfinite(rng):
                angle += scan.angle_increment
                continue
            if abs(angle) <= self.extended_half_fov and rng <= self.fov_extend_distance:
                if angle >= 0.0:
                    left_points += 1
                else:
                    right_points += 1
            angle += scan.angle_increment
        extend = left_points >= self.fov_extend_count and right_points >= self.fov_extend_count
        if extend != self._extended_fov:
            self._extended_fov = extend
            state = 'EXTENDED' if extend else 'DEFAULT'
            self.get_logger().info(
                'Visual FOV mode -> %s (left_count=%d, right_count=%d)',
                state,
                left_points,
                right_points,
            )
        return extend

    def _filter_center_out(
        self, points: List[Dict[str, float]], *, override_half_fov: Optional[float] = None
    ) -> tuple[List[Dict[str, float]], List[Dict[str, float]], List[Dict[str, float]]]:
        half_fov = override_half_fov if override_half_fov is not None else self.half_fov
        zero_index = next((i for i, p in enumerate(points) if p['angle'] >= 0.0), len(points))

        pos_indices: List[int] = []
        for i in range(zero_index, len(points)):
            if points[i]['angle'] > half_fov:
                break
            pos_indices.append(i)

        neg_indices: List[int] = []
        for i in range(zero_index - 1, -1, -1):
            if points[i]['angle'] < -half_fov:
                break
            neg_indices.append(i)

        def apply_filter(seq: List[int]) -> List[Dict[str, float]]:
            filtered: List[Dict[str, float]] = []
            prev_range: Optional[float] = None
            for idx in seq:
                point = points[idx]
                if prev_range is None:
                    filtered.append(point)
                    prev_range = point['range']
                    continue
                delta = point['range'] - prev_range
                if delta > self.gap_jump_threshold:
                    continue
                filtered.append(point)
                prev_range = point['range']
            return filtered

        positive_sequence = apply_filter(pos_indices)
        negative_sequence = apply_filter(neg_indices)
        filtered_points = positive_sequence + negative_sequence
        return filtered_points, negative_sequence, positive_sequence

    def _detect_gate(
        self,
        left_points: List[Dict[str, float]],
        right_points: List[Dict[str, float]],
    ) -> tuple[Optional[Dict[str, float]], Optional[Dict[str, float]]]:
        left_point: Optional[Dict[str, float]] = None
        if left_points:
            left_point = min(left_points, key=lambda p: p['range'])
            for i in range(len(left_points) - 1):
                delta = left_points[i + 1]['range'] - left_points[i]['range']
                if delta > self.gap_jump_threshold:
                    left_point = left_points[i]
                    break

        right_point: Optional[Dict[str, float]] = None
        if right_points:
            right_point = min(right_points, key=lambda p: p['range'])
            for i in range(len(right_points) - 1):
                delta = right_points[i + 1]['range'] - right_points[i]['range']
                if delta < -self.gap_jump_threshold:
                    right_point = right_points[i + 1]
                    break

        return left_point, right_point

    def _build_markers(
        self,
        left_point: Optional[Dict[str, float]],
        right_point: Optional[Dict[str, float]],
        *,
        frame_id: str,
        stamp,
    ) -> MarkerArray:
        marker_array = MarkerArray()
        idx = 0

        def append_marker(role: str, point: Dict[str, float]) -> None:
            nonlocal idx
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = 'viz_gap_points'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = point['x']
            marker.pose.position.y = point['y']
            marker.scale.x = marker.scale.y = marker.scale.z = 0.1
            if role == 'left':
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.2, 0.2
            elif role == 'right':
                marker.color.r, marker.color.g, marker.color.b = 0.2, 0.4, 1.0
            else:
                marker.color.r, marker.color.g, marker.color.b = 0.2, 1.0, 0.4
            marker.color.a = 0.9
            marker.lifetime = Duration(seconds=0.2).to_msg()
            marker_array.markers.append(marker)
            idx += 1

        if left_point:
            append_marker('left', left_point)
        if right_point:
            append_marker('right', right_point)
        if left_point and right_point:
            mid_point = {
                'x': 0.5 * (left_point['x'] + right_point['x']),
                'y': 0.5 * (left_point['y'] + right_point['y']),
            }
            append_marker('mid', mid_point)

        return marker_array

    def _cluster_filtered_points(
        self, points: List[Dict[str, float]]
    ) -> List[Dict[str, float]]:
        if len(points) < self.cluster_min_points:
            return []

        coords = [(p['x'], p['y']) for p in points]
        labels = [-1] * len(coords)
        cluster_id = 0
        for idx in range(len(coords)):
            if labels[idx] != -1:
                continue
            neighbors = self._region_query(coords, idx)
            if len(neighbors) < self.cluster_min_points:
                labels[idx] = -2
                continue
            labels[idx] = cluster_id
            queue = deque(neighbors)
            while queue:
                nbr = queue.popleft()
                if labels[nbr] == -2:
                    labels[nbr] = cluster_id
                if labels[nbr] != -1:
                    continue
                labels[nbr] = cluster_id
                nbr_neighbors = self._region_query(coords, nbr)
                if len(nbr_neighbors) >= self.cluster_min_points:
                    queue.extend(nbr_neighbors)
            cluster_id += 1

        clusters: List[Dict[str, float]] = []
        for cid in range(cluster_id):
            members = [points[i] for i, label in enumerate(labels) if label == cid]
            if not members:
                continue
            avg_x = sum(p['x'] for p in members) / len(members)
            avg_y = sum(p['y'] for p in members) / len(members)
            avg_range = sum(p['range'] for p in members) / len(members)
            clusters.append(
                {
                    'x': avg_x,
                    'y': avg_y,
                    'range': avg_range,
                    'angle': math.atan2(avg_y, avg_x),
                }
            )
        return clusters

    def _region_query(self, coords: List[tuple[float, float]], idx: int) -> List[int]:
        neighbors: List[int] = []
        x1, y1 = coords[idx]
        for j, (x2, y2) in enumerate(coords):
            if idx == j:
                continue
            if math.hypot(x1 - x2, y1 - y2) <= self.cluster_eps:
                neighbors.append(j)
        return neighbors

    def _publish_filtered_clusters(
        self,
        clusters: List[Dict[str, float]],
        *,
        frame_id: str,
        stamp,
        all_clusters: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        if not clusters and not all_clusters:
            return

        marker_array = MarkerArray()
        primary_clusters = list(clusters)
        selection_pool = list(all_clusters) if all_clusters is not None else primary_clusters
        far_left_cluster = None
        far_right_cluster = None
        near_left_cluster: Optional[Dict[str, float]] = None
        near_right_cluster: Optional[Dict[str, float]] = None
        if len(primary_clusters) >= 2:
            far_left_cluster = max(primary_clusters, key=lambda c: c['angle'])
            far_right_cluster = min(primary_clusters, key=lambda c: c['angle'])

        left_candidates = [c for c in selection_pool if c['angle'] >= 0.0]
        right_candidates = [c for c in selection_pool if c['angle'] < 0.0]
        if left_candidates:
            near_left_cluster = min(left_candidates, key=lambda c: c['range'])
        if right_candidates:
            near_right_cluster = min(right_candidates, key=lambda c: c['range'])

        def clusters_match(a: Optional[Dict[str, float]], b: Dict[str, float]) -> bool:
            if a is None:
                return False
            return (
                abs(a['x'] - b['x']) < 1e-3
                and abs(a['y'] - b['y']) < 1e-3
                and abs(a['range'] - b['range']) < 1e-3
            )

        highlighted_left = False
        highlighted_right = False
        target_point: Optional[tuple[float, float]] = None
        far_midpoint_coords: Optional[tuple[float, float]] = None

        for idx, cluster in enumerate(primary_clusters):
            marker = Marker()
            marker.header.frame_id = frame_id
            marker.header.stamp = stamp
            marker.ns = 'filtered_scan_points'
            marker.id = idx
            marker.type = Marker.SPHERE
            marker.action = Marker.ADD
            marker.pose.position.x = cluster['x']
            marker.pose.position.y = cluster['y']
            marker.pose.position.z = 0.0
            marker.scale.x = marker.scale.y = marker.scale.z = 0.08
            marker.color.r, marker.color.g, marker.color.b = 0.9, 0.1, 0.9
            if clusters_match(near_left_cluster, cluster):
                marker.color.r, marker.color.g, marker.color.b = 0.2, 0.4, 1.0
                highlighted_left = True
            elif clusters_match(near_right_cluster, cluster):
                marker.color.r, marker.color.g, marker.color.b = 0.2, 1.0, 0.4
                highlighted_right = True
            if far_left_cluster and clusters_match(far_left_cluster, cluster):
                marker.color.r, marker.color.g, marker.color.b = 1.0, 0.2, 0.2
            elif far_right_cluster and clusters_match(far_right_cluster, cluster):
                marker.color.r, marker.color.g, marker.color.b = 1.0, 1.0, 0.0
            marker.color.a = 0.9
            marker.lifetime = Duration(seconds=0.2).to_msg()
            marker_array.markers.append(marker)

        next_marker_id = len(primary_clusters)
        midpoint_marker = None
        if far_left_cluster and far_right_cluster:
            mid_x = 0.5 * (far_left_cluster['x'] + far_right_cluster['x'])
            mid_y = 0.5 * (far_left_cluster['y'] + far_right_cluster['y'])
            far_midpoint_coords = (mid_x, mid_y)
            midpoint_marker = Marker()
            midpoint_marker.header.frame_id = frame_id
            midpoint_marker.header.stamp = stamp
            midpoint_marker.ns = 'filtered_scan_points'
            midpoint_marker.id = next_marker_id
            midpoint_marker.type = Marker.SPHERE
            midpoint_marker.action = Marker.ADD
            midpoint_marker.pose.position.x = mid_x
            midpoint_marker.pose.position.y = mid_y
            midpoint_marker.pose.position.z = 0.0
            midpoint_marker.scale.x = midpoint_marker.scale.y = midpoint_marker.scale.z = 0.12
            midpoint_marker.color.r = 1.0
            midpoint_marker.color.g = 0.5
            midpoint_marker.color.b = 0.1
            midpoint_marker.color.a = 0.95
            midpoint_marker.lifetime = Duration(seconds=0.2).to_msg()
            marker_array.markers.append(midpoint_marker)

            # heading = math.degrees(math.atan2(mid_y, mid_x))
            # self.get_logger().info(
            #     f'Filtered midpoint heading {heading:.1f} deg '
            #     f"(left_y={far_left_cluster['y']:.2f}, right_y={far_right_cluster['y']:.2f})"
            # )
        if near_left_cluster and near_right_cluster:
            mid_x = 0.5 * (near_left_cluster['x'] + near_right_cluster['x'])
            mid_y = 0.5 * (near_left_cluster['y'] + near_right_cluster['y'])
            near_midpoint = Marker()
            near_midpoint.header.frame_id = frame_id
            near_midpoint.header.stamp = stamp
            near_midpoint.ns = 'filtered_scan_points'
            near_midpoint.id = next_marker_id + 1
            near_midpoint.type = Marker.SPHERE
            near_midpoint.action = Marker.ADD
            near_midpoint.pose.position.x = mid_x
            near_midpoint.pose.position.y = mid_y
            near_midpoint.pose.position.z = 0.0
            near_midpoint.scale.x = near_midpoint.scale.y = near_midpoint.scale.z = 0.1
            near_midpoint.color.r = near_midpoint.color.g = near_midpoint.color.b = 1.0
            near_midpoint.color.a = 0.9
            near_midpoint.lifetime = Duration(seconds=0.2).to_msg()
            marker_array.markers.append(near_midpoint)
            if midpoint_marker is not None:
                combo_midpoint = Marker()
                combo_midpoint.header.frame_id = frame_id
                combo_midpoint.header.stamp = stamp
                combo_midpoint.ns = 'filtered_scan_points'
                combo_midpoint.id = next_marker_id + 2
                combo_midpoint.type = Marker.SPHERE
                combo_midpoint.action = Marker.ADD
                w = self.midpoint_blend_weight
                combo_midpoint.pose.position.x = (1.0 - w) * midpoint_marker.pose.position.x + w * near_midpoint.pose.position.x
                combo_midpoint.pose.position.y = (1.0 - w) * midpoint_marker.pose.position.y + w * near_midpoint.pose.position.y
                combo_midpoint.pose.position.z = 0.0
                combo_midpoint.scale.x = combo_midpoint.scale.y = combo_midpoint.scale.z = 0.12
                combo_midpoint.color.r = 0.2
                combo_midpoint.color.g = 1.0
                combo_midpoint.color.b = 1.0
                combo_midpoint.color.a = 0.95
                combo_midpoint.lifetime = Duration(seconds=0.2).to_msg()
                marker_array.markers.append(combo_midpoint)
                target_point = (
                    combo_midpoint.pose.position.x,
                    combo_midpoint.pose.position.y,
                )

            if not highlighted_left and near_left_cluster is not None:
                marker_array.markers.append(
                    self._create_explicit_marker(
                        near_left_cluster,
                        frame_id,
                        stamp,
                        next_marker_id + 3,
                        (0.2, 0.4, 1.0),
                    )
                )
                next_marker_id += 1
            if not highlighted_right and near_right_cluster is not None:
                marker_array.markers.append(
                    self._create_explicit_marker(
                        near_right_cluster,
                        frame_id,
                        stamp,
                        next_marker_id + 4,
                        (0.2, 1.0, 0.4),
                    )
                )

        self.marker_pub.publish(marker_array)
        if target_point is None and far_midpoint_coords is not None:
            target_point = far_midpoint_coords
        self._publish_target_point(target_point, frame_id, stamp)

    def _create_explicit_marker(
        self,
        cluster: Dict[str, float],
        frame_id: str,
        stamp,
        marker_id: int,
        color: tuple[float, float, float],
    ) -> Marker:
        marker = Marker()
        marker.header.frame_id = frame_id
        marker.header.stamp = stamp
        marker.ns = 'filtered_scan_points'
        marker.id = marker_id
        marker.type = Marker.SPHERE
        marker.action = Marker.ADD
        marker.pose.position.x = cluster['x']
        marker.pose.position.y = cluster['y']
        marker.pose.position.z = 0.0
        marker.scale.x = marker.scale.y = marker.scale.z = 0.08
        marker.color.r, marker.color.g, marker.color.b = color
        marker.color.a = 0.9
        marker.lifetime = Duration(seconds=0.2).to_msg()
        return marker

    def _publish_target_point(
        self,
        point: Optional[tuple[float, float]],
        frame_id: str,
        stamp,
    ) -> None:
        if not self.target_pub or point is None:
            return
        msg = PointStamped()
        msg.header.frame_id = frame_id
        msg.header.stamp = stamp
        msg.point.x, msg.point.y = point
        msg.point.z = 0.0
        self.target_pub.publish(msg)

    def _publish_zero_cmd(self) -> None:
        twist = Twist()
        self.cmd_pub.publish(twist)

    def _maybe_log(
        self,
        points: List[Dict[str, float]],
        left_point: Optional[Dict[str, float]],
        right_point: Optional[Dict[str, float]],
    ) -> None:
        now = self.get_clock().now()
        if (
            self._last_log_time is not None
            and (now - self._last_log_time) < Duration(seconds=self.status_log_interval)
        ):
            return
        self._last_log_time = now
        if left_point and right_point:
            gap_width = math.hypot(
                left_point['x'] - right_point['x'], left_point['y'] - right_point['y']
            )
            # self.get_logger().info(
            #     f'Gate detected: left={left_point["range"]:.2f}m '
            #     f'right={right_point["range"]:.2f}m gap={gap_width:.2f}m (points={len(points)})'
            # )
        elif left_point or right_point:
            point = left_point if left_point is not None else right_point
            assert point is not None
            self.get_logger().info(
                f'Single cone at {point["range"]:.2f}m (points={len(points)})'
            )
        else:
            self.get_logger().info(f'No gate detected (points={len(points)})')


def main() -> None:
    rclpy.init()
    node = GapFinderVisualNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
