from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sensor_msgs.msg import LaserScan


@dataclass(frozen=True)
class GapFinderConfig:
    fov_min: float  # Left side of FOV in degrees
    fov_max: float  # Right side of FOV in degrees
    max_range: float
    cluster_eps: float
    cluster_min_points: int
    corridor_threshold: float
    desired_gap_width: float
    gap_width_tolerance: float


@dataclass
class ProcessResult:
    left_point: Optional[Tuple[float, float]]
    right_point: Optional[Tuple[float, float]]
    midpoint: Optional[Tuple[float, float]]
    gap_width: Optional[float]
    gap_within_tolerance: bool
    clusters: List[Tuple[float, float]]


class GapFinderProcessor:
    """Scan processor that detects the nearest left/right cones in the FOV."""

    def __init__(self, config: GapFinderConfig, logger=None) -> None:
        self._cfg = config
        
    def process_scan(self, scan: LaserScan) -> ProcessResult:
        forward_points = self._collect_points(scan)
        all_points = self._collect_points(scan)  # For all points (full scan)
        global_median = self._compute_global_median(scan)

        forward_clusters = self._cluster_points(forward_points)
        all_clusters = self._cluster_points(all_points)

        left_point: Optional[Tuple[float, float]] = None
        right_point: Optional[Tuple[float, float]] = None
        midpoint: Optional[Tuple[float, float]] = None
        gap_width: Optional[float] = None
        gap_within_tolerance = False
        cluster_points: List[Tuple[float, float]] = [(c['x'], c['y']) for c in forward_clusters]
        corridor_ready = global_median >= self._cfg.corridor_threshold
        if not corridor_ready:
            corridor_ready = self._has_balanced_corridor(all_clusters)
        if corridor_ready:
            roles = self._assign_roles(forward_clusters, all_clusters)
            near_left = roles.get('near_left')
            near_right = roles.get('near_right')
            if near_left:
                left_point = (near_left['x'], near_left['y'])
            if near_right:
                right_point = (near_right['x'], near_right['y'])
            if near_left and near_right:
                midpoint = self._midpoint(near_left, near_right)
                gap_width = math.hypot(
                    near_left['x'] - near_right['x'], near_left['y'] - near_right['y']
                )
                if gap_width is not None:
                    gap_within_tolerance = (
                        abs(gap_width - self._cfg.desired_gap_width)
                        <= self._cfg.gap_width_tolerance
                    )

        return ProcessResult(
            left_point=left_point,
            right_point=right_point,
            midpoint=midpoint,
            gap_width=gap_width,
            gap_within_tolerance=gap_within_tolerance,
            clusters=cluster_points,
        )

    def _collect_points(
        self,
        scan: LaserScan
    ) -> List[Dict[str, float]]:
        points: List[Dict[str, float]] = []
        angle = scan.angle_min
        # Filtering the points based on the fov_min and fov_max
        for rng in scan.ranges:
            if not math.isfinite(rng):
                angle += scan.angle_increment
                continue
            rng = min(rng, self._cfg.max_range)

            # Calculate angle in degrees
            angle_deg = math.degrees(angle)

            if self._cfg.fov_min <= angle_deg <= self._cfg.fov_max:
                x = rng * math.cos(angle)
                y = rng * math.sin(angle)
                points.append({'x': x, 'y': y, 'range': rng, 'angle': angle_deg})
            
            angle += scan.angle_increment
        return points

    def _compute_global_median(self, scan: LaserScan) -> float:
        values: List[float] = []
        angle = scan.angle_min
        for rng in scan.ranges:
            if not math.isfinite(rng):
                angle += scan.angle_increment
                continue
            values.append(min(rng, self._cfg.max_range))
            angle += scan.angle_increment
        if not values:
            return self._cfg.max_range
        values.sort()
        mid = len(values) // 2
        if len(values) % 2 == 0:
            return 0.5 * (values[mid - 1] + values[mid])
        return values[mid]

    def _cluster_points(self, points: Sequence[Dict[str, float]]) -> List[Dict[str, float]]:
        if len(points) < self._cfg.cluster_min_points:
            return []

        coords = [(p['x'], p['y']) for p in points]
        labels = [-1] * len(coords)
        cluster_id = 0
        for idx in range(len(coords)):
            if labels[idx] != -1:
                continue
            neighbors = self._region_query(coords, idx)
            if len(neighbors) < self._cfg.cluster_min_points:
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
                if len(nbr_neighbors) >= self._cfg.cluster_min_points:
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

    def _region_query(self, coords: Sequence[Tuple[float, float]], idx: int) -> List[int]:
        neighbors: List[int] = []
        x1, y1 = coords[idx]
        for j, (x2, y2) in enumerate(coords):
            if idx == j:
                continue
            if math.hypot(x1 - x2, y1 - y2) <= self._cfg.cluster_eps:
                neighbors.append(j)
        return neighbors

    def _assign_roles(
        self,
        forward_clusters: Sequence[Dict[str, float]],
        all_clusters: Sequence[Dict[str, float]],
    ) -> Dict[str, Optional[Dict[str, float]]]:
        roles: Dict[str, Optional[Dict[str, float]]] = {
            'far_left': None,
            'far_right': None,
            'near_left': None,
            'near_right': None,
        }
        if len(forward_clusters) >= 2:
            roles['far_left'] = max(forward_clusters, key=lambda c: c['angle'])
            roles['far_right'] = min(forward_clusters, key=lambda c: c['angle'])

        if all_clusters:
            left_candidates = [c for c in all_clusters if c['angle'] >= 0.0]
            right_candidates = [c for c in all_clusters if c['angle'] < 0.0]
            if left_candidates:
                roles['near_left'] = min(left_candidates, key=lambda c: c['range'])
            if right_candidates:
                roles['near_right'] = min(right_candidates, key=lambda c: c['range'])
        return roles

    def _has_balanced_corridor(self, clusters: Sequence[Dict[str, float]]) -> bool:
        """Detects straight corridors by checking for clusters on both sides of the robot."""
        if not clusters:
            return False
        left_present = any(cluster['angle'] > 0.0 for cluster in clusters)
        right_present = any(cluster['angle'] < 0.0 for cluster in clusters)
        return left_present and right_present

    def _midpoint(
        self,
        a: Optional[Dict[str, float]],
        b: Optional[Dict[str, float]],
    ) -> Optional[Tuple[float, float]]:
        if not a or not b:
            return None
        return (
            0.5 * (a['x'] + b['x']),
            0.5 * (a['y'] + b['y']),
        )
