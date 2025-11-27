from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from sensor_msgs.msg import LaserScan


@dataclass(frozen=True)
class GapFinderConfig:
    """Configuration for DBSCAN-based clustering on each side."""

    half_fov: float
    max_range: float
    cluster_eps: float
    cluster_min_points: int
    cluster_distance_threshold: float


@dataclass
class ProcessResult:
    left_point: Optional[Tuple[float, float]]
    right_point: Optional[Tuple[float, float]]
    gap_within_tolerance: bool
    clusters: List[Tuple[float, float]]


class GapFinderProcessor:
    """Scan processor that uses DBSCAN clusters on each half of the scan."""

    def __init__(self, config: GapFinderConfig, logger=None) -> None:
        self._cfg = config
        self._logger = logger

    def process_scan(self, scan: LaserScan) -> ProcessResult:
        left_points, right_points = self._split_points(scan)
        left_clusters = self._cluster_points(left_points)
        right_clusters = self._cluster_points(right_points)

        cluster_points: List[Tuple[float, float]] = [
            (c['x'], c['y']) for c in left_clusters + right_clusters
        ]
        left_cluster = self._nearest_cluster(left_clusters)
        right_cluster = self._nearest_cluster(right_clusters)
        left_point = self._cluster_to_point(left_cluster)
        right_point = self._cluster_to_point(right_cluster)

        threshold = self._cfg.cluster_distance_threshold
        gap_within_tolerance = (
            left_cluster is not None
            and right_cluster is not None
            and left_cluster['range'] <= threshold
            and right_cluster['range'] <= threshold
        )

        return ProcessResult(
            left_point=left_point,
            right_point=right_point,
            gap_within_tolerance=gap_within_tolerance,
            clusters=cluster_points,
        )

    def _split_points(
        self,
        scan: LaserScan,
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, float]]]:
        left_points: List[Dict[str, float]] = []
        right_points: List[Dict[str, float]] = []
        total = len(scan.ranges)
        if total == 0:
            return left_points, right_points
        midpoint_index = total // 2
        angle = scan.angle_min
        for idx, rng in enumerate(scan.ranges):
            if not math.isfinite(rng):
                angle += scan.angle_increment
                continue
            rng = min(rng, self._cfg.max_range)
            if abs(angle) > self._cfg.half_fov:
                angle += scan.angle_increment
                continue
            x = rng * math.cos(angle)
            y = rng * math.sin(angle)
            point = {'x': x, 'y': y, 'range': rng, 'angle': angle}
            if idx < midpoint_index:
                left_points.append(point)
            else:
                right_points.append(point)
            angle += scan.angle_increment
        return left_points, right_points

    def _cluster_points(
        self,
        points: Sequence[Dict[str, float]],
    ) -> List[Dict[str, float]]:
        if len(points) < self._cfg.cluster_min_points:
            return []

        coords = [(p['x'], p['y']) for p in points]
        labels = [-1] * len(coords)
        cluster_id = 0
        for idx in range(len(coords)):
            if labels[idx] != -1:
                continue
            neighbors = self._region_query(coords, idx)
            if len(neighbors) + 1 < self._cfg.cluster_min_points:
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
                if len(nbr_neighbors) + 1 >= self._cfg.cluster_min_points:
                    queue.extend(nbr_neighbors)
            cluster_id += 1

        clusters: List[Dict[str, float]] = []
        for cid in range(cluster_id):
            members = [points[i] for i, label in enumerate(labels) if label == cid]
            if not members:
                continue
            avg_x = sum(p['x'] for p in members) / len(members)
            avg_y = sum(p['y'] for p in members) / len(members)
            avg_range = math.hypot(avg_x, avg_y)
            clusters.append(
                {
                    'x': avg_x,
                    'y': avg_y,
                    'range': avg_range,
                }
            )
        return clusters

    def _region_query(
        self, coords: Sequence[Tuple[float, float]], idx: int
    ) -> List[int]:
        neighbors: List[int] = []
        x1, y1 = coords[idx]
        for j, (x2, y2) in enumerate(coords):
            if idx == j:
                continue
            if math.hypot(x1 - x2, y1 - y2) <= self._cfg.cluster_eps:
                neighbors.append(j)
        return neighbors

    def _nearest_cluster(
        self, clusters: Sequence[Dict[str, float]]
    ) -> Optional[Dict[str, float]]:
        if not clusters:
            return None
        return min(clusters, key=lambda c: c['range'])

    def _cluster_to_point(
        self,
        cluster: Optional[Dict[str, float]],
    ) -> Optional[Tuple[float, float]]:
        if cluster is None:
            return None
        return (cluster['x'], cluster['y'])