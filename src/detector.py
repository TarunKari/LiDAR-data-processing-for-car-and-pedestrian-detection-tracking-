"""
detector.py — DBSCAN Point Cloud Clustering & Bounding Box Extraction
Each cluster is a potential object (car, pedestrian, cyclist, etc.)
"""

import numpy as np
import logging
from dataclasses import dataclass, field
from typing import List, Tuple

from sklearn.cluster import DBSCAN

from config import CLUSTER

logger = logging.getLogger(__name__)


@dataclass
class Cluster:
    """Represents a single detected point cluster."""
    cluster_id: int
    frame_id: int
    points_xyz: np.ndarray      # (N, 3)
    intensity: np.ndarray       # (N,)
    distance: np.ndarray        # (N,)
    ambient: np.ndarray         # (N,)

    # Computed properties (filled by Detector)
    centroid: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_min: np.ndarray = field(default_factory=lambda: np.zeros(3))
    bbox_max: np.ndarray = field(default_factory=lambda: np.zeros(3))

    # Classification result (filled by Classifier)
    label: str = "unknown"
    label_id: int = 4
    confidence: float = 0.0

    # Tracking result (filled by Tracker)
    track_id: int = -1

    def __post_init__(self):
        if len(self.points_xyz) > 0:
            self.centroid = self.points_xyz.mean(axis=0)
            self.bbox_min = self.points_xyz.min(axis=0)
            self.bbox_max = self.points_xyz.max(axis=0)

    @property
    def num_points(self) -> int:
        return len(self.points_xyz)

    @property
    def bbox_dimensions(self) -> Tuple[float, float, float]:
        """Returns (length, width, height) = (dx, dy, dz) of bounding box."""
        diff = self.bbox_max - self.bbox_min
        return float(diff[0]), float(diff[1]), float(diff[2])

    @property
    def bbox_volume(self) -> float:
        l, w, h = self.bbox_dimensions
        return l * w * h

    @property
    def xy_extent(self) -> float:
        l, w, _ = self.bbox_dimensions
        return max(l, w)

    def to_dict(self) -> dict:
        l, w, h = self.bbox_dimensions
        cx, cy, cz = self.centroid
        return {
            "cluster_id": self.cluster_id,
            "frame_id": self.frame_id,
            "num_points": self.num_points,
            "centroid_x": round(cx, 3),
            "centroid_y": round(cy, 3),
            "centroid_z": round(cz, 3),
            "bbox_length": round(l, 3),
            "bbox_width": round(w, 3),
            "bbox_height": round(h, 3),
            "bbox_volume": round(self.bbox_volume, 4),
            "mean_intensity": round(float(self.intensity.mean()), 2),
            "mean_distance": round(float(self.distance.mean()), 3),
            "label": self.label,
            "label_id": self.label_id,
            "confidence": round(self.confidence, 4),
            "track_id": self.track_id,
        }


class Detector:
    """
    Applies DBSCAN clustering to a preprocessed point cloud and
    extracts individual object clusters with bounding boxes.
    """

    def __init__(self):
        self.cfg = CLUSTER
        self._dbscan = DBSCAN(
            eps=self.cfg["eps"],
            min_samples=self.cfg["min_samples"],
            algorithm="ball_tree",
            n_jobs=-1,
        )

    def detect(self, preprocess_result, frame_id: int = 0) -> List[Cluster]:
        """
        Run clustering on a PreprocessResult.
        Returns list of Cluster objects (one per detected object candidate).
        """
        pts = preprocess_result.points_xyz
        intensity = preprocess_result.intensity
        distance = preprocess_result.distance
        ambient = preprocess_result.ambient

        if len(pts) < self.cfg["min_samples"]:
            logger.warning(f"Frame {frame_id}: too few points for clustering ({len(pts)})")
            return []

        # Run DBSCAN on XYZ
        labels = self._dbscan.fit_predict(pts)
        unique_labels = set(labels) - {-1}  # -1 = noise

        clusters = []
        for lbl in unique_labels:
            mask = labels == lbl
            n_pts = mask.sum()

            # Size gate
            if n_pts < self.cfg["min_cluster_size"]:
                continue
            if n_pts > self.cfg["max_cluster_size"]:
                continue

            c = Cluster(
                cluster_id=int(lbl),
                frame_id=frame_id,
                points_xyz=pts[mask],
                intensity=intensity[mask],
                distance=distance[mask],
                ambient=ambient[mask],
            )
            clusters.append(c)

        noise_pts = (labels == -1).sum()
        logger.debug(
            f"Frame {frame_id}: {len(pts)} pts → "
            f"{len(unique_labels)} raw clusters → {len(clusters)} valid "
            f"({noise_pts} noise pts)"
        )
        return clusters