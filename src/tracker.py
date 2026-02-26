"""
tracker.py — Multi-Object Kalman Filter Tracker
State vector: [x, y, vx, vy]   (2D centroid position + velocity)
Measurement:  [x, y]           (centroid position from detector)

Uses Hungarian algorithm for optimal assignment.
Tracks confirmed after min_hits frames; deleted after max_age missed frames.
"""

import numpy as np
import logging
from typing import List, Dict, Tuple
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist

from config import TRACK
from src.detector import Cluster

logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────
#  KALMAN FILTER (constant velocity model)
# ──────────────────────────────────────────────

class KalmanFilter2D:
    """
    Simple constant-velocity 2D Kalman filter for object tracking.
    State: [x, y, vx, vy]
    """

    def __init__(self, initial_xy: np.ndarray, dt: float,
                 process_noise: float, measurement_noise: float):
        self.dt = dt
        n_state = 4
        n_meas = 2

        # State transition matrix (constant velocity)
        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1,  0],
            [0, 0, 0,  1],
        ], dtype=float)

        # Measurement matrix (we only observe x, y)
        self.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ], dtype=float)

        # Process noise covariance
        q = process_noise ** 2
        self.Q = np.diag([q, q, q * 4, q * 4])  # higher uncertainty on velocity

        # Measurement noise covariance
        r = measurement_noise ** 2
        self.R = np.diag([r, r])

        # Initial state
        self.x = np.array([initial_xy[0], initial_xy[1], 0.0, 0.0], dtype=float)

        # Initial covariance (high uncertainty)
        self.P = np.diag([10.0, 10.0, 100.0, 100.0])

    def predict(self) -> np.ndarray:
        """Predict next state."""
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        return self.x[:2].copy()

    def update(self, measurement: np.ndarray):
        """Update with new measurement."""
        z = measurement.reshape(2, 1)
        y = z - self.H @ self.x.reshape(-1, 1)           # innovation
        S = self.H @ self.P @ self.H.T + self.R           # innovation covariance
        K = self.P @ self.H.T @ np.linalg.inv(S)          # Kalman gain
        self.x = (self.x.reshape(-1, 1) + K @ y).flatten()
        n = len(self.x)
        self.P = (np.eye(n) - K @ self.H) @ self.P

    def mahalanobis_distance(self, measurement: np.ndarray) -> float:
        """Compute Mahalanobis distance between measurement and predicted state."""
        S = self.H @ self.P @ self.H.T + self.R
        diff = measurement - self.x[:2]
        try:
            dist = float(diff.T @ np.linalg.inv(S) @ diff)
        except np.linalg.LinAlgError:
            dist = float("inf")
        return dist

    @property
    def position(self) -> np.ndarray:
        return self.x[:2].copy()

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:4].copy()

    @property
    def speed(self) -> float:
        return float(np.linalg.norm(self.x[2:4]))


# ──────────────────────────────────────────────
#  TRACK
# ──────────────────────────────────────────────

class Track:
    """One tracked object with associated Kalman filter."""

    _id_counter = 0

    def __init__(self, cluster: Cluster, dt: float,
                 process_noise: float, measurement_noise: float):
        Track._id_counter += 1
        self.track_id = Track._id_counter
        self.kf = KalmanFilter2D(
            cluster.centroid[:2], dt, process_noise, measurement_noise
        )
        self.label = cluster.label
        self.label_id = cluster.label_id
        self.confidence = cluster.confidence
        self.hits = 1
        self.age = 0
        self.time_since_update = 0
        self.history: List[Dict] = []
        self._update_history(cluster)

    def predict(self):
        self.age += 1
        self.time_since_update += 1
        self.kf.predict()

    def update(self, cluster: Cluster):
        self.time_since_update = 0
        self.hits += 1
        self.kf.update(cluster.centroid[:2])
        # Update label with running vote (most recent wins with 70% weight)
        if cluster.confidence > 0.5:
            self.label = cluster.label
            self.label_id = cluster.label_id
            self.confidence = cluster.confidence
        self._update_history(cluster)

    def _update_history(self, cluster: Cluster):
        self.history.append({
            "frame_id": cluster.frame_id,
            "x": float(cluster.centroid[0]),
            "y": float(cluster.centroid[1]),
            "z": float(cluster.centroid[2]),
            "speed_mps": self.kf.speed,
            "label": cluster.label,
        })

    @property
    def is_confirmed(self) -> bool:
        return self.hits >= TRACK["min_hits"]

    @property
    def position(self) -> np.ndarray:
        return self.kf.position

    @property
    def velocity(self) -> np.ndarray:
        return self.kf.velocity

    @property
    def speed_kmh(self) -> float:
        return self.kf.speed * 3.6

    def to_dict(self) -> dict:
        pos = self.kf.position
        vel = self.kf.velocity
        return {
            "track_id": self.track_id,
            "label": self.label,
            "confidence": round(self.confidence, 4),
            "x": round(float(pos[0]), 3),
            "y": round(float(pos[1]), 3),
            "vx": round(float(vel[0]), 3),
            "vy": round(float(vel[1]), 3),
            "speed_mps": round(self.kf.speed, 3),
            "speed_kmh": round(self.speed_kmh, 1),
            "hits": self.hits,
            "age": self.age,
            "confirmed": self.is_confirmed,
        }


# ──────────────────────────────────────────────
#  TRACKER
# ──────────────────────────────────────────────

class Tracker:
    """
    Multi-object tracker using Kalman filters + Hungarian assignment.
    Handles:
      - Track initialisation for new detections
      - Track update when matched
      - Track deletion after max_age missed frames
      - Track confirmation after min_hits
    """

    def __init__(self):
        self.cfg = TRACK
        self.tracks: List[Track] = []
        self.frame_count = 0
        Track._id_counter = 0  # reset for fresh run

    def update(self, clusters: List[Cluster]) -> List[Track]:
        """
        Process one frame of detections.
        Returns list of all currently active (confirmed + tentative) tracks.
        """
        self.frame_count += 1

        # 1. Predict all existing tracks
        for track in self.tracks:
            track.predict()

        # 2. Match detections to tracks
        matched, unmatched_dets, unmatched_trks = self._associate(clusters)

        # 3. Update matched tracks
        for det_idx, trk_idx in matched:
            self.tracks[trk_idx].update(clusters[det_idx])
            clusters[det_idx].track_id = self.tracks[trk_idx].track_id

        # 4. Create new tracks for unmatched detections
        for det_idx in unmatched_dets:
            new_track = Track(
                clusters[det_idx],
                dt=self.cfg["dt"],
                process_noise=self.cfg["process_noise_std"],
                measurement_noise=self.cfg["measurement_noise_std"],
            )
            clusters[det_idx].track_id = new_track.track_id
            self.tracks.append(new_track)

        # 5. Delete old tracks
        self.tracks = [
            t for t in self.tracks
            if t.time_since_update <= self.cfg["max_age"]
        ]

        logger.debug(
            f"Frame {self.frame_count}: {len(clusters)} dets, "
            f"{len(matched)} matched, {len(unmatched_dets)} new, "
            f"{len(self.tracks)} active tracks"
        )

        return self.tracks

    def _associate(
        self, clusters: List[Cluster]
    ) -> Tuple[List[Tuple[int, int]], List[int], List[int]]:
        """
        Hungarian-algorithm assignment between detections and tracks.
        Uses Mahalanobis distance for gating.
        """
        n_dets = len(clusters)
        n_trks = len(self.tracks)

        if n_trks == 0:
            return [], list(range(n_dets)), []
        if n_dets == 0:
            return [], [], list(range(n_trks))

        # Build cost matrix (Mahalanobis distances)
        cost_matrix = np.full((n_dets, n_trks), fill_value=1e9)
        for di, cluster in enumerate(clusters):
            for ti, track in enumerate(self.tracks):
                mah = track.kf.mahalanobis_distance(cluster.centroid[:2])
                cost_matrix[di, ti] = mah

        # Hungarian assignment
        row_idx, col_idx = linear_sum_assignment(cost_matrix)

        matched, unmatched_dets, unmatched_trks = [], [], []

        # Gate assignments by Mahalanobis threshold
        for r, c in zip(row_idx, col_idx):
            if cost_matrix[r, c] <= self.cfg["mahal_threshold"]:
                matched.append((r, c))
            else:
                unmatched_dets.append(r)
                unmatched_trks.append(c)

        for di in range(n_dets):
            if di not in [m[0] for m in matched] and di not in unmatched_dets:
                unmatched_dets.append(di)

        for ti in range(n_trks):
            if ti not in [m[1] for m in matched] and ti not in unmatched_trks:
                unmatched_trks.append(ti)

        return matched, unmatched_dets, unmatched_trks

    def confirmed_tracks(self) -> List[Track]:
        return [t for t in self.tracks if t.is_confirmed]

    def reset(self):
        self.tracks = []
        self.frame_count = 0
        Track._id_counter = 0