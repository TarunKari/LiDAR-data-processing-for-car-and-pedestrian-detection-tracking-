"""
preprocessor.py — Point Cloud Preprocessing
Steps:
  1. Range filtering (sensor spec limits)
  2. Intensity filtering
  3. Statistical outlier removal
  4. Ground plane removal (RANSAC)
  5. (Optional) Voxel downsampling
"""

import numpy as np
import logging
from dataclasses import dataclass
from typing import Tuple, Optional

from config import PREPROCESS, SENSOR

logger = logging.getLogger(__name__)


@dataclass
class PreprocessResult:
    """Output of preprocessing a single frame."""
    points_xyz: np.ndarray          # filtered non-ground points (N, 3)
    intensity: np.ndarray           # corresponding intensity (N,)
    distance: np.ndarray            # corresponding distance (N,)
    ambient: np.ndarray             # corresponding ambient (N,)
    ground_mask: np.ndarray         # bool mask over original — True = ground
    valid_mask: np.ndarray          # bool mask over original — True = kept
    ground_plane_coeffs: Optional[np.ndarray]  # [a,b,c,d] of ax+by+cz+d=0

    @property
    def num_points(self): return len(self.points_xyz)


class Preprocessor:
    """
    Cleans and filters a raw LiDAR frame before clustering.
    All thresholds are driven by config.py.
    """

    def __init__(self):
        self.cfg = PREPROCESS

    # ──────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────

    def process(self, frame) -> PreprocessResult:
        """
        Run full preprocessing pipeline on a LiDARFrame.
        Returns PreprocessResult with cleaned point cloud.
        """
        pts = frame.points_xyz.copy()
        dist = frame.distance.copy()
        intensity = frame.intensity.copy()
        ambient = frame.ambient.copy()
        N = len(pts)

        # Step 1: Range filter
        range_mask = self._range_filter(dist)

        # Step 2: Intensity filter
        intensity_mask = self._intensity_filter(intensity)

        # Step 3: Combined valid mask so far
        valid = range_mask & intensity_mask

        # Step 4: Statistical outlier removal (on valid points)
        if self.cfg["outlier_removal_enabled"] and valid.sum() > 20:
            outlier_valid = np.zeros(N, dtype=bool)
            idx = np.where(valid)[0]
            keep = self._statistical_outlier_removal(pts[idx])
            outlier_valid[idx[keep]] = True
            valid = valid & outlier_valid

        # Step 5: Ground removal
        ground_mask = np.zeros(N, dtype=bool)
        ground_coeffs = None
        if self.cfg["ground_removal_enabled"] and valid.sum() > 50:
            idx = np.where(valid)[0]
            gnd, coeffs = self._ransac_ground_removal(pts[idx])
            ground_mask_local = np.zeros(N, dtype=bool)
            ground_mask_local[idx[gnd]] = True
            ground_mask = ground_mask_local
            valid = valid & ~ground_mask
            ground_coeffs = coeffs

        # Extract filtered subset
        pts_clean = pts[valid]
        dist_clean = dist[valid]
        int_clean = intensity[valid]
        amb_clean = ambient[valid]

        logger.debug(
            f"Preprocessed: {N} → {valid.sum()} pts "
            f"(ground={ground_mask.sum()}, removed={N - valid.sum() - ground_mask.sum()})"
        )

        return PreprocessResult(
            points_xyz=pts_clean,
            intensity=int_clean,
            distance=dist_clean,
            ambient=amb_clean,
            ground_mask=ground_mask,
            valid_mask=valid,
            ground_plane_coeffs=ground_coeffs,
        )

    # ──────────────────────────────────────────────
    #  INTERNAL STEPS
    # ──────────────────────────────────────────────

    def _range_filter(self, distance: np.ndarray) -> np.ndarray:
        """Keep points within the configured range band."""
        return (distance >= self.cfg["range_min"]) & (distance <= self.cfg["range_max"])

    def _intensity_filter(self, intensity: np.ndarray) -> np.ndarray:
        """Keep points with valid intensity values."""
        return (intensity >= self.cfg["intensity_min"]) & (intensity <= self.cfg["intensity_max"])

    def _statistical_outlier_removal(self, pts: np.ndarray) -> np.ndarray:
        """
        Remove points whose mean distance to k-nearest neighbours
        is more than std_ratio standard deviations above the global mean.
        Returns boolean mask (True = keep).
        """
        from scipy.spatial import KDTree

        k = min(self.cfg["outlier_neighbors"], len(pts) - 1)
        if k < 2:
            return np.ones(len(pts), dtype=bool)

        tree = KDTree(pts)
        dists, _ = tree.query(pts, k=k + 1)  # includes self
        mean_dists = dists[:, 1:].mean(axis=1)  # exclude self

        mu = mean_dists.mean()
        sigma = mean_dists.std()
        threshold = mu + self.cfg["outlier_std_ratio"] * sigma

        return mean_dists <= threshold

    def _ransac_ground_removal(
        self, pts: np.ndarray
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Fit a ground plane using RANSAC.
        Returns:
            ground_mask: bool array (True = ground point)
            coeffs: [a,b,c,d] of best plane ax+by+cz+d=0
        """
        if len(pts) < 10:
            return np.zeros(len(pts), dtype=bool), None

        best_inliers = np.zeros(len(pts), dtype=bool)
        best_coeffs = None
        best_count = 0
        dist_thresh = self.cfg["ransac_distance_threshold"]
        max_trials = self.cfg["ransac_max_trials"]
        rng = np.random.default_rng(seed=42)

        # Only consider low-z candidate points for plane fitting
        z_cand = pts[:, 2]
        low_z_mask = z_cand <= np.percentile(z_cand, 30)
        low_z_idx = np.where(low_z_mask)[0]

        if len(low_z_idx) < 3:
            return best_inliers, best_coeffs

        for _ in range(max_trials):
            # Sample 3 points
            sample_idx = rng.choice(low_z_idx, 3, replace=False)
            p1, p2, p3 = pts[sample_idx[0]], pts[sample_idx[1]], pts[sample_idx[2]]

            # Plane normal
            v1 = p2 - p1
            v2 = p3 - p1
            normal = np.cross(v1, v2)
            norm_len = np.linalg.norm(normal)
            if norm_len < 1e-8:
                continue
            normal /= norm_len

            a, b, c = normal
            d = -np.dot(normal, p1)

            # Distance of all points to plane
            distances = np.abs(pts @ normal + d)
            inliers = distances <= dist_thresh

            # Extra constraint: inliers must be near the ground (z-level)
            z_inlier = pts[inliers, 2]
            if len(z_inlier) > 0:
                z_mean = z_inlier.mean()
                inliers = inliers & (np.abs(pts[:, 2] - z_mean) <= self.cfg["ground_z_max"])

            count = inliers.sum()
            if count > best_count:
                best_count = count
                best_inliers = inliers
                best_coeffs = np.array([a, b, c, d])

        logger.debug(f"RANSAC ground: {best_count}/{len(pts)} inlier points")
        return best_inliers, best_coeffs

    def voxel_downsample(self, pts: np.ndarray, voxel_size: float) -> np.ndarray:
        """
        Simple voxel grid downsampling — keeps one point per voxel.
        Returns indices of kept points.
        """
        voxel_indices = np.floor(pts / voxel_size).astype(int)
        _, unique_idx = np.unique(voxel_indices, axis=0, return_index=True)
        return unique_idx