"""
data_loader.py — LiDAR CSV Frame Loader
Handles reading Blickfeld Cube 1 CSV exports (all 152 frames or subset).
Columns: X;Y;Z;DISTANCE;INTENSITY;POINT_ID;RETURN_ID;AMBIENT;TIMESTAMP
"""

import os
import glob
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Optional, Iterator
import logging

from config import DATA, SENSOR

logger = logging.getLogger(__name__)


@dataclass
class LiDARFrame:
    """Container for one LiDAR point cloud frame."""
    frame_id: int
    filename: str
    points_xyz: np.ndarray        # shape (N, 3) — X, Y, Z in metres
    distance: np.ndarray          # shape (N,)   — radial distance in metres
    intensity: np.ndarray         # shape (N,)   — return intensity 0-255
    point_id: np.ndarray          # shape (N,)   — point index from sensor
    return_id: np.ndarray         # shape (N,)   — return number (multi-echo)
    ambient: np.ndarray           # shape (N,)   — ambient light level
    timestamp: np.ndarray         # shape (N,)   — nanosecond UNIX timestamps
    num_points: int = field(init=False)

    def __post_init__(self):
        self.num_points = len(self.points_xyz)

    @property
    def x(self): return self.points_xyz[:, 0]
    @property
    def y(self): return self.points_xyz[:, 1]
    @property
    def z(self): return self.points_xyz[:, 2]

    @property
    def frame_timestamp_s(self) -> float:
        """Return median frame timestamp in seconds."""
        return float(np.median(self.timestamp)) * 1e-9

    def sanity_report(self) -> dict:
        """Return a dict of sanity-check statistics for this frame."""
        return {
            "frame_id": self.frame_id,
            "num_points": self.num_points,
            "x_range": (float(self.x.min()), float(self.x.max())),
            "y_range": (float(self.y.min()), float(self.y.max())),
            "z_range": (float(self.z.min()), float(self.z.max())),
            "distance_range": (float(self.distance.min()), float(self.distance.max())),
            "intensity_range": (float(self.intensity.min()), float(self.intensity.max())),
            "within_sensor_range": bool(
                self.distance.min() >= SENSOR["range_min_m"] - 1 and
                self.distance.max() <= SENSOR["range_max_m"] + 1
            ),
            "timestamp_span_ms": float(
                (self.timestamp.max() - self.timestamp.min()) * 1e-6
            ),
        }


class DataLoader:
    """
    Discovers and loads all LiDAR CSV frames from a given directory.
    Designed for 152-frame dataset (uses whatever frames are present).
    """

    def __init__(self, data_dir: str):
        self.data_dir = data_dir
        self.frame_paths = self._discover_frames()
        logger.info(f"DataLoader: found {len(self.frame_paths)} frames in '{data_dir}'")

    def _discover_frames(self) -> List[str]:
        """Find all CSV files, sorted by frame number."""
        pattern = os.path.join(self.data_dir, "*.csv")
        files = sorted(glob.glob(pattern))
        if not files:
            logger.warning(f"No CSV files found in {self.data_dir}")
        return files

    @property
    def num_frames(self) -> int:
        return len(self.frame_paths)

    def _parse_frame_id(self, filepath: str) -> int:
        """Extract frame number from filename like '..._frame-2415.csv'."""
        basename = os.path.basename(filepath)
        try:
            return int(basename.split("frame-")[-1].replace(".csv", ""))
        except (ValueError, IndexError):
            return -1

    def load_frame(self, filepath: str) -> Optional[LiDARFrame]:
        """Load a single CSV file into a LiDARFrame object."""
        try:
            df = pd.read_csv(
                filepath,
                delimiter=DATA["delimiter"],
                names=DATA["columns"],
                header=0,
                dtype=float,
                on_bad_lines="skip",
            )

            # Drop rows with any NaN
            df.dropna(inplace=True)

            if len(df) == 0:
                logger.warning(f"Empty frame after parsing: {filepath}")
                return None

            frame = LiDARFrame(
                frame_id=self._parse_frame_id(filepath),
                filename=os.path.basename(filepath),
                points_xyz=df[["X", "Y", "Z"]].values.astype(np.float32),
                distance=df["DISTANCE"].values.astype(np.float32),
                intensity=df["INTENSITY"].values.astype(np.float32),
                point_id=df["POINT_ID"].values.astype(np.int32),
                return_id=df["RETURN_ID"].values.astype(np.int32),
                ambient=df["AMBIENT"].values.astype(np.float32),
                timestamp=df["TIMESTAMP"].values.astype(np.float64),
            )

            logger.debug(f"Loaded frame {frame.frame_id}: {frame.num_points} points")
            return frame

        except Exception as e:
            logger.error(f"Failed to load {filepath}: {e}")
            return None

    def iter_frames(self) -> Iterator[LiDARFrame]:
        """Iterate through all discovered frames in order."""
        for path in self.frame_paths:
            frame = self.load_frame(path)
            if frame is not None:
                yield frame

    def load_all(self) -> List[LiDARFrame]:
        """Load all frames into memory and return as a list."""
        frames = []
        for frame in self.iter_frames():
            frames.append(frame)
        logger.info(f"Loaded {len(frames)} frames total.")
        return frames

    def sanity_check_all(self) -> pd.DataFrame:
        """Run sanity checks on all frames and return a summary DataFrame."""
        reports = []
        for frame in self.iter_frames():
            reports.append(frame.sanity_report())
        df = pd.DataFrame(reports)
        return df

    def dataset_summary(self) -> dict:
        """Return high-level summary statistics across all frames."""
        frames = self.load_all()
        if not frames:
            return {}

        point_counts = [f.num_points for f in frames]
        all_distances = np.concatenate([f.distance for f in frames])
        all_intensities = np.concatenate([f.intensity for f in frames])
        all_z = np.concatenate([f.z for f in frames])

        return {
            "total_frames": len(frames),
            "total_points": int(sum(point_counts)),
            "mean_points_per_frame": float(np.mean(point_counts)),
            "std_points_per_frame": float(np.std(point_counts)),
            "min_points_per_frame": int(min(point_counts)),
            "max_points_per_frame": int(max(point_counts)),
            "global_distance_min_m": float(all_distances.min()),
            "global_distance_max_m": float(all_distances.max()),
            "global_distance_mean_m": float(all_distances.mean()),
            "global_intensity_min": float(all_intensities.min()),
            "global_intensity_max": float(all_intensities.max()),
            "global_intensity_mean": float(all_intensities.mean()),
            "global_z_min_m": float(all_z.min()),
            "global_z_max_m": float(all_z.max()),
            "sensor_spec_range_ok": bool(
                all_distances.min() >= SENSOR["range_min_m"] - 2 and
                all_distances.max() <= SENSOR["range_max_m"]
            ),
        }