"""
config.py — Central configuration for LiDAR Detection & Tracking Pipeline
Blickfeld Cube 1 | Automotive Research Division
"""

# ─────────────────────────────────────────────
#  SENSOR SPECIFICATION  (Blickfeld Cube 1)
# ─────────────────────────────────────────────
SENSOR = {
    "range_min_m": 5.0,
    "range_max_m": 250.0,
    "range_resolution_m": 0.01,
    "range_precision_m": 0.02,
    "h_fov_deg": 70.0,
    "v_fov_deg": 30.0,
    "scan_lines_per_sec_min": 500,
    "update_rate_min_hz": 1,
    "update_rate_max_hz": 30,
    "wavelength_nm": 900,
}

# ─────────────────────────────────────────────
#  DATA LOADING
# ─────────────────────────────────────────────
DATA = {
    "delimiter": ";",
    "columns": ["X", "Y", "Z", "DISTANCE", "INTENSITY", "POINT_ID", "RETURN_ID", "AMBIENT", "TIMESTAMP"],
    "total_expected_frames": 152,
    "frame_prefix": "192_168_26_26_2020-11-25_20-01-45_frame-",
}

# ─────────────────────────────────────────────
#  PREPROCESSING
# ─────────────────────────────────────────────
PREPROCESS = {
    # Range filtering (per sensor spec)
    "range_min": 5.0,
    "range_max": 100.0,       # practical scene range

    # Ground removal via RANSAC plane fitting
    "ground_removal_enabled": True,
    "ransac_distance_threshold": 0.15,   # metres
    "ransac_max_trials": 1000,
    "ground_z_max": 0.3,                 # points below this height above ground plane are road

    # Statistical outlier removal
    "outlier_removal_enabled": True,
    "outlier_neighbors": 10,
    "outlier_std_ratio": 2.0,

    # Intensity filter
    "intensity_min": 0,
    "intensity_max": 255,

    # Voxel downsampling (optional, speeds up clustering)
    "voxel_downsample": False,
    "voxel_size": 0.1,
}

# ─────────────────────────────────────────────
#  CLUSTERING  (DBSCAN)
# ─────────────────────────────────────────────
CLUSTER = {
    "algorithm": "DBSCAN",
    "eps": 0.6,              # neighbourhood radius (metres)
    "min_samples": 5,        # minimum points to form a core point
    "min_cluster_size": 10,  # discard clusters smaller than this
    "max_cluster_size": 8000,
}

# ─────────────────────────────────────────────
#  CLASSIFICATION  (object type)
# ─────────────────────────────────────────────
CLASS = {
    # Class labels
    "labels": {0: "background", 1: "car", 2: "pedestrian", 3: "cyclist", 4: "unknown"},
    "label_ids": {"background": 0, "car": 1, "pedestrian": 2, "cyclist": 3, "unknown": 4},

    # Rule-based thresholds (bounding box dimensions in metres)
    "car": {
        "length_min": 2.5, "length_max": 6.0,
        "width_min": 1.2,  "width_max": 2.8,
        "height_min": 0.8, "height_max": 2.5,
        "points_min": 30,  "points_max": 3000,
    },
    "pedestrian": {
        "length_min": 0.3, "length_max": 1.2,
        "width_min": 0.3,  "width_max": 1.0,
        "height_min": 1.0, "height_max": 2.2,
        "points_min": 10,  "points_max": 300,
    },
    "cyclist": {
        "length_min": 1.0, "length_max": 2.5,
        "width_min": 0.4,  "width_max": 1.2,
        "height_min": 1.2, "height_max": 2.2,
        "points_min": 15,  "points_max": 500,
    },

    # ML classifier settings (Random Forest)
    "use_ml_classifier": True,
    "n_estimators": 200,
    "random_state": 42,
    "confidence_threshold": 0.6,
}

# ─────────────────────────────────────────────
#  TRACKING  (Kalman Filter)
# ─────────────────────────────────────────────
TRACK = {
    "max_age": 5,               # frames before track is deleted without association
    "min_hits": 2,              # frames before track is confirmed
    "iou_threshold": 0.3,       # IoU for bounding-box association
    "mahal_threshold": 5.991,   # Chi-squared 95% for 2-DOF gating
    "dt": 1.0 / 10.0,           # assumed inter-frame time (s) → 10 Hz nominal
    # Kalman noise
    "process_noise_std": 0.5,
    "measurement_noise_std": 0.3,
}

# ─────────────────────────────────────────────
#  PERFORMANCE TARGETS
# ─────────────────────────────────────────────
PERFORMANCE = {
    "target_ccr": 0.99,    # correct classification rate
    "target_far": 0.01,    # false alarm rate per hour
}

# ─────────────────────────────────────────────
#  OUTPUT & VISUALIZATION
# ─────────────────────────────────────────────
OUTPUT = {
    "save_plots": True,
    "save_csv": True,
    "save_report": True,
    "dpi": 150,
    "figure_size": (14, 8),
    "class_colors": {
        "car": "#2196F3",         # blue
        "pedestrian": "#FF5722",  # orange-red
        "cyclist": "#4CAF50",     # green
        "unknown": "#9E9E9E",     # grey
        "background": "#212121",  # dark
    },
}