"""
visualizer.py — Blickfeld-Style LiDAR Visualization
Matches the reference screenshots:
  - Dark grid ground plane
  - Distance-based colormap (blue=near → green → yellow → red=far)
  - 3D perspective view (elev=25, azim=-45)
  - Cyan annotation boxes with leader lines for detected objects
  - Distance [m] colorbar on left side
  - Per-frame 3D overview + labeled detection view
"""

import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Line3DCollection
import logging
from typing import List, Optional

from config import OUTPUT, CLASS
from src.detector import Cluster
from src.tracker import Track
from src.data_loader import LiDARFrame
from src.preprocessor import PreprocessResult

logger = logging.getLogger(__name__)

DPI = OUTPUT["dpi"]

# ── Blickfeld distance colormap: blue(0) → cyan → green → yellow → red(50m+)
DIST_CMAP = "jet"
DIST_VMIN = 0
DIST_VMAX = 50

# ── Label box style matching screenshots (cyan/grey frosted boxes)
LABEL_STYLES = {
    "car":        {"fc": "#5ad8e6cc", "ec": "#5ad8e6", "tc": "#000000"},
    "pedestrian": {"fc": "#5ad8e6cc", "ec": "#5ad8e6", "tc": "#000000"},
    "cyclist":    {"fc": "#5ad8e6cc", "ec": "#5ad8e6", "tc": "#000000"},
    "unknown":    {"fc": "#aaaaaa99", "ec": "#aaaaaa", "tc": "#000000"},
}

CLASS_POINT_COLORS = {
    "car":        "#2196F3",
    "pedestrian": "#00e5ff",
    "cyclist":    "#76ff03",
    "unknown":    "#90a4ae",
    "background": "#37474f",
}


def _get_point_color(label: str) -> str:
    return CLASS_POINT_COLORS.get(label, CLASS_POINT_COLORS["unknown"])


def _draw_grid(ax, x_range=(-25, 25), y_range=(0, 60), z_val=0,
               step=5, color="#2a2a2a", lw=0.4):
    """Draw a flat grid on the ground plane like the Blickfeld viewer."""
    xs = np.arange(x_range[0], x_range[1] + step, step)
    ys = np.arange(y_range[0], y_range[1] + step, step)
    # Lines along Y
    for x in xs:
        ax.plot([x, x], [y_range[0], y_range[1]], [z_val, z_val],
                color=color, lw=lw, zorder=0)
    # Lines along X
    for y in ys:
        ax.plot([x_range[0], x_range[1]], [y, y], [z_val, z_val],
                color=color, lw=lw, zorder=0)


def _add_colorbar_distance(fig, ax, vmin=0, vmax=50):
    """Add Distance [m] colorbar on the left, matching Blickfeld style."""
    sm = plt.cm.ScalarMappable(cmap=DIST_CMAP,
                               norm=plt.Normalize(vmin=vmin, vmax=vmax))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.02, pad=0.01,
                        location="left", shrink=0.6)
    cbar.set_label("Distance [m]", color="white", fontsize=9, labelpad=8)
    cbar.ax.yaxis.set_tick_params(color="white", labelsize=8)
    cbar.ax.set_facecolor("#111")
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color="white")
    # Only show 0 and max
    cbar.set_ticks([vmin, vmax])
    cbar.set_ticklabels([str(vmin), str(vmax)])
    return cbar


def _style_3d_ax(ax, x_range=(-25, 25), y_range=(0, 60), z_range=(-2, 20)):
    """Apply dark Blickfeld-style styling to a 3D axes."""
    ax.set_facecolor("#111111")
    ax.xaxis.pane.fill = False
    ax.yaxis.pane.fill = False
    ax.zaxis.pane.fill = False
    ax.xaxis.pane.set_edgecolor("#222")
    ax.yaxis.pane.set_edgecolor("#222")
    ax.zaxis.pane.set_edgecolor("#222")
    ax.grid(False)
    ax.set_xlim(*x_range)
    ax.set_ylim(*y_range)
    ax.set_zlim(*z_range)
    ax.tick_params(colors="#555", labelsize=7)
    ax.xaxis.label.set_color("white")
    ax.yaxis.label.set_color("white")
    ax.zaxis.label.set_color("white")
    ax.set_xlabel("X [m]", labelpad=2)
    ax.set_ylabel("Y [m]", labelpad=2)
    ax.set_zlabel("Z [m]", labelpad=2)
    # Match Blickfeld view angle
    ax.view_init(elev=25, azim=-45)


def _annotate_cluster_3d(ax, cluster: Cluster, label_text: str):
    """
    Draw a cyan annotation box with a leader line from centroid,
    matching the Blickfeld screenshot style.
    """
    cx, cy, cz = cluster.centroid
    style = LABEL_STYLES.get(cluster.label, LABEL_STYLES["unknown"])

    # Offset the label box away from the centroid
    ox = cx + 3.0
    oy = cy - 2.0
    oz = cz + 2.0

    # Leader line from centroid to label anchor
    ax.plot([cx, ox], [cy, oy], [cz + 0.5, oz],
            color="#5ad8e6", lw=0.8, alpha=0.85, zorder=10)

    # Text annotation box
    ax.text(ox, oy, oz, label_text,
            fontsize=7.5, color=style["tc"], fontweight="bold",
            ha="left", va="center",
            bbox=dict(
                boxstyle="round,pad=0.3",
                facecolor=style["fc"],
                edgecolor=style["ec"],
                linewidth=1.0,
                alpha=0.88,
            ),
            zorder=11)


# ═══════════════════════════════════════════════════════
#  MAIN: BLICKFELD-STYLE 3D OVERVIEW  (matches Image 1,2,4)
# ═══════════════════════════════════════════════════════

def plot_blickfeld_3d(
    frame: LiDARFrame,
    preprocess_result: PreprocessResult,
    clusters: List[Cluster],
    tracks: List[Track],
    save_path: Optional[str] = None,
    title: str = "OVERVIEW",
) -> plt.Figure:
    """
    Full Blickfeld-style 3D visualization of one frame.
    - Background points coloured by distance (jet colormap)
    - Detected objects highlighted in class colour
    - Cyan annotation labels with leader lines
    - Dark grid ground plane
    - Distance [m] colorbar
    """
    fig = plt.figure(figsize=(16, 9), facecolor="#111111")
    fig.patch.set_facecolor("#111111")

    # Title bar (matches screenshot header)
    fig.text(0.03, 0.95, title, color="white",
             fontsize=18, fontweight="bold", va="top", family="monospace")

    ax = fig.add_subplot(111, projection="3d")
    _style_3d_ax(ax)

    # ── 1. Draw ground grid
    _draw_grid(ax, x_range=(-25, 25), y_range=(0, 60), z_val=-0.5)

    # ── 2. Draw all background points coloured by distance
    all_pts = frame.points_xyz
    all_dist = frame.distance

    if len(all_pts) > 0:
        # Subsample to keep rendering fast (max 15k points)
        max_pts = 15000
        if len(all_pts) > max_pts:
            idx = np.random.choice(len(all_pts), max_pts, replace=False)
            pts_show = all_pts[idx]
            dist_show = all_dist[idx]
        else:
            pts_show = all_pts
            dist_show = all_dist

        norm_dist = np.clip(dist_show / DIST_VMAX, 0, 1)
        cmap = plt.cm.get_cmap(DIST_CMAP)
        pt_colors = cmap(norm_dist)

        ax.scatter(pts_show[:, 0], pts_show[:, 1], pts_show[:, 2],
                   c=pt_colors, s=0.8, alpha=0.55, zorder=2, depthshade=True)

    # ── 3. Highlight detected clusters with class colours + brighter points
    labeled_classes = set()
    for c in clusters:
        if c.label in ("unknown", "background"):
            continue
        col = _get_point_color(c.label)
        ax.scatter(c.points_xyz[:, 0], c.points_xyz[:, 1], c.points_xyz[:, 2],
                   c=col, s=3.0, alpha=0.9, zorder=5, depthshade=False)
        labeled_classes.add(c.label)

    # ── 4. Annotation labels for important clusters
    # Only label confirmed, non-unknown clusters; avoid overcrowding
    labeled = set()
    for c in sorted(clusters, key=lambda x: x.num_points, reverse=True):
        if c.label in ("unknown", "background"):
            continue
        if c.label in labeled and len([x for x in clusters
                                       if x.label == c.label]) < 3:
            continue
        track_suffix = f" (T{c.track_id})" if c.track_id >= 0 else ""
        _annotate_cluster_3d(ax, c, c.label + track_suffix)
        labeled.add(c.label)

    # ── 5. Colorbar
    _add_colorbar_distance(fig, ax, vmin=DIST_VMIN, vmax=DIST_VMAX)

    # ── 6. Frame info box (bottom-left like Blickfeld)
    n_cars = sum(1 for c in clusters if c.label == "car")
    n_peds = sum(1 for c in clusters if c.label == "pedestrian")
    n_cycs = sum(1 for c in clusters if c.label == "cyclist")
    n_conf = sum(1 for t in tracks if t.is_confirmed)
    info = (f"Frame {frame.frame_id}  |  "
            f"Cars: {n_cars}  Pedestrians: {n_peds}  Cyclists: {n_cycs}  |  "
            f"Confirmed tracks: {n_conf}  |  "
            f"Points: {frame.num_points:,}")
    fig.text(0.03, 0.03, info, color="#888888", fontsize=8, family="monospace")

    plt.tight_layout(rect=[0.04, 0.04, 1.0, 0.93])

    if save_path and OUTPUT["save_plots"]:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight",
                    facecolor="#111111")
        logger.info(f"Saved: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════
#  PARKING CAR PANEL  (matches Image 3 — three sub-views)
# ═══════════════════════════════════════════════════════

def plot_parking_car_panel(
    frames_data: list,   # list of (frame, pre_result, clusters, tracks, subtitle)
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Three-panel zoomed view of parking car sequence (matches screenshot Image 3).
    frames_data: list of up to 3 tuples (frame, pre_result, clusters, tracks, subtitle)
    """
    n = min(len(frames_data), 3)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5),
                              subplot_kw={"projection": "3d"},
                              facecolor="#111")
    fig.patch.set_facecolor("#111")
    fig.text(0.03, 0.96, "PARKING CAR", color="white",
             fontsize=16, fontweight="bold", va="top", family="monospace")

    if n == 1:
        axes = [axes]

    for i, (ax, (frame, pre, clusters, tracks, subtitle)) in enumerate(
            zip(axes, frames_data)):
        _style_3d_ax(ax, x_range=(-15, 5), y_range=(5, 25), z_range=(-1, 5))
        _draw_grid(ax, x_range=(-15, 5), y_range=(5, 25), z_val=-0.5, step=3)

        # All points by distance
        pts = frame.points_xyz
        dist = frame.distance
        if len(pts) > 0:
            idx = np.random.choice(len(pts), min(8000, len(pts)), replace=False)
            norm_d = np.clip(dist[idx] / DIST_VMAX, 0, 1)
            cols = plt.cm.get_cmap(DIST_CMAP)(norm_d)
            ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                       c=cols, s=1.0, alpha=0.6, depthshade=True)

        # Highlight clusters
        for c in clusters:
            if c.label == "car":
                ax.scatter(c.points_xyz[:, 0], c.points_xyz[:, 1],
                           c.points_xyz[:, 2],
                           c="#2196F3", s=4, alpha=0.9, depthshade=False)
                _annotate_cluster_3d(ax, c, subtitle)

        ax.set_title("", pad=0)
        ax.view_init(elev=20, azim=-50)

    plt.tight_layout(rect=[0, 0, 1, 0.92])

    if save_path and OUTPUT["save_plots"]:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="#111")
        logger.info(f"Saved: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════
#  PER-CLASS FOCUSED VIEW  (pedestrian / car+cyclist views)
# ═══════════════════════════════════════════════════════

def plot_class_focus_3d(
    frame: LiDARFrame,
    preprocess_result: PreprocessResult,
    clusters: List[Cluster],
    tracks: List[Track],
    focus_labels: List[str],
    title: str = "DETECTION VIEW",
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    Focused 3D view highlighting specific classes (e.g. pedestrian, cyclist+car).
    Matches Image 1 (pedestrian) and Image 2 (car & cyclist).
    """
    fig = plt.figure(figsize=(16, 9), facecolor="#111111")
    fig.patch.set_facecolor("#111111")
    fig.text(0.03, 0.95, title, color="white",
             fontsize=18, fontweight="bold", va="top", family="monospace")

    ax = fig.add_subplot(111, projection="3d")
    _style_3d_ax(ax)
    _draw_grid(ax, x_range=(-25, 25), y_range=(0, 60), z_val=-0.5)

    # All points coloured by distance
    all_pts = frame.points_xyz
    all_dist = frame.distance
    if len(all_pts) > 0:
        max_pts = 15000
        if len(all_pts) > max_pts:
            idx = np.random.choice(len(all_pts), max_pts, replace=False)
            pts_s = all_pts[idx]; dist_s = all_dist[idx]
        else:
            pts_s = all_pts; dist_s = all_dist
        norm_d = np.clip(dist_s / DIST_VMAX, 0, 1)
        cols = plt.cm.get_cmap(DIST_CMAP)(norm_d)
        ax.scatter(pts_s[:, 0], pts_s[:, 1], pts_s[:, 2],
                   c=cols, s=0.8, alpha=0.5, depthshade=True, zorder=2)

    # Highlight focus clusters more prominently
    for c in clusters:
        if c.label in focus_labels:
            col = _get_point_color(c.label)
            ax.scatter(c.points_xyz[:, 0], c.points_xyz[:, 1],
                       c.points_xyz[:, 2],
                       c=col, s=4.0, alpha=0.95, depthshade=False, zorder=6)
            track_sfx = f" (T{c.track_id})" if c.track_id >= 0 else ""
            _annotate_cluster_3d(ax, c, c.label + track_sfx)

    _add_colorbar_distance(fig, ax)

    n_focus = sum(1 for c in clusters if c.label in focus_labels)
    n_conf = sum(1 for t in tracks if t.is_confirmed)
    fig.text(0.03, 0.03,
             f"Frame {frame.frame_id}  |  "
             f"Highlighted: {', '.join(focus_labels)} ({n_focus})  |  "
             f"Confirmed tracks: {n_conf}  |  Points: {frame.num_points:,}",
             color="#888888", fontsize=8, family="monospace")

    plt.tight_layout(rect=[0.04, 0.04, 1.0, 0.93])

    if save_path and OUTPUT["save_plots"]:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="#111111")
        logger.info(f"Saved: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════
#  TRACK HISTORY (top-down, dark style)
# ═══════════════════════════════════════════════════════

def plot_track_history(tracks: List[Track], save_path: Optional[str] = None) -> plt.Figure:
    """Top-down track trajectory plot, dark styled."""
    fig, ax = plt.subplots(figsize=(14, 9), facecolor="#111111")
    ax.set_facecolor("#111111")

    confirmed = [t for t in tracks if t.is_confirmed]

    if not confirmed:
        ax.text(0, 0, "No confirmed tracks yet", color="white",
                ha="center", fontsize=14)
    else:
        for track in confirmed:
            col = _get_point_color(track.label)
            hist = track.history
            xs = [h["x"] for h in hist]
            ys = [h["y"] for h in hist]
            ax.plot(xs, ys, "-", color=col, alpha=0.8, lw=2.0)
            ax.scatter(xs[0], ys[0], c=col, s=60, marker="o", zorder=5)
            ax.scatter(xs[-1], ys[-1], c=col, s=120, marker="*", zorder=6)
            spd = track.history[-1].get("speed_mps", 0) * 3.6
            ax.annotate(
                f"T{track.track_id} · {track.label}\n{spd:.1f} km/h",
                xy=(xs[-1], ys[-1]),
                xytext=(xs[-1] + 1.2, ys[-1] + 0.5),
                color="white", fontsize=7,
                bbox=dict(boxstyle="round,pad=0.25",
                          facecolor="#5ad8e6bb", edgecolor="#5ad8e6",
                          linewidth=0.8),
                arrowprops=dict(arrowstyle="-", color="#5ad8e6", lw=0.7),
            )

    # Draw grid lines
    for gx in range(-30, 31, 5):
        ax.axvline(gx, color="#1e1e1e", lw=0.5)
    for gy in range(-10, 70, 5):
        ax.axhline(gy, color="#1e1e1e", lw=0.5)

    ax.set_xlim(-25, 25); ax.set_ylim(-5, 65)
    ax.set_xlabel("X [m]", color="white"); ax.set_ylabel("Y [m]", color="white")
    ax.set_title(f"Track Histories — {len(confirmed)} Confirmed Tracks",
                 color="white", fontsize=14, fontweight="bold")
    ax.tick_params(colors="#666")
    for sp in ax.spines.values():
        sp.set_edgecolor("#333")

    legend_items = [
        mpatches.Patch(color=_get_point_color(l), label=l.capitalize())
        for l in ["car", "pedestrian", "cyclist", "unknown"]
    ]
    ax.legend(handles=legend_items, facecolor="#1a1a1a",
              labelcolor="white", fontsize=9, loc="upper right")

    ax.text(0, -3, "▲ LiDAR origin", color="#555", ha="center", fontsize=8)

    plt.tight_layout()
    if save_path and OUTPUT["save_plots"]:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="#111111")
        logger.info(f"Saved: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════
#  DATASET STATISTICS
# ═══════════════════════════════════════════════════════

def plot_dataset_statistics(loader_summary: dict,
                             save_path: Optional[str] = None) -> plt.Figure:
    """Dataset statistics — dark themed."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6), facecolor="#111111")
    fig.suptitle("DATASET STATISTICS", color="white",
                 fontsize=14, fontweight="bold", family="monospace")

    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    # Left: bar chart
    ax = axes[0]
    cats = ["Available\nFrames", "Expected\nFrames", "Mean pts\n/ frame (÷100)"]
    vals = [
        loader_summary.get("total_frames", 0),
        152,
        loader_summary.get("mean_points_per_frame", 0) / 100,
    ]
    colors = ["#00e5ff", "#2196F3", "#76ff03"]
    bars = ax.bar(cats, vals, color=colors, alpha=0.85, width=0.5)
    for bar, val, orig in zip(bars, vals, [loader_summary.get("total_frames", 0),
                                            152,
                                            loader_summary.get("mean_points_per_frame", 0)]):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.3,
                f"{orig:.0f}", color="white", ha="center", fontsize=11,
                fontweight="bold")
    ax.set_ylabel("Count", color="white"); ax.yaxis.label.set_color("white")
    ax.set_title("Frame & Point Statistics", color="white", fontsize=11)

    # Right: distance distribution
    ax = axes[1]
    d_min = loader_summary.get("global_distance_min_m", 5)
    d_max = loader_summary.get("global_distance_max_m", 80)
    d_mean = loader_summary.get("global_distance_mean_m", 25)
    x = np.linspace(0, 110, 500)
    sigma = max((d_max - d_min) / 4, 1)
    y = np.exp(-0.5 * ((x - d_mean) / sigma) ** 2)
    # Colour by distance using jet
    cmap = plt.cm.get_cmap(DIST_CMAP)
    for i in range(len(x) - 1):
        col = cmap(np.clip(x[i] / DIST_VMAX, 0, 1))
        ax.fill_between(x[i:i+2], y[i:i+2], alpha=0.7, color=col)
    ax.axvline(5,   color="white", ls="--", lw=1.2, label="Spec min 5 m")
    ax.axvline(d_mean, color="yellow", ls="-", lw=1.5,
               label=f"Mean {d_mean:.1f} m")
    ax.set_xlabel("Distance [m]", color="white")
    ax.set_ylabel("Relative Frequency", color="white")
    ax.set_title("Distance Distribution", color="white", fontsize=11)
    ax.legend(facecolor="#222", labelcolor="white", fontsize=8)
    ax.xaxis.label.set_color("white")

    plt.tight_layout()
    if save_path and OUTPUT["save_plots"]:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="#111111")
        logger.info(f"Saved: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════
#  PERFORMANCE SUMMARY
# ═══════════════════════════════════════════════════════

def plot_performance_summary(report,
                              save_path: Optional[str] = None) -> plt.Figure:
    """CCR / FAR gauges + class distribution pie, dark themed."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 6), facecolor="#111111")
    fig.suptitle("PERFORMANCE SUMMARY", color="white",
                 fontsize=14, fontweight="bold", family="monospace")

    for ax in axes:
        ax.set_facecolor("#1a1a1a")
        ax.tick_params(colors="white")
        for sp in ax.spines.values():
            sp.set_edgecolor("#333")

    theta = np.linspace(0, np.pi, 300)

    # CCR gauge
    ax = axes[0]
    ax.plot(np.cos(theta), np.sin(theta), color="#333", lw=12)
    filled = np.linspace(0, np.pi * report.ccr, 300)
    color_ccr = "#00e5ff" if report.ccr_ok else "#FF5722"
    ax.plot(np.cos(filled), np.sin(filled), color=color_ccr, lw=12)
    ax.text(0, 0.3, f"{report.ccr:.3f}", color="white",
            ha="center", va="center", fontsize=24, fontweight="bold")
    ax.text(0, -0.15, f"Target ≥ {report.target_ccr}",
            color="#888", ha="center", fontsize=10)
    ax.text(0, -0.4, "CCR", color="white", ha="center",
            fontsize=15, fontweight="bold")
    ax.text(0, -0.65,
            "✓ MET" if report.ccr_ok else "✗ NOT MET",
            color=color_ccr, ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.8, 1.2)
    ax.set_aspect("equal"); ax.axis("off")

    # FAR gauge
    ax = axes[1]
    ax.plot(np.cos(theta), np.sin(theta), color="#333", lw=12)
    far_norm = min(report.far_per_hour / max(report.target_far * 2, 1e-9), 1.0)
    filled2 = np.linspace(0, np.pi * (1 - far_norm), 300)
    color_far = "#00e5ff" if report.far_ok else "#FF5722"
    ax.plot(np.cos(filled2), np.sin(filled2), color=color_far, lw=12)
    ax.text(0, 0.3, f"{report.far_per_hour:.3f}",
            color="white", ha="center", va="center",
            fontsize=24, fontweight="bold")
    ax.text(0, -0.15, f"Target ≤ {report.target_far}/hr",
            color="#888", ha="center", fontsize=10)
    ax.text(0, -0.4, "FAR [/hr]", color="white",
            ha="center", fontsize=15, fontweight="bold")
    ax.text(0, -0.65,
            "✓ MET" if report.far_ok else "✗ NOT MET",
            color=color_far, ha="center", fontsize=12, fontweight="bold")
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-0.8, 1.2)
    ax.set_aspect("equal"); ax.axis("off")

    # Class pie
    ax = axes[2]
    cls_counts = {k: v for k, v in report.class_counts.items() if v > 0}
    if cls_counts:
        pie_colors = [_get_point_color(k) for k in cls_counts]
        wedges, texts, autotexts = ax.pie(
            list(cls_counts.values()),
            labels=[k.capitalize() for k in cls_counts],
            colors=pie_colors, autopct="%1.1f%%", startangle=90,
        )
        for t in texts + autotexts:
            t.set_color("white")
        ax.set_title("Class Distribution", color="white", fontsize=11)
    else:
        ax.text(0, 0, "No detections", color="white", ha="center")
        ax.axis("off")

    plt.tight_layout()
    if save_path and OUTPUT["save_plots"]:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="#111111")
        logger.info(f"Saved: {save_path}")

    return fig


# ═══════════════════════════════════════════════════════
#  LEGACY ALIASES (keep main.py calls working)
# ═══════════════════════════════════════════════════════

def plot_frame_topdown(frame, preprocess_result, clusters, tracks,
                       save_path=None):
    """Alias → Blickfeld 3D overview."""
    return plot_blickfeld_3d(frame, preprocess_result, clusters, tracks,
                              save_path=save_path, title="OVERVIEW")


def plot_frame_3d(preprocess_result, clusters, frame_id, save_path=None):
    """Alias kept for backward compatibility — now returns full Blickfeld view."""
    fig = plt.figure(figsize=(16, 9), facecolor="#111111")
    ax = fig.add_subplot(111, projection="3d")
    _style_3d_ax(ax)
    _draw_grid(ax)

    pts = preprocess_result.points_xyz
    dist = preprocess_result.distance
    if len(pts) > 0:
        idx = np.random.choice(len(pts), min(12000, len(pts)), replace=False)
        norm_d = np.clip(dist[idx] / DIST_VMAX, 0, 1)
        cols = plt.cm.get_cmap(DIST_CMAP)(norm_d)
        ax.scatter(pts[idx, 0], pts[idx, 1], pts[idx, 2],
                   c=cols, s=1.2, alpha=0.6, depthshade=True)

    for c in clusters:
        if c.label not in ("unknown", "background"):
            col = _get_point_color(c.label)
            ax.scatter(c.points_xyz[:, 0], c.points_xyz[:, 1],
                       c.points_xyz[:, 2], c=col, s=4, alpha=0.9,
                       depthshade=False)
            _annotate_cluster_3d(ax, c, c.label)

    _add_colorbar_distance(fig, ax)
    ax.set_title(f"Frame {frame_id} — 3D Point Cloud",
                 color="white", fontsize=13)

    if save_path and OUTPUT["save_plots"]:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight", facecolor="#111111")

    return fig


def close_all():
    plt.close("all")