"""
main.py — LiDAR Detection & Tracking Pipeline — Full Runner
Usage:
    python main.py --data_dir data/ --output_dir output/ [--visualize] [--eval_only]
"""

import os
import sys

# ── Set matplotlib backend FIRST before any other import ──
# This must happen before importing visualizer or any matplotlib-using module.
# On Windows "Agg" (non-interactive file renderer) avoids display/Tk errors.
import matplotlib
matplotlib.use("Agg")

import time
import argparse
import logging
import json
import csv
import pandas as pd
import numpy as np

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import PERFORMANCE, OUTPUT
from src.data_loader import DataLoader, LiDARFrame
from src.preprocessor import Preprocessor
from src.detector import Detector
from src.classifier import Classifier
from src.tracker import Tracker
from src.evaluator import Evaluator, FrameResult
from src import visualizer

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────
#  PIPELINE
# ─────────────────────────────────────────────

def run_pipeline(data_dir: str, output_dir: str, visualize: bool = False):
    """
    Execute the full LiDAR detection & tracking pipeline on all available frames.
    Works with 4 provided frames; designed for all 152.
    """
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "frames"), exist_ok=True)
    os.makedirs(os.path.join(output_dir, "reports"), exist_ok=True)

    # ── Step 0: Load data ──────────────────────────────────────
    logger.info("=" * 60)
    logger.info("  LiDAR DETECTION & TRACKING PIPELINE — START")
    logger.info("=" * 60)
    logger.info(f"Data directory : {data_dir}")
    logger.info(f"Output directory: {output_dir}")

    loader = DataLoader(data_dir)
    logger.info(f"Found {loader.num_frames} frames (expected 152 for full dataset)")

    # ── Dataset sanity check ───────────────────────────────────
    logger.info("Running dataset sanity check...")
    sanity_df = loader.sanity_check_all()
    summary = loader.dataset_summary()

    logger.info(f"Total points in dataset: {summary.get('total_points', 0):,}")
    logger.info(f"Mean points/frame: {summary.get('mean_points_per_frame', 0):.0f}")
    logger.info(f"Distance range: {summary.get('global_distance_min_m', 0):.2f}–{summary.get('global_distance_max_m', 0):.2f} m")
    logger.info(f"Sensor spec range OK: {summary.get('sensor_spec_range_ok', '?')}")

    # Save sanity check to CSV
    sanity_df.to_csv(os.path.join(output_dir, "sanity_check.csv"), index=False)
    with open(os.path.join(output_dir, "dataset_summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    if visualize:
        fig = visualizer.plot_dataset_statistics(
            summary, save_path=os.path.join(output_dir, "reports", "dataset_statistics.png")
        )
        visualizer.close_all()

    # ── Initialise pipeline components ────────────────────────
    preprocessor = Preprocessor()
    detector = Detector()
    classifier = Classifier()
    tracker = Tracker()
    evaluator = Evaluator()
    stored_frames = []  # for parking car panel

    # Train classifier (synthetic data since we don't have GT labels)
    logger.info("Training ML classifier on synthetic data...")
    classifier.train()
    logger.info("Classifier ready.")

    # ── Main processing loop ───────────────────────────────────
    all_detection_rows = []
    total_start_time = time.time()

    for frame_idx, frame in enumerate(loader.iter_frames()):
        t_frame_start = time.time()
        logger.info(f"Processing frame {frame.frame_id} ({frame_idx+1}/{loader.num_frames})...")

        # 1. Preprocess
        pre_result = preprocessor.process(frame)
        logger.debug(f"  Preprocessed: {frame.num_points} → {pre_result.num_points} pts")

        # 2. Detect (cluster)
        clusters = detector.detect(pre_result, frame_id=frame.frame_id)
        logger.info(f"  Detected {len(clusters)} cluster(s)")

        # 3. Classify
        clusters = classifier.classify_all(clusters)

        # Count by label
        label_counts = {}
        for c in clusters:
            label_counts[c.label] = label_counts.get(c.label, 0) + 1
        for lbl, cnt in label_counts.items():
            if lbl not in ["unknown", "background"]:
                logger.info(f"    {lbl}: {cnt}")

        # 4. Track
        active_tracks = tracker.update(clusters)
        confirmed = tracker.confirmed_tracks()
        logger.info(f"  Active tracks: {len(active_tracks)}, Confirmed: {len(confirmed)}")

        # 5. Record results
        t_frame_end = time.time()
        result = FrameResult(
            frame_id=frame.frame_id,
            clusters=clusters,
            tracks=confirmed,
            num_input_points=frame.num_points,
            num_preprocessed_points=pre_result.num_points,
            processing_time_s=t_frame_end - t_frame_start,
        )
        evaluator.add_frame_result(result)
        evaluator.add_tracks(active_tracks)

        # Save detection rows
        for c in clusters:
            all_detection_rows.append(c.to_dict())

        stored_frames.append((frame, pre_result, clusters, confirmed))

        # 6. Visualize (Blickfeld-style 3D outputs)
        if visualize:
            fdir = os.path.join(output_dir, "frames")

            # Overview (Image 4 style)
            visualizer.plot_blickfeld_3d(
                frame, pre_result, clusters, confirmed,
                save_path=os.path.join(fdir, f"frame_{frame.frame_id:04d}_overview.png"),
                title="OVERVIEW",
            )
            visualizer.close_all()

            # Pedestrian focus (Image 1 style)
            if any(c.label == "pedestrian" for c in clusters):
                visualizer.plot_class_focus_3d(
                    frame, pre_result, clusters, confirmed,
                    focus_labels=["pedestrian"],
                    title="PEDESTRIAN",
                    save_path=os.path.join(fdir, f"frame_{frame.frame_id:04d}_pedestrian.png"),
                )
                visualizer.close_all()

            # Car & Cyclist focus (Image 2 style)
            if any(c.label in ("car", "cyclist") for c in clusters):
                visualizer.plot_class_focus_3d(
                    frame, pre_result, clusters, confirmed,
                    focus_labels=["car", "cyclist"],
                    title="CAR & CYCLIST",
                    save_path=os.path.join(fdir, f"frame_{frame.frame_id:04d}_car_cyclist.png"),
                )
                visualizer.close_all()

            logger.info(f"  Visualizations saved for frame {frame.frame_id}")

        logger.info(f"  Frame done in {t_frame_end - t_frame_start:.3f}s")

    total_time = time.time() - total_start_time
    logger.info(f"\nAll {loader.num_frames} frames processed in {total_time:.2f}s")

    # ── Evaluation ─────────────────────────────────────────────
    total_hours = loader.num_frames / 10.0 / 3600.0  # assume 10 Hz
    report = evaluator.compute_report(total_time_hours=total_hours, data_dir=data_dir)

    logger.info("\n" + report.summary_str())

    # ── Save outputs ───────────────────────────────────────────

    # 1. Detection CSV
    if all_detection_rows and OUTPUT["save_csv"]:
        det_csv_path = os.path.join(output_dir, "detections.csv")
        df_det = pd.DataFrame(all_detection_rows)
        df_det.to_csv(det_csv_path, index=False)
        logger.info(f"Detections saved: {det_csv_path}")

    # 2. Tracks CSV
    track_rows = [t.to_dict() for t in evaluator.all_tracks]
    if track_rows and OUTPUT["save_csv"]:
        trk_csv_path = os.path.join(output_dir, "tracks.csv")
        pd.DataFrame(track_rows).to_csv(trk_csv_path, index=False)
        logger.info(f"Tracks saved: {trk_csv_path}")

    # 3. Sanity findings
    sanity_findings = evaluator.sanity_check_dataset(summary)
    logger.info("\nDataset Sanity Findings:")
    for f_line in sanity_findings:
        logger.info(f"  {f_line}")

    # 4. Visualizations
    if visualize:
        logger.info("Generating summary visualizations...")

        visualizer.plot_track_history(
            evaluator.all_tracks,
            save_path=os.path.join(output_dir, "reports", "track_history.png"),
        )
        visualizer.close_all()

        visualizer.plot_performance_summary(
            report,
            save_path=os.path.join(output_dir, "reports", "performance_summary.png"),
        )
        visualizer.close_all()

        # Parking car panel (Image 3 style) — use available frames with cars
        car_frames = [(fr, pr, cl, tr) for (fr, pr, cl, tr)
                      in stored_frames if any(c.label == "car" for c in cl)]
        if car_frames:
            panel_data = []
            subtitles = ["driver approaching car", "driver approaching car", "car door"]
            for i, (fr, pr, cl, tr) in enumerate(car_frames[:3]):
                panel_data.append((fr, pr, cl, tr, subtitles[i % len(subtitles)]))
            visualizer.plot_parking_car_panel(
                panel_data,
                save_path=os.path.join(output_dir, "reports", "parking_car_panel.png"),
            )
            visualizer.close_all()
            logger.info("Saved parking car panel")

    # 5. Text report
    if OUTPUT["save_report"]:
        report_path = os.path.join(output_dir, "reports", "evaluation_report.txt")
        with open(report_path, "w") as f:
            f.write(report.summary_str())
            f.write("\n\nDataset Sanity Findings:\n")
            for line in sanity_findings:
                f.write(f"  {line}\n")
            f.write(f"\nTotal processing time: {total_time:.2f}s\n")
            f.write(f"Frames processed: {loader.num_frames}\n")
        logger.info(f"Report saved: {report_path}")

    logger.info("=" * 60)
    logger.info("  PIPELINE COMPLETE")
    logger.info(f"  CCR  = {report.ccr:.4f}  (target {report.target_ccr})")
    logger.info(f"  FAR  = {report.far_per_hour:.5f}/hr (target {report.target_far})")
    logger.info(f"  Targets met: {'YES ✓' if report.targets_met else 'NO ✗'}")
    logger.info("=" * 60)

    return report


# ─────────────────────────────────────────────
#  ENTRY POINT
# ─────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LiDAR Detection & Tracking Pipeline")
    parser.add_argument("--data_dir", default="data/", help="Directory containing CSV frame files")
    parser.add_argument("--output_dir", default="output/", help="Directory for output files")
    parser.add_argument("--visualize", action="store_true", help="Generate and save visualizations")
    parser.add_argument("--eval_only", action="store_true", help="Only run evaluation, skip pipeline")
    args = parser.parse_args()

    if not os.path.isdir(args.data_dir):
        print(f"ERROR: Data directory not found: {args.data_dir}")
        sys.exit(1)

    report = run_pipeline(
        data_dir=args.data_dir,
        output_dir=args.output_dir,
        visualize=args.visualize,
    )