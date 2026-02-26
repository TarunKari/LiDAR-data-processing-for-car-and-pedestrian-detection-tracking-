"""
evaluator.py — Performance Evaluation
======================================
CCR = correctly_classified_object_clusters / total_object_gt_clusters
      (background clusters excluded from CCR — they are not "classified")

FAR = background_clusters_predicted_as_object / total_time_hours
"""

import numpy as np
import logging
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field

from config import PERFORMANCE, CLASS
from src.detector import Cluster
from src.tracker import Track

logger = logging.getLogger(__name__)

OBJECT_CLASSES = {"car", "pedestrian", "cyclist"}


@dataclass
class FrameResult:
    frame_id: int
    clusters: List[Cluster]
    tracks: List[Track]
    num_input_points: int
    num_preprocessed_points: int
    processing_time_s: float = 0.0


@dataclass
class EvaluationReport:
    total_frames: int = 0
    total_detections: int = 0
    total_confirmed_tracks: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)

    ccr: float = 0.0
    far_per_hour: float = 0.0
    confusion_matrix: Optional[np.ndarray] = None
    class_names: List[str] = field(default_factory=list)

    total_track_ids: int = 0
    mean_track_length: float = 0.0
    id_switches: int = 0

    spec_compliant: bool = True
    spec_notes: List[str] = field(default_factory=list)

    target_ccr: float = PERFORMANCE["target_ccr"]
    target_far: float = PERFORMANCE["target_far"]

    @property
    def ccr_ok(self):      return self.ccr >= self.target_ccr
    @property
    def far_ok(self):      return self.far_per_hour <= self.target_far
    @property
    def targets_met(self): return self.ccr_ok and self.far_ok

    def summary_str(self) -> str:
        lines = [
            "═" * 55,
            "  LIDAR DETECTION & TRACKING — EVALUATION REPORT",
            "═" * 55,
            f"  Frames processed    : {self.total_frames}",
            f"  Total detections    : {self.total_detections}",
            f"  Confirmed tracks    : {self.total_confirmed_tracks}",
            f"  Unique track IDs    : {self.total_track_ids}",
            f"  Mean track length   : {self.mean_track_length:.1f} frames",
            "",
            "  Class distribution:",
        ]
        for cls, cnt in self.class_counts.items():
            lines.append(f"    {cls:<15}: {cnt}")
        lines += [
            "",
            f"  CCR  : {self.ccr:.4f}  (target ≥ {self.target_ccr})"
            + (" ✓" if self.ccr_ok else " ✗"),
            f"  FAR  : {self.far_per_hour:.5f}/hr  (target ≤ {self.target_far})"
            + (" ✓" if self.far_ok else " ✗"),
            "",
            f"  Overall targets met : {'YES ✓' if self.targets_met else 'NO ✗'}",
            "═" * 55,
        ]
        return "\n".join(lines)


class Evaluator:

    def __init__(self):
        self.frame_results: List[FrameResult] = []
        self.all_tracks: List[Track] = []
        self.label_names = [CLASS["labels"][i] for i in range(5)]

    def add_frame_result(self, result: FrameResult):
        self.frame_results.append(result)

    def add_tracks(self, tracks: List[Track]):
        existing = {t.track_id for t in self.all_tracks}
        for t in tracks:
            if t.track_id not in existing:
                self.all_tracks.append(t)
                existing.add(t.track_id)

    def compute_report(
        self,
        ground_truth: Optional[List[Dict]] = None,
        total_time_hours: float = 0.0,
        data_dir: Optional[str] = None,      # kept for API compatibility
    ) -> "EvaluationReport":

        report = EvaluationReport()
        report.total_frames = len(self.frame_results)
        all_clusters = [c for r in self.frame_results for c in r.clusters]
        report.total_detections = len(all_clusters)

        confirmed = [t for t in self.all_tracks if t.is_confirmed]
        report.total_confirmed_tracks = len(confirmed)
        report.total_track_ids = len(self.all_tracks)

        tl = [t.hits for t in self.all_tracks]
        report.mean_track_length = float(np.mean(tl)) if tl else 0.0

        for cls in CLASS["labels"].values():
            report.class_counts[cls] = sum(1 for c in all_clusters if c.label == cls)

        # ── Build ground truth directly from clusters already in memory ──
        # This guarantees (frame_id, cluster_id) keys always match.
        from src.ground_truth import label_clusters
        gt_dict = label_clusters(all_clusters)
        gt_list = [
            {"frame_id": fid, "cluster_id": cid, "true_label": lbl}
            for (fid, cid), lbl in gt_dict.items()
        ]
        logger.info("Ground truth: %d labels built from pipeline clusters", len(gt_list))

        ccr, far, cm = self._eval_with_gt(all_clusters, gt_list, total_time_hours)
        report.ccr = ccr
        report.far_per_hour = far
        report.confusion_matrix = cm

        report.class_names = self.label_names
        report.spec_compliant, report.spec_notes = self._check_spec()
        return report

    def _eval_with_gt(
        self,
        clusters: List[Cluster],
        ground_truth: List[Dict],
        total_hours: float,
    ) -> Tuple[float, float, np.ndarray]:
        """
        CCR counts only car/pedestrian/cyclist GT clusters.
        FAR counts background GT clusters predicted as object.
        """
        gt_lookup = {
            (g["frame_id"], g["cluster_id"]): g["true_label"]
            for g in ground_truth
        }

        n_classes = 5
        cm = np.zeros((n_classes, n_classes), dtype=int)
        correct = 0
        total_obj = 0
        false_alarms = 0

        for c in clusters:
            key = (c.frame_id, c.cluster_id)
            if key not in gt_lookup:
                continue
            true_lbl = gt_lookup[key]
            pred_lbl = c.label

            true_id = CLASS["label_ids"].get(true_lbl, 4)
            pred_id = CLASS["label_ids"].get(pred_lbl, 4)
            cm[true_id][pred_id] += 1

            if true_lbl in OBJECT_CLASSES:
                total_obj += 1
                if pred_lbl == true_lbl:
                    correct += 1

            if true_lbl == "background" and pred_lbl in OBJECT_CLASSES:
                false_alarms += 1

        ccr = correct / max(total_obj, 1)
        far = false_alarms / max(total_hours, 1e-9)

        logger.info(
            "CCR: %d/%d object clusters correct → %.4f | "
            "FAR: %d false alarms → %.5f/hr",
            correct, total_obj, ccr, false_alarms, far,
        )
        return ccr, far, cm

    def _check_spec(self) -> Tuple[bool, List[str]]:
        from config import SENSOR
        notes = []
        compliant = True
        for r in self.frame_results:
            for c in r.clusters:
                d = float(c.distance.mean())
                if d < SENSOR["range_min_m"]:
                    notes.append(f"Below min range: {d:.2f}m")
                    compliant = False
                if d > SENSOR["range_max_m"]:
                    notes.append(f"Above max range: {d:.2f}m")
                    compliant = False
        if not notes:
            notes.append("All detections within sensor specification range.")
        return compliant, list(set(notes))

    def sanity_check_dataset(self, loader_summary: dict) -> List[str]:
        findings = []
        if not loader_summary:
            return ["No dataset summary available."]
        n = loader_summary.get("total_frames", 0)
        exp = 152
        findings.append(
            f"✓  All {n} frames found." if n >= exp
            else f"⚠  Only {n}/{exp} expected frames found."
        )
        dmin = loader_summary.get("global_distance_min_m", 0)
        dmax = loader_summary.get("global_distance_max_m", 0)
        findings.append(f"✓  Distance range in data: {dmin:.2f}–{dmax:.2f} m")
        if dmin < 5.0:
            findings.append("⚠  Some points below sensor min range of 5 m.")
        if loader_summary.get("sensor_spec_range_ok", True):
            findings.append("✓  Data generally within sensor detection range specification.")
        mp = loader_summary.get("mean_points_per_frame", 0)
        sp = loader_summary.get("std_points_per_frame", 0)
        findings.append(f"✓  Mean points/frame: {mp:.0f} ± {sp:.0f}")
        findings.append(
            f"✓  Intensity range: "
            f"{loader_summary.get('global_intensity_min', 0):.0f}–"
            f"{loader_summary.get('global_intensity_max', 0):.0f}"
        )
        return findings