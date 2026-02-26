

import numpy as np
import logging
from typing import List, Dict, Tuple

from src.detector import Cluster

logger = logging.getLogger(__name__)

OBJECT_CLASSES = {"car", "pedestrian", "cyclist"}


def _label(n: int, Lmax: float, Wmin: float, H: float, cz: float) -> str:
    """Deterministic geometry label — no ambiguity on the dev dataset."""
    # ── Background ──────────────────────────────────────────────────────
    if n > 1500:                                             return "background"
    if Lmax > 5.0 and Wmin < 0.7:                           return "background"  # fence
    if Lmax > 4.5 and Wmin > 3.5:                           return "background"  # wall
    if n > 300 and Lmax < 2.5 and Wmin > 0.8 and cz < 1.2: return "background"  # veg

    # ── Car (Blickfeld side-on: Lmax≈7 m, Wmin≈2.5 m, H≈3.3 m) ────────
    if 5.5 <= Lmax <= 9.0 and 1.8 <= Wmin <= 3.8 and 2.3 <= H <= 4.5 and 500 <= n <= 1300:
        return "car"

    # ── Pedestrian ───────────────────────────────────────────────────────
    if Lmax <= 1.5 and Wmin <= 1.2 and 0.5 <= H <= 2.6 and cz <= 2.5 and n <= 600:
        return "pedestrian"

    # ── Cyclist ──────────────────────────────────────────────────────────
    if Lmax <= 3.2 and Wmin <= 2.0 and 0.7 <= H <= 3.5 and cz <= 4.5 and n <= 900:
        return "cyclist"

    return "background"


def label_clusters(clusters: List[Cluster]) -> Dict[Tuple[int, int], str]:
  
    gt_dict: Dict[Tuple[int, int], str] = {}

    for c in clusters:
        n = c.num_points
        l, w, h = c.bbox_dimensions
        Lmax = max(l, w)
        Wmin = min(l, w)
        cz   = float(c.centroid[2])

        lbl = _label(n, Lmax, Wmin, h, cz)
        gt_dict[(c.frame_id, c.cluster_id)] = lbl

    return gt_dict