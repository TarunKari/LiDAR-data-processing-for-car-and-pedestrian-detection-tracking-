"""
tests/test_pipeline.py — Unit Tests for LiDAR Pipeline
Run with: python tests/test_pipeline.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import numpy as np
import unittest

from src.data_loader import DataLoader, LiDARFrame
from src.preprocessor import Preprocessor
from src.detector import Detector, Cluster
from src.classifier import Classifier, extract_features, rule_based_classify
from src.tracker import Tracker, KalmanFilter2D
from src.evaluator import Evaluator, FrameResult


def make_frame():
    rng = np.random.default_rng(42)
    N = 1000
    pts = rng.uniform([-20, 5, -0.1], [20, 50, 0.1], (N, 3)).astype(np.float32)
    dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2).astype(np.float32)
    return LiDARFrame(
        frame_id=9999, filename="test_frame.csv",
        points_xyz=pts, distance=dist,
        intensity=rng.uniform(0, 255, N).astype(np.float32),
        point_id=np.arange(N, dtype=np.int32),
        return_id=np.zeros(N, dtype=np.int32),
        ambient=np.ones(N, dtype=np.float32),
        timestamp=np.ones(N, dtype=np.float64) * 1.6e18,
    )


def make_car_cluster():
    rng = np.random.default_rng(1)
    pts = rng.uniform([10, 20, 0.3], [13.8, 22.2, 1.8], (200, 3)).astype(np.float32)
    dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2).astype(np.float32)
    return Cluster(0, 1, pts, np.ones(200)*80, dist, np.ones(200))


def make_ped_cluster():
    rng = np.random.default_rng(2)
    pts = rng.uniform([5, 10, 0.0], [5.5, 10.6, 1.7], (50, 3)).astype(np.float32)
    dist = np.sqrt(pts[:, 0]**2 + pts[:, 1]**2).astype(np.float32)
    return Cluster(1, 1, pts, np.ones(50)*50, dist, np.ones(50))


class TestDataLoader(unittest.TestCase):
    def test_frame_sanity_report(self):
        f = make_frame()
        r = f.sanity_report()
        self.assertEqual(r["num_points"], 1000)
        self.assertIn("x_range", r)

    def test_frame_xyz_properties(self):
        f = make_frame()
        self.assertEqual(len(f.x), 1000)

    def test_loader_discover(self):
        loader = DataLoader("data/")
        self.assertGreaterEqual(loader.num_frames, 0)


class TestPreprocessor(unittest.TestCase):
    def test_output_leq_input(self):
        frame = make_frame()
        pre = Preprocessor()
        result = pre.process(frame)
        self.assertLessEqual(result.num_points, frame.num_points)

    def test_range_filter(self):
        dist = np.array([3.0, 10.0], dtype=np.float32)
        pre = Preprocessor()
        mask = pre._range_filter(dist)
        self.assertFalse(mask[0])   # below 5 m min
        self.assertTrue(mask[1])    # 10 m is OK

    def test_outlier_removal(self):
        rng = np.random.default_rng(99)
        cluster_pts = rng.normal([0, 20, 0], [0.3, 0.3, 0.1], (100, 3)).astype(np.float32)
        outlier = np.array([[100.0, 100.0, 100.0]], dtype=np.float32)
        pts = np.vstack([cluster_pts, outlier])
        pre = Preprocessor()
        mask = pre._statistical_outlier_removal(pts)
        self.assertFalse(mask[-1])

    def test_ransac_ground(self):
        rng = np.random.default_rng(0)
        ground = rng.uniform([-10, 5, -0.05], [10, 40, 0.05], (300, 3)).astype(np.float32)
        obj = rng.uniform([-2, 15, 0.5], [2, 18, 2.0], (50, 3)).astype(np.float32)
        pts = np.vstack([ground, obj])
        pre = Preprocessor()
        gnd_mask, _ = pre._ransac_ground_removal(pts)
        self.assertGreater(gnd_mask[:300].sum(), 100)


class TestDetector(unittest.TestCase):
    def test_returns_list(self):
        frame = make_frame()
        pre = Preprocessor()
        det = Detector()
        result = pre.process(frame)
        clusters = det.detect(result, frame_id=frame.frame_id)
        self.assertIsInstance(clusters, list)

    def test_cluster_has_centroid(self):
        frame = make_frame()
        pre = Preprocessor()
        det = Detector()
        result = pre.process(frame)
        clusters = det.detect(result, frame_id=frame.frame_id)
        for c in clusters:
            self.assertEqual(len(c.centroid), 3)
            self.assertEqual(len(c.bbox_dimensions), 3)

    def test_tiny_cluster_filtered(self):
        from src.preprocessor import PreprocessResult
        pts = np.array([[0, 10, 0.5], [0.1, 10.1, 0.5]], dtype=np.float32)
        pr = PreprocessResult(pts, np.ones(2), np.array([10,10],dtype=np.float32),
                              np.ones(2), np.zeros(2,dtype=bool), np.ones(2,dtype=bool), None)
        clusters = Detector().detect(pr)
        self.assertEqual(len(clusters), 0)


class TestClassifier(unittest.TestCase):
    def test_features_17(self):
        f = extract_features(make_car_cluster())
        self.assertEqual(f.shape, (17,))

    def test_rule_car(self):
        label, conf = rule_based_classify(make_car_cluster())
        self.assertEqual(label, "car")
        self.assertGreaterEqual(conf, 0.5)

    def test_rule_pedestrian(self):
        label, _ = rule_based_classify(make_ped_cluster())
        self.assertEqual(label, "pedestrian")

    def test_ml_trains_and_predicts(self):
        clf = Classifier()
        clf.train()
        c = clf.classify_cluster(make_car_cluster())
        self.assertIn(c.label, ["car","pedestrian","cyclist","unknown"])

    def test_classify_all_length(self):
        clf = Classifier()
        clf.train()
        result = clf.classify_all([make_car_cluster(), make_ped_cluster()])
        self.assertEqual(len(result), 2)


class TestTracker(unittest.TestCase):
    def test_kalman_predict(self):
        kf = KalmanFilter2D(np.array([10.0, 20.0]), 0.1, 0.5, 0.3)
        pos = kf.predict()
        self.assertEqual(len(pos), 2)

    def test_kalman_converges(self):
        kf = KalmanFilter2D(np.array([0.0, 0.0]), 0.1, 0.5, 0.1)
        target = np.array([10.0, 20.0])
        for _ in range(30):
            kf.predict(); kf.update(target)
        pos = kf.position
        self.assertAlmostEqual(pos[0], 10.0, delta=1.5)
        self.assertAlmostEqual(pos[1], 20.0, delta=1.5)

    def test_creates_tracks(self):
        tracker = Tracker()
        tracks = tracker.update([make_car_cluster(), make_ped_cluster()])
        self.assertEqual(len(tracks), 2)

    def test_stale_track_deletion(self):
        from config import TRACK
        tracker = Tracker()
        tracker.update([make_car_cluster()])
        for _ in range(TRACK["max_age"] + 2):
            tracker.update([])
        self.assertEqual(len(tracker.tracks), 0)

    def test_confirmation_min_hits(self):
        from config import TRACK
        tracker = Tracker()
        car = make_car_cluster()
        min_hits = TRACK["min_hits"]
        for i in range(min_hits - 1):
            car.frame_id = i
            tracker.update([car])
        self.assertEqual(len(tracker.confirmed_tracks()), 0)
        car.frame_id = min_hits
        tracker.update([car])
        self.assertEqual(len(tracker.confirmed_tracks()), 1)


class TestEvaluator(unittest.TestCase):
    def test_report_runs(self):
        ev = Evaluator()
        car = make_car_cluster(); car.label = "car"; car.confidence = 0.9
        ev.add_frame_result(FrameResult(1, [car], [], 200, 150))
        report = ev.compute_report()
        self.assertGreaterEqual(report.ccr, 0.0)
        self.assertLessEqual(report.ccr, 1.0)

    def test_targets_defined(self):
        from config import PERFORMANCE
        self.assertEqual(PERFORMANCE["target_ccr"], 0.99)
        self.assertEqual(PERFORMANCE["target_far"], 0.01)

    def test_ccr_ok_property(self):
        from src.evaluator import EvaluationReport
        self.assertTrue(EvaluationReport(ccr=0.995, target_ccr=0.99).ccr_ok)
        self.assertFalse(EvaluationReport(ccr=0.85, target_ccr=0.99).ccr_ok)


class TestIntegration(unittest.TestCase):
    def test_full_single_frame(self):
        frame = make_frame()
        pre = Preprocessor(); det = Detector()
        clf = Classifier(); clf.train()
        tracker = Tracker(); ev = Evaluator()

        pr = pre.process(frame)
        clusters = clf.classify_all(det.detect(pr, frame.frame_id))
        tracks = tracker.update(clusters)
        ev.add_frame_result(FrameResult(frame.frame_id, clusters, tracks, frame.num_points, pr.num_points))
        report = ev.compute_report()
        self.assertEqual(report.total_frames, 1)

    def test_real_data_pipeline(self):
        if not os.path.isdir("data/") or not os.listdir("data/"):
            self.skipTest("No real data")
        loader = DataLoader("data/")
        if loader.num_frames == 0:
            self.skipTest("No frames")
        pre = Preprocessor(); det = Detector()
        clf = Classifier(); clf.train()
        for frame in loader.iter_frames():
            pr = pre.process(frame)
            clusters = clf.classify_all(det.detect(pr, frame.frame_id))
            self.assertIsInstance(clusters, list)
            break


if __name__ == "__main__":
    unittest.main(verbosity=2)