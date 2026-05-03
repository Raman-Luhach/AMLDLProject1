"""Unit tests for Ablation Study Framework."""

import numpy as np

from src.evaluation.ablation import AblationFramework, DENSITY_BUCKETS


class TestAblationFramework:
    def test_density_buckets_cover_all(self):
        """Ensure density buckets cover the full range."""
        for count in [0, 25, 50, 100, 150, 200, 500]:
            found = False
            for name, (lo, hi) in DENSITY_BUCKETS.items():
                if lo <= count < hi:
                    found = True
                    break
            assert found, f"Count {count} not covered by any bucket"

    def test_compute_iou_identical(self):
        """IoU of identical boxes should be 1."""
        boxes = np.array([[0, 0, 100, 100], [50, 50, 150, 150]], dtype=np.float32)
        iou = AblationFramework._compute_iou(boxes, boxes)
        assert iou.shape == (2, 2)
        assert np.allclose(np.diag(iou), 1.0)

    def test_compute_iou_no_overlap(self):
        """IoU of non-overlapping boxes should be 0."""
        boxes1 = np.array([[0, 0, 10, 10]], dtype=np.float32)
        boxes2 = np.array([[20, 20, 30, 30]], dtype=np.float32)
        iou = AblationFramework._compute_iou(boxes1, boxes2)
        assert iou[0, 0] == 0.0

    def test_compute_iou_partial(self):
        """IoU of partially overlapping boxes should be between 0 and 1."""
        boxes1 = np.array([[0, 0, 100, 100]], dtype=np.float32)
        boxes2 = np.array([[50, 50, 150, 150]], dtype=np.float32)
        iou = AblationFramework._compute_iou(boxes1, boxes2)
        assert 0 < iou[0, 0] < 1

    def test_variants_defined(self):
        """All expected variants should be defined."""
        expected = [
            "full_hybrid",
            "dl_only",
            "no_recalibrator",
            "no_spatial_attention",
            "no_row_model",
            "no_density_field",
            "hard_nms",
            "no_cbam",
        ]
        for name in expected:
            assert name in AblationFramework.VARIANTS

    def test_empty_predictions(self):
        """Metrics should handle empty predictions gracefully."""
        framework = AblationFramework.__new__(AblationFramework)
        metrics = framework._compute_metrics(
            predictions=[{"boxes": np.zeros((0, 4)), "scores": np.array([])}],
            targets=[{"boxes": np.array([[0, 0, 10, 10]])}],
            iou_threshold=0.5,
        )
        assert metrics["mAP_50"] == 0.0
        assert metrics["FN"] == 1

    def test_perfect_predictions(self):
        """Perfect predictions should yield high metrics."""
        framework = AblationFramework.__new__(AblationFramework)
        gt_box = np.array([[10, 10, 50, 50]], dtype=np.float32)
        metrics = framework._compute_metrics(
            predictions=[{"boxes": gt_box, "scores": np.array([0.99])}],
            targets=[{"boxes": gt_box}],
            iou_threshold=0.5,
        )
        assert metrics["precision"] == 1.0
        assert metrics["recall"] == 1.0
        assert metrics["F1"] == 1.0
