"""Unit tests for Spatial Reasoning Engine."""

import numpy as np
import pytest
import tempfile

from src.models.spatial_reasoning import SpatialReasoningEngine


@pytest.fixture
def config():
    return {
        "num_row_components": 4,
        "kde_bandwidth": 0.1,
        "min_detections": 3,
        "save_dir": tempfile.mkdtemp(),
    }


@pytest.fixture
def engine(config):
    return SpatialReasoningEngine(config)


@pytest.fixture
def synthetic_boxes():
    """Create synthetic shelf-like GT boxes (3 rows of products)."""
    boxes = []
    image_size = 550

    for row_y in [100, 250, 400]:  # 3 rows
        for col_x in range(50, 500, 60):  # products in each row
            x1 = col_x
            y1 = row_y
            x2 = col_x + 50
            y2 = row_y + 80
            boxes.append([x1, y1, x2, y2])

    return [np.array(boxes, dtype=np.float32)]


class TestSpatialReasoningEngine:
    def test_init(self, engine):
        assert engine is not None
        assert not engine.is_fitted

    def test_fit(self, engine, synthetic_boxes):
        engine.fit(synthetic_boxes, image_size=550)
        assert engine.is_fitted
        assert engine._mean_objects_per_image > 0

    def test_compute_spatial_features_shape(self, engine, synthetic_boxes):
        engine.fit(synthetic_boxes, image_size=550)

        detections = synthetic_boxes[0]
        features = engine.compute_spatial_features(detections, image_size=550)

        assert features.shape == (8,)
        assert features.dtype == np.float32

    def test_spatial_features_empty(self, engine):
        features = engine.compute_spatial_features(np.zeros((0, 4)), image_size=550)
        assert features.shape == (8,)
        assert np.allclose(features, 0.0)

    def test_spatial_features_few_detections(self, engine):
        # Fewer than min_detections
        boxes = np.array([[100, 100, 200, 200]], dtype=np.float32)
        features = engine.compute_spatial_features(boxes, image_size=550)
        assert features.shape == (8,)

    def test_generate_density_field_shape(self, engine, synthetic_boxes):
        engine.fit(synthetic_boxes, image_size=550)

        detections = synthetic_boxes[0]
        density = engine.generate_density_field(detections, shape=(69, 69), image_size=550)

        assert density.shape == (69, 69)
        assert density.dtype == np.float32
        assert density.min() >= 0.0
        assert density.max() <= 1.0

    def test_density_field_unfitted(self, engine):
        boxes = np.array([[100, 100, 200, 200]] * 10, dtype=np.float32)
        density = engine.generate_density_field(boxes, shape=(32, 32), image_size=550)
        # Unfitted → uniform 0.5
        assert np.allclose(density, 0.5)

    def test_density_field_empty_detections(self, engine, synthetic_boxes):
        engine.fit(synthetic_boxes, image_size=550)
        density = engine.generate_density_field(np.zeros((0, 4)), shape=(32, 32), image_size=550)
        assert np.allclose(density, 0.5)

    def test_bayesian_update(self, engine, synthetic_boxes):
        engine.fit(synthetic_boxes, image_size=550)
        scores = np.array([0.9, 0.5, 0.1], dtype=np.float32)
        features = engine.compute_spatial_features(synthetic_boxes[0], image_size=550)
        updated = engine.bayesian_update(scores, features)

        assert updated.shape == scores.shape
        assert (updated >= 0).all() and (updated <= 1).all()

    def test_bayesian_update_unfitted(self, engine):
        scores = np.array([0.9, 0.5], dtype=np.float32)
        features = np.zeros(8, dtype=np.float32)
        updated = engine.bayesian_update(scores, features)
        # Unfitted → return original scores
        assert np.array_equal(updated, scores)

    def test_save_load(self, engine, synthetic_boxes):
        engine.fit(synthetic_boxes, image_size=550)

        # Compute features before save
        features_before = engine.compute_spatial_features(synthetic_boxes[0], image_size=550)

        # Save
        engine.save()

        # Create new engine and load
        new_engine = SpatialReasoningEngine(engine.config)
        new_engine.load()

        assert new_engine.is_fitted

        # Features should be identical
        features_after = new_engine.compute_spatial_features(synthetic_boxes[0], image_size=550)
        assert np.allclose(features_before, features_after)

    def test_gmm_detects_rows(self, engine, synthetic_boxes):
        engine.fit(synthetic_boxes, image_size=550)

        # GMM should detect approximately 3 rows
        assert engine.gmm.n_components >= 2
        assert engine.gmm.n_components <= 6  # Reasonable range

    def test_coverage_feature(self, engine, synthetic_boxes):
        engine.fit(synthetic_boxes, image_size=550)
        features = engine.compute_spatial_features(synthetic_boxes[0], image_size=550)
        # Coverage should be between 0 and 1
        assert 0 <= features[6] <= 1.0
