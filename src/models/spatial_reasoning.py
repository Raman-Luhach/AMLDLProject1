"""Spatial Reasoning Engine for Hybrid Neuro-Symbolic Detection.

Probabilistic ML module that captures 'shelf grammar' — the structural
regularities in retail shelf environments:
    1. Products are arranged in horizontal rows (modeled by GMM)
    2. Within rows, products have regular spacing (regularity score)
    3. Product density is spatially predictable (modeled by KDE)
    4. Local context informs detection confidence (Bayesian update)

This is the ML core of the hybrid system. It uses scikit-learn models
(NOT neural networks) to provide probabilistic spatial reasoning that
complements the DL visual features from YOLACT.

Usage:
    engine = SpatialReasoningEngine(config)
    engine.fit(all_gt_boxes)  # fit on training GT
    features = engine.compute_spatial_features(detections)
    density = engine.generate_density_field(detections, (H, W))
"""

import logging
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.ndimage import gaussian_filter
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KernelDensity

logger = logging.getLogger(__name__)


class SpatialReasoningEngine:
    """Probabilistic spatial reasoning for retail shelf scenes.

    Models shelf structure using GMM (row detection) and KDE (density estimation).
    Extracts spatial features and generates density fields that inform the
    neural network's attention mechanism.

    Args:
        config: Dictionary with spatial reasoning parameters.
    """

    def __init__(self, config: Dict) -> None:
        self.config = config
        self.num_row_components = config.get("num_row_components", 8)
        self.kde_bandwidth = config.get("kde_bandwidth", 0.05)
        self.min_detections = config.get("min_detections", 5)
        self.save_dir = Path(config.get("save_dir", "results/hybrid/spatial_models"))

        # GMM for row structure detection
        self.gmm = GaussianMixture(
            n_components=self.num_row_components,
            covariance_type="full",
            random_state=42,
            max_iter=200,
            n_init=3,
        )

        # KDE for spatial density estimation
        self.kde = KernelDensity(
            bandwidth=self.kde_bandwidth,
            kernel="gaussian",
        )

        self.is_fitted = False
        self._mean_objects_per_image = 0.0
        self._mean_box_area = 0.0
        self._std_box_area = 0.0

    def fit(self, all_gt_boxes: List[np.ndarray], image_size: int = 550) -> None:
        """Fit spatial models on training ground-truth boxes.

        Fits the GMM on normalized y-center coordinates to learn row structure,
        and fits the KDE on normalized (x, y) centers for density estimation.

        Args:
            all_gt_boxes: List of arrays, each (N_i, 4) in [x1, y1, x2, y2] format.
            image_size: Input image size for normalization.
        """
        logger.info("Fitting spatial reasoning engine on %d images...", len(all_gt_boxes))

        all_centers_y = []
        all_centers_xy = []
        all_areas = []
        objects_per_image = []

        for boxes in all_gt_boxes:
            if len(boxes) == 0:
                objects_per_image.append(0)
                continue

            # Normalize to [0, 1]
            boxes_norm = boxes / image_size

            # Compute centers
            cx = (boxes_norm[:, 0] + boxes_norm[:, 2]) / 2.0
            cy = (boxes_norm[:, 1] + boxes_norm[:, 3]) / 2.0

            # Compute areas
            w = boxes_norm[:, 2] - boxes_norm[:, 0]
            h = boxes_norm[:, 3] - boxes_norm[:, 1]
            areas = w * h

            all_centers_y.append(cy)
            all_centers_xy.append(np.stack([cx, cy], axis=1))
            all_areas.extend(areas.tolist())
            objects_per_image.append(len(boxes))

        if not all_centers_y:
            logger.warning("No ground-truth boxes found. Spatial engine not fitted.")
            return

        # Concatenate all data
        centers_y = np.concatenate(all_centers_y).reshape(-1, 1)
        centers_xy = np.concatenate(all_centers_xy)

        # Fit GMM on y-centers (row detection)
        # Use BIC to select optimal number of components
        best_n = self._select_gmm_components(centers_y)
        self.gmm = GaussianMixture(
            n_components=best_n,
            covariance_type="full",
            random_state=42,
            max_iter=200,
            n_init=3,
        )
        self.gmm.fit(centers_y)
        logger.info("GMM fitted with %d row components (BIC-selected)", best_n)

        # Fit KDE on (x, y) centers — subsample to cap inference cost
        max_kde_samples = 5000
        if len(centers_xy) > max_kde_samples:
            rng = np.random.RandomState(42)
            idx = rng.choice(len(centers_xy), max_kde_samples, replace=False)
            kde_data = centers_xy[idx]
        else:
            kde_data = centers_xy
        self.kde.fit(kde_data)
        logger.info("KDE fitted on %d center points (subsampled from %d)", len(kde_data), len(centers_xy))

        # Store statistics for feature computation
        self._mean_objects_per_image = np.mean(objects_per_image)
        self._mean_box_area = np.mean(all_areas)
        self._std_box_area = np.std(all_areas)

        self.is_fitted = True
        logger.info("Spatial reasoning engine fitted successfully.")

    def _select_gmm_components(self, data: np.ndarray, max_k: int = 12) -> int:
        """Select optimal GMM components using BIC."""
        best_bic = np.inf
        best_k = 2

        for k in range(2, min(max_k + 1, self.num_row_components + 1)):
            gmm = GaussianMixture(
                n_components=k,
                covariance_type="full",
                random_state=42,
                max_iter=100,
            )
            gmm.fit(data)
            bic = gmm.bic(data)
            if bic < best_bic:
                best_bic = bic
                best_k = k

        return best_k

    def compute_spatial_features(self, detections: np.ndarray, image_size: int = 550) -> np.ndarray:
        """Extract 8-dimensional spatial feature vector from detections.

        Features capture scene-level statistics that characterize the
        spatial layout of detected objects.

        Args:
            detections: Array of shape (N, 4) in [x1, y1, x2, y2] format.
            image_size: Image size for normalization.

        Returns:
            8-dimensional feature vector:
                [n_objects_norm, mean_area, std_area, n_rows,
                 mean_row_spacing, regularity, coverage, density_uniformity]
        """
        features = np.zeros(8, dtype=np.float32)

        if len(detections) < self.min_detections:
            return features

        # Normalize coordinates
        dets_norm = detections / image_size

        # 1. Number of objects (normalized by training mean)
        n_objects = len(detections)
        features[0] = n_objects / max(self._mean_objects_per_image, 1.0)

        # 2-3. Area statistics
        w = dets_norm[:, 2] - dets_norm[:, 0]
        h = dets_norm[:, 3] - dets_norm[:, 1]
        areas = w * h
        features[1] = np.mean(areas) / max(self._mean_box_area, 1e-6)
        features[2] = np.std(areas) / max(self._std_box_area, 1e-6)

        # 4. Number of detected rows (from GMM)
        if self.is_fitted:
            cy = ((dets_norm[:, 1] + dets_norm[:, 3]) / 2.0).reshape(-1, 1)
            row_assignments = self.gmm.predict(cy)
            n_rows = len(np.unique(row_assignments))
            features[3] = n_rows / self.gmm.n_components
        else:
            features[3] = 0.5

        # 5. Mean row spacing
        if self.is_fitted and n_rows > 1:
            row_centers = []
            for r in np.unique(row_assignments):
                mask = row_assignments == r
                row_centers.append(np.mean(cy[mask]))
            row_centers = sorted(row_centers)
            spacings = np.diff(row_centers)
            features[4] = np.mean(spacings) if len(spacings) > 0 else 0.0
        else:
            features[4] = 0.0

        # 6. Regularity score (within-row spacing consistency)
        if self.is_fitted and n_rows >= 1:
            regularity_scores = []
            for r in np.unique(row_assignments):
                mask = row_assignments == r
                if np.sum(mask) < 3:
                    continue
                row_cx = (dets_norm[mask, 0] + dets_norm[mask, 2]) / 2.0
                row_cx_sorted = np.sort(row_cx)
                gaps = np.diff(row_cx_sorted)
                if len(gaps) > 0 and np.mean(gaps) > 0:
                    # CV (coefficient of variation) — lower = more regular
                    cv = np.std(gaps) / np.mean(gaps)
                    regularity_scores.append(1.0 / (1.0 + cv))
            features[5] = np.mean(regularity_scores) if regularity_scores else 0.0
        else:
            features[5] = 0.0

        # 7. Coverage ratio (fraction of image covered by detections)
        total_area = np.sum(areas)
        features[6] = min(total_area, 1.0)

        # 8. Density uniformity (how evenly distributed detections are)
        if n_objects >= 4:
            # Divide image into 4x4 grid and measure uniformity
            grid = np.zeros((4, 4))
            cx_all = (dets_norm[:, 0] + dets_norm[:, 2]) / 2.0
            cy_all = (dets_norm[:, 1] + dets_norm[:, 3]) / 2.0
            for i in range(n_objects):
                gi = min(int(cy_all[i] * 4), 3)
                gj = min(int(cx_all[i] * 4), 3)
                grid[gi, gj] += 1
            grid = grid / max(n_objects, 1)
            # Uniformity: 1 - normalized entropy deviation
            uniform = 1.0 / 16.0
            features[7] = 1.0 - np.std(grid.flatten() - uniform)
        else:
            features[7] = 0.0

        return features

    def generate_density_field(
        self, detections: np.ndarray, shape: Tuple[int, int], image_size: int = 550
    ) -> np.ndarray:
        """Generate spatial density map from detections.

        Produces a smooth density field indicating where objects are
        expected based on current detections and learned spatial priors.

        Args:
            detections: Array of shape (N, 4) in [x1, y1, x2, y2] format.
            shape: Output shape (H, W) for the density field.
            image_size: Image size for normalization.

        Returns:
            Density field of shape (H, W), normalized to [0, 1].
        """
        H, W = shape

        if len(detections) < self.min_detections or not self.is_fitted:
            # Return uniform density if insufficient data
            return np.ones((H, W), dtype=np.float32) * 0.5

        # Normalize detection centers
        dets_norm = detections / image_size
        cx = (dets_norm[:, 0] + dets_norm[:, 2]) / 2.0
        cy = (dets_norm[:, 1] + dets_norm[:, 3]) / 2.0
        centers = np.stack([cx, cy], axis=1)

        # Generate grid of evaluation points
        x_grid = np.linspace(0, 1, W)
        y_grid = np.linspace(0, 1, H)
        xx, yy = np.meshgrid(x_grid, y_grid)
        grid_points = np.stack([xx.ravel(), yy.ravel()], axis=1)

        # Evaluate KDE at grid points
        log_density = self.kde.score_samples(grid_points)
        density = np.exp(log_density).reshape(H, W)

        # Smooth slightly for stability
        density = gaussian_filter(density, sigma=1.0)

        # Normalize to [0, 1]
        d_min, d_max = density.min(), density.max()
        if d_max - d_min > 1e-8:
            density = (density - d_min) / (d_max - d_min)
        else:
            density = np.ones_like(density) * 0.5

        return density.astype(np.float32)

    def bayesian_update(self, scores: np.ndarray, spatial_features: np.ndarray) -> np.ndarray:
        """Bayesian confidence recalibration using spatial prior.

        Updates detection confidence using Bayes rule:
            P(object | score, spatial) ∝ P(score | object) * P(object | spatial)

        The spatial prior is derived from the density field and regularity.

        Args:
            scores: Original detection confidences, shape (N,).
            spatial_features: 8-dim spatial feature vector.

        Returns:
            Recalibrated scores, shape (N,).
        """
        if not self.is_fitted or len(scores) == 0:
            return scores

        # Compute spatial prior strength from features
        # Higher density regions → higher prior for objects
        coverage = spatial_features[6]
        regularity = spatial_features[5]
        density_uniformity = spatial_features[7]

        # Spatial prior: scenes with high regularity and coverage
        # are more likely to have real objects
        spatial_prior = 0.5 + 0.3 * regularity + 0.2 * coverage

        # Bayesian update: blend original score with spatial prior
        # This is a simplified conjugate update
        alpha = 0.7  # weight of original score
        beta = 0.3  # weight of spatial prior
        updated_scores = alpha * scores + beta * spatial_prior

        # Clip to valid range
        updated_scores = np.clip(updated_scores, 0.0, 1.0)

        return updated_scores

    def save(self, path: Optional[Path] = None) -> None:
        """Save fitted models to disk."""
        save_dir = path or self.save_dir
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        state = {
            "gmm": self.gmm,
            "kde": self.kde,
            "is_fitted": self.is_fitted,
            "mean_objects_per_image": self._mean_objects_per_image,
            "mean_box_area": self._mean_box_area,
            "std_box_area": self._std_box_area,
            "config": self.config,
        }

        save_path = save_dir / "spatial_engine.pkl"
        with open(save_path, "wb") as f:
            pickle.dump(state, f)
        logger.info("Spatial reasoning engine saved to %s", save_path)

    def load(self, path: Optional[Path] = None) -> None:
        """Load fitted models from disk."""
        save_dir = path or self.save_dir
        save_dir = Path(save_dir)
        load_path = save_dir / "spatial_engine.pkl"

        if not load_path.exists():
            raise FileNotFoundError(f"No spatial engine found at {load_path}")

        with open(load_path, "rb") as f:
            state = pickle.load(f)

        self.gmm = state["gmm"]
        self.kde = state["kde"]
        self.is_fitted = state["is_fitted"]
        self._mean_objects_per_image = state["mean_objects_per_image"]
        self._mean_box_area = state["mean_box_area"]
        self._std_box_area = state["std_box_area"]
        logger.info("Spatial reasoning engine loaded from %s", load_path)
