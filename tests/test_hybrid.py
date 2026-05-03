"""Unit tests for Hybrid Neuro-Symbolic Detector."""

import pytest
import torch

from src.models.hybrid import HybridDetector
from src.models.spatial_attention import SpatialPriorAttention
from src.models.confidence_recalibrator import ConfidenceRecalibrator


@pytest.fixture
def device():
    return torch.device("cpu")


@pytest.fixture
def hybrid_config():
    return {
        "spatial_reasoning": {
            "num_row_components": 4,
            "kde_bandwidth": 0.1,
            "min_detections": 2,
            "save_dir": "/tmp/test_spatial",
        },
        "recalibrator": {
            "spatial_dim": 8,
            "visual_dim": 64,
            "hidden_dims": [64, 32],
            "dropout": 0.1,
        },
        "spatial_attention": {
            "gate_init": 0.1,
            "conv_channels": 8,
        },
    }


@pytest.fixture
def yolact_config():
    return {
        "input_size": 550,
        "num_classes": 2,
        "pretrained_backbone": False,
        "fpn_out_channels": 256,
        "num_prototypes": 32,
        "conf_threshold": 0.05,
        "nms_sigma": 0.5,
        "max_detections": 100,
    }


class TestSpatialPriorAttention:
    def test_forward_shape(self, device):
        module = SpatialPriorAttention(conv_channels=8, gate_init=0.1).to(device)
        fpn_feat = torch.randn(2, 256, 69, 69, device=device)
        density = torch.rand(2, 1, 69, 69, device=device)
        output = module(fpn_feat, density)
        assert output.shape == fpn_feat.shape

    def test_gate_value(self, device):
        module = SpatialPriorAttention(gate_init=0.1).to(device)
        gate_val = module.get_gate_value()
        assert 0 < gate_val < 1
        assert abs(gate_val - torch.sigmoid(torch.tensor(0.1)).item()) < 1e-5

    def test_identity_when_gate_zero(self, device):
        module = SpatialPriorAttention(gate_init=-100.0).to(device)
        fpn_feat = torch.randn(1, 256, 10, 10, device=device)
        density = torch.rand(1, 1, 10, 10, device=device)
        output = module(fpn_feat, density)
        # Gate ≈ 0, so output ≈ fpn_feat
        assert torch.allclose(output, fpn_feat, atol=1e-4)

    def test_density_resize(self, device):
        module = SpatialPriorAttention(conv_channels=8).to(device)
        fpn_feat = torch.randn(1, 256, 69, 69, device=device)
        # Density at different resolution
        density = torch.rand(1, 1, 32, 32, device=device)
        output = module(fpn_feat, density)
        assert output.shape == fpn_feat.shape

    def test_backward(self, device):
        module = SpatialPriorAttention(conv_channels=8).to(device)
        fpn_feat = torch.randn(1, 256, 10, 10, device=device, requires_grad=True)
        density = torch.rand(1, 1, 10, 10, device=device)
        output = module(fpn_feat, density)
        loss = output.sum()
        loss.backward()
        assert fpn_feat.grad is not None
        assert module.gate.grad is not None


class TestConfidenceRecalibrator:
    def test_forward_shape(self, device):
        model = ConfidenceRecalibrator(spatial_dim=8, visual_dim=64).to(device)
        scores = torch.rand(10, 1, device=device)
        spatial = torch.randn(10, 8, device=device)
        visual = torch.randn(10, 64, device=device)
        output = model(scores, spatial, visual)
        assert output.shape == (10, 1)

    def test_output_range(self, device):
        model = ConfidenceRecalibrator().to(device)
        scores = torch.rand(5, 1, device=device)
        spatial = torch.randn(5, 8, device=device)
        visual = torch.randn(5, 64, device=device)
        output = model(scores, spatial, visual)
        assert (output >= 0).all() and (output <= 1).all()

    def test_backward(self, device):
        model = ConfidenceRecalibrator().to(device)
        scores = torch.rand(5, 1, device=device, requires_grad=True)
        spatial = torch.randn(5, 8, device=device)
        visual = torch.randn(5, 64, device=device)
        output = model(scores, spatial, visual)
        loss = output.sum()
        loss.backward()
        assert scores.grad is not None

    def test_empty_input(self, device):
        model = ConfidenceRecalibrator().to(device)
        scores = torch.rand(0, 1, device=device)
        spatial = torch.randn(0, 8, device=device)
        visual = torch.randn(0, 64, device=device)
        output = model(scores, spatial, visual)
        assert output.shape == (0, 1)


class TestHybridDetector:
    def test_instantiation(self, yolact_config, hybrid_config, device):
        model = HybridDetector(
            yolact_config=yolact_config,
            hybrid_config=hybrid_config,
        ).to(device)
        assert model is not None

    def test_parameter_count(self, yolact_config, hybrid_config, device):
        model = HybridDetector(
            yolact_config=yolact_config,
            hybrid_config=hybrid_config,
        ).to(device)
        params = model.count_parameters()
        assert params["total"] > 0
        assert params["trainable"] > 0
        assert "yolact" in params
        assert "spatial_attention" in params
        assert "recalibrator" in params

    def test_freeze_unfreeze(self, yolact_config, hybrid_config, device):
        model = HybridDetector(
            yolact_config=yolact_config,
            hybrid_config=hybrid_config,
        ).to(device)

        # Freeze YOLACT
        model.freeze_yolact()
        for param in model.yolact.parameters():
            assert not param.requires_grad

        # Fusion components should still be trainable
        assert model.spatial_attention.gate.requires_grad
        for param in model.recalibrator.parameters():
            assert param.requires_grad

        # Unfreeze YOLACT
        model.unfreeze_yolact()
        for param in model.yolact.parameters():
            assert param.requires_grad

    def test_training_forward(self, yolact_config, hybrid_config, device):
        model = HybridDetector(
            yolact_config=yolact_config,
            hybrid_config=hybrid_config,
        ).to(device)
        model.train()

        images = torch.randn(1, 3, 550, 550, device=device)
        targets = [
            {
                "boxes": torch.tensor([[100, 100, 200, 200], [300, 300, 400, 400]], dtype=torch.float32, device=device),
                "labels": torch.tensor([1, 1], dtype=torch.long, device=device),
                "masks": torch.zeros(2, 550, 550, dtype=torch.uint8, device=device),
            }
        ]

        output = model(images, targets)
        assert "class_preds" in output
        assert "box_preds" in output
        assert "mask_coeffs" in output
        assert "prototypes" in output
        assert "anchors" in output
        assert "spatial_features" in output
        assert "density_maps" in output
        assert "gate_value" in output

    def test_inference_forward(self, yolact_config, hybrid_config, device):
        model = HybridDetector(
            yolact_config=yolact_config,
            hybrid_config=hybrid_config,
        ).to(device)
        model.eval()

        images = torch.randn(1, 3, 550, 550, device=device)

        with torch.no_grad():
            output = model(images)

        assert isinstance(output, list)
        assert len(output) == 1
        assert "boxes" in output[0]
        assert "scores" in output[0]
