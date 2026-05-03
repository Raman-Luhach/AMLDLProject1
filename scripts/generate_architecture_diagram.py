#!/usr/bin/env python3
"""Generate publication-quality architecture diagram for the Hybrid Detector.

Produces a professional diagram showing the full neuro-symbolic pipeline
with color-coded components, tensor shapes, and data flow arrows.

Output: results/figures/architecture_diagram.png (and .pdf)

Color scheme:
    Blue (#4ECDC4):   DL components (YOLACT backbone, FPN, ProtoNet, Pred Head)
    Green (#95E864):  ML components (GMM, KDE, Regularity, Spatial Features)
    Purple (#9B59B6): Fusion components (Spatial Attention, Recalibrator)
    Orange (#F39C12): Input/Output
"""

import os
import sys

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

# Color scheme
DL_COLOR = "#4ECDC4"  # Teal for DL
DL_LIGHT = "#A8E6CF"
ML_COLOR = "#95E864"  # Green for ML
ML_LIGHT = "#C5F0A4"
FUSION_COLOR = "#9B59B6"  # Purple for Fusion
FUSION_LIGHT = "#D2B4DE"
IO_COLOR = "#F39C12"  # Orange for I/O
IO_LIGHT = "#FAD7A0"
BG_COLOR = "#FAFAFA"
ARROW_COLOR = "#2C3E50"
FEEDBACK_COLOR = "#E74C3C"


def draw_box(ax, x, y, w, h, text, color, fontsize=8, text_color="black", alpha=0.9):
    """Draw a rounded rectangle with centered text."""
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.05",
        facecolor=color,
        edgecolor="#2C3E50",
        linewidth=1.2,
        alpha=alpha,
        zorder=2,
    )
    ax.add_patch(box)
    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color=text_color,
        fontweight="bold",
        zorder=3,
        wrap=True,
    )
    return box


def draw_arrow(ax, x1, y1, x2, y2, color=ARROW_COLOR, style="->", lw=1.5, dashed=False):
    """Draw an arrow between two points."""
    ls = "--" if dashed else "-"
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        color=color,
        linewidth=lw,
        linestyle=ls,
        mutation_scale=15,
        zorder=1,
        connectionstyle="arc3,rad=0.0",
    )
    ax.add_patch(arrow)


def draw_tensor_label(ax, x, y, text, fontsize=6):
    """Draw tensor shape annotation."""
    ax.text(
        x,
        y,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#7F8C8D",
        fontstyle="italic",
        zorder=4,
        bbox=dict(boxstyle="round,pad=0.15", facecolor="white", edgecolor="#BDC3C7", linewidth=0.5, alpha=0.9),
    )


def generate_phase3_diagram():
    """Generate the Phase 3 Hybrid Architecture diagram."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(-0.5, 10.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    # Title
    ax.text(
        8,
        10.2,
        "Phase 3: Hybrid Neuro-Symbolic Architecture",
        ha="center",
        va="center",
        fontsize=16,
        fontweight="bold",
        color="#2C3E50",
    )
    ax.text(
        8,
        9.8,
        "YOLACT + Spatial Reasoning Engine + Differentiable Feedback",
        ha="center",
        va="center",
        fontsize=10,
        color="#7F8C8D",
    )

    # ====== Input ======
    draw_box(ax, 0.2, 7.5, 2.0, 1.2, "Input Image\n550x550x3", IO_COLOR, fontsize=9)

    # ====== DL Components (Blue) ======
    # Backbone
    draw_box(ax, 3.0, 7.5, 2.5, 1.2, "MobileNetV3-Large\nBackbone\n(3.0M params)", DL_COLOR, fontsize=8)
    draw_arrow(ax, 2.2, 8.1, 3.0, 8.1)
    draw_tensor_label(ax, 2.6, 8.5, "(B,3,550,550)")

    # FPN
    draw_box(ax, 6.2, 7.5, 2.5, 1.2, "FPN + CBAM\nAttention\n(3.3M params)", DL_COLOR, fontsize=8)
    draw_arrow(ax, 5.5, 8.1, 6.2, 8.1)
    draw_tensor_label(ax, 5.85, 8.6, "C3,C4,C5")

    # FPN outputs
    draw_tensor_label(ax, 7.45, 7.2, "P3: (B,256,69,69)\nP4-P7...")

    # ProtoNet branch
    draw_box(ax, 6.2, 5.5, 2.5, 1.0, "ProtoNet\n(2.4M params)", DL_COLOR, fontsize=8)
    draw_arrow(ax, 7.45, 7.5, 7.45, 6.5, style="->")
    draw_tensor_label(ax, 8.9, 6.0, "(B,32,138,138)")

    # Prediction Head
    draw_box(ax, 9.5, 7.5, 2.5, 1.2, "Prediction\nHead\n(1.4M params)", DL_COLOR, fontsize=8)
    draw_arrow(ax, 8.7, 8.1, 9.5, 8.1)

    # Raw detections output
    draw_tensor_label(ax, 10.75, 7.2, "cls + box + mask_coeff")

    # ====== ML Components (Green) ======
    # Spatial Reasoning Engine
    draw_box(ax, 0.2, 3.5, 3.5, 2.5, "", ML_COLOR, fontsize=8, alpha=0.3)
    ax.text(2.0, 5.7, "Spatial Reasoning Engine (ML)", ha="center", fontsize=9, fontweight="bold", color="#27AE60")

    # Sub-components
    draw_box(ax, 0.4, 5.0, 1.5, 0.7, "GMM Row\nDetector", ML_LIGHT, fontsize=7)
    draw_box(ax, 2.0, 5.0, 1.5, 0.7, "KDE Density\nEstimator", ML_LIGHT, fontsize=7)
    draw_box(ax, 0.4, 4.0, 1.5, 0.7, "Regularity\nScorer", ML_LIGHT, fontsize=7)
    draw_box(ax, 2.0, 4.0, 1.5, 0.7, "Spatial\nFeatures (8d)", ML_LIGHT, fontsize=7)

    # Arrow from raw detections to ML
    draw_arrow(ax, 10.75, 7.5, 10.75, 6.8, style="->")
    ax.text(11.0, 7.15, "raw\ndets", fontsize=7, color="#7F8C8D")
    draw_arrow(ax, 10.75, 6.8, 2.0, 6.1, style="->", lw=1.2)

    # ====== Fusion Components (Purple) ======
    # Spatial Prior Attention (FEEDBACK)
    draw_box(
        ax,
        5.0,
        3.5,
        3.0,
        1.2,
        "Spatial Prior Attention\n(learnable gate)\nFEEDBACK LOOP",
        FUSION_COLOR,
        fontsize=8,
        text_color="white",
    )

    # Arrow from KDE to Spatial Attention
    draw_arrow(ax, 3.5, 5.3, 5.0, 4.4, style="->")
    draw_tensor_label(ax, 4.25, 5.0, "density\nfield (H,W)")

    # FEEDBACK arrow back to FPN
    draw_arrow(ax, 6.5, 4.7, 7.0, 7.5, color=FEEDBACK_COLOR, style="->", lw=2.0, dashed=True)
    ax.text(5.8, 6.2, "FEEDBACK\n(modulates P3)", fontsize=7, color=FEEDBACK_COLOR, fontweight="bold", rotation=70)

    # Confidence Recalibrator
    draw_box(
        ax,
        9.0,
        3.5,
        3.0,
        1.5,
        "Confidence\nRecalibrator\nMLP: [73] → 128 → 64 → 1",
        FUSION_COLOR,
        fontsize=8,
        text_color="white",
    )

    # Arrows to Recalibrator
    draw_arrow(ax, 3.5, 4.3, 9.0, 4.3, style="->")  # spatial features
    draw_tensor_label(ax, 6.25, 4.6, "spatial feats (8d)")

    draw_arrow(ax, 10.75, 7.5, 10.5, 5.0, style="->")  # scores + visual
    draw_tensor_label(ax, 11.3, 6.2, "score (1d)\nvisual (64d)")

    # ====== Output ======
    draw_box(ax, 12.5, 3.5, 2.5, 1.5, "Final Refined\nDetections\n(boxes, scores,\nmasks)", IO_COLOR, fontsize=8)
    draw_arrow(ax, 12.0, 4.25, 12.5, 4.25)

    # ====== Legend ======
    legend_y = 1.5
    legend_items = [
        (DL_COLOR, "DL (Neural Network)"),
        (ML_COLOR, "ML (Probabilistic/Statistical)"),
        (FUSION_COLOR, "Fusion (Differentiable)"),
        (IO_COLOR, "Input/Output"),
    ]

    ax.text(0.5, legend_y + 0.8, "Legend:", fontsize=10, fontweight="bold", color="#2C3E50")
    for i, (color, label) in enumerate(legend_items):
        x = 0.5 + i * 3.5
        patch = FancyBboxPatch(
            (x, legend_y), 0.6, 0.4, boxstyle="round,pad=0.05", facecolor=color, edgecolor="#2C3E50", linewidth=0.8
        )
        ax.add_patch(patch)
        ax.text(x + 0.8, legend_y + 0.2, label, fontsize=8, va="center", color="#2C3E50")

    # Feedback arrow legend
    draw_arrow(ax, 0.5, legend_y - 0.3, 1.3, legend_y - 0.3, color=FEEDBACK_COLOR, style="->", lw=2.0, dashed=True)
    ax.text(1.5, legend_y - 0.3, "Feedback Loop (ML → DL)", fontsize=8, va="center", color=FEEDBACK_COLOR)

    # Total params
    ax.text(
        8,
        0.5,
        "Total: ~10.0M (YOLACT) + ~25K (Fusion) = ~10.03M params",
        ha="center",
        fontsize=10,
        color="#2C3E50",
        fontstyle="italic",
    )

    # Save
    output_dir = os.path.join(project_root, "results", "figures")
    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(os.path.join(output_dir, "architecture_phase3.png"), dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.savefig(os.path.join(output_dir, "architecture_phase3.pdf"), bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Phase 3 diagram saved to {output_dir}/architecture_phase3.png")


def generate_phase1_diagram():
    """Generate Phase 1 HOG+SVM diagram."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.set_xlim(-0.5, 14)
    ax.set_ylim(0, 5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    ax.text(
        7,
        4.7,
        "Phase 1: Classical ML Baseline (HOG + SVM)",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#2C3E50",
    )

    # Pipeline
    draw_box(ax, 0, 2, 2.0, 1.2, "Input\nImage", IO_COLOR, fontsize=9)
    draw_box(ax, 2.5, 2, 2.2, 1.2, "Sliding\nWindow\n(64x64)", ML_COLOR, fontsize=8)
    draw_box(ax, 5.2, 2, 2.2, 1.2, "HOG Feature\nExtraction\n(1764-dim)", ML_COLOR, fontsize=8)
    draw_box(ax, 7.9, 2, 2.0, 1.2, "Linear\nSVM\nClassifier", ML_COLOR, fontsize=8)
    draw_box(ax, 10.4, 2, 2.0, 1.2, "Hard NMS\nPost-\nprocessing", ML_COLOR, fontsize=8)
    draw_box(ax, 12.8, 2, 1.2, 1.2, "Sparse\nDets", IO_COLOR, fontsize=8)

    draw_arrow(ax, 2.0, 2.6, 2.5, 2.6)
    draw_arrow(ax, 4.7, 2.6, 5.2, 2.6)
    draw_arrow(ax, 7.4, 2.6, 7.9, 2.6)
    draw_arrow(ax, 9.9, 2.6, 10.4, 2.6)
    draw_arrow(ax, 12.4, 2.6, 12.8, 2.6)

    # Annotations
    draw_tensor_label(ax, 3.6, 1.6, "Multi-scale\npyramid")
    draw_tensor_label(ax, 6.3, 1.6, "9 orient bins\n8x8 cells")
    draw_tensor_label(ax, 8.9, 1.6, "C=0.01")
    draw_tensor_label(ax, 11.4, 1.6, "IoU=0.3")

    ax.text(
        7,
        0.5,
        "Result: mAP@0.5 = 3.09% | Precision = 86.4% | Recall = 2.1%",
        ha="center",
        fontsize=10,
        color="#E74C3C",
        fontstyle="italic",
    )

    output_dir = os.path.join(project_root, "results", "figures")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "architecture_phase1.png"), dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.savefig(os.path.join(output_dir, "architecture_phase1.pdf"), bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Phase 1 diagram saved to {output_dir}/architecture_phase1.png")


def generate_phase2_diagram():
    """Generate Phase 2 YOLACT diagram."""
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(-0.5, 15.5)
    ax.set_ylim(0, 8.5)
    ax.set_aspect("equal")
    ax.axis("off")
    fig.patch.set_facecolor(BG_COLOR)

    ax.text(
        8,
        8.2,
        "Phase 2: YOLACT + MobileNetV3 + FPN + CBAM + Soft-NMS",
        ha="center",
        fontsize=14,
        fontweight="bold",
        color="#2C3E50",
    )

    # Backbone
    draw_box(ax, 0.2, 5.5, 2.0, 1.2, "Input\n(B,3,550,550)", IO_COLOR, fontsize=8)
    draw_box(ax, 2.8, 5.5, 2.5, 1.2, "MobileNetV3\nLarge\n(3.0M)", DL_COLOR, fontsize=8)
    draw_arrow(ax, 2.2, 6.1, 2.8, 6.1)

    # Multi-scale features
    draw_box(ax, 6.0, 6.5, 1.5, 0.7, "C3\n(40ch)", DL_LIGHT, fontsize=7)
    draw_box(ax, 6.0, 5.5, 1.5, 0.7, "C4\n(112ch)", DL_LIGHT, fontsize=7)
    draw_box(ax, 6.0, 4.5, 1.5, 0.7, "C5\n(960ch)", DL_LIGHT, fontsize=7)
    draw_arrow(ax, 5.3, 6.1, 6.0, 6.1)

    # FPN
    draw_box(ax, 8.2, 5.0, 2.5, 2.5, "FPN\n+\nCBAM\nAttention\n(3.3M)", DL_COLOR, fontsize=8)
    draw_arrow(ax, 7.5, 6.1, 8.2, 6.1)

    # FPN outputs
    draw_tensor_label(ax, 11.5, 7.5, "P3:(B,256,69,69)")
    draw_tensor_label(ax, 11.5, 7.0, "P4:(B,256,35,35)")
    draw_tensor_label(ax, 11.5, 6.5, "P5:(B,256,18,18)")
    draw_tensor_label(ax, 11.5, 6.0, "P6:(B,256,9,9)")
    draw_tensor_label(ax, 11.5, 5.5, "P7:(B,256,5,5)")
    draw_arrow(ax, 10.7, 6.25, 10.8, 6.25)

    # ProtoNet
    draw_box(ax, 8.2, 3.0, 2.5, 1.0, "ProtoNet (2.4M)\n32 prototypes", DL_COLOR, fontsize=8)
    draw_arrow(ax, 9.45, 5.0, 9.45, 4.0)
    draw_tensor_label(ax, 11.0, 3.5, "(B,32,138,138)")

    # Prediction Head
    draw_box(ax, 12.5, 5.0, 2.5, 2.5, "Prediction\nHead\n(1.4M)\n\ncls + box\n+ mask_coeff", DL_COLOR, fontsize=8)
    draw_arrow(ax, 10.7, 6.25, 12.5, 6.25)

    # Soft-NMS
    draw_box(ax, 12.5, 3.0, 2.5, 1.0, "Soft-NMS\n(Gaussian, σ=0.5)", DL_COLOR, fontsize=8)
    draw_arrow(ax, 13.75, 5.0, 13.75, 4.0)

    # Output
    draw_box(ax, 12.5, 1.5, 2.5, 1.0, "Detections\n(boxes, masks)", IO_COLOR, fontsize=8)
    draw_arrow(ax, 13.75, 3.0, 13.75, 2.5)

    # Mask assembly note
    draw_box(ax, 8.2, 1.5, 3.5, 1.0, "Mask Assembly\nmask = coeff @ proto^T", DL_LIGHT, fontsize=8)
    draw_arrow(ax, 11.7, 2.0, 12.5, 2.0)

    ax.text(
        8,
        0.5,
        "Total: ~10.0M params | ONNX INT8: 9.9MB | 8.7 FPS (CPU)",
        ha="center",
        fontsize=10,
        color="#2C3E50",
        fontstyle="italic",
    )

    output_dir = os.path.join(project_root, "results", "figures")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, "architecture_phase2.png"), dpi=300, bbox_inches="tight", facecolor=BG_COLOR)
    plt.savefig(os.path.join(output_dir, "architecture_phase2.pdf"), bbox_inches="tight", facecolor=BG_COLOR)
    plt.close()
    print(f"Phase 2 diagram saved to {output_dir}/architecture_phase2.png")


if __name__ == "__main__":
    generate_phase1_diagram()
    generate_phase2_diagram()
    generate_phase3_diagram()
    print("\nAll architecture diagrams generated successfully!")
