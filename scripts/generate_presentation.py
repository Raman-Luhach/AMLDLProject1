#!/usr/bin/env python3
"""Generate a professional PDF presentation for Phase 2."""

import os
import json
from reportlab.lib.pagesizes import landscape, A4
from reportlab.lib.units import inch, cm
from reportlab.lib.colors import HexColor, white, black
from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
from reportlab.lib.styles import ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle,
    PageBreak, Frame, PageTemplate, BaseDocTemplate
)
from reportlab.pdfgen import canvas as pdfcanvas
from reportlab.platypus.flowables import Flowable

PROJECT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(PROJECT, "presentation", "Phase2_Presentation.pdf")
os.makedirs(os.path.dirname(OUT), exist_ok=True)

W, H = landscape(A4)  # 841.89 x 595.27

# Colors
BG = HexColor("#FFFFFF")
PRIMARY = HexColor("#18181B")    # zinc-900
ACCENT = HexColor("#2563EB")    # blue-600
ACCENT2 = HexColor("#D97706")   # amber-600
SUBTLE = HexColor("#71717A")    # zinc-500
LIGHT = HexColor("#F4F4F5")     # zinc-100
BORDER = HexColor("#E4E4E7")    # zinc-200
DARK_BG = HexColor("#18181B")   # for dark slides
GREEN = HexColor("#16A34A")

# Styles
TITLE_STYLE = ParagraphStyle("Title", fontName="Helvetica-Bold", fontSize=28,
                              textColor=PRIMARY, leading=34, alignment=TA_LEFT)
SUBTITLE_STYLE = ParagraphStyle("Subtitle", fontName="Helvetica", fontSize=14,
                                 textColor=SUBTLE, leading=20, alignment=TA_LEFT)
HEADING_STYLE = ParagraphStyle("Heading", fontName="Helvetica-Bold", fontSize=22,
                                textColor=PRIMARY, leading=28, alignment=TA_LEFT)
SUBHEADING_STYLE = ParagraphStyle("SubHeading", fontName="Helvetica-Bold", fontSize=14,
                                   textColor=ACCENT, leading=18, alignment=TA_LEFT)
BODY_STYLE = ParagraphStyle("Body", fontName="Helvetica", fontSize=11,
                             textColor=PRIMARY, leading=16, alignment=TA_LEFT)
BODY_SMALL = ParagraphStyle("BodySmall", fontName="Helvetica", fontSize=9.5,
                             textColor=SUBTLE, leading=13, alignment=TA_LEFT)
BULLET_STYLE = ParagraphStyle("Bullet", fontName="Helvetica", fontSize=11,
                               textColor=PRIMARY, leading=16, leftIndent=20,
                               bulletIndent=8, alignment=TA_LEFT)
CODE_STYLE = ParagraphStyle("Code", fontName="Courier", fontSize=9,
                             textColor=PRIMARY, leading=13, alignment=TA_LEFT,
                             backColor=LIGHT)
CENTER_STYLE = ParagraphStyle("Center", fontName="Helvetica", fontSize=11,
                               textColor=PRIMARY, leading=16, alignment=TA_CENTER)
CAPTION_STYLE = ParagraphStyle("Caption", fontName="Helvetica-Oblique", fontSize=9,
                                textColor=SUBTLE, leading=12, alignment=TA_CENTER)
TABLE_HEADER = ParagraphStyle("TH", fontName="Helvetica-Bold", fontSize=10,
                               textColor=white, leading=14, alignment=TA_CENTER)
TABLE_CELL = ParagraphStyle("TD", fontName="Helvetica", fontSize=10,
                             textColor=PRIMARY, leading=14, alignment=TA_CENTER)
TABLE_CELL_L = ParagraphStyle("TDL", fontName="Helvetica", fontSize=10,
                               textColor=PRIMARY, leading=14, alignment=TA_LEFT)
WHITE_TITLE = ParagraphStyle("WhiteTitle", fontName="Helvetica-Bold", fontSize=32,
                              textColor=white, leading=40, alignment=TA_CENTER)
WHITE_SUB = ParagraphStyle("WhiteSub", fontName="Helvetica", fontSize=14,
                            textColor=HexColor("#A1A1AA"), leading=20, alignment=TA_CENTER)


class SlideBackground(Flowable):
    """Draw a slide background with optional dark mode."""
    def __init__(self, width, height, dark=False, accent_bar=True):
        Flowable.__init__(self)
        self.width = width
        self.height = height
        self.dark = dark
        self.accent_bar = accent_bar

    def draw(self):
        c = self.canv
        # Background
        bg = DARK_BG if self.dark else BG
        c.setFillColor(bg)
        c.rect(0, 0, self.width, self.height, fill=1, stroke=0)
        if self.accent_bar and not self.dark:
            # Top accent line
            c.setStrokeColor(ACCENT)
            c.setLineWidth(3)
            c.line(0, self.height, self.width, self.height)
            # Bottom subtle line
            c.setStrokeColor(BORDER)
            c.setLineWidth(0.5)
            c.line(40, 30, self.width - 40, 30)


def make_table(headers, rows, col_widths=None):
    """Create a styled table."""
    header_row = [Paragraph(h, TABLE_HEADER) for h in headers]
    data = [header_row]
    for row in rows:
        data.append([Paragraph(str(cell), TABLE_CELL if i > 0 else TABLE_CELL_L)
                     for i, cell in enumerate(row)])

    if col_widths is None:
        col_widths = [W * 0.85 / len(headers)] * len(headers)

    t = Table(data, colWidths=col_widths)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), ACCENT),
        ('TEXTCOLOR', (0, 0), (-1, 0), white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, 0), (-1, 0), 8),
        ('TOPPADDING', (0, 0), (-1, 0), 8),
        ('BACKGROUND', (0, 1), (-1, -1), white),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [white, LIGHT]),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('TOPPADDING', (0, 1), (-1, -1), 6),
        ('BOTTOMPADDING', (0, 1), (-1, -1), 6),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))
    return t


def bullet(text, style=BULLET_STYLE):
    return Paragraph(f"&bull; {text}", style)


def img_path(relpath):
    p = os.path.join(PROJECT, relpath)
    return p if os.path.exists(p) else None


def safe_image(path, width=None, height=None):
    if path and os.path.exists(path):
        return Image(path, width=width, height=height)
    return Spacer(1, 0)


def slide_number_canvas(canvas, doc):
    """Add slide numbers to each page."""
    canvas.saveState()
    canvas.setFont("Helvetica", 8)
    canvas.setFillColor(SUBTLE)
    canvas.drawRightString(W - 40, 15, f"{doc.page}")
    canvas.restoreState()


def build():
    doc = BaseDocTemplate(OUT, pagesize=landscape(A4),
                          leftMargin=50, rightMargin=50,
                          topMargin=40, bottomMargin=45)

    frame = Frame(50, 45, W - 100, H - 85, id='main')
    template = PageTemplate(id='slide', frames=frame, onPage=slide_number_canvas)
    doc.addPageTemplates([template])

    story = []
    content_width = W - 100

    # ========================================================
    # SLIDE 1: Title
    # ========================================================
    story.append(Spacer(1, 80))
    story.append(Paragraph("High-Density Object Segmentation", TITLE_STYLE))
    story.append(Paragraph("for Retail Environments", TITLE_STYLE))
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        '<font color="#2563EB">YOLACT</font> + MobileNetV3-Large + CBAM Attention + Soft-NMS',
        ParagraphStyle("s", fontName="Helvetica", fontSize=15, textColor=SUBTLE, leading=20)
    ))
    story.append(Spacer(1, 30))
    story.append(Paragraph("Raman Luhach (230107)&nbsp;&nbsp;&nbsp;|&nbsp;&nbsp;&nbsp;Rachit Kumar (230128)",
                           ParagraphStyle("s", fontName="Helvetica-Bold", fontSize=12, textColor=PRIMARY, leading=16)))
    story.append(Spacer(1, 8))
    story.append(Paragraph("Applied Machine Learning / Deep Learning &mdash; Phase 2",
                           ParagraphStyle("s", fontName="Helvetica", fontSize=11, textColor=SUBTLE)))
    story.append(Spacer(1, 8))
    story.append(Paragraph("SKU-110K Dataset &nbsp;&bull;&nbsp; 11,743 Images &nbsp;&bull;&nbsp; 1.73M Annotations &nbsp;&bull;&nbsp; ~147 Objects/Image",
                           ParagraphStyle("s", fontName="Helvetica", fontSize=10, textColor=ACCENT)))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 2: Problem Motivation
    # ========================================================
    story.append(Paragraph("Problem Motivation", HEADING_STYLE))
    story.append(Spacer(1, 5))
    story.append(Paragraph('<font color="#2563EB">Why does dense retail object detection matter?</font>', SUBHEADING_STYLE))
    story.append(Spacer(1, 10))
    story.append(bullet("Retail shelf images contain an average of <b>147 objects per frame</b> &mdash; one of the densest detection benchmarks in computer vision"))
    story.append(Spacer(1, 4))
    story.append(bullet("Products are tightly packed, heavily occluded, and visually similar"))
    story.append(Spacer(1, 4))
    story.append(bullet("Standard NMS <b>aggressively kills correct overlapping detections</b> &mdash; destroying recall in dense scenes"))
    story.append(Spacer(1, 12))
    story.append(Paragraph('<font color="#2563EB">The Three Gaps We Address</font>', SUBHEADING_STYLE))
    story.append(Spacer(1, 8))

    gaps = [
        ["1", "Instance segmentation (not just detection) in ultra-dense retail scenes"],
        ["2", "Lightweight backbone for edge/mobile deployment (< 10M parameters vs 44.5M)"],
        ["3", "Density-aware post-processing (Soft-NMS) that preserves overlapping detections"],
    ]
    gt = Table(
        [[Paragraph(f'<font color="#FFFFFF"><b>{r[0]}</b></font>', TABLE_CELL),
          Paragraph(r[1], TABLE_CELL_L)] for r in gaps],
        colWidths=[40, content_width - 60]
    )
    gt.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (0, -1), ACCENT),
        ('BACKGROUND', (1, 0), (1, -1), LIGHT),
        ('GRID', (0, 0), (-1, -1), 0.5, BORDER),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(gt)
    story.append(PageBreak())

    # ========================================================
    # SLIDE 3: Literature Review
    # ========================================================
    story.append(Paragraph("Literature Review &amp; Positioning", HEADING_STYLE))
    story.append(Spacer(1, 10))

    lit_headers = ["Paper", "Contribution", "Limitation"]
    lit_rows = [
        ["Goldman et al. (2019)\nSKU-110K", "First large-scale dense\ndetection benchmark", "Detection only,\nno segmentation"],
        ["Bolya et al. (2019)\nYOLACT", "Real-time instance\nsegmentation via prototypes", "ResNet-101 (44.5M params)\ntoo heavy for edge"],
        ["Howard et al. (2019)\nMobileNetV3", "Efficient backbone with\nNAS-optimized blocks", "Not evaluated for\ninstance segmentation"],
        ["Bodla et al. (2017)\nSoft-NMS", "Gaussian score decay\npreserves overlapping boxes", "Not combined with\ninstance segmentation"],
        ["Woo et al. (2018)\nCBAM", "Channel + Spatial attention\nfor feature refinement", "Not applied to\ndense detection FPN"],
    ]
    story.append(make_table(lit_headers, lit_rows, col_widths=[140, 200, 180]))
    story.append(Spacer(1, 15))
    story.append(Paragraph(
        '<b>Our Contribution:</b> Combining 5 innovations &mdash; YOLACT + MobileNetV3 (88% fewer backbone params) + CBAM on FPN + Soft-NMS + ONNX INT8 deployment',
        ParagraphStyle("s", fontName="Helvetica", fontSize=11, textColor=ACCENT, leading=16, alignment=TA_CENTER)
    ))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 4: Dataset & EDA
    # ========================================================
    story.append(Paragraph("Dataset &mdash; SKU-110K Deep Dive", HEADING_STYLE))
    story.append(Spacer(1, 8))

    ds_headers = ["Split", "Images", "Annotations", "Avg Objects/Image"]
    ds_rows = [
        ["Train", "8,219", "1,208,482", "147.0"],
        ["Val", "588", "90,968", "154.7"],
        ["Test", "2,936", "431,546", "147.0"],
        ["Total", "11,743", "1,730,996", "147.4"],
    ]
    story.append(make_table(ds_headers, ds_rows, col_widths=[100, 120, 150, 150]))
    story.append(Spacer(1, 10))

    story.append(Paragraph('<font color="#2563EB">Key EDA Findings</font>', SUBHEADING_STYLE))
    story.append(Spacer(1, 6))
    story.append(bullet("Object size: Mean normalized area = <b>0.285%</b> of image &mdash; objects are tiny"))
    story.append(bullet("Aspect ratio: Median = <b>0.534</b> &mdash; most products are taller than wide"))
    story.append(bullet("Pairwise IoU: Heavy overlap &rarr; justifies Soft-NMS over Hard-NMS"))
    story.append(bullet("K-Means anchors: K=9 achieves <b>mean IoU = 0.7156</b> with ground truth"))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 5: EDA Visualizations
    # ========================================================
    story.append(Paragraph("Exploratory Data Analysis &mdash; Visualizations", HEADING_STYLE))
    story.append(Spacer(1, 8))

    # Show 4 EDA images in a 2x2 grid
    eda_imgs = [
        ("results/eda/objects_per_image_histogram.png", "Objects Per Image Distribution"),
        ("results/eda/box_area_distribution.png", "Box Area Distribution"),
        ("results/eda/pairwise_iou_histogram.png", "Pairwise IoU (Justifies Soft-NMS)"),
        ("results/eda/anchor_kmeans_analysis.png", "K-Means Anchor Analysis"),
    ]
    img_w = (content_width - 30) / 2
    img_h = (H - 180) / 2

    grid_data = []
    row = []
    for i, (p, cap) in enumerate(eda_imgs):
        fp = img_path(p)
        cell = []
        if fp:
            cell.append(safe_image(fp, width=img_w, height=img_h - 15))
        cell.append(Paragraph(cap, CAPTION_STYLE))
        row.append(cell)
        if (i + 1) % 2 == 0:
            grid_data.append(row)
            row = []

    gt = Table(grid_data, colWidths=[img_w + 10, img_w + 10])
    gt.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(gt)
    story.append(PageBreak())

    # ========================================================
    # SLIDE 6: Architecture
    # ========================================================
    story.append(Paragraph("Model Architecture", HEADING_STYLE))
    story.append(Spacer(1, 6))

    arch_text = """<font face="Courier" size="9" color="#18181B">
    Input (3x550x550) &rarr; <font color="#2563EB"><b>MobileNetV3-Large</b></font> &rarr; C3(40ch) / C4(112ch) / C5(960ch)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&darr;<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<font color="#2563EB"><b>FPN + CBAM</b></font> &rarr; P3 / P4 / P5 / P6 / P7 (all 256ch)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&darr;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&darr;<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<font color="#16A34A"><b>ProtoNet</b></font> (32 masks)&nbsp;&nbsp;&nbsp;<font color="#D97706"><b>PredictionHead</b></font> (cls/box/coeff)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&darr;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&darr;<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;<font color="#DC2626"><b>Soft-NMS</b></font> (Gaussian, &sigma;=0.5)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&darr;<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Final Detections (boxes, scores, masks)</font>"""
    story.append(Paragraph(arch_text, ParagraphStyle("arch", fontSize=9, leading=14, alignment=TA_CENTER,
                                                       backColor=LIGHT, borderPadding=12)))
    story.append(Spacer(1, 12))

    param_headers = ["Component", "Parameters", "Purpose"]
    param_rows = [
        ["MobileNetV3-Large", "~3.0M", "Feature extraction (ImageNet pretrained)"],
        ["FPN + CBAM", "~3.3M", "Multi-scale features + attention"],
        ["ProtoNet", "~2.4M", "32 shared prototype masks (from P3)"],
        ["Prediction Head", "~1.4M", "Class / Box / Mask coefficient prediction"],
        ["Total", "9.98M", "78% fewer params than ResNet-101 YOLACT"],
    ]
    story.append(make_table(param_headers, param_rows, col_widths=[150, 100, 280]))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 7: Feature Engineering
    # ========================================================
    story.append(Paragraph("Feature Engineering &amp; Key Innovations", HEADING_STYLE))
    story.append(Spacer(1, 10))

    story.append(Paragraph('<font color="#2563EB">1. CBAM Attention on FPN</font>', SUBHEADING_STYLE))
    story.append(bullet("Channel attention learns <b>which</b> feature channels matter for product detection"))
    story.append(bullet("Spatial attention learns <b>where</b> to focus in cluttered shelf scenes"))
    story.append(bullet("Applied to P3, P4, P5 &mdash; the scales where most products appear"))
    story.append(Spacer(1, 8))

    story.append(Paragraph('<font color="#2563EB">2. K-Means Optimized Anchors</font>', SUBHEADING_STYLE))
    story.append(bullet("K-Means on all SKU-110K boxes &rarr; K=9 anchors, <b>mean IoU = 0.7156</b>"))
    story.append(bullet("Significantly better than default COCO anchors (~0.5 IoU)"))
    story.append(Spacer(1, 8))

    story.append(Paragraph('<font color="#2563EB">3. Prototype Mask Representation</font>', SUBHEADING_STYLE))
    story.append(bullet("32 shared prototypes from P3 &rarr; linear combination per detection = per-instance mask"))
    story.append(bullet("Far more efficient than Mask R-CNN's per-RoI mask head"))
    story.append(Spacer(1, 8))

    story.append(Paragraph('<font color="#2563EB">4. Soft-NMS Post-Processing</font>', SUBHEADING_STYLE))
    story.append(bullet('Hard NMS: score = 0 if IoU > threshold (binary kill)'))
    story.append(bullet('Soft-NMS: <b>score *= exp(-IoU&sup2;/&sigma;)</b> (gradual Gaussian decay, &sigma;=0.5)'))
    story.append(bullet("Preserves correct overlapping detections in dense shelf scenes"))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 8: Theoretical Rigor
    # ========================================================
    story.append(Paragraph("Training Methodology &amp; Theoretical Rigor", HEADING_STYLE))
    story.append(Spacer(1, 10))

    story.append(Paragraph('<font color="#2563EB">Loss Function</font>', SUBHEADING_STYLE))
    story.append(Spacer(1, 4))
    story.append(Paragraph(
        '<font face="Courier" size="11"><b>L = 1.0 &times; L<sub>cls</sub>(Focal) + 1.5 &times; L<sub>box</sub>(SmoothL1) + 6.125 &times; L<sub>mask</sub>(BCE)</b></font>',
        ParagraphStyle("s", fontSize=11, alignment=TA_CENTER, leading=18, backColor=LIGHT, borderPadding=8)
    ))
    story.append(Spacer(1, 10))

    story.append(Paragraph('<font color="#2563EB">Why Focal Loss?</font> (&alpha;=0.25, &gamma;=2.0)', SUBHEADING_STYLE))
    story.append(bullet("~147 objects vs thousands of anchors &rarr; <b>1:930 negative:positive ratio</b>"))
    story.append(bullet("Focal Loss down-weights easy negatives: <b>FL(p) = -&alpha;(1-p)<sup>&gamma;</sup> log(p)</b>"))
    story.append(Spacer(1, 6))

    story.append(Paragraph('<font color="#2563EB">Regularization Strategy</font>', SUBHEADING_STYLE))
    story.append(bullet("<b>MixUp</b> (&alpha;=0.2): Convex combinations &rarr; smoother decision boundaries"))
    story.append(bullet("<b>Label Smoothing</b> (&epsilon;=0.1): Prevents overconfident predictions"))
    story.append(bullet("<b>Gradient Clipping</b> (norm=10): Prevents exploding gradients"))
    story.append(Spacer(1, 6))

    story.append(Paragraph('<font color="#2563EB">Optimization</font>', SUBHEADING_STYLE))
    story.append(bullet("SGD + Momentum(0.9) + Weight Decay(5e-4)"))
    story.append(bullet("Cosine Annealing LR: 0.001 &rarr; 1e-6 after 3-epoch linear warmup"))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 9: Training Results
    # ========================================================
    story.append(Paragraph("Training Results &mdash; Loss Convergence", HEADING_STYLE))
    story.append(Spacer(1, 6))
    story.append(Paragraph("3,000 images &bull; 8 epochs &bull; batch size 4 &bull; Apple Silicon MPS", BODY_SMALL))
    story.append(Spacer(1, 8))

    train_headers = ["Epoch", "Train Loss", "Val Loss", "Cls Loss", "Box Loss", "Mask Loss", "LR"]
    train_rows = [
        ["1", "8.594", "—", "0.279", "3.476", "0.506", "0.000333"],
        ["2", "8.049", "6.536", "0.207", "3.297", "0.473", "0.000667"],
        ["4", "5.548", "4.300", "0.123", "2.228", "0.340", "0.001"],
        ["6", "4.701", "3.804", "0.104", "1.781", "0.314", "0.000655"],
        ["8", "4.355", "3.603", "0.097", "1.601", "0.303", "0.0000964"],
    ]
    story.append(make_table(train_headers, train_rows,
                            col_widths=[55, 90, 90, 90, 90, 90, 90]))
    story.append(Spacer(1, 12))

    story.append(Paragraph('<font color="#2563EB">Key Observations</font>', SUBHEADING_STYLE))
    story.append(bullet("<b>1.97&times; loss reduction</b> in just 8 epochs (8.594 &rarr; 4.355)"))
    story.append(bullet("Val loss still decreasing &rarr; model has <b>NOT converged</b> (room for improvement)"))
    story.append(bullet("Classification loss dropped <b>65%</b>, Box loss dropped <b>54%</b>"))
    story.append(bullet("No overfitting: validation loss tracks training loss closely"))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 10: Detection Results + Visualizations
    # ========================================================
    story.append(Paragraph("Detection Results &amp; Visualizations", HEADING_STYLE))
    story.append(Spacer(1, 6))

    # Results table on left, image on right
    eval_headers = ["Metric", "YOLACT", "HOG+SVM"]
    eval_rows = [
        ["mAP@0.50", "0.08%", "3.09%"],
        ["AP@[.50:.95]", "0.01%", "—"],
        ["AR@100", "0.41%", "—"],
        ["Precision", "2.71%", "86.36%"],
        ["Recall", "1.82%", "2.09%"],
        ["Total Detections", "10,000", "22"],
        ["Parameters", "9.98M", "1,764 features"],
    ]
    story.append(make_table(eval_headers, eval_rows, col_widths=[150, 130, 130]))
    story.append(Spacer(1, 8))
    story.append(Paragraph(
        "YOLACT is undertrained (8 epochs, 36% data) but detects far more objects. "
        "HOG+SVM is extremely conservative (22 detections across 50 images) &mdash; high precision but <b>misses 98%</b> of objects.",
        BODY_SMALL
    ))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 11: Detection Visualizations
    # ========================================================
    story.append(Paragraph("Detection Samples &amp; Grad-CAM Attention", HEADING_STYLE))
    story.append(Spacer(1, 8))

    det_imgs = [
        ("results/eval/detection_samples.png", "YOLACT Detection Samples"),
        ("results/eval/gradcam_grid.png", "Grad-CAM Attention Maps"),
    ]
    img_w2 = (content_width - 20) / 2
    img_h2 = H - 160

    row_data = []
    for p, cap in det_imgs:
        fp = img_path(p)
        cell = []
        if fp:
            cell.append(safe_image(fp, width=img_w2, height=img_h2 - 15))
        cell.append(Paragraph(cap, CAPTION_STYLE))
        row_data.append(cell)

    gt2 = Table([row_data], colWidths=[img_w2 + 5, img_w2 + 5])
    gt2.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(gt2)
    story.append(PageBreak())

    # ========================================================
    # SLIDE 12: Failure Analysis
    # ========================================================
    story.append(Paragraph("Failure Analysis &mdash; Where &amp; Why the Model Fails", HEADING_STYLE))
    story.append(Spacer(1, 8))

    story.append(Paragraph('<font color="#2563EB">Error Breakdown by Scene Density</font>', SUBHEADING_STYLE))
    story.append(Spacer(1, 6))

    err_headers = ["Density", "Images", "True Pos", "False Pos", "Precision", "Recall"]
    err_rows = [
        ["30-100 objects", "5", "19", "481", "3.80%", "4.70%"],
        ["100-200 objects", "85", "229", "8,271", "2.69%", "1.87%"],
        ["200+ objects", "10", "23", "977", "2.30%", "1.02%"],
    ]
    story.append(make_table(err_headers, err_rows,
                            col_widths=[110, 70, 80, 80, 90, 90]))
    story.append(Spacer(1, 10))

    story.append(Paragraph('<font color="#DC2626"><b>Key Finding: Recall degrades 4.6&times; from sparse to ultra-dense scenes</b></font>',
                           ParagraphStyle("s", fontSize=12, alignment=TA_CENTER, leading=16)))
    story.append(Spacer(1, 10))

    story.append(Paragraph('<font color="#2563EB">Mathematical Explanation</font>', SUBHEADING_STYLE))
    story.append(bullet("9 anchors &times; 5 FPN levels &times; spatial grid &asymp; <b>137,000 anchors</b> but only ~147 positives"))
    story.append(bullet("Class imbalance ratio: <b>1:930</b> &mdash; even Focal Loss needs more epochs to overcome this"))
    story.append(bullet("Soft-NMS preserves overlapping detections but cannot fix weak classification"))
    story.append(bullet("In ultra-dense scenes (200+), anchor-GT IoU matching becomes highly ambiguous"))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 13: Robustness & Ablation
    # ========================================================
    story.append(Paragraph("Robustness Analysis &amp; Ablation Study", HEADING_STYLE))
    story.append(Spacer(1, 6))

    # Robustness image
    rob_img = img_path("results/eval/robustness_analysis.png")
    pr_img = img_path("results/eval/precision_recall.png")

    rob_row = []
    if rob_img:
        rob_row.append([safe_image(rob_img, width=img_w2, height=(H - 310)), Paragraph("Robustness Under Noise/Blur/Brightness", CAPTION_STYLE)])
    if pr_img:
        rob_row.append([safe_image(pr_img, width=img_w2, height=(H - 310)), Paragraph("Precision-Recall Curve", CAPTION_STYLE)])

    if rob_row:
        gt3 = Table([rob_row], colWidths=[img_w2 + 5, img_w2 + 5])
        gt3.setStyle(TableStyle([('VALIGN', (0, 0), (-1, -1), 'TOP'), ('TOPPADDING', (0, 0), (-1, -1), 4)]))
        story.append(gt3)

    story.append(Spacer(1, 8))

    abl_headers = ["NMS Method", "AP@0.50", "AP@[.50:.95]", "AR@100"]
    abl_rows = [
        ["Soft-NMS (\u03c3=0.5)", "0.071%", "0.010%", "0.41%"],
        ["Hard-NMS (IoU=0.5)", "0.075%", "0.010%", "0.42%"],
    ]
    story.append(make_table(abl_headers, abl_rows, col_widths=[170, 140, 140, 140]))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 14: Deployment
    # ========================================================
    story.append(Paragraph("Deployment &mdash; Edge-Ready Model", HEADING_STYLE))
    story.append(Spacer(1, 10))

    story.append(Paragraph('<font color="#2563EB">ONNX Export + INT8 Quantization</font>', SUBHEADING_STYLE))
    story.append(Spacer(1, 8))

    dep_headers = ["Backend", "Model Size", "Latency", "FPS", "Speedup"]
    dep_rows = [
        ["PyTorch FP32 (MPS)", "38.1 MB", "318.3 ms", "3.1", "1\u00d7"],
        ["ONNX FP32 (CPU)", "0.6 MB", "120.3 ms", "8.3", "2.6\u00d7"],
        ["ONNX INT8 (CPU)", "9.9 MB", "115.2 ms", "8.7", "2.8\u00d7"],
    ]
    story.append(make_table(dep_headers, dep_rows, col_widths=[150, 110, 110, 80, 80]))
    story.append(Spacer(1, 15))

    story.append(Paragraph('<font color="#2563EB">Key Deployment Insights</font>', SUBHEADING_STYLE))
    story.append(Spacer(1, 6))
    story.append(bullet("ONNX INT8: <b>2.8&times; faster</b> than PyTorch, deployable on ARM/x86 edge devices"))
    story.append(bullet("Model fits in <b>< 10 MB</b> after quantization &mdash; suitable for embedded systems"))
    story.append(bullet("Total parameters: <b>9.98M</b> (vs 44.5M ResNet-101 YOLACT)"))
    story.append(bullet("Automated pipeline: <font face='Courier'>make export</font> runs ONNX export + INT8 + benchmarking"))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 15: Live Demo
    # ========================================================
    story.append(Paragraph("Live Demo &mdash; Web Application", HEADING_STYLE))
    story.append(Spacer(1, 10))

    demo_arch = """<font face="Courier" size="9">
    Browser (Next.js + React) &rarr; Upload Image<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&darr;<br/>
    Next.js API Route (/api/inference or /api/inference-baseline)<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&darr; Spawn Python subprocess<br/>
    inference_api.py &rarr; Load Model &rarr; Preprocess &rarr; Inference &rarr; JSON<br/>
    &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&darr;<br/>
    Browser renders detections on HTML Canvas<br/>
    </font>"""
    story.append(Paragraph(demo_arch, ParagraphStyle("s", fontSize=9, leading=14, alignment=TA_CENTER,
                                                      backColor=LIGHT, borderPadding=10)))
    story.append(Spacer(1, 12))

    story.append(Paragraph('<font color="#2563EB">Demo Features</font>', SUBHEADING_STYLE))
    story.append(Spacer(1, 6))
    story.append(bullet("<b>Model Toggle:</b> Switch between YOLACT (Deep Learning) and HOG+SVM (Classical ML)"))
    story.append(bullet("<b>Confidence Slider:</b> Filter detections 5% &ndash; 95% live, client-side, no re-inference"))
    story.append(bullet("<b>Sample Images:</b> Pre-loaded synthetic shelf images for instant testing"))
    story.append(bullet("<b>Fullscreen Viewer:</b> Scroll to zoom, drag to pan, download results as PNG"))
    story.append(bullet("<b>Color-coded boxes:</b> Green (high confidence) &rarr; Yellow &rarr; Red (low confidence)"))
    story.append(bullet("<b>Full logging:</b> Python &rarr; Node.js log pipeline for real-time debugging"))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 16: Repository & Code Quality
    # ========================================================
    story.append(Paragraph("Repository &amp; Code Quality", HEADING_STYLE))
    story.append(Spacer(1, 10))

    repo_headers = ["Component", "Details"]
    repo_rows = [
        ["Source Code", "7 model files, 3 data, 2 training, 2 eval, 3 deployment, 1 baseline"],
        ["Scripts", "8 entry-point scripts covering full pipeline"],
        ["Notebooks", "4 Jupyter notebooks (EDA, Baseline, Training, Deployment)"],
        ["Web App", "Next.js 16 with React, Tailwind, Framer Motion"],
        ["Report", "IEEE-format LaTeX paper with 18 references"],
        ["Automation", "Makefile with 10 targets: make all runs end-to-end"],
        ["Commits", "15+ meaningful commits showing steady progression"],
    ]
    story.append(make_table(repo_headers, repo_rows, col_widths=[130, 480]))
    story.append(Spacer(1, 12))

    story.append(Paragraph('<font color="#2563EB">Full Reproducibility</font>', SUBHEADING_STYLE))
    story.append(Spacer(1, 6))
    repro = """<font face="Courier" size="10">
    git clone &lt;repo&gt; &amp;&amp; cd AMLDLProject1<br/>
    make install&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# Install dependencies + editable package<br/>
    make download-data&nbsp;&nbsp;# Download SKU-110K dataset<br/>
    make all&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;# EDA &rarr; Baseline &rarr; Train &rarr; Eval &rarr; Export &rarr; Report
    </font>"""
    story.append(Paragraph(repro, ParagraphStyle("s", fontSize=10, leading=15, alignment=TA_LEFT,
                                                   backColor=LIGHT, borderPadding=10)))
    story.append(PageBreak())

    # ========================================================
    # SLIDE 17: Conclusion
    # ========================================================
    story.append(Spacer(1, 40))
    story.append(Paragraph("Conclusion", HEADING_STYLE))
    story.append(Spacer(1, 15))

    story.append(Paragraph('<font color="#2563EB">What We Built</font>', SUBHEADING_STYLE))
    story.append(Spacer(1, 6))
    story.append(bullet("End-to-end instance segmentation for dense retail scenes (SKU-110K)"))
    story.append(bullet("Custom YOLACT with MobileNetV3 + CBAM + Soft-NMS (<b>9.98M params</b>)"))
    story.append(bullet("Classical ML baseline (HOG+SVM) for direct comparison"))
    story.append(bullet("ONNX INT8 deployment pipeline (<b>&lt;10 MB</b>, ~9 FPS on CPU)"))
    story.append(bullet("Interactive web demo with model switching and live confidence filtering"))
    story.append(Spacer(1, 15))

    story.append(Paragraph('<font color="#2563EB">Key Takeaways</font>', SUBHEADING_STYLE))
    story.append(Spacer(1, 6))
    story.append(bullet("Dense detection is fundamentally harder &mdash; 147 objects/image creates <b>1:930 class imbalance</b>"))
    story.append(bullet("MobileNetV3 achieves <b>78% parameter reduction</b> vs ResNet-101"))
    story.append(bullet("Soft-NMS theoretically essential for overlapping objects in retail scenes"))
    story.append(bullet("Model undertrained but architecture &amp; pipeline are production-ready"))
    story.append(Spacer(1, 30))
    story.append(Paragraph("<b>Thank You &mdash; Questions?</b>",
                           ParagraphStyle("s", fontName="Helvetica-Bold", fontSize=20,
                                          textColor=ACCENT, alignment=TA_CENTER)))

    # Build PDF
    doc.build(story)
    print(f"\nPresentation saved to: {OUT}")
    print(f"Total slides: 17")


if __name__ == "__main__":
    build()
