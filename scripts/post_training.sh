#!/bin/bash
# Post-training evaluation pipeline - run immediately after training completes
set -e

cd /Users/aloksinghtomar/Desktop/AmlDlProject
source .venv/bin/activate

echo "=========================================="
echo "  POST-TRAINING EVALUATION PIPELINE"
echo "=========================================="

# 1. COCO-style evaluation on val set (use 200 images for speed)
echo ""
echo "--- Step 1: COCO Evaluation ---"
python scripts/evaluate.py --max-images 200 --batch-size 4

# 2. Advanced evaluation (Grad-CAM + Ablation + Robustness)
echo ""
echo "--- Step 2: Advanced Evaluation ---"
python scripts/advanced_evaluation.py --max-images 150 --batch-size 4

# 3. Re-export ONNX model
echo ""
echo "--- Step 3: ONNX Export ---"
python scripts/export.py || echo "ONNX export skipped (non-critical)"

echo ""
echo "=========================================="
echo "  ALL EVALUATIONS COMPLETE"
echo "=========================================="
