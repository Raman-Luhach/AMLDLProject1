#!/bin/bash
set -e

echo "=============================================="
echo "  SKU-110K Hybrid Neuro-Symbolic Detector"
echo "  One-Command Setup & Training Pipeline"
echo "=============================================="

# Check Python version
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")
echo "[1/10] Python version: $PYTHON_VERSION"

if python3 -c "import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)" 2>/dev/null; then
    echo "  OK (>= 3.10)"
else
    echo "  WARNING: Python 3.10+ recommended"
fi

# Create virtual environment
echo "[2/10] Creating virtual environment..."
if [ ! -d ".venv" ]; then
    python3 -m venv .venv
    echo "  Created .venv"
else
    echo "  .venv already exists"
fi
source .venv/bin/activate

# Install dependencies
echo "[3/10] Installing dependencies..."
pip install --quiet -r requirements.txt
pip install --quiet -e .
echo "  Dependencies installed"

# Check for dataset
echo "[4/10] Checking dataset..."
if [ -d "data/SKU110K_fixed" ]; then
    echo "  Dataset found"
else
    echo "  Dataset not found. Please download SKU-110K:"
    echo "    bash scripts/download_data.sh"
    echo "  Or manually place it in data/SKU110K_fixed/"
    echo "  Continuing with remaining setup..."
fi

# Run EDA
echo "[5/10] Running EDA..."
if [ -d "data/SKU110K_fixed" ]; then
    python scripts/run_eda.py 2>/dev/null || echo "  EDA skipped (data not ready)"
else
    echo "  Skipped (no dataset)"
fi

# Train baseline
echo "[6/10] Training HOG+SVM baseline..."
if [ -d "data/SKU110K_fixed" ]; then
    python scripts/run_baseline.py 2>/dev/null || echo "  Baseline skipped"
else
    echo "  Skipped (no dataset)"
fi

# Train YOLACT (Stage 1)
echo "[7/10] Training YOLACT (Stage 1)..."
if [ -d "data/SKU110K_fixed" ]; then
    if [ -f "results/training/checkpoints/best_model.pth" ]; then
        echo "  Resuming from existing checkpoint..."
        python scripts/train.py --config configs/default.yaml \
            --resume results/training/checkpoints/best_model.pth \
            --epochs 30
    else
        python scripts/train.py --config configs/default.yaml --epochs 30
    fi
else
    echo "  Skipped (no dataset)"
fi

# Fit Spatial Reasoning (Stage 2)
echo "[8/10] Fitting Spatial Reasoning Engine (Stage 2)..."
if [ -d "data/SKU110K_fixed" ]; then
    python scripts/train_hybrid.py --config configs/hybrid.yaml --stage fit-spatial
else
    echo "  Skipped (no dataset)"
fi

# Train Hybrid (Stage 3)
echo "[9/10] Training Hybrid Model (Stage 3)..."
if [ -d "data/SKU110K_fixed" ]; then
    python scripts/train_hybrid.py --config configs/hybrid.yaml --stage train-hybrid
else
    echo "  Skipped (no dataset)"
fi

# Run Ablation & Generate Diagrams
echo "[10/10] Running ablation studies & generating diagrams..."
if [ -f "results/hybrid/checkpoints/hybrid_best.pth" ]; then
    python scripts/run_ablation.py --config configs/hybrid.yaml
fi
python scripts/generate_architecture_diagram.py

echo ""
echo "=============================================="
echo "  Setup Complete!"
echo "=============================================="
echo ""
echo "Results saved in:"
echo "  results/training/    - YOLACT training logs"
echo "  results/hybrid/      - Hybrid model checkpoints"
echo "  results/ablation/    - Ablation study results"
echo "  results/figures/     - Architecture diagrams"
echo ""
echo "To start the web demo:"
echo "  cd web && npm run dev"
