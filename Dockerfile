FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

# System dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python dependencies (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Install package in editable mode
RUN pip install -e .

# Create results directories
RUN mkdir -p results/hybrid/checkpoints results/hybrid/spatial_models \
    results/ablation results/figures weights

# Default command: show help
CMD ["python", "-c", "print('SKU-110K Hybrid Detector\\n\\nCommands:\\n  python scripts/train_hybrid.py --config configs/hybrid.yaml\\n  python scripts/run_ablation.py\\n  python scripts/generate_architecture_diagram.py\\n  pytest tests/ -v')"]
