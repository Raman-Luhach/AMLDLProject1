.PHONY: install download-data eda baseline train evaluate export benchmark report clean all
.PHONY: train-hybrid fit-spatial ablation diagram test docker docker-push

install:
	pip install -r requirements.txt
	pip install -e .

download-data:
	bash scripts/download_data.sh

eda:
	python scripts/run_eda.py

baseline:
	python scripts/run_baseline.py

train:
	python scripts/train.py --config configs/default.yaml

evaluate:
	python scripts/evaluate.py --config configs/default.yaml

export:
	python scripts/export.py --config configs/default.yaml

benchmark:
	python src/deployment/benchmark.py

report:
	cd report && pdflatex main.tex && bibtex main && pdflatex main.tex && pdflatex main.tex

# Phase 3: Hybrid Neuro-Symbolic Detector
fit-spatial:
	python scripts/train_hybrid.py --config configs/hybrid.yaml --stage fit-spatial

train-hybrid:
	python scripts/train_hybrid.py --config configs/hybrid.yaml --stage train-hybrid

ablation:
	python scripts/run_ablation.py --config configs/hybrid.yaml

diagram:
	python scripts/generate_architecture_diagram.py

test:
	pytest tests/ -v --tb=short

# Docker
docker:
	docker build -t sku110k-hybrid .

docker-push:
	docker push sku110k-hybrid:latest

clean:
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.pyc" -delete 2>/dev/null || true
	rm -rf build/ dist/ *.egg-info/

all: install download-data eda baseline train evaluate export report

# Full Phase 3 pipeline
phase3: fit-spatial train-hybrid ablation diagram
