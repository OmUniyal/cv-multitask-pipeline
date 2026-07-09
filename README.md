---
title: CV Multitask Pipeline
emoji: 🔍
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: "4.25.0"
app_file: app.py
pinned: false
---

# CV Multitask Pipeline

> Simultaneous object classification and detection using a shared EfficientNet-B0 backbone with custom task heads — deployed as a production-grade FastAPI + Gradio application.

[![Live Demo](https://img.shields.io/badge/🤗-Live%20Demo-yellow)](https://huggingface.co/spaces/OmUniyal/cv-multitask-pipeline)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.12-orange)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110-green)](https://fastapi.tiangolo.com/)
[![Docker](https://img.shields.io/badge/Docker-ready-blue)](https://www.docker.com/)

---

## What this project does

Most CV tutorials pick one task — classify OR detect. This pipeline does both simultaneously from a single forward pass:

- **Classification** — identifies the primary object class (20 VOC categories)
- **Detection** — predicts the bounding box location of that object

One image in → class label + bounding box out.

---

## Architecture

```
Image [3, 224, 224]
↓
Frozen EfficientNet-B0 Backbone
(4M params, ImageNet pretrained)
↓
Feature Vector [1280]
↓              ↓
Classification    Detection
Head              Head
(667K params)     (329K params)
↓              ↓
Class Scores      BBox Coords
[20]              [4]
```

**Why frozen backbone?**
EfficientNet-B0 learned general visual features (edges, textures, shapes) from 1.2M ImageNet images. Freezing preserves these features while training only the task heads — faster training, less data needed, better generalisation.

**Why multi-task?**
Shared backbone features benefit both tasks simultaneously. Features useful for "this is a dog" also help locate where the dog is. Tasks reinforce each other.

---

## Results

| Metric | Value |
|--------|-------|
| Top-1 Accuracy | 79.9% |
| Mean IoU | 0.468 |
| IoU@0.5 Accuracy | 49.2% |
| Training epochs | 30 (best at epoch 5) |
| Training dataset | PASCAL VOC 2012 (5,717 train / 5,823 val) |
| Inference time (CPU) | ~200ms |
| Inference time (GPU) | ~20ms |

---

## Project structure

```
cv-multitask-pipeline/
├── src/
│   ├── data/          # VOC dataset, transforms, dataloader
│   ├── models/        # backbone, task heads, multi-task model, losses
│   ├── training/      # trainer, evaluator, config
│   └── api/           # FastAPI server, schemas, model manager
├── scripts/           # CLI training entrypoint
├── notebooks/         # EDA, model exploration, Colab training
├── models/            # checkpoints + registry.json
├── Dockerfile
└── docker-compose.yml
```

---

## Key engineering decisions

**1. Multi-task loss weighting**
Classification and detection losses have different natural scales (CrossEntropy ~2-3, SmoothL1 ~0.05-0.1). Without lambda weighting, classification dominates and the detection head gets starved of gradient signal. We use `λ_cls=1.0, λ_det=5.0` to balance contribution.

**2. Model versioning via registry**
`models/registry.json` maps version names to checkpoint files. The API loads versions dynamically — swap checkpoints without restarting the server. Supports `/v1/predict` and `/v2/predict` simultaneously.

**3. Bbox-aware transforms**
Every geometric augmentation (flip, crop, resize) updates both image AND bounding box coordinates. Standard torchvision transforms only handle images — incorrect boxes would poison detection training.

**4. Production API design**
- Async batch queue — accumulates requests for 50ms, processes as single batch
- Pydantic schemas — validates all inputs before reaching the model
- Health + readiness endpoints — liveness vs model-loaded distinction
- Structured logging — every request logged with inference time

---

## Setup and usage

### Local development

```bash
# clone and setup
git clone https://github.com/OmUniyal/cv-multitask-pipeline.git
cd cv-multitask-pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\Activate.ps1
pip install -r requirements.txt
pip install -e .

# download VOC 2012 dataset
# Linux/Mac:
wget http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar
tar -xf VOCtrainval_11-May-2012.tar -C data/

# train (small local test)
python scripts/train.py --epochs 5 --max-train-samples 200 --max-val-samples 50

# start API server
uvicorn src.api.main:app --reload --port 8000

# start Gradio UI (separate terminal)
python src/ui/gradio_app.py
```

### Docker

```bash
docker-compose up --build
# API: http://localhost:8000
# UI:  http://localhost:7860
# Docs: http://localhost:8000/docs
```

### Full training (Colab)

Open `notebooks/colab_train.py` in Google Colab with T4 GPU runtime and run cells sequentially. Downloads VOC 2012, trains for 30 epochs, downloads best checkpoint.

---

## API reference

### POST /v1/predict

```json
// Request
{
  "image_base64": "<base64 encoded image>",
  "model_version": "v1"
}

// Response
{
  "class_name": "horse",
  "class_idx": 12,
  "confidence": 0.847,
  "bbox": {
    "x_min": 0.12,
    "y_min": 0.08,
    "x_max": 0.89,
    "y_max": 0.95
  },
  "model_version": "v1",
  "inference_time_ms": 41.2
}
```

Full API docs at `/docs` (Swagger UI auto-generated by FastAPI).

---

## Tech stack

| Component | Technology |
|-----------|------------|
| Model | PyTorch + EfficientNet-B0 |
| Dataset | PASCAL VOC 2012 |
| API | FastAPI + Uvicorn |
| UI | Gradio |
| Containerization | Docker + Docker Compose |
| Deployment | HuggingFace Spaces |
| Training | Google Colab (T4 GPU) |

---

## About

Built by [Om Uniyal](https://github.com/OmUniyal) as part of a production ML portfolio.

---

## Other projects

| Project | Description | Demo |
|---------|-------------|------|
| [RAG Document Q&A](https://github.com/OmUniyal/rag-document-qa) | Self-hosted RAG pipeline with Ollama + Groq fallback, built without LangChain | [Live Demo](https://huggingface.co/spaces/omUniyal/rag-document-qa) |

