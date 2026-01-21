# Development & Deployment Guide

## Architecture

```
Input Text
    ↓
Tokenizer (DistilBERT)
    ↓
DistilBERT Encoder
    ↓
Pooled Output [CLS]
    ↓
Linear Layer (768 → 2)
    ↓
Softmax
    ↓
[Not Clickbait, Clickbait]
```

## Deployment Options

We provide three deployment variants optimized for different use cases:

| Variant | Docker Image | Memory | Best For                          |
| ------- | ------------ | ------ | --------------------------------- |
| PyTorch | 4.27 GB      | 4 Gi   | Full flexibility                  |
| ONNX    | 568 MB       | 2 Gi   | Resource-constrained environments |
| BentoML | 1.93 GB      | 2 Gi   | High-throughput batch processing  |

### Benchmark Results (1000 requests)

| Metric       | PyTorch   | ONNX   | BentoML |
| ------------ | --------- | ------ | ------- |
| Mean latency | 284 ms    | 349 ms | 111 ms  |
| P95 latency  | 264 ms    | 450 ms | 180 ms  |
| Max latency  | 61,835 ms | 710 ms | -       |
| Requests/sec | 3.52      | 2.87   | 26.91   |

PyTorch is faster on average but ONNX is more consistent (no cold-start spikes). BentoML excels at batch processing with adaptive batching.

---

## Prerequisites

- Docker Desktop installed and running
- Python + uv (for `invoke` tasks)
- Google Cloud SDK (for cloud deployment)

## Project Setup

```bash
uv sync
```

## Pre-commit Hooks

Pre-commit runs automatic checks (formatting, linting, config validation) before each commit.

```bash
# Run all hooks
uv run pre-commit run --all-files

# Skip hooks (not recommended)
git commit -m "your commit message" --no-verify
```

## Run Tests

```bash
uv run invoke test
```

---

# Local Development

## Preprocess Data

```bash
uv run invoke preprocess-data
```

## Train Model

```bash
uv run invoke train
```

## Evaluate Model

```bash
uv run invoke evaluate
```

## Run API Locally

```bash
uv run invoke dev-api
```

API will be available at http://127.0.0.1:8000

**Endpoints:**

| Endpoint          | Method | Description             |
| ----------------- | ------ | ----------------------- |
| `/`               | GET    | Health check            |
| `/docs`           | GET    | Interactive API docs    |
| `/classify`       | POST   | Classify single text    |
| `/classify/batch` | POST   | Classify multiple texts |

**Example requests:**

```bash
# Health check
curl http://127.0.0.1:8000/

# Classify single text
curl -X POST http://127.0.0.1:8000/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "Scientists publish research findings"}'

# Batch classify
curl -X POST http://127.0.0.1:8000/classify/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["Breaking news headline", "You Will NEVER Believe This"]}'
```

## Run Frontend Locally

```bash
uv run streamlit run src/clickbait_classifier/frontend.py
```

The app will be available at http://localhost:8501

---

# Docker

## Docker Compose

```bash
# Start all services
docker compose up --build

# Start development container
docker compose up -d dev

# Run API
docker compose up api

# Stop all containers
docker compose down
```

## Training in Docker

```bash
# Default configuration
uv run invoke docker-train

# Custom configuration
uv run invoke docker-train --args="--config configs/my_experiment.yaml"

# With CLI overrides
uv run invoke docker-train --args="--config configs/config.yaml --epochs 5 --batch-size 64 --lr 1e-4"
```

## Evaluation in Docker

```bash
uv run invoke docker-evaluate
```

## Build API Image

```bash
uv run invoke build-api
```

---

# Model Registry (W&B)

This project uses W&B Model Registry with GitHub Actions for automated model testing and promotion.

## Workflow

1. **Train a model:**

   ```bash
   uv run invoke train --epochs 3
   ```

2. **Upload to W&B:**

   ```bash
   uv run invoke upload-model
   ```

3. **Review metrics** in the [W&B dashboard](https://wandb.ai/group-19/clickbait-classifier)

4. **Stage the model** (triggers CI pipeline):

   ```bash
   uv run invoke stage-model
   ```

5. **Automated testing:** GitHub Actions downloads the staged model, runs performance tests, and auto-promotes to `production` if tests pass.

## Model Commands

| Command                                                   | Description                           |
| --------------------------------------------------------- | ------------------------------------- |
| `uv run invoke upload-model`                              | Upload latest model checkpoint to W&B |
| `uv run invoke stage-model`                               | Add staging alias to trigger CI tests |
| `uv run invoke stage-model --artifact=clickbait-model:v2` | Stage a specific version              |

---

# Cloud Deployment (GCP)

## Prerequisites

```bash
gcloud auth login
gcloud config set project dtumlops-484212
```

## Train on Vertex AI (GPU)

The Docker image is automatically built and pushed to Artifact Registry when you push to `main`.

```bash
gcloud ai custom-jobs create \
  --region=europe-west1 \
  --display-name=clickbait-train \
  --config=configs/config_gpu.yaml \
  --args=--processed-path=/gcs/dtumlops-clickbait-data/data/processed \
  --args=--output=/gcs/dtumlops-clickbait-data/models \
  --args=--epochs=3
```

Monitor the job:

```bash
gcloud ai custom-jobs stream-logs <JOB_NAME>
```

---

## Deploy API to Cloud Run

### PyTorch API

```bash
# Build and push
docker build --platform linux/amd64 \
  -f dockerfiles/api_gcp.dockerfile \
  -t europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/clickbait-api-gcp:latest .

docker push europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/clickbait-api-gcp:latest

# Deploy
gcloud run deploy clickbait-api-gcp \
  --image europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/clickbait-api-gcp:latest \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 4Gi
```

**Live URL:** https://clickbait-api-gcp-136485552734.europe-west1.run.app/docs

### ONNX API (Lightweight)

| Metric          | PyTorch | ONNX   |
| --------------- | ------- | ------ |
| Docker image    | 4.27 GB | 568 MB |
| Model file      | 759 MB  | 266 MB |
| Memory required | 4 Gi    | 2 Gi   |

```bash
# Export model to ONNX
PYTHONPATH=src uv run python scripts/export_onnx.py

# Build and push
docker build -f dockerfiles/api_onnx_gcp.dockerfile \
  --platform linux/amd64 \
  -t europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/api-onnx:latest .

docker push europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/api-onnx:latest

# Deploy
gcloud run deploy clickbait-api-onnx \
  --image=europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/api-onnx:latest \
  --region=europe-west1 \
  --memory=2Gi \
  --cpu=1
```

**Live URL:** https://clickbait-api-onnx-136485552734.europe-west1.run.app

### BentoML API (Adaptive Batching)

```bash
# Build and containerize
uv run bentoml build
uv run bentoml containerize clickbait_classifier:latest --opt platform=linux/amd64

# Tag and push
docker tag clickbait_classifier:latest \
  europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/bentoml-service:latest
docker push europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/bentoml-service:latest

# Deploy
gcloud run deploy clickbait-bentoml \
  --image=europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/bentoml-service:latest \
  --region=europe-west1 \
  --memory=2Gi \
  --port=3000
```

**Live URL:** https://clickbait-bentoml-136485552734.europe-west1.run.app

---

## Deploy Frontend to Cloud Run

```bash
# Build using Cloud Build
gcloud builds submit --config cloudbuild_frontend.yaml --timeout=600

# Deploy
gcloud run deploy clickbait-frontend \
  --image europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/frontend:light \
  --region europe-west1 \
  --allow-unauthenticated \
  --memory 512Mi
```

**Live URL:** https://clickbait-frontend-136485552734.europe-west1.run.app

---

# Load Testing

## Start API first

```bash
uv run invoke dev-api
```

## Run Locust with Web UI

```bash
uv run locust -f tests/performancetests/locustfile.py
```

Open http://localhost:8089, set host to `http://localhost:8000`.

## Run Locust Headless (CI/CD)

```bash
uv run locust -f tests/performancetests/locustfile.py \
    --headless --users 10 --spawn-rate 2 --run-time 30s --host http://localhost:8000
```

## Cloud Benchmark

```bash
PYTHONPATH=src uv run python scripts/benchmark_cloud.py \
  --url https://clickbait-api-onnx-136485552734.europe-west1.run.app \
  --requests 1000
```

---

# Profiling

```bash
uv run python src/clickbait_classifier/train.py --profile
```

---

# Documentation

```bash
# Build
uv run invoke build-docs

# Serve
uv run invoke serve-docs
```
