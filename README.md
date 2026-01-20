# Clickbait Classifier

## Overall Goal

We want to build a clickbait detector, a model that can tell whether a headline is a genuine news or sensationalized garbage. The idea is to train a text classifier and deploy it as an API that can score headlines in real time.

The ML task itself isn't groundbreaking, but that's kind of the point. We want to spend our time on the MLOps side: setting up reproducible training, experiment tracking, CI/CD, containerization, and cloud deployment.

## Framework

We'll use PyTorch as our deep learning framework and HuggingFace Transformers for pretrained models and tokenizers.

## Data

We're using the [Clickbait Dataset](https://www.kaggle.com/datasets/amananandrai/clickbait-dataset) from Kaggle. It has around 32,000 headlines labeled as clickbait or not. Small enough to iterate quickly, clean enough that we won't spend days on preprocessing or training.

## Models

We haven't fully decided on our approach yet. Options we're considering:

- Fine-tuning a pre-trained model using HuggingFace Transformers, something like DistilBERT, BERT, or RoBERTa. This is probably the most practical path.
- Training a simpler model from scratch to compare and understand the difference.
- Both, if time allows. Comparing fine-tuned transformers against a baseline we build ourselves.

Model candidates include `distilbert-base-uncased` (fast, good enough), `bert-base-uncased` (standard choice), or potentially something smaller if inference speed matters for deployment.

We'll track experiments in W&B and let the results guide our final choice. The priority is getting the pipeline working end-to-end, not squeezing out the last percentage of accuracy.

## Docker

This project uses Docker and Docker Compose to run the training and API
in a reproducible environment.

### Prerequisites

- Docker Desktop installed and running
- - Python + uv (for `invoke` tasks)

---

# Commands

## Project Setup

Ensure your environment is synchronized before running any tasks:

```bash
uv sync
```

## Pre-commit setup

```bash
uv run pre-commit run --all-files
```

## Run tests

```bash
uv run invoke test
```

## Load testing

Run load tests with locust against the local API.

### Start the API first

```bash
uv run invoke dev-api
```

### Run locust with web UI

```bash
uv run locust -f tests/performancetests/locustfile.py
```

Open http://localhost:8089, set host to `http://localhost:8000`, configure users and spawn rate.

### Run locust headless (CI/CD)

```bash
uv run locust -f tests/performancetests/locustfile.py \
    --headless --users 10 --spawn-rate 2 --run-time 30s --host http://localhost:8000
```

## Local development

### Preprocess data

```bash
uv run invoke preprocess-data
```

### Train model locally

```bash
uv run invoke train
```

### Evaluate model locally

```bash
uv run invoke evaluate
```

### Run API locally

```bash
uv run invoke dev-api
```

API will be available at http://127.0.0.1:8000

**Endpoints:**

| Endpoint          | Method | Description                   |
| ----------------- | ------ | ----------------------------- |
| `/`               | GET    | Health check                  |
| `/docs`           | GET    | Interactive API documentation |
| `/classify`       | POST   | Classify single text          |
| `/classify/batch` | POST   | Classify multiple texts       |

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

## Model Registry Workflow

This project uses W&B Model Registry with GitHub Actions for automated model testing and promotion.

### Workflow

1. **Train a model:**

```bash
   uvr train --epochs 3
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

Or add the `staging` alias manually in the W&B UI.

5. **Automated testing:** GitHub Actions downloads the staged model, runs performance tests, and auto-promotes to `production` if tests pass.

### Model Commands

| Command                                                   | Description                           |
| --------------------------------------------------------- | ------------------------------------- |
| `uv run invoke upload-model`                              | Upload latest model checkpoint to W&B |
| `uv run invoke stage-model`                               | Add staging alias to trigger CI tests |
| `uv run invoke stage-model --artifact=clickbait-model:v2` | Stage a specific version              |

# Docker Compose

### Start Services

```bash
docker compose up --build
```

### Development Container

Start a persistent development container:

```bash
docker compose up -d dev
```

### Run training in docker

```bash
uv run invoke docker-train --args="--config configs/config.yaml --epochs 5"
```

### Run evaluation in docker

```bash
uv run invoke docker-evaluate
```

### Build API dockerimage

```bash
uv run invoke build-api
```

### Run API in docker

```bash
uv run invoke run-api-docker
```

### Stop all contrainers

```bash
docker compose down
```

# Profiling

### training with profiling

```bash
uv run python src/clickbait_classifier/train.py --profile
```

# Documentation

### Build documentation

```bash
uv run invoke build-docs
```

### Serve documentation

```bash
uv run invoke serve-docs
```

# Pre commits

### Run all hooks

```bash
uv run pre-commit run --all-files
```

### Skip hooks

```bash
git commit -m "your commit message" --no-verify
```

### Preprocess data

Runs the preprocessing pipeline from raw to processed data.

```bash
uv run invoke preprocess-data
```

### Development container (recommended)

Start a persistent development container (runs in the background):

```bash
docker compose up -d dev
```

### Run training

Training can be run with configuration files (recommended) or with CLI arguments:

**Using default configuration:**

```bash
uv run invoke docker-train
```

**Using a custom configuration file:**

```bash
uv run invoke docker-train --args="--config configs/my_experiment.yaml"
```

**Using configuration file with CLI overrides:**

```bash
uv run invoke docker-train --args="--config configs/config.yaml --epochs 5 --batch-size 64 --lr 1e-4"
```

**Using only CLI arguments (backward compatible):**

```bash
uv run invoke docker-train --args="--epochs 3 --batch-size 32 --lr 2e-5"
```

Note: CLI arguments override values from the configuration file when both are provided.

### Run API

```bash
docker compose up api
```

### Stop all containers when finished

```bash
docker compose down
```

## Profiling

Run training with profiling enabled. This profiles only the first batch of the first epoch.

```bash
uv run python src/clickbait_classifier/train.py --profile
```

## Pre-commits

Pre-commit runs automatic checks (formatting, linting, config validation) before each commit.
Commits are blocked if checks fail.

### Run all hooks:

```bash
uv run pre-commit run --all-files
```

### Skip hooks:

```bash
git commit -m"your commit message" --no-verify
```

## Run training on Google Cloud Platform (GCP) Vertex AI

Training on Google Cloud (Vertex AI) using GPU.

### Prerequisites

Make sure you have authenticated with Google Cloud:

```bash
gcloud auth login
gcloud config set project dtumlops-484212
```

### Start Training Job

The Docker image is automatically built and pushed to Artifact Registry when you push to `main` (via Cloud Build trigger).

Run this command to start training on Vertex AI with GPU:

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

When finished, the model and config are saved to: `gs://dtumlops-clickbait-data/models/`

## ☁️ Deployment (Cloud Run)

The API loads the model from GCS bucket on startup.

### 1. Build and push image

```bash
docker buildx build --platform linux/amd64 \
  -f dockerfiles/api_gcp.dockerfile \
  -t europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/api-gcp:v1 \
  --push .

```

### 2. Deploy to Cloud Run

```bash
gcloud run deploy clickbait-api-gcp \
  --image=europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/api-gcp:v1 \
  --region=europe-west1 \
  --allow-unauthenticated \
  --port=8000 \
  --memory=4Gi \
  --timeout=300

```

3. Test API
https://<din-url>.run.app/docs



## 🖥️ Frontend (Streamlit)

### 🏃‍♀️ Run Locally
To run the frontend on your machine during development:

```bash
uv run streamlit run src/clickbait_classifier/frontend.py
```
The app will be available at http://localhost:8501.

### 🐳 Deploy to Cloud Run

To deploy the frontend to Google Cloud Run, follow these steps:

1. Build and Push Image
```bash
docker buildx build --platform linux/amd64 \
  -f dockerfiles/frontend.dockerfile \
  -t europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/frontend:v1 \
  --push .
  ```
2. Deploy Service Note: Streamlit runs on port 8501 by default.

```bash
gcloud run deploy clickbait-frontend \
  --image=europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/frontend:v1 \
  --region=europe-west1 \
  --allow-unauthenticated \
  --port=8501
```

After deployment, click the URL provided in the terminal to open the app.

## ONNX Deployment (Lightweight)

ONNX provides a lighter-weight alternative to PyTorch for deployment.

### Benefits

| Metric          | PyTorch | ONNX   |
| --------------- | ------- | ------ |
| Docker image    | 4.27 GB | 568 MB |
| Model file      | 759 MB  | 266 MB |
| Memory required | 4 Gi    | 2 Gi   |
| CPU required    | 2       | 1      |

### Export model to ONNX
```bash
PYTHONPATH=src uv run python scripts/export_onnx.py
```

### Run ONNX API locally
```bash
# Start API
PYTHONPATH=src uv run uvicorn clickbait_classifier.api_onnx:app --reload --port 8001

# Test
curl -X POST http://localhost:8001/classify \
  -H "Content-Type: application/json" \
  -d '{"text": "You Will NEVER Believe What Happened Next!"}'
```

### Deploy ONNX to Cloud Run
```bash
# Build and push
docker build -f dockerfiles/api_onnx_gcp.dockerfile \
  --platform linux/amd64 \
  -t europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/api-onnx:latest .

docker push europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/api-onnx:latest

# Deploy (note: less resources needed)
gcloud run deploy clickbait-api-onnx \
  --image=europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/api-onnx:latest \
  --region=europe-west1 \
  --platform=managed \
  --memory=2Gi \
  --cpu=1 \
  --timeout=300
```

### Benchmark comparison (1000 requests)

| Metric         | ONNX   | PyTorch   |
| -------------- | ------ | --------- |
| Mean latency   | 349 ms | 284 ms    |
| Median latency | 327 ms | 219 ms    |
| P95 latency    | 450 ms | 264 ms    |
| Max latency    | 710 ms | 61,835 ms |
| Requests/sec   | 2.87   | 3.52      |

PyTorch is faster but ONNX is more consistent (no cold-start spikes) and uses half the resources.

### Run cloud benchmark
```bash
PYTHONPATH=src uv run python scripts/benchmark_cloud.py \
  --url https://clickbait-api-onnx-136485552734.europe-west1.run.app \
  --requests 1000
```

## BentoML Service

BentoML provides ML-optimized serving with adaptive batching.

### Run locally
```bash
uv run bentoml serve src.clickbait_classifier.bentoml_service:ClickbaitClassifier
```

### Test endpoint
```bash
curl -X POST http://localhost:3000/classify \
  -H "Content-Type: application/json" \
  -d '{"texts": ["You Will NEVER Believe What Happened Next!"]}'
```

### Build and containerize
```bash
uv run bentoml build
uv run bentoml containerize clickbait_classifier:latest --opt platform=linux/amd64
```

### Deploy to Cloud Run
```bash
docker tag clickbait_classifier:latest \
  europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/bentoml-service:latest
docker push europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/bentoml-service:latest

gcloud run deploy clickbait-bentoml \
  --image=europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/bentoml-service:latest \
  --region=europe-west1 \
  --memory=2Gi \
  --port=3000
```

Service URL: https://clickbait-bentoml-136485552734.europe-west1.run.app

### Load test results (10 users, 30s)

| Metric       | BentoML |
| ------------ | ------- |
| Requests/sec | 26.91   |
| Mean latency | 111 ms  |
| P95 latency  | 180 ms  |
| Errors       | 0%      |
