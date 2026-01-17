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


### Run AIP locally
```bash
uv run invoke dev-api
```
The API health check will be available at https://www.google.com/search?q=http://127.0.0.1:8000/

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

| Command | Description |
|---------|-------------|
| `uv run invoke upload-model` | Upload latest model checkpoint to W&B |
| `uv run invoke stage-model` | Add staging alias to trigger CI tests |
| `uv run invoke stage-model --artifact=clickbait-model:v2` | Stage a specific version |

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
Training on Google Cloud (Vertex AI)
Follow these steps to train the model on Vertex AI using GPU.

Prerequisites:
Make sure you have authenticated with Google Cloud:
gcloud auth login
gcloud auth configure-docker europe-west1-docker.pkg.dev

1. Build and Push Docker Image

First, build the Docker image and push it to the cloud. Important: Change the version tag (e.g., v5, v6) for every new build.

docker buildx build --platform linux/amd64 \
  -f dockerfiles/train.dockerfile \
  -t europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/train:<YOUR_TAG> \
  --push .

  (Replace <YOUR_TAG> with your version, e.g., v6)

2. Update Configuration

Open configs/config_gpu.yaml and update the imageUri to match the tag you just pushed:

# Example inside config_gpu.yaml
imageUri: europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/train:<YOUR_TAG>

3. Start Training Job

Run this command to start the training on Vertex AI. The model will be saved directly to our Google Cloud Storage bucket.

gcloud ai custom-jobs create \
  --region=europe-west1 \
  --display-name=clickbait-train-run \
  --config=configs/config_gpu.yaml \
  --command=uv \
  --args=run \
  --args=-m \
  --args=clickbait_classifier.train \
  --args=--config=configs/config.yaml \
  --args=--processed-path=/gcs/dtumlops-clickbait-data/data/processed \
  --args=--output=/gcs/dtumlops-clickbait-data/models

When the job is finished, the model (.ckpt) and the config file are automatically saved to our storage bucket.

Bucket Path: gs://dtumlops-clickbait-data/models/



## ☁️ Deployment (Cloud Run)

Loading the model from GCS bucket

1. Build and push image

```bash
docker buildx build --platform linux/amd64 \
  -f dockerfiles/api_gcp.dockerfile \
  -t europe-west1-docker.pkg.dev/dtumlops-484212/container-reg/api-gcp:v1 \
  --push .

```

2. Deploy to cloud run

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
