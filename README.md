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



### Commands

## Project Setup
Ensure your environment is synchronized before running any tasks:
```bash
uv sync
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

### Run tests
```bash
uv run invoke test
```

### Run AIP locally
```bash
uv run invoke dev-api
```
The API health check will be available at https://www.google.com/search?q=http://127.0.0.1:8000/




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
