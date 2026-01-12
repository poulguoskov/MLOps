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


# How to run

## Docker

This project uses Docker and Docker Compose to run the training and API
in a reproducible environment.


### Prerequisites
- Docker Desktop installed and running
- - Python + uv (for `invoke` tasks)
---

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

## Configuration Management

This project uses [Hydra](https://hydra.cc/) for hyperparameter management. All hyperparameters are defined in YAML configuration files, making experiments reproducible and easy to track.

### Configuration Files

The main configuration file is located at `configs/config.yaml` and contains all hyperparameters organized by category:

- **model**: Model architecture parameters (model_name, num_labels, dropout)
- **training**: Training hyperparameters (epochs, batch_size, lr, device, optimizer, loss, seed)
- **data**: Data preprocessing parameters (paths, tokenizer settings, train/val/test splits, random_state)
- **paths**: Output paths for models and configs

### Using Configuration Files

**Train with default config:**
```bash
uv run train
# or
uv run train --config configs/config.yaml
```

**Train with custom config:**
```bash
uv run train --config configs/my_experiment.yaml
```

**Override config values via CLI:**
```bash
uv run train --config configs/config.yaml --epochs 10 --batch-size 64 --lr 1e-4
```

**Preprocess data with config:**
```bash
uv run preprocess --config configs/config.yaml
```

**Preprocess with CLI overrides:**
```bash
uv run preprocess --config configs/config.yaml --max-length 256 --train-split 0.8
```

### Config Saving

When a model is saved, the complete configuration used for training is automatically saved alongside the model weights in the same directory as `config.yaml`. This ensures full reproducibility - you can always see exactly which hyperparameters were used for any trained model.

### Creating New Experiments

To create a new experiment configuration:

1. Copy the default config: `cp configs/config.yaml configs/experiment1.yaml`
2. Modify the hyperparameters you want to change
3. Run training: `uv run train --config configs/experiment1.yaml`

This approach keeps your experiments organized and makes it easy to compare different configurations.
