"""Lightning-based training script for clickbait classifier."""

import random
from datetime import datetime
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import pytorch_lightning as pl
import torch
import torch.profiler
import typer
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf
from pytorch_lightning.callbacks import ModelCheckpoint, RichProgressBar
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import DataLoader

from clickbait_classifier.data import load_data
from clickbait_classifier.lightning_module import ClickbaitLightningModule
from clickbait_classifier.load_from_env_file import api_key
from clickbait_classifier.utils import save_config

app = typer.Typer()


def _set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)


def _load_config(config_path: Optional[Path]) -> OmegaConf:
    """Load configuration from file using Hydra."""
    if config_path is None:
        config_path = Path("configs/config.yaml")

    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    config_name = config_path.stem

    logger.debug(f"Loading configuration from {config_path}")
    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
    logger.debug("Configuration loaded successfully")
    return cfg


@app.command()
def train(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = None,
    processed_path: Annotated[
        Optional[Path],
        typer.Option(help="Path to processed data (overrides config)"),
    ] = None,
    epochs: Annotated[
        Optional[int],
        typer.Option(help="Number of training epochs (overrides config)"),
    ] = None,
    batch_size: Annotated[
        Optional[int],
        typer.Option("--batch-size", "-b", help="Batch size (overrides config)"),
    ] = None,
    lr: Annotated[
        Optional[float],
        typer.Option("--lr", "-l", help="Learning rate (overrides config)"),
    ] = None,
    device: Annotated[
        Optional[str],
        typer.Option(help="Device to use: auto, cpu, cuda, mps (overrides config)"),
    ] = None,
    output: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Model output path (overrides config)"),
    ] = None,
    profile_run: Annotated[bool, typer.Option("--profile", "-p", help="Run torch profiler")] = False,
) -> None:
    """Train the clickbait classifier using PyTorch Lightning."""
    # Load configuration
    cfg = _load_config(config)

    # Create unique run directory with timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = Path("models") / timestamp
    cfg.paths.model_output = str(run_dir / "clickbait_model.pt")

    # Override config with CLI arguments (CLI takes precedence)
    if processed_path is not None:
        cfg.data.processed_path = str(processed_path)
    if epochs is not None:
        cfg.training.epochs = epochs
    if batch_size is not None:
        cfg.training.batch_size = batch_size
    if lr is not None:
        cfg.training.lr = lr
    if device is not None:
        cfg.training.device = device
    if output is not None:
        cfg.paths.model_output = str(output)

    # Set seed for reproducibility
    pl.seed_everything(cfg.training.seed)
    logger.info(f"Set random seed to {cfg.training.seed}")

    # Determine accelerator
    device_str = cfg.training.device
    if device_str == "auto":
        accelerator = "auto"
    elif device_str == "mps":
        accelerator = "mps"
    elif device_str == "cuda":
        accelerator = "gpu"
    else:
        accelerator = "cpu"
    logger.info(f"Using accelerator: {accelerator}")

    # Load data
    processed_path = Path(cfg.data.processed_path)
    train_set, val_set, test_set = load_data(processed_path)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
        num_workers=0,
    )
    val_loader = DataLoader(val_set, batch_size=cfg.training.batch_size, num_workers=0)

    logger.info(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    logger.info(
        f"Number of epochs: {cfg.training.epochs}, "
        f"Batch size: {cfg.training.batch_size}, "
        f"Learning rate: {cfg.training.lr}"
    )
    weight_decay = getattr(getattr(cfg.training, "optimizer", None), "weight_decay", 0.0)

    # Initialize Lightning model
    logger.info(f"Initializing model: {cfg.model.model_name}")
    model = ClickbaitLightningModule(
        model_name=cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        dropout=cfg.model.dropout,
        lr=cfg.training.lr,
        weight_decay=weight_decay,
    )

    # Callbacks
    callbacks = [
        RichProgressBar(),
        ModelCheckpoint(
            dirpath=run_dir,
            filename="clickbait_model",
            monitor="val_acc",
            mode="max",
            save_top_k=1,
        ),
    ]

    # W&B Logger
    wandb_logger = None
    if api_key:
        wandb_logger = WandbLogger(
            project="clickbait-classifier",
            name=f"lightning-{cfg.model.model_name}-epochs{cfg.training.epochs}-lr{cfg.training.lr}",
            log_model=True,
        )
        logger.info("W&B logging enabled")
    else:
        logger.warning("WANDB_API_KEY not found, wandb logging will be disabled")

    # Trainer
    trainer = pl.Trainer(
        max_epochs=cfg.training.epochs,
        accelerator=accelerator,
        callbacks=callbacks,
        logger=wandb_logger,
        deterministic=True,
        profiler="simple" if profile_run else None,
    )

    # Train!
    logger.info("Starting training...")
    trainer.fit(model, train_loader, val_loader)

    # Save config alongside model
    run_dir.mkdir(parents=True, exist_ok=True)
    config_output_path = run_dir / "config.yaml"
    save_config(cfg, config_output_path)
    logger.info(f"Configuration saved to {config_output_path}")
    logger.info(f"Training complete. Best model saved to {run_dir}")


if __name__ == "__main__":
    app()
