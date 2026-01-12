import logging
import random
from pathlib import Path
from typing import Annotated, Optional

import numpy as np
import torch
import typer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from torch.utils.data import DataLoader

from clickbait_classifier.data import load_data
from clickbait_classifier.model import ClickbaitClassifier
from clickbait_classifier.utils import save_config

app = typer.Typer()
log = logging.getLogger(__name__)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)


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

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
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
) -> None:
    """Train the clickbait classifier."""
    # Load configuration
    cfg = _load_config(config)

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
    seed = cfg.training.seed
    _set_seed(seed)
    log.info(f"Set random seed to {seed}")

    # Set device
    device_str = cfg.training.device
    if device_str == "auto":
        device_str = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device = torch.device(device_str)
    log.info(f"Using device: {device}")

    # Load data
    processed_path = Path(cfg.data.processed_path)
    train_set, val_set, test_set = load_data(processed_path)
    train_loader = DataLoader(
        train_set,
        batch_size=cfg.training.batch_size,
        shuffle=cfg.training.shuffle,
    )
    val_loader = DataLoader(val_set, batch_size=cfg.training.batch_size)

    log.info(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")
    log.info(
        f"Number of epochs: {cfg.training.epochs}, "
        f"Batch size: {cfg.training.batch_size}, "
        f"Learning rate: {cfg.training.lr}"
    )

    # Model
    model = ClickbaitClassifier(
        model_name=cfg.model.model_name,
        num_labels=cfg.model.num_labels,
        dropout=cfg.model.dropout,
    ).to(device)

    # Optimizer - use Hydra instantiate if _target_ is present, otherwise fallback
    if hasattr(cfg.training.optimizer, "_target_"):
        from hydra.utils import instantiate

        # Ensure optimizer lr matches training.lr (in case it was overridden)
        cfg.training.optimizer.lr = cfg.training.lr
        optimizer = instantiate(cfg.training.optimizer, params=model.parameters())
    else:
        optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.training.lr)

    # Loss function - use Hydra instantiate if _target_ is present, otherwise fallback
    if hasattr(cfg.training.loss, "_target_"):
        from hydra.utils import instantiate

        criterion = instantiate(cfg.training.loss)
    else:
        criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(cfg.training.epochs):
        model.train()
        total_loss = 0

        for batch in train_loader:
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(train_loader)

        # Validation
        model.eval()
        correct, total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                input_ids, attention_mask, labels = batch
                input_ids = input_ids.to(device)
                attention_mask = attention_mask.to(device)
                labels = labels.to(device)

                logits = model(input_ids, attention_mask)
                preds = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        val_acc = correct / total
        log.info(f"Epoch {epoch + 1}/{cfg.training.epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Save model
    output_path = Path(cfg.paths.model_output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output_path)
    log.info(f"Model saved to {output_path}")

    # Save config alongside model
    config_output_path = output_path.parent / "config.yaml"
    save_config(cfg, config_output_path)
    log.info(f"Configuration saved to {config_output_path}")


if __name__ == "__main__":
    app()
