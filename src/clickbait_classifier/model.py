from pathlib import Path
from typing import Annotated, Optional

import torch
import typer
from hydra import compose, initialize_config_dir
from omegaconf import OmegaConf
from torch import nn
from transformers import AutoModel

app = typer.Typer()


class ClickbaitClassifier(nn.Module):
    """DistilBERT-based classifier for clickbait detection."""

    def __init__(
        self,
        model_name: Optional[str] = None,
        num_labels: Optional[int] = None,
        dropout: Optional[float] = None,
        config: Optional[OmegaConf] = None,
    ) -> None:
        """
        Initialize the ClickbaitClassifier.

        Args:
            model_name: Name of the pretrained model (overridden by config if provided)
            num_labels: Number of output labels (overridden by config if provided)
            dropout: Dropout rate (overridden by config if provided)
            config: OmegaConf config object. If provided, model_name, num_labels, and dropout
                    will be taken from config.model.*
        """
        super().__init__()

        # If config is provided, use it; otherwise use individual parameters or defaults
        if config is not None:
            model_name = config.model.model_name
            num_labels = config.model.num_labels
            dropout = config.model.dropout
        else:
            # Use provided values or defaults
            model_name = model_name or "distilbert-base-uncased"
            num_labels = num_labels or 2
            dropout = dropout if dropout is not None else 0.1

        self.transformer = AutoModel.from_pretrained(model_name)
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(self.transformer.config.hidden_size, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        outputs = self.transformer(input_ids=input_ids, attention_mask=attention_mask)
        # Use [CLS] token representation (first token)
        pooled = outputs.last_hidden_state[:, 0, :]
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)
        return logits


def _load_config(config_path: Optional[Path]) -> Optional[OmegaConf]:
    """Load configuration from file using Hydra."""
    if config_path is None:
        return None

    config_path = Path(config_path).resolve()
    config_dir = config_path.parent
    config_name = config_path.stem

    with initialize_config_dir(config_dir=str(config_dir), version_base=None):
        cfg = compose(config_name=config_name)
    return cfg


@app.command()
def info(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = None,
) -> None:
    """Show model architecture and parameter count."""
    cfg = _load_config(config)
    model = ClickbaitClassifier(config=cfg)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model_name = cfg.model.model_name if cfg else "distilbert-base-uncased"
    print("Model: ClickbaitClassifier")
    print(f"Base: {model_name}")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")


@app.command()
def test(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = None,
) -> None:
    """Run a quick forward pass test."""
    cfg = _load_config(config)
    model = ClickbaitClassifier(config=cfg)
    model.eval()

    batch_size = 4
    seq_length = 128
    input_ids = torch.randint(0, 1000, (batch_size, seq_length))
    attention_mask = torch.ones(batch_size, seq_length, dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    print(f"Input shape: ({batch_size}, {seq_length})")
    print(f"Output shape: {logits.shape}")
    print("Forward pass successful!")


if __name__ == "__main__":
    app()
