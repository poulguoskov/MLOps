from pathlib import Path
from typing import Annotated, Optional

import pandas as pd
import torch
import typer
from hydra import compose, initialize_config_dir
from loguru import logger
from omegaconf import OmegaConf
from torch.utils.data import Dataset, TensorDataset
from transformers import AutoTokenizer

app = typer.Typer()


class ClickbaitDataset(Dataset):
    """Dataset for clickbait classification (raw text)."""

    def __init__(self, data_path: Path) -> None:
        self.data = pd.read_csv(data_path)
        self.headlines = self.data["headline"].tolist()
        self.labels = torch.tensor(self.data["clickbait"].values, dtype=torch.long)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int) -> tuple[str, torch.Tensor]:
        return self.headlines[index], self.labels[index]


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


def load_data(
    processed_path: Path = Path("data/processed"),
) -> tuple[TensorDataset, TensorDataset, TensorDataset]:
    """Load preprocessed train, val, and test sets."""
    logger.info(f"Loading processed data from {processed_path}")
    train_data = torch.load(processed_path / "train.pt", weights_only=True)
    val_data = torch.load(processed_path / "val.pt", weights_only=True)
    test_data = torch.load(processed_path / "test.pt", weights_only=True)
    logger.info(
        f"Loaded train: {train_data['input_ids'].shape[0]} samples, val: {val_data['input_ids'].shape[0]} samples, "
        f"test: {test_data['input_ids'].shape[0]} samples"
    )

    train_set = TensorDataset(
        train_data["input_ids"],
        train_data["attention_mask"],
        train_data["labels"],
    )
    val_set = TensorDataset(
        val_data["input_ids"],
        val_data["attention_mask"],
        val_data["labels"],
    )
    test_set = TensorDataset(
        test_data["input_ids"],
        test_data["attention_mask"],
        test_data["labels"],
    )

    return train_set, val_set, test_set


@app.command()
def preprocess(
    config: Annotated[
        Optional[Path],
        typer.Option("--config", "-c", help="Path to configuration file"),
    ] = None,
    raw_path: Annotated[
        Optional[Path],
        typer.Option(help="Path to raw CSV (overrides config)"),
    ] = None,
    output_path: Annotated[
        Optional[Path],
        typer.Option("--output", "-o", help="Output directory (overrides config)"),
    ] = None,
    model_name: Annotated[
        Optional[str],
        typer.Option(help="Tokenizer model name (overrides config)"),
    ] = None,
    max_length: Annotated[
        Optional[int],
        typer.Option(help="Max sequence length (overrides config)"),
    ] = None,
    train_split: Annotated[
        Optional[float],
        typer.Option(help="Train split ratio (overrides config)"),
    ] = None,
    val_split: Annotated[
        Optional[float],
        typer.Option(help="Validation split ratio (overrides config)"),
    ] = None,
) -> None:
    """Tokenize raw data and save train/val/test splits as tensors."""
    # Load configuration
    cfg = _load_config(config)

    # Override config with CLI arguments (CLI takes precedence)
    if raw_path is not None:
        cfg.data.raw_path = str(raw_path)
    if output_path is not None:
        cfg.data.output_path = str(output_path)
    if model_name is not None:
        cfg.data.tokenizer_model_name = model_name
    if max_length is not None:
        cfg.data.max_length = max_length
    if train_split is not None:
        cfg.data.train_split = train_split
    if val_split is not None:
        cfg.data.val_split = val_split

    raw_path = Path(cfg.data.raw_path)
    output_path = Path(cfg.data.output_path)
    model_name = cfg.data.tokenizer_model_name
    max_length = cfg.data.max_length
    train_split = cfg.data.train_split
    val_split = cfg.data.val_split
    random_state = cfg.data.random_state

    logger.info(f"Loading data from {raw_path}")
    df = pd.read_csv(raw_path)
    logger.info(f"Loaded {len(df)} samples")
    logger.info(f"Clickbait: {df['clickbait'].sum()}, Non-clickbait: {(df['clickbait'] == 0).sum()}")

    # Shuffle the data
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Split indices
    n = len(df)
    train_end = int(train_split * n)
    val_end = int((train_split + val_split) * n)

    train_df = df[:train_end]
    val_df = df[train_end:val_end]
    test_df = df[val_end:]

    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

    # Tokenize
    logger.info(f"Tokenizing with {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    output_path.mkdir(parents=True, exist_ok=True)

    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        encodings = tokenizer(
            split_df["headline"].tolist(),
            truncation=True,
            padding="max_length",
            max_length=max_length,
            return_tensors="pt",
        )

        data = {
            "input_ids": encodings["input_ids"],
            "attention_mask": encodings["attention_mask"],
            "labels": torch.tensor(split_df["clickbait"].values, dtype=torch.long),
        }

        torch.save(data, output_path / f"{split_name}.pt")
        logger.info(f"Saved {split_name}.pt with shape {encodings['input_ids'].shape}")


if __name__ == "__main__":
    app()
