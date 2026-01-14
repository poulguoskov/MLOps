from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn
from torch.utils.data import TensorDataset

import clickbait_classifier.train as train_module


class DummyTransformer(nn.Module):
    """Fake transformer to avoid HuggingFace downloads."""

    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = type("cfg", (), {"hidden_size": hidden_size})()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        batch_size, seq_len = input_ids.shape
        h = self.config.hidden_size
        last_hidden_state = torch.randn(batch_size, seq_len, h)
        return type("out", (), {"last_hidden_state": last_hidden_state})()


def _dummy_cfg(tmp_path: Path):
    """Create a config that matches the fields used by train.py."""
    return SimpleNamespace(
        data=SimpleNamespace(processed_path=str(tmp_path / "data" / "processed")),
        model=SimpleNamespace(model_name="dummy", num_labels=2, dropout=0.0),
        training=SimpleNamespace(
            seed=123,
            device="cpu",
            epochs=1,
            batch_size=4,
            lr=1e-3,
            shuffle=False,
            optimizer=SimpleNamespace(weight_decay=0.01),
            loss=SimpleNamespace(),
        ),
        paths=SimpleNamespace(model_output=str(tmp_path / "models" / "clickbait_model.ckpt")),
    )


def _dummy_datasets():
    """Create small dummy datasets for testing."""
    num_samples, seq_len = 8, 16
    input_ids = torch.randint(0, 1000, (num_samples, seq_len), dtype=torch.long)
    attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)
    labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)
    ds = TensorDataset(input_ids, attention_mask, labels)
    return ds, ds, ds


@pytest.fixture
def patch_transformer(monkeypatch):
    """Patch AutoModel to avoid downloading real models."""
    def _fake_from_pretrained(_name: str):
        return DummyTransformer(hidden_size=16)

    monkeypatch.setattr(model_module.AutoModel, "from_pretrained", _fake_from_pretrained)


def test_train_runs_with_lightning(monkeypatch, tmp_path, patch_transformer):
    """Test that Lightning training runs and saves checkpoint."""
    cfg = _dummy_cfg(tmp_path)

    # Track what was saved
    saved = {"config_path": None}

    def fake_save_config(_cfg, path):
        saved["config_path"] = Path(path)
        # Create the file so assertions pass
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text("# dummy config")

    # Avoid hydra config from disk
    monkeypatch.setattr(train_module, "_load_config", lambda _config_path: cfg)

    # Avoid reading actual data files
    monkeypatch.setattr(train_module, "load_data", lambda _processed_path: _dummy_datasets())

    # Mock save_config to avoid OmegaConf error with SimpleNamespace
    monkeypatch.setattr(train_module, "save_config", fake_save_config)

    # Disable wandb
    monkeypatch.setattr(train_module, "api_key", None)

    # Run training
    train_module.train()

    # Check that config was saved
    assert saved["config_path"] is not None, "Config was not saved"

    # Check that model checkpoint was saved
    run_dirs = list(Path("models").glob("2*"))  # Timestamp dirs start with 2
    assert len(run_dirs) > 0, "No model directory created"

    # Find the most recent run
    latest_run = max(run_dirs, key=lambda p: p.stat().st_mtime)
    ckpt_files = list(latest_run.glob("*.ckpt"))
    assert len(ckpt_files) > 0, f"No checkpoint saved in {latest_run}"
