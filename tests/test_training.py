from pathlib import Path
from types import SimpleNamespace

import torch
from torch import nn
from torch.utils.data import TensorDataset

import clickbait_classifier.train as train_module


class DummyModel(nn.Module):
    """Small model that aligns with the signature model(input_ids, attention_mask)."""

    def __init__(self, num_labels: int = 2):
        super().__init__()
        self.classifier = nn.Linear(1, num_labels)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        # input_ids and attention_mask are [B, L]. Make a simple feature from them: mean over tokens.
        x = input_ids.float().mean(dim=1, keepdim=True)  # [B, 1]
        return self.classifier(x)  # [B, num_labels]


def _dummy_cfg(tmp_path: Path):
    # Create a "cfg" that matches the fields used by train.py
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
            optimizer=SimpleNamespace(),  # no _target_, so default optimizer is used
            loss=SimpleNamespace(),  # no _target_, so default loss is used
        ),
        paths=SimpleNamespace(model_output=str(tmp_path / "models" / "clickbait_model.pt")),
    )


def _dummy_datasets():
    num_samples, seq_len = 8, 16
    input_ids = torch.randint(0, 1000, (num_samples, seq_len), dtype=torch.long)
    attention_mask = torch.ones((num_samples, seq_len), dtype=torch.long)
    labels = torch.randint(0, 2, (num_samples,), dtype=torch.long)

    ds = TensorDataset(input_ids, attention_mask, labels)
    # train/val/test can be the same for the test
    return ds, ds, ds


def test_train_runs_and_saves_model_and_config(monkeypatch, tmp_path):
    cfg = _dummy_cfg(tmp_path)

    # Avoid hydra-config from disk
    monkeypatch.setattr(train_module, "_load_config", lambda _config_path: cfg)

    # Avoid reading actual data files
    monkeypatch.setattr(train_module, "load_data", lambda _processed_path: _dummy_datasets())

    # Avoid HuggingFace download: replace ClickbaitClassifier -> DummyModel
    monkeypatch.setattr(train_module, "ClickbaitClassifier", lambda **kwargs: DummyModel(num_labels=2))

    # Disable wandb (since api_key is imported as a variable in the module)
    monkeypatch.setattr(train_module, "api_key", None)

    # Mock torch.save and save_config without writing to disk
    saved = {"model_path": None, "config_path": None}

    def fake_torch_save(state_dict, path):
        saved["model_path"] = Path(path)

    def fake_save_config(_cfg, path):
        saved["config_path"] = Path(path)

    monkeypatch.setattr(train_module.torch, "save", fake_torch_save)
    monkeypatch.setattr(train_module, "save_config", fake_save_config)

    # Run training
    train_module.train()

    # Check that we tried to save model and config where cfg says
    assert saved["model_path"] == Path(cfg.paths.model_output), "Model was not saved to expected path"
    assert (
        saved["config_path"] == Path(cfg.paths.model_output).parent / "config.yaml"
    ), "Config was not saved next to model"
