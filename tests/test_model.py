import pytest
import torch
from torch import nn

import clickbait_classifier.model as model_module
from clickbait_classifier.lightning_module import ClickbaitLightningModule
from clickbait_classifier.model import ClickbaitClassifier


class DummyTransformer(nn.Module):
    def __init__(self, hidden_size: int = 16):
        super().__init__()
        self.config = type("cfg", (), {"hidden_size": hidden_size})()

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        # Simuler last_hidden_state: [B, L, H]
        batch_size, seq_len = input_ids.shape
        h = self.config.hidden_size
        last_hidden_state = torch.randn(batch_size, seq_len, h)
        return type("out", (), {"last_hidden_state": last_hidden_state})()


@pytest.fixture
def patch_transformer(monkeypatch):
    def _fake_from_pretrained(_name: str):
        return DummyTransformer(hidden_size=16)

    monkeypatch.setattr(model_module.AutoModel, "from_pretrained", _fake_from_pretrained)


@pytest.mark.usefixtures("patch_transformer")
@pytest.mark.parametrize("batch_size", [1, 4, 8])
@pytest.mark.parametrize("seq_len", [8, 32])
def test_model_forward_output_shape(batch_size: int, seq_len: int):
    model = ClickbaitClassifier(num_labels=2, dropout=0.0)
    model.eval()

    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    assert logits.shape == (batch_size, 2), "Expected logits shape [B, num_labels]"
    assert logits.dtype.is_floating_point, "Logits should be floating point"


@pytest.mark.usefixtures("patch_transformer")
def test_lightning_module_forward():
    """Test that LightningModule wraps model correctly."""
    model = ClickbaitLightningModule(num_labels=2, dropout=0.0, lr=1e-5)
    model.eval()

    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)

    with torch.no_grad():
        logits = model(input_ids, attention_mask)

    assert logits.shape == (batch_size, 2), "Expected logits shape [B, num_labels]"


@pytest.mark.usefixtures("patch_transformer")
def test_lightning_module_training_step():
    """Test that training_step returns a loss."""
    model = ClickbaitLightningModule(num_labels=2, dropout=0.0, lr=1e-5)

    batch_size, seq_len = 4, 32
    input_ids = torch.randint(0, 1000, (batch_size, seq_len))
    attention_mask = torch.ones((batch_size, seq_len), dtype=torch.long)
    labels = torch.randint(0, 2, (batch_size,))

    batch = (input_ids, attention_mask, labels)
    loss = model.training_step(batch, batch_idx=0)

    assert isinstance(loss, torch.Tensor), "training_step should return a tensor"
    assert loss.shape == (), "Loss should be a scalar"
