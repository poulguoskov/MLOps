import pytest
import torch
from torch import nn

import clickbait_classifier.model as model_module
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
