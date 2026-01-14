from pathlib import Path

import pytest
import torch

from clickbait_classifier.data import ClickbaitDataset, load_data
from tests import _PATH_DATA

PROCESSED_DIR = Path(_PATH_DATA) / "processed"
RAW_CSV = Path(_PATH_DATA) / "raw" / "clickbait_data.csv"


@pytest.mark.skipif(not PROCESSED_DIR.exists(), reason="Processed data not found (data/processed)")
def test_load_data_returns_three_nonempty_splits():
    train_set, val_set, test_set = load_data(PROCESSED_DIR)

    assert len(train_set) > 0, "Train set is empty"
    assert len(val_set) > 0, "Val set is empty"
    assert len(test_set) > 0, "Test set is empty"


@pytest.mark.skipif(not PROCESSED_DIR.exists(), reason="Processed data not found (data/processed)")
@pytest.mark.parametrize("split", ["train", "val", "test"])
def test_processed_files_exist(split: str):
    p = PROCESSED_DIR / f"{split}.pt"
    assert p.exists(), f"Missing file: {p}"


@pytest.mark.skipif(not PROCESSED_DIR.exists(), reason="Processed data not found (data/processed)")
@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_all_splits_have_correct_shapes_and_labels(split_name: str):
    train_set, val_set, test_set = load_data(PROCESSED_DIR)
    ds = {"train": train_set, "val": val_set, "test": test_set}[split_name]

    input_ids, attention_mask, labels = ds.tensors

    assert input_ids.ndim == 2, f"{split_name}: input_ids should be 2D [N, L]"
    assert attention_mask.ndim == 2, f"{split_name}: attention_mask should be 2D [N, L]"
    assert labels.ndim == 1, f"{split_name}: labels should be 1D [N]"

    n, seq_len = input_ids.shape
    assert attention_mask.shape == (n, seq_len), f"{split_name}: attention_mask shape mismatch"
    assert labels.shape == (n,), f"{split_name}: labels length mismatch"

    unique_mask = torch.unique(attention_mask).tolist()
    assert all(v in [0, 1] for v in unique_mask), f"{split_name}: attention_mask not binary: {unique_mask}"

    unique_labels = torch.unique(labels).tolist()
    assert all(v in [0, 1] for v in unique_labels), f"{split_name}: labels not binary: {unique_labels}"


@pytest.mark.skipif(not PROCESSED_DIR.exists(), reason="Processed data not found (data/processed)")
@pytest.mark.parametrize("split_name", ["train", "val", "test"])
def test_getitem_triplet_for_all_splits(split_name: str):
    train_set, val_set, test_set = load_data(PROCESSED_DIR)
    ds = {"train": train_set, "val": val_set, "test": test_set}[split_name]

    input_ids, attention_mask, label = ds[0]
    assert input_ids.shape == attention_mask.shape, f"{split_name}: input_ids and attention_mask shape mismatch"
    assert int(label.item()) in [0, 1], f"{split_name}: label is not 0/1"


@pytest.mark.skipif(not RAW_CSV.exists(), reason="Raw CSV not found (data/raw/clickbait_data.csv)")
def test_raw_clickbait_dataset_loads_and_formats_items():
    ds = ClickbaitDataset(RAW_CSV)

    assert len(ds) > 0, "Raw dataset is empty"

    headline, label = ds[0]
    assert isinstance(headline, str), "headline should be a string"
    assert isinstance(label, torch.Tensor), "label should be a torch.Tensor"

    y = int(label.item())
    assert y in [0, 1], f"Expected label 0/1, got {y}"
