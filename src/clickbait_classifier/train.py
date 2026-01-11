from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from clickbait_classifier.data import load_data
from clickbait_classifier.model import ClickbaitClassifier


def train(
    processed_path: Path = Path("data/processed"),
    epochs: int = 3,
    batch_size: int = 32,
    lr: float = 2e-5,
    device: str = "auto",
) -> None:
    """Train the clickbait classifier."""
    # Set device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    # Load data
    train_set, val_set, test_set = load_data(processed_path)
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)

    print(f"Train: {len(train_set)}, Val: {len(val_set)}, Test: {len(test_set)}")

    # Model
    model = ClickbaitClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
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
        print(f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.4f} - Val Acc: {val_acc:.4f}")

    # Save model
    models_path = Path("models")
    models_path.mkdir(exist_ok=True)
    torch.save(model.state_dict(), models_path / "clickbait_model.pt")
    print(f"Model saved to {models_path / 'clickbait_model.pt'}")


if __name__ == "__main__":
    #train()
    print("Training script executed")
