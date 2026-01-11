from pathlib import Path
from typing import Annotated

import torch
import typer
from torch import nn
from torch.utils.data import DataLoader
import torch.profiler

from clickbait_classifier.data import load_data
from clickbait_classifier.model import ClickbaitClassifier

app = typer.Typer()


@app.command()
def train(
    processed_path: Path = Path("data/processed"),
    epochs: Annotated[int, typer.Option(help="Number of training epochs")] = 3,
    batch_size: Annotated[int, typer.Option("--batch-size", "-b", help="Batch size")] = 32,
    lr: Annotated[float, typer.Option("--lr", "-l", help="Learning rate")] = 2e-5,
    device: Annotated[str, typer.Option(help="Device to use (auto, cpu, cuda, mps)")] = "auto",
    output: Annotated[Path, typer.Option("--output", "-o", help="Model output path")] = Path(
        "models/clickbait_model.pt"
    ),
    profile_run: Annotated[bool, typer.Option("--profile", help="Run torch profiler")] = False,
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
    print(f"Number of epochs: {epochs}, Batch size: {batch_size}, Learning rate: {lr}")

    # Model
    model = ClickbaitClassifier().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss()

    # Training loop
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for i, batch in enumerate(train_loader):
            input_ids, attention_mask, labels = batch
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            if profile_run and i == 0 and epoch == 0:
                activities = [torch.profiler.ProfilerActivity.CPU]
                if device == "cuda":
                    activities.append(torch.profiler.ProfilerActivity.CUDA)
                
                with torch.profiler.profile(
                    activities = activities,
                    record_shapes=True,
                ) as profiler:
                    logits = model(input_ids, attention_mask)
                    loss = criterion(logits, labels)
                    loss.backward()
                    optimizer.step()

                print(profiler.key_averages().table(sort_by="cpu_time_total", row_limit=10))

                if device == "cuda":
                    print(profiler.key_averages().table(sort_by="cuda_time_total", row_limit=10))

            else:
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
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), output)
    print(f"Model saved to {output}")


if __name__ == "__main__":
    app()
