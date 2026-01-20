"""Export trained model to ONNX format for faster inference."""

import argparse
import glob
from pathlib import Path

import torch

from clickbait_classifier.lightning_module import ClickbaitLightningModule


def export_to_onnx(checkpoint_path: str, output_path: str = "models/clickbait_model.onnx"):
    """Export PyTorch Lightning model to ONNX format."""
    print(f"Loading model from {checkpoint_path}")
    model = ClickbaitLightningModule.load_from_checkpoint(checkpoint_path, map_location="cpu")
    model.eval()

    # Dummy inputs matching tokenizer output
    batch_size = 1
    seq_len = 128
    dummy_input_ids = torch.randint(0, 30522, (batch_size, seq_len))
    dummy_attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)

    # Export to ONNX (without external data)
    print(f"Exporting to {output_path}")
    torch.onnx.export(
        model,
        (dummy_input_ids, dummy_attention_mask),
        output_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence"},
            "attention_mask": {0: "batch_size", 1: "sequence"},
            "logits": {0: "batch_size"},
        },
        opset_version=14,
        do_constant_folding=True,
    )

    # Verify the export
    import onnx

    onnx_model = onnx.load(output_path)
    onnx.checker.check_model(onnx_model)
    print(f"✅ ONNX model exported and validated: {output_path}")

    # Print model size
    onnx_size = Path(output_path).stat().st_size / (1024 * 1024)
    print(f"ONNX model size: {onnx_size:.1f} MB")

    # Check for external data file
    data_file = Path(output_path + ".data")
    if data_file.exists():
        data_size = data_file.stat().st_size / (1024 * 1024)
        print(f"External data file: {data_size:.1f} MB")
        print("⚠️  Model has external data - may need both files for deployment")
    else:
        print("✅ No external data file - single file deployment")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export model to ONNX")
    parser.add_argument("--checkpoint", type=str, help="Path to .ckpt file")
    parser.add_argument("--output", type=str, default="models/clickbait_model.onnx")
    args = parser.parse_args()

    if not args.checkpoint:
        checkpoints = sorted(glob.glob("models/**/*.ckpt", recursive=True))
        if not checkpoints:
            print("No checkpoints found in models/")
            exit(1)
        args.checkpoint = checkpoints[-1]
        print(f"Using latest checkpoint: {args.checkpoint}")

    export_to_onnx(args.checkpoint, args.output)
