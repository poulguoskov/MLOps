"""Compare PyTorch vs ONNX inference speed across devices."""

import time

import numpy as np
import onnxruntime as ort
import torch
from transformers import DistilBertTokenizer

from clickbait_classifier.lightning_module import ClickbaitLightningModule


def get_available_devices():
    """Get list of available PyTorch devices."""
    devices = ["cpu"]
    if torch.cuda.is_available():
        devices.append("cuda")
    if torch.backends.mps.is_available():
        devices.append("mps")
    return devices


def benchmark_pytorch(model, input_ids, attention_mask, device, n_runs=100):
    """Benchmark PyTorch inference on specified device."""
    model = model.to(device)
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    model.eval()

    # Warmup
    with torch.no_grad():
        for _ in range(10):
            _ = model(input_ids, attention_mask)

    # Sync before timing
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    start = time.time()
    with torch.no_grad():
        for _ in range(n_runs):
            _ = model(input_ids, attention_mask)

    # Sync after timing
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()

    elapsed = time.time() - start
    return elapsed / n_runs * 1000  # ms per inference


def benchmark_onnx(onnx_path, input_ids, attention_mask, n_runs=100):
    """Benchmark ONNX inference (CPU only)."""
    session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])

    inputs = {
        "input_ids": input_ids.numpy(),
        "attention_mask": attention_mask.numpy(),
    }

    # Warmup
    for _ in range(10):
        _ = session.run(None, inputs)

    start = time.time()
    for _ in range(n_runs):
        _ = session.run(None, inputs)
    elapsed = time.time() - start

    return elapsed / n_runs * 1000  # ms per inference


def main():
    checkpoint_path = "models/2026-01-17_12-21-53/clickbait_model.ckpt"
    onnx_path = "models/clickbait_model.onnx"
    n_runs = 100

    # Load tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")

    # Test text
    text = "You Won't BELIEVE What Happened Next!"
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", max_length=128, truncation=True)
    input_ids = inputs["input_ids"]
    attention_mask = inputs["attention_mask"]

    # Load PyTorch model
    print("Loading PyTorch model...")
    pytorch_model = ClickbaitLightningModule.load_from_checkpoint(checkpoint_path, map_location="cpu")

    # Get available devices
    pytorch_devices = get_available_devices()
    print(f"\nAvailable PyTorch devices: {pytorch_devices}")

    # Verify outputs match (on CPU)
    print("\n" + "=" * 50)
    print("Verifying outputs match (CPU)...")
    print("=" * 50)
    with torch.no_grad():
        pytorch_out = pytorch_model(input_ids, attention_mask)

    onnx_session = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    onnx_out = onnx_session.run(None, {"input_ids": input_ids.numpy(), "attention_mask": attention_mask.numpy()})[0]

    print(f"PyTorch output: {pytorch_out.numpy()}")
    print(f"ONNX output:    {onnx_out}")
    print(f"Max difference: {np.abs(pytorch_out.numpy() - onnx_out).max():.6f}")

    # Benchmark PyTorch on all devices
    print("\n" + "=" * 50)
    print(f"Benchmarking PyTorch ({n_runs} runs each)...")
    print("=" * 50)
    pytorch_times = {}
    for device in pytorch_devices:
        try:
            time_ms = benchmark_pytorch(pytorch_model, input_ids.clone(), attention_mask.clone(), device, n_runs)
            pytorch_times[device] = time_ms
            print(f"PyTorch [{device:5}]: {time_ms:6.2f} ms/inference")
        except Exception as e:
            print(f"PyTorch [{device:5}]: Failed - {e}")

    # Benchmark ONNX (CPU only)
    print("\n" + "=" * 50)
    print(f"Benchmarking ONNX ({n_runs} runs, CPU only)...")
    print("=" * 50)
    try:
        onnx_time = benchmark_onnx(onnx_path, input_ids, attention_mask, n_runs)
        print(f"ONNX    [cpu  ]: {onnx_time:6.2f} ms/inference")
    except Exception as e:
        print(f"ONNX    [cpu  ]: Failed - {e}")
        onnx_time = None

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)
    if "cpu" in pytorch_times and onnx_time:
        speedup = pytorch_times["cpu"] / onnx_time
        print(f"ONNX vs PyTorch (CPU): {speedup:.2f}x {'faster' if speedup > 1 else 'slower'}")

    if pytorch_times:
        best_pytorch = min(pytorch_times.items(), key=lambda x: x[1])
        print(f"Best PyTorch: {best_pytorch[1]:.2f} ms ({best_pytorch[0]})")
    if onnx_time:
        print(f"Best ONNX:    {onnx_time:.2f} ms (cpu)")


if __name__ == "__main__":
    main()
