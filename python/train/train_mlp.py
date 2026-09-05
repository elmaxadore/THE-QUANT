#!/usr/bin/env python3
"""THE QUANT — PyTorch MLP Trainer & ONNX Export.

Trains a deep Neural Network (SiLU activations, batch norm, dropout) on GPU/CPU
using PyTorch, matching Rust's 12 feature pipeline (src/features.rs), and exports
the model to `models/mlp.onnx` and `models/latest.onnx`.
"""
import os
import json
import shutil
import numpy as np

N_FEATURES = 12
ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def make_data(n=50_000, seed=42):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    momentum = 0.35 * X[:, 0] + 0.30 * X[:, 1] + 0.20 * X[:, 11]
    trend = 0.40 * X[:, 5]
    score = momentum + trend + 0.15 * rng.standard_normal(n)
    y = (score > 0.0).astype(np.float32).reshape(-1, 1)
    return X, y


def train_mlp(n_samples=50_000, epochs=25, batch_size=512, lr=0.001):
    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError:
        print("[!] PyTorch is not installed. Install with `pip install torch` to run MLP trainer.")
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[MLP Train] Using device: {device} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")

    X, y = make_data(n_samples)
    dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    class QuantMLP(nn.Module):
        def __init__(self):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(N_FEATURES, 64),
                nn.BatchNorm1d(64),
                nn.SiLU(),
                nn.Dropout(0.1),
                nn.Linear(64, 32),
                nn.BatchNorm1d(32),
                nn.SiLU(),
                nn.Linear(32, 1),
                nn.Tanh()
            )
        def forward(self, x):
            return self.net(x)

    model = QuantMLP().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    for epoch in range(1, epochs + 1):
        model.train()
        total_loss = 0.0
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            optimizer.zero_grad()
            out = model(bx)
            loss = criterion(out, by)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * len(bx)
        avg_loss = total_loss / n_samples
        if epoch % 5 == 0 or epoch == 1 or epoch == epochs:
            print(f"[MLP Train] Epoch {epoch:02d}/{epochs} - Loss: {avg_loss:.6f}")

    # Export to ONNX
    model.eval()
    model_cpu = model.to("cpu")
    dummy_input = torch.randn(1, N_FEATURES, dtype=torch.float32)

    models_dir = os.path.join(ROOT, "models")
    os.makedirs(models_dir, exist_ok=True)

    out_onnx = os.path.join(models_dir, "mlp.onnx")
    latest_onnx = os.path.join(models_dir, "latest.onnx")

    torch.onnx.export(
        model_cpu,
        dummy_input,
        out_onnx,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
        opset_version=17
    )
    shutil.copyfile(out_onnx, latest_onnx)
    print(f"[MLP Train] Successfully exported PyTorch ONNX model -> {out_onnx} and {latest_onnx}")


if __name__ == "__main__":
    train_mlp()
