#!/usr/bin/env python3
"""THE QUANT — Hybrid ML bridge: train a cheap model and export .onnx.

This is the OFFLINE training side of the hybrid architecture. It never touches
the live hot path. It generates a synthetic 12-feature dataset mirroring the
Rust feature pipeline (see src/features.rs), trains a small sklearn model, and
exports it to ONNX so Rust can serve it via the `ort` crate.
"""
import os

import numpy as np

N_FEATURES = 12
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def make_data(n=20_000, seed=7):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, N_FEATURES)).astype(np.float32)
    # Learnable target from momentum + trend + volume features (indices 0,1,11,5).
    momentum = 0.35 * X[:, 0] + 0.30 * X[:, 1] + 0.20 * X[:, 11]
    trend = 0.40 * X[:, 5]
    score = momentum + trend + 0.15 * rng.standard_normal(n)
    y = np.where(score > 0.02, 1.0, np.where(score < -0.02, -1.0, 0.0))
    return X, y


def train(n=20_000):
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.metrics import accuracy_score
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType

    X, y = make_data(n)
    model = GradientBoostingClassifier(
        n_estimators=80, max_depth=3, learning_rate=0.08, random_state=1
    )
    model.fit(X, y)
    acc = accuracy_score(y, model.predict(X))
    print(f"[train] fit GradientBoosting, train acc={acc:.3f}")

    initial_type = [("input", FloatTensorType([None, N_FEATURES]))]
    onx = convert_sklearn(model, initial_types=initial_type, target_opset=17)
    out_path = os.path.join(ROOT, "models", "latest.onnx")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        f.write(onx.SerializeToString())
    print(f"[train] exported ONNX -> {out_path}")


if __name__ == "__main__":
    train()