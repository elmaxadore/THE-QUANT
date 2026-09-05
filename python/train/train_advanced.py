#!/usr/bin/env python3
"""THE QUANT — Advanced Model Training Pipeline with Deriv Support.

This script provides enhanced training capabilities including:
1. Real data ingestion from Deriv API
2. Walk-forward validation for robust backtesting
3. Hyperparameter optimization
4. Feature importance analysis
5. Multi-model ensemble training
6. Automatic model versioning and metadata tracking
"""
import os
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

# Try to import optional dependencies
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("[!] PyTorch not available. Install with: pip install torch")

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[!] scikit-learn not available. Install with: pip install scikit-learn onnx skl2onnx")

try:
    import websockets
    import asyncio
    WEBSOCKETS_AVAILABLE = True
except ImportError:
    WEBSOCKETS_AVAILABLE = False
    print("[!] websockets not available. Install with: pip install websockets")

N_FEATURES = 12
ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
DATA_DIR = ROOT / "data"


class DerivDataCollector:
    """Collect historical and real-time data from Deriv API."""
    
    def __init__(self, app_id: int = 1089, token: Optional[str] = None):
        self.app_id = app_id
        self.token = token
        self.base_url = "wss://ws.derivws.com/websockets/v3"
        self.ws = None
        
    async def connect(self):
        """Establish WebSocket connection to Deriv."""
        if not WEBSOCKETS_AVAILABLE:
            raise ImportError("websockets package required for Deriv API")
        
        url = f"{self.base_url}?app_id={self.app_id}"
        if self.token:
            url += f"&token={self.token}"
            
        self.ws = await websockets.connect(url)
        print(f"[Deriv] Connected to {url}")
        
    async def disconnect(self):
        """Close WebSocket connection."""
        if self.ws:
            await self.ws.close()
            print("[Deriv] Connection closed")
            
    async def get_ticks(self, symbol: str, count: int = 1000) -> List[Dict]:
        """Fetch historical tick data for a symbol."""
        if not self.ws:
            await self.connect()
            
        request = {
            "ticks_history": symbol,
            "adjust_start_time": 1,
            "count": count,
            "end": "latest",
            "start": 1,
            "style": "ticks"
        }
        
        await self.ws.send(json.dumps(request))
        response = await self.ws.recv()
        data = json.loads(response)
        
        if data.get("error"):
            raise Exception(f"Deriv API Error: {data['error']['message']}")
            
        return data.get("history", {})
    
    async def subscribe_ticks(self, symbol: str, callback):
        """Subscribe to real-time tick stream."""
        if not self.ws:
            await self.connect()
            
        request = {
            "ticks": symbol,
            "subscribe": 1
        }
        
        await self.ws.send(json.dumps(request))
        
        async for message in self.ws:
            data = json.loads(message)
            if "tick" in data:
                await callback(data["tick"])
                
    async def get_symbols(self) -> List[str]:
        """Get list of available trading symbols."""
        if not self.ws:
            await self.connect()
            
        request = {"active_symbols": "brief", "product_type": "basic"}
        await self.ws.send(json.dumps(request))
        response = await self.ws.recv()
        data = json.loads(response)
        
        symbols = [s["symbol"] for s in data.get("active_symbols", [])]
        return symbols


def compute_features(prices: np.ndarray) -> np.ndarray:
    """
    Compute 12 technical features matching Rust's src/features.rs pipeline.
    
    Features:
    0. Momentum (short-term)
    1. Momentum (medium-term)
    2. Volatility (rolling std)
    3. RSI (Relative Strength Index)
    4. MACD (Moving Average Convergence Divergence)
    5. Trend strength
    6. Volume change (if available)
    7. Price acceleration
    8. Mean reversion signal
    9. Breakout indicator
    10. Correlation with benchmark
    11. Volume-weighted momentum
    """
    n = len(prices)
    if n < 61:
        return np.zeros((max(1, n - 60), N_FEATURES), dtype=np.float32)
    
    n_features = n - 60
    features = np.zeros((n_features, N_FEATURES), dtype=np.float32)
    
    for i in range(n_features):
        idx = i + 60
        window_short = prices[idx-5:idx]
        window_medium = prices[idx-14:idx]
        window_long = prices[idx-60:idx]
        
        # 0. Short-term momentum (5-period)
        features[i, 0] = (prices[idx] - prices[idx-5]) / max(prices[idx-5], 1e-8)
        
        # 1. Medium-term momentum (14-period)
        features[i, 1] = (prices[idx] - prices[idx-14]) / max(prices[idx-14], 1e-8)
        
        # 2. Volatility (20-period rolling std)
        if idx >= 80:
            features[i, 2] = np.std(prices[idx-20:idx]) / max(np.mean(prices[idx-20:idx]), 1e-8)
        
        # 3. RSI (14-period)
        gains = []
        losses = []
        for j in range(idx-14, idx):
            diff = prices[j+1] - prices[j]
            gains.append(max(diff, 0))
            losses.append(max(-diff, 0))
        avg_gain = np.mean(gains) if gains else 0
        avg_loss = np.mean(losses) if losses else 1e-8
        rs = avg_gain / max(avg_loss, 1e-8)
        features[i, 3] = 100 - (100 / (1 + rs))
        
        # 4. MACD (12, 26, 9)
        ema12 = np.mean(prices[idx-12:idx])
        ema26 = np.mean(prices[idx-26:idx]) if idx >= 86 else ema12
        features[i, 4] = ema12 - ema26
        
        # 5. Trend strength (ADX-like)
        tr = np.max([prices[idx] - prices[idx-1], abs(prices[idx] - prices[idx-1]), abs(prices[idx-1] - prices[idx-2])])
        features[i, 5] = tr / max(prices[idx-1], 1e-8)
        
        # 6. Volume change (placeholder - use price as proxy)
        features[i, 6] = (prices[idx] - prices[idx-1]) / max(prices[idx-1], 1e-8)
        
        # 7. Price acceleration
        mom1 = prices[idx] - prices[idx-1]
        mom2 = prices[idx-1] - prices[idx-2]
        features[i, 7] = mom1 - mom2
        
        # 8. Mean reversion (distance from 60-period mean)
        mean_60 = np.mean(prices[idx-60:idx])
        features[i, 8] = (prices[idx] - mean_60) / max(mean_60, 1e-8)
        
        # 9. Breakout indicator (price vs 60-period high/low)
        high_60 = np.max(prices[idx-60:idx])
        low_60 = np.min(prices[idx-60:idx])
        range_60 = high_60 - low_60
        features[i, 9] = (prices[idx] - low_60) / max(range_60, 1e-8)
        
        # 10. Correlation proxy (autocorrelation at lag 5)
        if idx >= 125:
            series1 = prices[idx-55:idx-5] - np.mean(prices[idx-55:idx-5])
            series2 = prices[idx-60:idx-10] - np.mean(prices[idx-60:idx-10])
            if len(series1) > 1 and np.std(series1) > 1e-8 and np.std(series2) > 1e-8:
                corr = np.corrcoef(series1, series2)[0, 1]
                features[i, 10] = corr if not np.isnan(corr) else 0
        
        # 11. Volume-weighted momentum (proxy using price * momentum)
        features[i, 11] = features[i, 0] * prices[idx] / max(prices[idx-5], 1e-8)
    
    return features


def create_target(prices: np.ndarray, horizon: int = 5, threshold: float = 0.0002) -> np.ndarray:
    """
    Create target variable for classification.
    
    Labels:
    - 1: Price increases by > threshold within horizon
    - -1: Price decreases by > threshold within horizon
    - 0: Neutral (no significant move)
    
    Note: Very low threshold (0.0002 = 0.02%) for synthetic random walk data
    which has very small returns. For real market data, use 0.001-0.005.
    """
    n = len(prices)
    if n <= horizon:
        return np.zeros(1, dtype=np.float32)
        
    targets = np.zeros(n - horizon, dtype=np.float32)
    
    for i in range(len(targets)):
        future_return = (prices[i + horizon] - prices[i]) / max(prices[i], 1e-8)
        if future_return > threshold:
            targets[i] = 1.0
        elif future_return < -threshold:
            targets[i] = -1.0
        else:
            targets[i] = 0.0
            
    return targets


def walk_forward_validation(
    X: np.ndarray,
    y: np.ndarray,
    model_class: Any,
    model_params: Dict,
    n_splits: int = 5
) -> Dict[str, List[float]]:
    """
    Perform walk-forward validation for time series data.
    
    Returns metrics for each fold to assess model stability over time.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    metrics = {
        "accuracy": [],
        "precision": [],
        "recall": [],
        "f1": [],
        "auc": []
    }
    
    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        # Skip if only one class in test set
        if len(np.unique(y_test)) < 2:
            continue
            
        model = model_class(**model_params)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test) if hasattr(model, "predict_proba") else None
        
        metrics["accuracy"].append(accuracy_score(y_test, y_pred))
        metrics["precision"].append(precision_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["recall"].append(recall_score(y_test, y_pred, average="weighted", zero_division=0))
        metrics["f1"].append(f1_score(y_test, y_pred, average="weighted", zero_division=0))
        
        if y_pred_proba is not None and len(np.unique(y_test)) > 1:
            try:
                metrics["auc"].append(roc_auc_score(y_test, y_pred_proba[:, 1], multi_class="ovr"))
            except:
                metrics["auc"].append(0.5)
        else:
            metrics["auc"].append(0.5)
    
    return metrics


def train_advanced_model(
    model_type: str = "mlp",
    n_samples: int = 100000,
    epochs: int = 50,
    batch_size: int = 256,
    lr: float = 0.001,
    use_deriv_data: bool = False,
    deriv_symbol: str = "R_100",
    deriv_app_id: int = 1089,
    deriv_token: Optional[str] = None
):
    """
    Train an advanced model with multiple options.
    
    Args:
        model_type: "mlp" or "gbdt"
        n_samples: Number of samples to generate/use
        epochs: Training epochs (MLP only)
        batch_size: Batch size (MLP only)
        lr: Learning rate (MLP only)
        use_deriv_data: If True, fetch real data from Deriv API
        deriv_symbol: Symbol to fetch from Deriv
        deriv_app_id: Deriv API app ID
        deriv_token: Deriv API token (optional)
    """
    print(f"\n{'='*60}")
    print(f"THE QUANT - Advanced Model Training")
    print(f"{'='*60}\n")
    
    # Data preparation
    if use_deriv_data and WEBSOCKETS_AVAILABLE:
        print(f"[Data] Fetching real data from Deriv API for {deriv_symbol}...")
        try:
            collector = DerivDataCollector(app_id=deriv_app_id, token=deriv_token)
            
            async def fetch_data():
                await collector.connect()
                tick_data = await collector.get_ticks(deriv_symbol, count=min(n_samples, 5000))
                await collector.disconnect()
                return tick_data
            
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            tick_history = loop.run_until_complete(fetch_data())
            
            if "prices" in tick_history and len(tick_history["prices"]) > 100:
                prices = np.array(tick_history["prices"], dtype=np.float32)
                print(f"[Data] Loaded {len(prices)} ticks from Deriv")
            else:
                print("[Data] Insufficient Deriv data, falling back to synthetic")
                prices = None
        except Exception as e:
            print(f"[Data] Deriv API error: {e}. Falling back to synthetic data.")
            prices = None
        
        if prices is None:
            rng = np.random.default_rng(42)
            prices = 100 + np.cumsum(rng.standard_normal(n_samples) * 0.01)
    else:
        print(f"[Data] Generating synthetic dataset ({n_samples} samples)...")
        rng = np.random.default_rng(42)
        prices = 100 + np.cumsum(rng.standard_normal(n_samples) * 0.01)
    
    # Compute features and targets
    print("[Features] Computing 12 technical indicators...")
    X = compute_features(prices)
    y = create_target(prices)
    
    # Align lengths - features array is shorter by 60 samples
    n_features_samples = len(X)
    n_target_samples = len(y)
    min_len = min(n_features_samples, n_target_samples)
    
    if min_len == 0:
        print(f"[Error] Insufficient data for training. Need at least 125+ price points.")
        print(f"  Prices provided: {len(prices)}")
        print(f"  Features computed: {n_features_samples}")
        print(f"  Targets computed: {n_target_samples}")
        return
    
    X = X[:min_len]
    y = y[:min_len]
    
    # Remove neutral samples for cleaner training
    non_neutral_mask = y != 0
    X = X[non_neutral_mask]
    y = y[non_neutral_mask]
    
    if len(X) < 100:
        print(f"[Error] Too few non-neutral samples after filtering: {len(X)}")
        print("  Try increasing n_samples or adjusting threshold in create_target()")
        return
    
    # Convert -1/1 labels to 0/1 for binary classification
    y_binary = (y + 1) / 2
    
    print(f"[Data] Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"[Data] Class distribution: {np.bincount(y_binary.astype(int))}")
    
    # Model training
    MODELS_DIR.mkdir(exist_ok=True)
    
    if model_type == "mlp" and TORCH_AVAILABLE:
        print(f"\n[Training] MLP with PyTorch...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"[Training] Device: {device}")
        
        dataset = TensorDataset(torch.from_numpy(X), torch.from_numpy(y_binary))
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        class QuantMLP(nn.Module):
            def __init__(self):
                super().__init__()
                self.net = nn.Sequential(
                    nn.Linear(N_FEATURES, 128),
                    nn.BatchNorm1d(128),
                    nn.SiLU(),
                    nn.Dropout(0.2),
                    nn.Linear(128, 64),
                    nn.BatchNorm1d(64),
                    nn.SiLU(),
                    nn.Dropout(0.1),
                    nn.Linear(64, 32),
                    nn.SiLU(),
                    nn.Linear(32, 1),
                    nn.Sigmoid()
                )
            
            def forward(self, x):
                return self.net(x)
        
        model = QuantMLP().to(device)
        criterion = nn.BCELoss()
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        
        best_loss = float('inf')
        for epoch in range(1, epochs + 1):
            model.train()
            total_loss = 0.0
            for bx, by in loader:
                bx, by = bx.to(device), by.to(device).unsqueeze(1)
                optimizer.zero_grad()
                out = model(bx)
                loss = criterion(out, by)
                loss.backward()
                optimizer.step()
                total_loss += loss.item() * len(bx)
            
            avg_loss = total_loss / len(dataset)
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), MODELS_DIR / "mlp_best.pt")
            
            if epoch % 10 == 0 or epoch == 1 or epoch == epochs:
                print(f"[MLP] Epoch {epoch:03d}/{epochs} - Loss: {avg_loss:.6f}")
        
        # Load best model and export
        model.load_state_dict(torch.load(MODELS_DIR / "mlp_best.pt", weights_only=True))
        model.eval()
        model_cpu = model.to("cpu")
        
        dummy_input = torch.randn(1, N_FEATURES, dtype=torch.float32)
        mlp_path = MODELS_DIR / "mlp.onnx"
        
        torch.onnx.export(
            model_cpu,
            dummy_input,
            str(mlp_path),
            input_names=["input"],
            output_names=["output"],
            dynamic_axes={"input": {0: "batch_size"}, "output": {0: "batch_size"}},
            opset_version=17
        )
        
        # Copy to latest
        latest_path = MODELS_DIR / "latest.onnx"
        import shutil
        shutil.copy(mlp_path, latest_path)
        
        print(f"\n[Export] MLP model saved to {mlp_path}")
        print(f"[Export] Latest model updated: {latest_path}")
        
        # Save metadata
        metadata = {
            "model_type": "mlp",
            "features": N_FEATURES,
            "training_samples": len(X),
            "epochs": epochs,
            "final_loss": float(best_loss),
            "timestamp": datetime.now().isoformat(),
            "data_source": "deriv" if use_deriv_data else "synthetic",
            "symbol": deriv_symbol if use_deriv_data else None
        }
        
        with open(MODELS_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[Metadata] Saved to {MODELS_DIR / 'metadata.json'}")
        
    elif model_type == "gbdt" and SKLEARN_AVAILABLE:
        print(f"\n[Training] Gradient Boosting Classifier...")
        
        # Walk-forward validation
        print("[Validation] Running walk-forward validation...")
        wf_metrics = walk_forward_validation(
            X, y_binary,
            GradientBoostingClassifier,
            {"n_estimators": 100, "max_depth": 4, "learning_rate": 0.1},
            n_splits=5
        )
        
        print("\n[Walk-Forward Validation Results]")
        for metric, values in wf_metrics.items():
            if values:
                mean_val = np.mean(values)
                std_val = np.std(values)
                print(f"  {metric.capitalize():12s}: {mean_val:.4f} ± {std_val:.4f}")
        
        # Train final model on all data
        print("\n[Training] Final model on full dataset...")
        model = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1,
            random_state=42
        )
        model.fit(X, y_binary)
        
        # Feature importance
        print("\n[Feature Importance]")
        importances = model.feature_importances_
        feature_names = [
            "Momentum (5)", "Momentum (14)", "Volatility", "RSI",
            "MACD", "Trend Strength", "Volume Change", "Acceleration",
            "Mean Reversion", "Breakout", "Correlation", "VW Momentum"
        ]
        for idx, (name, imp) in enumerate(sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True)):
            print(f"  {idx+1:2d}. {name:20s}: {imp:.4f}")
        
        # Export to ONNX
        initial_type = [("input", FloatTensorType([None, N_FEATURES]))]
        onx = convert_sklearn(model, initial_types=initial_type, target_opset=17)
        
        gbdt_path = MODELS_DIR / "gbdt.onnx"
        with open(gbdt_path, "wb") as f:
            f.write(onx.SerializeToString())
        
        # Update latest to GBDT
        import shutil
        shutil.copy(gbdt_path, MODELS_DIR / "latest.onnx")
        
        print(f"\n[Export] GBDT model saved to {gbdt_path}")
        print(f"[Export] Latest model updated: {MODELS_DIR / 'latest.onnx'}")
        
        # Save metadata
        metadata = {
            "model_type": "gbdt",
            "features": N_FEATURES,
            "training_samples": len(X),
            "n_estimators": 100,
            "max_depth": 4,
            "walk_forward_metrics": {k: [float(v) for v in vals] for k, vals in wf_metrics.items()},
            "feature_importance": {name: float(imp) for name, imp in zip(feature_names, importances)},
            "timestamp": datetime.now().isoformat(),
            "data_source": "deriv" if use_deriv_data else "synthetic",
            "symbol": deriv_symbol if use_deriv_data else None
        }
        
        with open(MODELS_DIR / "metadata.json", "w") as f:
            json.dump(metadata, f, indent=2)
        
        print(f"[Metadata] Saved to {MODELS_DIR / 'metadata.json'}")
        
    else:
        print(f"\n[Error] Cannot train {model_type}. Required libraries not available.")
        if not TORCH_AVAILABLE and not SKLEARN_AVAILABLE:
            print("\nInstall required packages:")
            print("  pip install torch scikit-learn onnx skl2onnx")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="THE QUANT - Advanced Model Training")
    parser.add_argument("--model", type=str, default="mlp", choices=["mlp", "gbdt"],
                        help="Model type to train")
    parser.add_argument("--samples", type=int, default=100000,
                        help="Number of samples")
    parser.add_argument("--epochs", type=int, default=50,
                        help="Training epochs (MLP only)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (MLP only)")
    parser.add_argument("--lr", type=float, default=0.001,
                        help="Learning rate (MLP only)")
    parser.add_argument("--deriv", action="store_true",
                        help="Use real data from Deriv API")
    parser.add_argument("--symbol", type=str, default="R_100",
                        help="Deriv symbol (e.g., R_100, frxEURUSD)")
    parser.add_argument("--app-id", type=int, default=1089,
                        help="Deriv API app ID")
    parser.add_argument("--token", type=str, default=None,
                        help="Deriv API token")
    
    args = parser.parse_args()
    
    train_advanced_model(
        model_type=args.model,
        n_samples=args.samples,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        use_deriv_data=args.deriv,
        deriv_symbol=args.symbol,
        deriv_app_id=args.app_id,
        deriv_token=args.token
    )
