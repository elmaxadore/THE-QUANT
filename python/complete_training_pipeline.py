#!/usr/bin/env python3
"""THE QUANT — Complete Training Pipeline with Real Market Data.

This script orchestrates the full training pipeline:
1. Unsupervised Learning: Market regime detection and anomaly detection
2. Supervised Learning: Multi-strategy training for different regimes
3. Reinforcement Learning: Risk management optimization
4. Backtesting: Walk-forward validation and performance metrics
5. Model Export: ONNX models for Rust integration
"""
import os
import sys
import json
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unsupervised_insights import (
    MarketRegimeDetector, 
    DeepFeatureExtractor, 
    AnomalyDetector,
    fetch_market_data,
    preprocess_features
)

try:
    from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("[!] scikit-learn not available. Install with: pip install scikit-learn")

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

try:
    from skl2onnx import convert_sklearn
    from skl2onnx.common.data_types import FloatTensorType
    ONNX_AVAILABLE = True
except ImportError:
    ONNX_AVAILABLE = False

ROOT = Path(__file__).parent.parent.parent
MODELS_DIR = ROOT / "models"
RESULTS_DIR = ROOT / "results"
DATA_DIR = ROOT / "data"

# Major Assets Configuration for THE QUANT v4.1
MAJOR_ASSETS = {
    "FOREX": [
        "EURUSD=X", "GBPUSD=X", "USDJPY=X", "USDCHF=X", 
        "AUDUSD=X", "USDCAD=X", "NZDUSD=X", "EURGBP=X",
        "EURJPY=X", "GBPJPY=X"
    ],
    "CRYPTO": [
        "BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", 
        "XRPUSDT", "ADAUSDT", "DOGEUSDT", "AVAXUSDT"
    ],
    "INDICES_COMMODITIES": [
        "GC=F",      # Gold
        "SI=F",      # Silver
        "CL=F",      # Crude Oil
        "NG=F",      # Natural Gas
        "^US500",    # S&P 500 (Yahoo proxy)
        "^NDX",      # NASDAQ 100 (Yahoo proxy)
        "^DJI"       # Dow Jones (Yahoo proxy)
    ]
}

def get_all_major_symbols():
    """Get list of all major asset symbols."""
    all_symbols = []
    for category, symbols in MAJOR_ASSETS.items():
        all_symbols.extend(symbols)
    return all_symbols


class StrategyTrainer:
    """Train specialized strategies for different market regimes."""
    
    def __init__(self):
        self.strategies = {}
        self.scalers = {}
        self.regime_models = {}
        
    def train_regime_strategies(
        self,
        X: np.ndarray,
        y: np.ndarray,
        regimes: np.ndarray,
        strategy_types: List[str] = ['trend', 'mean_reversion', 'breakout']
    ) -> Dict[str, Dict]:
        """Train different strategies for each market regime."""
        
        if not SKLEARN_AVAILABLE:
            print("[!] scikit-learn required for strategy training")
            return {}
        
        unique_regimes = np.unique(regimes)
        results = {}
        
        for regime_id in unique_regimes:
            regime_mask = regimes == regime_id
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            if len(X_regime) < 100:
                print(f"[!] Insufficient samples for regime {regime_id}: {len(X_regime)}")
                continue
            
            # Split data temporally
            split_idx = int(len(X_regime) * 0.8)
            X_train, X_test = X_regime[:split_idx], X_regime[split_idx:]
            y_train, y_test = y_regime[:split_idx], y_regime[split_idx:]
            
            # Scale features
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            regime_results = {
                'regime_id': int(regime_id),
                'samples': len(X_regime),
                'strategies': {}
            }
            
            for strategy_type in strategy_types:
                if strategy_type == 'trend':
                    model = GradientBoostingClassifier(
                        n_estimators=100,
                        max_depth=5,
                        learning_rate=0.1,
                        random_state=42
                    )
                elif strategy_type == 'mean_reversion':
                    model = RandomForestClassifier(
                        n_estimators=100,
                        max_depth=7,
                        min_samples_split=10,
                        random_state=42
                    )
                elif strategy_type == 'breakout':
                    model = GradientBoostingClassifier(
                        n_estimators=150,
                        max_depth=4,
                        learning_rate=0.05,
                        subsample=0.8,
                        random_state=42
                    )
                else:
                    continue
                
                model.fit(X_train_scaled, y_train)
                y_pred = model.predict(X_test_scaled)
                
                # Calculate metrics
                metrics = {
                    'accuracy': accuracy_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred, average='weighted', zero_division=0),
                    'recall': recall_score(y_test, y_pred, average='weighted', zero_division=0),
                    'f1': f1_score(y_test, y_pred, average='weighted', zero_division=0)
                }
                
                # Store model
                key = f"regime_{regime_id}_{strategy_type}"
                self.strategies[key] = model
                self.scalers[key] = scaler
                
                regime_results['strategies'][strategy_type] = metrics
                print(f"✓ Trained {strategy_type} strategy for regime {regime_id}: F1={metrics['f1']:.3f}")
            
            results[f'regime_{regime_id}'] = regime_results
        
        return results
    
    def export_strategies(self, output_dir: Path):
        """Export trained strategies to ONNX format."""
        if not ONNX_AVAILABLE:
            print("[!] ONNX export not available")
            return
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for key, model in self.strategies.items():
            try:
                initial_type = [('float_input', FloatTensorType([None, model.n_features_in_]))]
                onnx_model = convert_sklearn(model, initial_types=initial_type)
                
                filename = f"{key}.onnx"
                filepath = output_dir / filename
                with open(filepath, "wb") as f:
                    f.write(onnx_model.SerializeToString())
                
                print(f"✓ Exported strategy: {filename}")
            except Exception as e:
                print(f"[!] Failed to export {key}: {e}")


class RiskManagerRL:
    """Reinforcement Learning for dynamic risk management."""
    
    def __init__(self, state_dim: int = 10, action_dim: int = 3):
        self.state_dim = state_dim
        self.action_dim = action_dim  # Reduce position, Hold, Increase position
        self.policy_network = None
        
    def create_policy_network(self, hidden_dims: List[int] = [64, 32]) -> nn.Module:
        """Create RL policy network for position sizing."""
        if not TORCH_AVAILABLE:
            print("[!] PyTorch required for RL")
            return None
        
        layers = []
        prev_dim = self.state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, self.action_dim))
        layers.append(nn.Softmax(dim=-1))
        
        self.policy_network = nn.Sequential(*layers)
        return self.policy_network
    
    def train_policy(
        self,
        states: np.ndarray,
        rewards: np.ndarray,
        episodes: int = 100,
        lr: float = 0.001
    ) -> List[float]:
        """Train policy using REINFORCE algorithm."""
        if not TORCH_AVAILABLE or self.policy_network is None:
            return []
        
        optimizer = optim.Adam(self.policy_network.parameters(), lr=lr)
        episode_rewards = []
        
        # Set network to training mode
        self.policy_network.train()
        
        for episode in range(episodes):
            episode_reward = 0
            log_probs = []
            
            # Sample mini-batch trajectory (at least 2 samples for BatchNorm)
            batch_size = min(32, len(states))
            indices = np.random.choice(len(states), size=batch_size, replace=False)
            
            for idx in indices:
                state = torch.FloatTensor(states[idx]).unsqueeze(0)
                probs = self.policy_network(state)
                dist = torch.distributions.Categorical(probs)
                action = dist.sample()
                log_prob = dist.log_prob(action)
                log_probs.append(log_prob)
                
                episode_reward += rewards[idx]
            
            # Update policy
            optimizer.zero_grad()
            policy_loss = (-torch.stack(log_probs) * episode_reward).mean()
            policy_loss.backward()
            optimizer.step()
            
            episode_rewards.append(episode_reward)
            
            if episode % 10 == 0:
                print(f"  Episode {episode}: Avg Reward = {episode_reward:.3f}")
        
        return episode_rewards
    
    def get_position_size(self, state: np.ndarray, base_size: float = 1.0) -> float:
        """Get optimal position size based on current state."""
        if not TORCH_AVAILABLE or self.policy_network is None:
            return base_size
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0)
            probs = self.policy_network(state_tensor)
            action = torch.argmax(probs).item()
        
        # Action mapping: 0=Reduce (0.5x), 1=Hold (1.0x), 2=Increase (1.5x)
        multipliers = [0.5, 1.0, 1.5]
        return base_size * multipliers[action]


def backtest_strategies(
    X: np.ndarray,
    y: np.ndarray,
    regimes: np.ndarray,
    trainer: StrategyTrainer,
    transaction_cost: float = 0.001
) -> Dict:
    """Perform walk-forward backtest of regime-aware strategies."""
    
    tscv = TimeSeriesSplit(n_splits=5)
    equity_curve = [10000]  # Start with $10,000
    trades = []
    
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        regimes_test = regimes[test_idx]
        
        # Determine dominant regime in test set
        dominant_regime = np.bincount(regimes_test).argmax()
        
        # Select best strategy for this regime
        best_strategy_key = f"regime_{dominant_regime}_trend"
        if best_strategy_key not in trainer.strategies:
            best_strategy_key = f"regime_{dominant_regime}_mean_reversion"
        
        if best_strategy_key not in trainer.strategies:
            continue
        
        model = trainer.strategies[best_strategy_key]
        scaler = trainer.scalers[best_strategy_key]
        
        X_test_scaled = scaler.transform(X_test)
        predictions = model.predict(X_test_scaled)
        
        # Simulate trading
        for i, pred in enumerate(predictions):
            if pred == 0:  # No trade
                continue
            
            actual_return = y_test[i]
            trade_return = pred * actual_return - transaction_cost
            
            new_equity = equity_curve[-1] * (1 + trade_return)
            equity_curve.append(new_equity)
            
            trades.append({
                'fold': fold,
                'prediction': pred,
                'actual': actual_return,
                'return': trade_return,
                'equity': new_equity
            })
    
    # Calculate metrics
    if len(equity_curve) < 2:
        return {'error': 'Insufficient trades'}
    
    returns = np.diff(equity_curve) / equity_curve[:-1]
    
    metrics = {
        'total_trades': len(trades),
        'final_equity': equity_curve[-1],
        'total_return': (equity_curve[-1] - 10000) / 10000,
        'sharpe_ratio': np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0,
        'max_drawdown': min(0, (np.minimum.accumulate(np.maximum.accumulate(equity_curve) - equity_curve) / np.maximum.accumulate(equity_curve)).min()),
        'win_rate': len([t for t in trades if t['return'] > 0]) / len(trades) if trades else 0
    }
    
    return metrics


def run_complete_pipeline(
    symbols: List[str] = ['BTCUSDT'],
    start_date: str = '2023-01-01',
    end_date: str = '2024-12-01',
    n_regimes: int = 4
):
    """Execute the complete training and validation pipeline."""
    
    print("="*70)
    print("THE QUANT v4.1 - COMPLETE TRAINING PIPELINE")
    print("="*70)
    
    # Create directories
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    
    all_results = {
        'timestamp': datetime.now().isoformat(),
        'symbols': {},
        'summary': {}
    }
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Processing: {symbol}")
        print(f"{'='*60}")
        
        # Step 1: Fetch Data
        print("\n[1/6] Fetching market data...")
        df = fetch_market_data(symbol, start_date, end_date)
        
        if df is None or len(df) < 200:
            print(f"[!] Insufficient data for {symbol}")
            continue
        
        print(f"✓ Loaded {len(df)} candles")
        
        # Step 2: Feature Engineering
        print("\n[2/6] Engineering features...")
        X, feature_names = preprocess_features(df)
        
        if X is None or 'target' not in df.columns:
            print("[!] Target variable not found, creating from price data")
            # Create target from returns
            df['returns'] = df['close'].pct_change(5)
            df['target'] = (df['returns'] > 0.001).astype(int) - (df['returns'] < -0.001).astype(int)
            df = df.dropna()
            X, feature_names = preprocess_features(df)
        
        if X is None:
            continue
        
        y = df['target'].values[-len(X):]
        print(f"✓ Created {X.shape[1]} features")
        
        # Step 3: Unsupervised Learning (Regime Detection)
        print("\n[3/6] Detecting market regimes...")
        regime_detector = MarketRegimeDetector(n_regimes=n_regimes)
        regime_detector.fit(X)
        regimes = regime_detector.regime_labels
        
        regime_stats = regime_detector.analyze_regimes(y, np.std(X, axis=1))
        print(f"✓ Detected {n_regimes} regimes")
        for regime_id in range(n_regimes):
            if f'regime_{regime_id}' in regime_stats:
                stats = regime_stats[f'regime_{regime_id}']
                print(f"  Regime {regime_id}: {stats.get('occurrence_pct', 0):.1f}% occurrence")
        
        # Step 4: Supervised Learning (Strategy Training)
        print("\n[4/6] Training regime-specific strategies...")
        trainer = StrategyTrainer()
        strategy_results = trainer.train_regime_strategies(X, y, regimes)
        
        # Export strategies
        strategy_dir = MODELS_DIR / "strategies" / symbol
        trainer.export_strategies(strategy_dir)
        
        # Step 5: Reinforcement Learning (Risk Management)
        print("\n[5/6] Optimizing risk management with RL...")
        
        # Convert X to numpy if it's a DataFrame
        X_np = X.values if hasattr(X, 'values') else X
        
        # Ensure state_features has the right shape - use first 10 features consistently
        n_features_for_rl = min(10, X_np.shape[1]) if len(X_np.shape) > 1 else 10
        rl_manager = RiskManagerRL(state_dim=n_features_for_rl)
        rl_manager.create_policy_network()
        
        # Create simplified state representation - exactly 10 features
        if len(X_np.shape) > 1:
            state_features = X_np[:, :n_features_for_rl]
            # Pad if less than 10 features
            if state_features.shape[1] < 10:
                padding = np.zeros((state_features.shape[0], 10 - state_features.shape[1]))
                state_features = np.hstack([state_features, padding])
        else:
            state_features = X_np.reshape(-1, 1)
            if state_features.shape[1] < 10:
                padding = np.zeros((state_features.shape[0], 10 - state_features.shape[1]))
                state_features = np.hstack([state_features, padding])
        
        simple_rewards = np.abs(y)  # Use absolute returns as reward signal
        
        rl_rewards = rl_manager.train_policy(state_features, simple_rewards, episodes=50)
        print(f"✓ RL Policy trained, final avg reward: {np.mean(rl_rewards[-10:]):.3f}")
        
        # Step 6: Backtesting
        print("\n[6/6] Backtesting strategies...")
        backtest_results = backtest_strategies(X_np, y, regimes, trainer)
        
        print(f"\nBacktest Results:")
        print(f"  Total Trades: {backtest_results.get('total_trades', 0)}")
        print(f"  Total Return: {backtest_results.get('total_return', 0)*100:.2f}%")
        print(f"  Sharpe Ratio: {backtest_results.get('sharpe_ratio', 0):.2f}")
        print(f"  Max Drawdown: {backtest_results.get('max_drawdown', 0)*100:.2f}%")
        print(f"  Win Rate: {backtest_results.get('win_rate', 0)*100:.1f}%")
        
        # Save results
        all_results['symbols'][symbol] = {
            'regime_stats': regime_stats,
            'strategy_results': strategy_results,
            'backtest': backtest_results,
            'n_samples': len(X),
            'n_features': X.shape[1]
        }
    
    # Save comprehensive results
    results_file = RESULTS_DIR / "training_results.json"
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("PIPELINE COMPLETE")
    print(f"{'='*70}")
    print(f"Results saved to: {results_file}")
    print(f"Models saved to: {MODELS_DIR}")
    
    # Summary
    print("\nSUMMARY:")
    for symbol, data in all_results['symbols'].items():
        bt = data.get('backtest', {})
        print(f"  {symbol}:")
        print(f"    - Return: {bt.get('total_return', 0)*100:.2f}%")
        print(f"    - Sharpe: {bt.get('sharpe_ratio', 0):.2f}")
        print(f"    - Trades: {bt.get('total_trades', 0)}")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Complete Training Pipeline")
    parser.add_argument('--symbols', nargs='+', default=None, help='Symbols to train (default: all major assets)')
    parser.add_argument('--start_date', default='2023-01-01', help='Start date')
    parser.add_argument('--end_date', default='2024-12-01', help='End date')
    parser.add_argument('--n_regimes', type=int, default=4, help='Number of market regimes')
    parser.add_argument('--all_major', action='store_true', help='Train on all major assets')
    
    args = parser.parse_args()
    
    # Determine symbols to use
    if args.all_major or args.symbols is None:
        symbols_to_train = get_all_major_symbols()
        print(f"Training on all {len(symbols_to_train)} major assets...")
    else:
        symbols_to_train = args.symbols
    
    results = run_complete_pipeline(
        symbols=symbols_to_train,
        start_date=args.start_date,
        end_date=args.end_date,
        n_regimes=args.n_regimes
    )
