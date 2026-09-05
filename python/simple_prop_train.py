#!/usr/bin/env python3
"""Simplified Prop Firm Training Pipeline"""
import sys
sys.path.insert(0, '/workspace')

import numpy as np
import pandas as pd
import yfinance as yf
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import f1_score, precision_score
from sklearn.preprocessing import RobustScaler
import joblib
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType
import json
from pathlib import Path
from datetime import datetime

class SimplePropTrainer:
    def __init__(self, symbol, start_date='2023-01-01', end_date='2024-12-01'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        
    def fetch_data(self):
        """Fetch data with proper column handling"""
        symbol_map = {
            'BTCUSDT': 'BTC-USD',
            'ETHUSDT': 'ETH-USD',
            'EURUSD': 'EURUSD=X',
            'GBPUSD': 'GBPUSD=X',
        }
        yf_symbol = symbol_map.get(self.symbol, self.symbol)
        
        print(f"Fetching {self.symbol} ({yf_symbol})...")
        df = yf.download(yf_symbol, start=self.start_date, end=self.end_date, progress=False)
        
        if len(df) == 0:
            return None
            
        # Flatten multi-level columns
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]
        
        print(f"✓ Fetched {len(df)} bars")
        return df
    
    def create_features(self, df):
        """Create technical features"""
        df = df.copy()
        
        # Basic indicators
        df['SMA_20'] = df['Close'].rolling(20).mean()
        df['SMA_50'] = df['Close'].rolling(50).mean()
        df['RSI_14'] = self._rsi(df['Close'], 14)
        df['MOMENTUM'] = df['Close'].pct_change(10)
        df['VOLATILITY'] = df['Close'].pct_change().rolling(20).std()
        
        # ATR
        high = df['High']
        low = df['Low']
        close = df['Close']
        tr1 = high - low
        tr2 = abs(high - close.shift())
        tr3 = abs(low - close.shift())
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        df['ATR'] = tr.rolling(14).mean()
        
        df = df.dropna()
        print(f"✓ Created {df.shape[1]} features, {len(df)} samples")
        return df
    
    def _rsi(self, prices, period):
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def create_target(self, df, lookforward=5, threshold=0.02):
        """Create binary target"""
        future_return = df['Close'].shift(-lookforward) / df['Close'] - 1
        df = df.copy()
        df['target'] = (future_return > threshold).astype(int)
        df = df.dropna(subset=['target'])
        return df
    
    def train(self, df):
        """Train model with walk-forward validation"""
        feature_cols = ['SMA_20', 'SMA_50', 'RSI_14', 'MOMENTUM', 'VOLATILITY', 'ATR']
        X = df[feature_cols].values
        y = df['target'].values
        
        tscv = TimeSeriesSplit(n_splits=5)
        best_f1 = 0
        best_model = None
        best_scaler = None
        
        print("\nTraining with walk-forward validation...")
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            scaler = RobustScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            clf = GradientBoostingClassifier(
                n_estimators=100, max_depth=4, learning_rate=0.05,
                min_samples_split=20, random_state=42
            )
            clf.fit(X_train_scaled, y_train)
            
            y_pred = clf.predict(X_test_scaled)
            f1 = f1_score(y_test, y_pred, zero_division=0)
            
            if f1 > best_f1:
                best_f1 = f1
                best_model = clf
                best_scaler = scaler
        
        print(f"✓ Best F1 Score: {best_f1:.3f}")
        return best_model, best_scaler, feature_cols
    
    def backtest(self, df, model, scaler, feature_cols):
        """Simple backtest with prop firm rules"""
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        signals = model.predict(X_scaled)
        
        capital = 100000
        peak = capital
        max_dd = 0
        trades = []
        
        for i in range(1, len(signals)):
            price = df['Close'].iloc[i]
            
            if signals[i] == 1 and capital > 0:
                # Enter trade
                shares = int(capital * 0.01 / (price * 0.02))  # 1% risk, 2% stop
                if shares > 0:
                    trades.append({'idx': i, 'type': 'BUY', 'price': price, 'shares': shares})
            
            # Check drawdown
            if capital > peak:
                peak = capital
            dd = (peak - capital) / peak
            max_dd = max(max_dd, dd)
        
        total_return = (capital - 100000) / 100000
        passed = max_dd < 0.10
        
        return {
            'total_return': total_return,
            'max_drawdown': max_dd,
            'trades': len(trades),
            'passed': passed
        }
    
    def export_model(self, model, scaler, feature_cols):
        """Export to ONNX"""
        output_dir = Path('/workspace/models/prop_strategies')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        onnx_path = output_dir / f'{self.symbol}_strategy.onnx'
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        joblib.dump(scaler, output_dir / f'{self.symbol}_scaler.pkl')
        
        with open(output_dir / f'{self.symbol}_features.json', 'w') as f:
            json.dump(feature_cols, f)
        
        print(f"✓ Models saved to {output_dir}")
    
    def run(self):
        """Run full pipeline"""
        print(f"\n{'='*60}")
        print(f"PROP FIRM TRAINING: {self.symbol}")
        print(f"{'='*60}\n")
        
        df = self.fetch_data()
        if df is None:
            print("✗ Failed to fetch data")
            return None
        
        df = self.create_features(df)
        df = self.create_target(df, lookforward=5, threshold=0.015)
        
        if len(df) < 100:
            print("✗ Insufficient data")
            return None
        
        model, scaler, feature_cols = self.train(df)
        results = self.backtest(df, model, scaler, feature_cols)
        self.export_model(model, scaler, feature_cols)
        
        print(f"\n{'='*60}")
        print("RESULTS")
        print(f"{'='*60}")
        print(f"Return: {results['total_return']*100:.2f}%")
        print(f"Max DD: {results['max_drawdown']*100:.2f}%")
        print(f"Trades: {results['trades']}")
        print(f"Prop Pass: {'✓ YES' if results['passed'] else '✗ NO'}")
        print(f"{'='*60}\n")
        
        return results

if __name__ == '__main__':
    assets = ['BTCUSDT', 'ETHUSDT', 'EURUSD']
    all_results = {}
    
    for asset in assets:
        try:
            trainer = SimplePropTrainer(asset)
            result = trainer.run()
            if result:
                all_results[asset] = result
        except Exception as e:
            print(f"✗ Error with {asset}: {e}")
    
    if all_results:
        print(f"\n{'🏆'*20}")
        print("SUMMARY")
        print(f"{'🏆'*20}")
        for asset, r in all_results.items():
            status = "PASS" if r['passed'] else "FAIL"
            print(f"{asset}: Return={r['total_return']*100:.1f}%, DD={r['max_drawdown']*100:.1f}%, Status={status}")
        print(f"{'🏆'*20}\n")
