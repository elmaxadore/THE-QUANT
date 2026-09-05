#!/usr/bin/env python3
"""
Prop Firm Compliant Training Pipeline
Trains models with strict adherence to prop firm rules:
- Max Daily Drawdown: 5%
- Max Total Drawdown: 10%
- Risk per Trade: 0.5-1%
- Minimum Risk/Reward: 1:1.5
- Consistency Rules
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from unsupervised_insights import (
    MarketRegimeDetector, DeepFeatureExtractor, AnomalyDetector
)

# Import data fetching classes - define them if not available
try:
    from unsupervised_insights import DataFetcher, FeatureEngineer
except ImportError:
    # Define minimal versions
    class DataFetcher:
        def fetch_symbol(self, symbol, start_date, end_date):
            """Fetch data using yfinance directly with correct symbol formats"""
            import yfinance as yf
            
            # Map symbols to yfinance format
            symbol_map = {
                'BTCUSDT': 'BTC-USD',
                'ETHUSDT': 'ETH-USD',
                'EURUSD': 'EURUSD=X',
                'GBPUSD': 'GBPUSD=X',
                'XAUUSD': 'GC=F',  # Gold futures
                'US30': 'YM=F',    # Dow Jones futures
                'NAS100': 'NQ=F',  # Nasdaq futures
                'SPX500': 'ES=F',  # S&P futures
                'OIL': 'CL=F',     # Crude oil futures
            }
            
            yf_symbol = symbol_map.get(symbol.upper(), symbol)
            
            print(f"  Fetching data for {symbol} using yfinance symbol: {yf_symbol}...")
            
            try:
                df = yf.download(yf_symbol, start=start_date, end=end_date, progress=False)
                if len(df) > 0:
                    print(f"  ✓ Fetched {len(df)} bars")
                    return df
                else:
                    print(f"  ⚠ No data returned for {yf_symbol}")
                    return None
            except Exception as e:
                print(f"  ✗ Error fetching {yf_symbol}: {e}")
                return None
    
    class FeatureEngineer:
        def create_all_features(self, df):
            """Create basic technical features"""
            df = df.copy()
            
            # Basic indicators
            df['SMA_20'] = df['Close'].rolling(20).mean()
            df['SMA_50'] = df['Close'].rolling(50).mean()
            df['RSI'] = self._calculate_rsi(df['Close'], 14)
            df['MACD'] = df['Close'].ewm(span=12).mean() - df['Close'].ewm(span=26).mean()
            df['ATR'] = self._calculate_atr(df, 14)
            
            # Volatility
            df['volatility'] = df['Close'].pct_change().rolling(20).std()
            
            # Momentum
            df['momentum'] = df['Close'].pct_change(periods=10)
            
            return df
        
        def _calculate_rsi(self, prices, period):
            delta = prices.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
            rs = gain / loss
            return 100 - (100 / (1 + rs))
        
        def _calculate_atr(self, df, period):
            high = df['High']
            low = df['Low']
            close = df['Close']
            tr1 = high - low
            tr2 = abs(high - close.shift())
            tr3 = abs(low - close.shift())
            tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
            return tr.rolling(period).mean()

from sklearn.model_selection import TimeSeriesSplit
# walk_forward_validation is not a direct import, we use TimeSeriesSplit instead
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score
)
from sklearn.preprocessing import StandardScaler, RobustScaler
import joblib
import onnxruntime as rt
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

class PropFirmBacktester:
    """Backtester with strict prop firm rule enforcement"""
    
    def __init__(self, initial_capital=100000, config=None):
        self.initial_capital = initial_capital
        self.config = config or {
            'max_daily_dd': 0.05,      # 5% max daily drawdown
            'max_total_dd': 0.10,      # 10% max total drawdown
            'risk_per_trade': 0.01,    # 1% risk per trade
            'min_risk_reward': 1.5,    # Minimum 1:1.5 RR
            'max_trades_per_day': 10,
            'consistency_threshold': 0.30,  # No single day >30% of profits
        }
        
    def run_backtest(self, signals, prices, regime_labels, strategy_confidence):
        """
        Run backtest with prop firm constraints
        
        Args:
            signals: array of -1, 0, 1 (sell, hold, buy)
            prices: DataFrame with OHLCV data
            regime_labels: array of regime assignments
            strategy_confidence: array of confidence scores [0,1]
        """
        capital = self.initial_capital
        peak_capital = capital
        daily_peaks = {}
        trades = []
        daily_pnl = {}
        equity_curve = [capital]
        
        position = None
        shares = 0
        entry_price = 0
        stop_loss = 0
        take_profit = 0
        
        for i in range(1, len(signals)):
            current_date = prices.index[i].strftime('%Y-%m-%d')
            price = prices['Close'].iloc[i]
            
            # Track daily peaks for drawdown calculation
            if current_date not in daily_peaks:
                daily_peaks[current_date] = capital
            
            # Update peak capital
            if capital > peak_capital:
                peak_capital = capital
            
            # Check daily drawdown limit
            daily_dd = (daily_peaks[current_date] - capital) / daily_peaks[current_date]
            if daily_dd >= self.config['max_daily_dd']:
                # Close any open position
                if position is not None:
                    pnl = (price - entry_price) * shares if position == 1 else (entry_price - price) * shares
                    capital += pnl
                    trades.append({
                        'date': current_date,
                        'type': 'FORCE_CLOSE_DAILY_DD',
                        'pnl': pnl,
                        'capital': capital
                    })
                    position = None
                continue
            
            # Check total drawdown limit
            total_dd = (peak_capital - capital) / peak_capital
            if total_dd >= self.config['max_total_dd']:
                break  # Stop trading
            
            # Get dynamic position size based on confidence and regime
            confidence = strategy_confidence[i] if i < len(strategy_confidence) else 0.5
            regime = regime_labels[i] if i < len(regime_labels) else 0
            
            # Adjust risk based on regime volatility
            regime_risk_mult = {0: 0.5, 1: 1.0, 2: 1.0, 3: 0.3}.get(regime, 0.5)
            effective_risk = self.config['risk_per_trade'] * confidence * regime_risk_mult
            
            # Calculate position size
            if position is None and signals[i] != 0:
                # Calculate stop loss and take profit based on ATR
                atr = prices['ATR'].iloc[i] if 'ATR' in prices.columns else price * 0.02
                
                if signals[i] == 1:  # Buy signal
                    stop_loss = price - (atr * 1.5)
                    take_profit = price + (atr * self.config['min_risk_reward'] * 1.5)
                    
                    # Check if RR meets minimum requirement
                    risk_amount = price - stop_loss
                    reward_amount = take_profit - price
                    if reward_amount / risk_amount < self.config['min_risk_reward']:
                        continue  # Skip trade if RR insufficient
                    
                    shares = int((capital * effective_risk) / risk_amount)
                    if shares > 0:
                        position = 1
                        entry_price = price
                        trades.append({
                            'date': current_date,
                            'type': 'BUY',
                            'shares': shares,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'capital': capital
                        })
                
                elif signals[i] == -1:  # Sell signal
                    stop_loss = price + (atr * 1.5)
                    take_profit = price - (atr * self.config['min_risk_reward'] * 1.5)
                    
                    risk_amount = stop_loss - price
                    reward_amount = price - take_profit
                    if reward_amount / risk_amount < self.config['min_risk_reward']:
                        continue
                    
                    shares = int((capital * effective_risk) / risk_amount)
                    if shares > 0:
                        position = -1
                        entry_price = price
                        trades.append({
                            'date': current_date,
                            'type': 'SELL',
                            'shares': shares,
                            'entry_price': entry_price,
                            'stop_loss': stop_loss,
                            'take_profit': take_profit,
                            'capital': capital
                        })
            
            # Check for stop loss or take profit hit
            elif position is not None:
                pnl = 0
                exit_reason = None
                
                if position == 1:  # Long position
                    if price <= stop_loss:
                        pnl = (stop_loss - entry_price) * shares
                        exit_reason = 'STOP_LOSS'
                        position = None
                    elif price >= take_profit:
                        pnl = (take_profit - entry_price) * shares
                        exit_reason = 'TAKE_PROFIT'
                        position = None
                
                elif position == -1:  # Short position
                    if price >= stop_loss:
                        pnl = (entry_price - stop_loss) * shares
                        exit_reason = 'STOP_LOSS'
                        position = None
                    elif price <= take_profit:
                        pnl = (entry_price - take_profit) * shares
                        exit_reason = 'TAKE_PROFIT'
                        position = None
                
                if exit_reason:
                    capital += pnl
                    trades.append({
                        'date': current_date,
                        'type': exit_reason,
                        'pnl': pnl,
                        'capital': capital,
                        'exit_price': price if exit_reason in ['STOP_LOSS', 'TAKE_PROFIT'] else None
                    })
            
            equity_curve.append(capital)
            
            # Track daily PnL
            if current_date not in daily_pnl:
                daily_pnl[current_date] = 0
            if position is None and len(trades) > 0 and trades[-1]['date'] == current_date:
                if 'pnl' in trades[-1]:
                    daily_pnl[current_date] += trades[-1]['pnl']
        
        # Calculate metrics
        equity_series = pd.Series(equity_curve)
        returns = equity_series.pct_change().dropna()
        
        total_return = (equity_curve[-1] - self.initial_capital) / self.initial_capital
        
        # Maximum drawdown
        rolling_max = equity_series.expanding().max()
        drawdowns = (equity_series - rolling_max) / rolling_max
        max_drawdown = abs(drawdowns.min())
        
        # Sharpe ratio (annualized)
        sharpe_ratio = (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() > 0 else 0
        
        # Win rate
        winning_trades = [t for t in trades if t.get('pnl', 0) > 0]
        losing_trades = [t for t in trades if t.get('pnl', 0) < 0]
        win_rate = len(winning_trades) / len(trades) if trades else 0
        
        # Profit factor
        gross_profit = sum(t['pnl'] for t in winning_trades)
        gross_loss = abs(sum(t['pnl'] for t in losing_trades))
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average R:R
        avg_rr = np.mean([
            abs(t.get('pnl', 0)) / ((t['entry_price'] - t['stop_loss']) * t['shares']) 
            if t['type'] in ['TAKE_PROFIT'] and t.get('shares', 0) > 0 else 0
            for t in trades
        ]) if trades else 0
        
        # Consistency check
        if daily_pnl:
            total_profit = sum(v for v in daily_pnl.values() if v > 0)
            max_daily_profit = max(v for v in daily_pnl.values()) if daily_pnl else 0
            consistency_ratio = max_daily_profit / total_profit if total_profit > 0 else 0
        else:
            consistency_ratio = 0
        
        return {
            'total_return': total_return,
            'max_drawdown': max_drawdown,
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'avg_risk_reward': avg_rr,
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'consistency_ratio': consistency_ratio,
            'final_capital': equity_curve[-1],
            'equity_curve': equity_curve,
            'trades': trades,
            'daily_pnl': daily_pnl,
            'passed_prop_rules': max_drawdown < 0.10 and consistency_ratio < 0.30
        }


class PropFirmStrategyTrainer:
    """Train strategies optimized for prop firm challenges"""
    
    def __init__(self, symbol, start_date='2022-01-01', end_date='2024-12-01'):
        self.symbol = symbol
        self.start_date = start_date
        self.end_date = end_date
        self.fetcher = DataFetcher()
        self.feature_engineer = FeatureEngineer()
        self.regime_detector = MarketRegimeDetector(n_regimes=4)
        self.backtester = PropFirmBacktester()
        
    def fetch_and_prepare_data(self):
        """Fetch real market data and prepare features"""
        print(f"\n{'='*60}")
        print(f"Fetching data for {self.symbol}")
        print(f"{'='*60}")
        
        # Fetch data
        df = self.fetcher.fetch_symbol(self.symbol, self.start_date, self.end_date)
        if df is None or len(df) < 100:
            print(f"⚠️  Insufficient data for {self.symbol}")
            return None
        
        print(f"✓ Fetched {len(df)} bars from {df.index[0]} to {df.index[-1]}")
        
        # Engineer features
        df_features = self.feature_engineer.create_all_features(df)
        if df_features is None:
            return None
        
        # Drop NaN values
        df_features = df_features.dropna()
        if len(df_features) < 200:
            print(f"⚠️  Too few samples after feature engineering for {self.symbol}")
            return None
        
        print(f"✓ Created {df_features.shape[1]} features, {len(df_features)} samples")
        
        return df_features
    
    def detect_regimes(self, df):
        """Detect market regimes using unsupervised learning"""
        print("\n📊 Detecting Market Regimes...")
        
        feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume']]
        X = df[feature_cols].values
        
        regime_detector = MarketRegimeDetector(n_regimes=4)
        regime_detector.fit(X)
        regimes = regime_detector.predict(X)
        
        df['regime'] = regimes
        
        # Print regime statistics
        unique, counts = np.unique(regimes, return_counts=True)
        print("Regime Distribution:")
        regime_names = {0: "CALM", 1: "BULL", 2: "BEAR", 3: "HIGH_VOL"}
        for u, c in zip(unique, counts):
            pct = c / len(regimes) * 100
            regime_name = regime_names.get(u, f"REGIME_{u}")
            print(f"  Regime {u} ({regime_name}): {c} samples ({pct:.1f}%)")
        
        return df
    
    def create_target(self, df, lookforward=5, threshold=0.02):
        """Create target variable with proper risk/reward consideration"""
        # Forward return
        future_return = df['Close'].shift(-lookforward) / df['Close'] - 1
        
        # Create multi-class target considering transaction costs
        # 0: Hold/Sell, 1: Buy
        df = df.copy()
        target = np.zeros(len(df))
        
        # Handle multi-dimensional arrays from yfinance
        fr_values = future_return.values.flatten() if len(future_return.values.shape) > 1 else future_return.values
        mask = fr_values > threshold
        valid_idx = np.where(mask)[0]
        target[valid_idx] = 1  # Buy signal
        
        df['target'] = target
        
        # Drop last rows where we can't calculate forward return
        df = df.dropna(subset=['target'])
        
        return df
    
    def train_strategies(self, df):
        """Train multiple strategies per regime with walk-forward validation"""
        print("\n🎯 Training Strategies with Walk-Forward Validation...")
        
        feature_cols = [c for c in df.columns if c not in ['Open', 'High', 'Low', 'Close', 'Volume', 'regime', 'target']]
        X = df[feature_cols].values
        y = df['target'].values
        regimes = df['regime'].values
        
        # Time series split for walk-forward validation
        tscv = TimeSeriesSplit(n_splits=5)
        
        all_results = {}
        best_models = {}
        scalers = {}
        
        for regime_id in sorted(np.unique(regimes)):
            print(f"\n  Training for Regime {regime_id}...")
            
            regime_mask = regimes == regime_id
            X_regime = X[regime_mask]
            y_regime = y[regime_mask]
            
            if len(X_regime) < 100:
                print(f"    ⚠️  Insufficient samples for regime {regime_id}")
                continue
            
            # Try different classifiers
            classifiers = {
                'GradientBoosting': GradientBoostingClassifier(
                    n_estimators=100, 
                    max_depth=4, 
                    learning_rate=0.05,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    subsample=0.8,
                    random_state=42
                ),
                'RandomForest': RandomForestClassifier(
                    n_estimators=100,
                    max_depth=6,
                    min_samples_split=20,
                    min_samples_leaf=10,
                    class_weight='balanced',
                    random_state=42
                ),
                'MLP': MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation='relu',
                    solver='adam',
                    alpha=0.001,
                    batch_size=32,
                    learning_rate='adaptive',
                    max_iter=500,
                    random_state=42,
                    early_stopping=True
                )
            }
            
            regime_results = {}
            
            for clf_name, clf in classifiers.items():
                fold_scores = []
                
                for train_idx, test_idx in tscv.split(X_regime):
                    X_train, X_test = X_regime[train_idx], X_regime[test_idx]
                    y_train, y_test = y_regime[train_idx], y_regime[test_idx]
                    
                    # Scale features
                    scaler = RobustScaler()
                    X_train_scaled = scaler.fit_transform(X_train)
                    X_test_scaled = scaler.transform(X_test)
                    
                    # Train
                    clf.fit(X_train_scaled, y_train)
                    
                    # Predict
                    y_pred = clf.predict(X_test_scaled)
                    y_prob = clf.predict_proba(X_test_scaled)[:, 1] if hasattr(clf, 'predict_proba') else y_pred
                    
                    # Calculate metrics
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    
                    fold_scores.append({
                        'f1': f1,
                        'precision': precision,
                        'y_pred': y_pred,
                        'y_prob': y_prob,
                        'y_test': y_test,
                        'X_test': X_test_scaled
                    })
                
                # Average metrics across folds
                avg_f1 = np.mean([s['f1'] for s in fold_scores])
                avg_precision = np.mean([s['precision'] for s in fold_scores])
                
                regime_results[clf_name] = {
                    'avg_f1': avg_f1,
                    'avg_precision': avg_precision,
                    'fold_results': fold_scores,
                    'best_fold_idx': np.argmax([s['f1'] for s in fold_scores])
                }
                
                print(f"    {clf_name}: F1={avg_f1:.3f}, Precision={avg_precision:.3f}")
            
            # Select best classifier for this regime
            best_clf_name = max(regime_results.keys(), key=lambda k: regime_results[k]['avg_f1'])
            best_result = regime_results[best_clf_name]
            
            all_results[f'regime_{regime_id}'] = {
                'classifier': best_clf_name,
                'metrics': best_result,
                'full_results': regime_results
            }
            
            # Retrain best model on full regime data
            best_clf = classifiers[best_clf_name]
            scaler = RobustScaler()
            X_regime_scaled = scaler.fit_transform(X_regime)
            best_clf.fit(X_regime_scaled, y_regime)
            
            best_models[f'regime_{regime_id}'] = best_clf
            scalers[f'regime_{regime_id}'] = scaler
        
        return all_results, best_models, scalers, feature_cols
    
    def generate_signals(self, df, best_models, scalers, feature_cols):
        """Generate trading signals using trained models"""
        print("\n📡 Generating Trading Signals...")
        
        X = df[feature_cols].values
        regimes = df['regime'].values
        
        signals = np.zeros(len(df))
        confidences = np.zeros(len(df))
        
        for i in range(len(df)):
            regime_id = regimes[i]
            regime_key = f'regime_{regime_id}'
            
            if regime_key in best_models:
                model = best_models[regime_key]
                scaler = scalers[regime_key]
                
                x_sample = X[i:i+1]
                x_scaled = scaler.transform(x_sample)
                
                pred = model.predict(x_scaled)[0]
                prob = model.predict_proba(x_scaled)[0][1] if hasattr(model, 'predict_proba') else 0.5
                
                signals[i] = pred  # 0 or 1
                confidences[i] = prob if pred == 1 else (1 - prob)
        
        # Convert binary signals to -1, 0, 1
        # For simplicity, we'll use 1 for buy, 0 for hold/neutral
        # In production, you'd have separate models for long/short
        
        return signals, confidences
    
    def run_backtest(self, df, signals, confidences):
        """Run prop-firm compliant backtest"""
        print("\n💰 Running Prop-Firm Backtest...")
        
        results = self.backtester.run_backtest(
            signals=signals,
            prices=df[['Open', 'High', 'Low', 'Close', 'Volume']].copy(),
            regime_labels=df['regime'].values,
            strategy_confidence=confidences
        )
        
        return results
    
    def export_models(self, best_models, scalers, feature_cols, output_dir='models/prop_strategies'):
        """Export models to ONNX format"""
        print(f"\n💾 Exporting Models to {output_dir}...")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for regime_key, model in best_models.items():
            # Export to ONNX
            initial_type = [('float_input', FloatTensorType([None, len(feature_cols)]))]
            onnx_model = convert_sklearn(model, initial_types=initial_type)
            
            onnx_path = os.path.join(output_dir, f'{self.symbol}_{regime_key}.onnx')
            with open(onnx_path, "wb") as f:
                f.write(onnx_model.SerializeToString())
            
            # Save scaler
            scaler_path = os.path.join(output_dir, f'{self.symbol}_{regime_key}_scaler.pkl')
            joblib.dump(scalers[regime_key], scaler_path)
            
            print(f"  ✓ Saved {regime_key} model and scaler")
        
        # Save feature columns
        features_path = os.path.join(output_dir, f'{self.symbol}_features.json')
        with open(features_path, 'w') as f:
            json.dump(feature_cols, f)
        
        print(f"  ✓ Saved feature list")
    
    def run_full_pipeline(self):
        """Execute complete training pipeline"""
        print(f"\n{'🔥'*30}")
        print(f"PROP FIRM COMPLIANT TRAINING PIPELINE")
        print(f"Symbol: {self.symbol}")
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"{'🔥'*30}\n")
        
        # Step 1: Fetch and prepare data
        df = self.fetch_and_prepare_data()
        if df is None:
            return None
        
        # Step 2: Detect regimes
        df = self.detect_regimes(df)
        
        # Step 3: Create target variable
        df = self.create_target(df, lookforward=5, threshold=0.015)
        
        # Step 4: Train strategies
        results, best_models, scalers, feature_cols = self.train_strategies(df)
        
        # Step 5: Generate signals
        signals, confidences = self.generate_signals(df, best_models, scalers, feature_cols)
        
        # Step 6: Run backtest
        backtest_results = self.run_backtest(df, signals, confidences)
        
        # Step 7: Export models
        self.export_models(best_models, scalers, feature_cols)
        
        # Step 8: Print summary
        self.print_summary(backtest_results, results)
        
        # Save detailed results
        self.save_results(backtest_results, results, df)
        
        return backtest_results
    
    def print_summary(self, backtest_results, training_results):
        """Print comprehensive summary"""
        print(f"\n{'='*60}")
        print("📊 PROP FIRM BACKTEST RESULTS")
        print(f"{'='*60}")
        
        status = "✅ PASSED" if backtest_results['passed_prop_rules'] else "❌ FAILED"
        print(f"Prop Firm Compliance: {status}")
        
        print(f"\n📈 Performance Metrics:")
        print(f"  Total Return: {backtest_results['total_return']*100:.2f}%")
        print(f"  Max Drawdown: {backtest_results['max_drawdown']*100:.2f}%")
        print(f"  Sharpe Ratio: {backtest_results['sharpe_ratio']:.2f}")
        print(f"  Win Rate: {backtest_results['win_rate']*100:.1f}%")
        print(f"  Profit Factor: {backtest_results['profit_factor']:.2f}")
        print(f"  Avg Risk/Reward: {backtest_results['avg_risk_reward']:.2f}")
        
        print(f"\n📉 Trade Statistics:")
        print(f"  Total Trades: {backtest_results['total_trades']}")
        print(f"  Winning: {backtest_results['winning_trades']}")
        print(f"  Losing: {backtest_results['losing_trades']}")
        
        print(f"\n⚖️  Consistency Check:")
        print(f"  Max Daily Profit %: {backtest_results['consistency_ratio']*100:.1f}%")
        print(f"  (Must be < 30% for prop firms)")
        
        print(f"\n🎯 Strategy Performance by Regime:")
        for regime_key, result in training_results.items():
            metrics = result['metrics']
            print(f"  {regime_key}: F1={metrics['avg_f1']:.3f} ({result['classifier']})")
        
        print(f"\n💰 Capital:")
        print(f"  Initial: ${self.backtester.initial_capital:,.2f}")
        print(f"  Final: ${backtest_results['final_capital']:,.2f}")
        print(f"  Profit: ${backtest_results['final_capital'] - self.backtester.initial_capital:,.2f}")
        
        print(f"\n{'='*60}\n")
    
    def save_results(self, backtest_results, training_results, df):
        """Save detailed results to files"""
        output_dir = Path('results/prop_training')
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Save JSON results
        results_dict = {
            'symbol': self.symbol,
            'period': {'start': self.start_date, 'end': self.end_date},
            'timestamp': datetime.now().isoformat(),
            'backtest': {k: v for k, v in backtest_results.items() if k != 'equity_curve' and k != 'trades' and k != 'daily_pnl'},
            'training': {k: {'classifier': v['classifier'], 'f1': v['metrics']['avg_f1']} 
                        for k, v in training_results.items()},
            'prop_rules': {
                'max_daily_dd_limit': 0.05,
                'max_total_dd_limit': 0.10,
                'min_risk_reward': 1.5,
                'risk_per_trade': 0.01
            }
        }
        
        results_path = output_dir / f'{self.symbol}_results.json'
        with open(results_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        # Save equity curve
        equity_df = pd.DataFrame({
            'date': df.index[:len(backtest_results['equity_curve'])],
            'equity': backtest_results['equity_curve']
        })
        equity_df.to_csv(output_dir / f'{self.symbol}_equity.csv', index=False)
        
        # Save trades
        if backtest_results['trades']:
            trades_df = pd.DataFrame(backtest_results['trades'])
            trades_df.to_csv(output_dir / f'{self.symbol}_trades.csv', index=False)
        
        # Save regime-labeled data
        df.to_csv(output_dir / f'{self.symbol}_data_with_regimes.csv')
        
        print(f"💾 Results saved to {output_dir}/")


def main():
    """Main execution function"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Prop Firm Compliant Training Pipeline')
    parser.add_argument('--assets', type=str, default='BTCUSDT,ETHUSDT,EURUSD',
                       help='Comma-separated list of assets')
    parser.add_argument('--start_date', type=str, default='2022-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default='2024-12-01',
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--prop_mode', type=str, default='strict',
                       choices=['strict', 'moderate'],
                       help='Prop firm rule strictness')
    
    args = parser.parse_args()
    
    assets = [a.strip() for a in args.assets.split(',')]
    
    all_results = {}
    
    for asset in assets:
        try:
            trainer = PropFirmStrategyTrainer(
                symbol=asset,
                start_date=args.start_date,
                end_date=args.end_date
            )
            
            results = trainer.run_full_pipeline()
            
            if results:
                all_results[asset] = results
            
            # Small delay between assets
            import time
            time.sleep(1)
            
        except Exception as e:
            print(f"\n❌ Error training {asset}: {str(e)}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print summary across all assets
    if all_results:
        print(f"\n{'🏆'*30}")
        print("AGGREGATE RESULTS ACROSS ALL ASSETS")
        print(f"{'🏆'*30}\n")
        
        print(f"{'Asset':<12} {'Return':>10} {'DD':>10} {'Sharpe':>10} {'Win%':>10} {'Trades':>10} {'Status':>10}")
        print("-" * 82)
        
        for asset, results in all_results.items():
            status = "PASS" if results['passed_prop_rules'] else "FAIL"
            print(f"{asset:<12} {results['total_return']*100:>9.2f}% {results['max_drawdown']*100:>9.2f}% "
                  f"{results['sharpe_ratio']:>10.2f} {results['win_rate']*100:>9.1f}% "
                  f"{results['total_trades']:>10} {status:>10}")
        
        # Calculate aggregate stats
        avg_return = np.mean([r['total_return'] for r in all_results.values()])
        avg_dd = np.mean([r['max_drawdown'] for r in all_results.values()])
        avg_sharpe = np.mean([r['sharpe_ratio'] for r in all_results.values()])
        pass_rate = sum(1 for r in all_results.values() if r['passed_prop_rules']) / len(all_results)
        
        print("-" * 82)
        print(f"{'AVERAGE':<12} {avg_return*100:>9.2f}% {avg_dd*100:>9.2f}% {avg_sharpe:>10.2f}")
        print(f"\nPass Rate: {pass_rate*100:.1f}% ({sum(1 for r in all_results.values() if r['passed_prop_rules'])}/{len(all_results)} assets)")
        print(f"{'🏆'*30}\n")
    
    return all_results


if __name__ == '__main__':
    main()
