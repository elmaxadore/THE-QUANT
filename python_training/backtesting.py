"""
Walk-Forward Validation and Backtesting Module
Implements rigorous backtesting with walk-forward analysis to prevent overfitting
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import warnings
warnings.filterwarnings('ignore')


class WalkForwardValidator:
    """
    Walk-forward validation for time series models.
    Prevents look-ahead bias and provides realistic performance estimates.
    """
    
    def __init__(self, 
                 train_window: int = 252,  # ~1 year of daily data
                 test_window: int = 63,    # ~3 months
                 step: int = 21):          # Step forward by ~1 month
        """
        Args:
            train_window: Size of training window in periods
            test_window: Size of test window in periods
            step: Step size for rolling forward
        """
        self.train_window = train_window
        self.test_window = test_window
        self.step = step
        
    def split(self, data: pd.DataFrame) -> List[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Generate walk-forward splits.
        
        Returns:
            List of (train_df, test_df) tuples
        """
        splits = []
        n = len(data)
        
        start_idx = 0
        while start_idx + self.train_window + self.test_window <= n:
            train_end = start_idx + self.train_window
            test_end = train_end + self.test_window
            
            train_df = data.iloc[start_idx:train_end].copy()
            test_df = data.iloc[train_end:test_end].copy()
            
            splits.append((train_df, test_df))
            
            start_idx += self.step
        
        print(f"Generated {len(splits)} walk-forward splits")
        print(f"  Train window: {self.train_window} periods")
        print(f"  Test window: {self.test_window} periods")
        print(f"  Step size: {self.step} periods")
        
        return splits


class Backtester:
    """
    Simple backtesting engine for evaluating trading strategies.
    """
    
    def __init__(self, 
                 initial_capital: float = 10000.0,
                 transaction_cost: float = 0.0001,  # 0.01% per trade
                 slippage: float = 0.0001):         # 0.01% slippage
        """
        Args:
            initial_capital: Starting capital
            transaction_cost: Transaction cost as fraction of trade value
            slippage: Slippage as fraction of trade value
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.slippage = slippage
    
    def run_backtest(self, 
                    data: pd.DataFrame,
                    predictions: np.ndarray,
                    position_col: str = 'position',
                    target_col: str = 'forward_return') -> pd.DataFrame:
        """
        Run backtest given predictions.
        
        Args:
            data: DataFrame with price data
            predictions: Array of predicted positions (-1, 0, 1)
            position_col: Column name for positions
            target_col: Column name for actual returns
        
        Returns:
            DataFrame with backtest results
        """
        
        df = data.copy()
        df['position'] = predictions[:len(df)]
        
        # Shift position to avoid look-ahead bias (trade on next bar)
        df['position_shifted'] = df['position'].shift(1)
        
        # Calculate strategy returns
        df['strategy_return'] = df['position_shifted'] * df[target_col]
        
        # Apply transaction costs
        position_changes = df['position_shifted'].diff().abs()
        df['transaction_cost'] = position_changes * self.transaction_cost
        df['slippage_cost'] = position_changes * self.slippage
        
        # Net returns
        df['net_return'] = df['strategy_return'] - df['transaction_cost'] - df['slippage_cost']
        
        # Cumulative returns
        df['cumulative_strategy'] = (1 + df['net_return']).cumprod()
        df['cumulative_market'] = (1 + df[target_col]).cumprod()
        
        # Equity curve
        df['equity'] = self.initial_capital * df['cumulative_strategy']
        
        return df
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """
        Calculate comprehensive performance metrics.
        """
        net_returns = results['net_return'].dropna()
        equity = results['equity']
        
        if len(net_returns) == 0 or equity.iloc[-1] == equity.iloc[0]:
            return self._empty_metrics()
        
        # Basic statistics
        total_return = equity.iloc[-1] / equity.iloc[0] - 1
        num_periods = len(net_returns)
        annualization_factor = 252 * 24  # Assuming hourly data
        
        # Annualized return
        ann_return = (1 + total_return) ** (annualization_factor / num_periods) - 1
        
        # Volatility
        volatility = net_returns.std() * np.sqrt(annualization_factor)
        
        # Sharpe Ratio (assuming risk-free rate = 0)
        sharpe = ann_return / volatility if volatility > 0 else 0
        
        # Maximum Drawdown
        rolling_max = equity.cummax()
        drawdown = (equity - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        # Calmar Ratio
        calmar = ann_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        # Win rate
        winning_trades = net_returns[net_returns > 0]
        losing_trades = net_returns[net_returns < 0]
        win_rate = len(winning_trades) / len(net_returns) if len(net_returns) > 0 else 0
        
        # Profit factor
        gross_profit = winning_trades.sum() if len(winning_trades) > 0 else 0
        gross_loss = abs(losing_trades.sum()) if len(losing_trades) > 0 else 0
        profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
        
        # Average win/loss ratio
        avg_win = winning_trades.mean() if len(winning_trades) > 0 else 0
        avg_loss = abs(losing_trades.mean()) if len(losing_trades) > 0 else 0
        win_loss_ratio = avg_win / avg_loss if avg_loss > 0 else float('inf')
        
        # Number of trades
        num_trades = len(net_returns[net_returns != 0])
        
        return {
            'total_return': total_return,
            'annualized_return': ann_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'calmar_ratio': calmar,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            'win_loss_ratio': win_loss_ratio,
            'num_trades': num_trades,
            'final_equity': equity.iloc[-1],
            'initial_capital': self.initial_capital
        }
    
    def _empty_metrics(self) -> Dict:
        """Return empty metrics dict for edge cases."""
        return {
            'total_return': 0.0,
            'annualized_return': 0.0,
            'volatility': 0.0,
            'sharpe_ratio': 0.0,
            'max_drawdown': 0.0,
            'calmar_ratio': 0.0,
            'win_rate': 0.0,
            'profit_factor': 0.0,
            'win_loss_ratio': 0.0,
            'num_trades': 0,
            'final_equity': self.initial_capital,
            'initial_capital': self.initial_capital
        }


def evaluate_model_predictions(y_true: np.ndarray, 
                               y_pred: np.ndarray,
                               y_prob: Optional[np.ndarray] = None) -> Dict:
    """
    Evaluate classification model predictions.
    """
    metrics = {}
    
    # Convert to binary (1 vs -1/0) for some metrics
    y_true_binary = (y_true == 1).astype(int)
    y_pred_binary = (y_pred == 1).astype(int)
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(y_true_binary, y_pred_binary)
    
    # Precision, Recall, F1 (for class 1)
    try:
        metrics['precision'] = precision_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['recall'] = recall_score(y_true_binary, y_pred_binary, zero_division=0)
        metrics['f1'] = f1_score(y_true_binary, y_pred_binary, zero_division=0)
    except:
        metrics['precision'] = 0.0
        metrics['recall'] = 0.0
        metrics['f1'] = 0.0
    
    # AUC-ROC (if probabilities available)
    if y_prob is not None and len(np.unique(y_true_binary)) > 1:
        try:
            metrics['auc_roc'] = roc_auc_score(y_true_binary, y_prob)
        except:
            metrics['auc_roc'] = 0.5
    else:
        metrics['auc_roc'] = 0.5
    
    # Class distribution
    unique, counts = np.unique(y_true, return_counts=True)
    metrics['class_distribution'] = dict(zip(unique.astype(int), counts.astype(int)))
    
    return metrics


def run_walk_forward_backtest(model_class,
                              X: np.ndarray,
                              y: np.ndarray,
                              prices: pd.Series,
                              validator: WalkForwardValidator,
                              threshold: float = 0.3) -> Dict:
    """
    Run complete walk-forward validation and backtesting.
    
    Args:
        model_class: Scikit-learn compatible model class
        X: Feature matrix
        y: Target variable
        prices: Price series for backtesting
        validator: WalkForwardValidator instance
        threshold: Probability threshold for taking positions
    
    Returns:
        Dictionary with all metrics and results
    """
    
    splits = validator.split(pd.DataFrame({'X': list(X), 'y': y, 'price': prices}))
    
    all_predictions = []
    all_actuals = []
    all_probs = []
    backtest_results = []
    
    print(f"\nRunning walk-forward validation on {len(splits)} splits...")
    
    for i, (train_df, test_df) in enumerate(splits):
        print(f"  Split {i+1}/{len(splits)}...", end=' ')
        
        # Get train/test data
        X_train = np.array(train_df['X'].tolist())
        y_train = np.array(train_df['y'].tolist())
        X_test = np.array(test_df['X'].tolist())
        y_test = np.array(test_df['y'].tolist())
        
        # Handle NaN values
        mask_train = ~np.isnan(y_train) & ~np.any(np.isnan(X_train), axis=1)
        mask_test = ~np.isnan(y_test) & ~np.any(np.isnan(X_test), axis=1)
        
        X_train_clean = X_train[mask_train]
        y_train_clean = y_train[mask_train]
        X_test_clean = X_test[mask_test]
        y_test_clean = y_test[mask_test]
        
        if len(X_train_clean) == 0 or len(X_test_clean) == 0:
            print("Skipped (no valid data)")
            continue
        
        # Train model
        try:
            model = model_class()
            model.fit(X_train_clean, y_train_clean)
            
            # Predict
            y_pred = model.predict(X_test_clean)
            y_prob = model.predict_proba(X_test_clean)[:, 1] if hasattr(model, 'predict_proba') else None
            
            # Store results
            all_predictions.extend(y_pred)
            all_actuals.extend(y_test_clean)
            if y_prob is not None:
                all_probs.extend(y_prob[:, 1] if len(y_prob.shape) > 1 else y_prob)
            
            # Create position signals
            positions = np.zeros(len(y_pred))
            positions[y_pred == 1] = 1
            positions[y_pred == -1] = -1
            
            # Backtest this split
            test_prices = test_df['price'].iloc[mask_test].reset_index(drop=True)
            test_returns = test_prices.pct_change().fillna(0)
            
            backtest_df = pd.DataFrame({
                'close': test_prices.values,
                'forward_return': test_returns.values,
                'position': positions
            })
            
            backtester = Backtester()
            bt_results = backtester.run_backtest(backtest_df, positions)
            metrics = backtester.calculate_metrics(bt_results)
            backtest_results.append(metrics)
            
            print(f"Sharpe={metrics['sharpe_ratio']:.2f}, Return={metrics['total_return']*100:.1f}%")
            
        except Exception as e:
            print(f"Error: {e}")
            continue
    
    # Aggregate results
    if not backtest_results:
        return {'error': 'No valid backtest results'}
    
    # Average backtest metrics
    avg_metrics = {}
    for key in backtest_results[0].keys():
        values = [r[key] for r in backtest_results if isinstance(r[key], (int, float))]
        if values:
            avg_metrics[f'avg_{key}'] = np.mean(values)
            avg_metrics[f'std_{key}'] = np.std(values)
    
    # Overall prediction metrics
    if all_predictions:
        pred_metrics = evaluate_model_predictions(
            np.array(all_actuals),
            np.array(all_predictions),
            np.array(all_probs) if all_probs else None
        )
        avg_metrics.update(pred_metrics)
    
    avg_metrics['num_splits'] = len(backtest_results)
    avg_metrics['split_results'] = backtest_results
    
    return avg_metrics


if __name__ == "__main__":
    # Example usage
    print("Walk-Forward Backtesting Module")
    print("=" * 50)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 1000
    
    X_sample = np.random.randn(n_samples, 10)
    y_sample = np.random.choice([-1, 0, 1], n_samples)
    prices_sample = pd.Series(100 + np.cumsum(np.random.randn(n_samples) * 0.5))
    
    # Setup validator
    validator = WalkForwardValidator(train_window=500, test_window=100, step=50)
    
    # Test with dummy model
    from sklearn.dummy import DummyClassifier
    
    def create_dummy_model():
        return DummyClassifier(strategy='stratified')
    
    results = run_walk_forward_backtest(
        create_dummy_model,
        X_sample, y_sample, prices_sample,
        validator
    )
    
    print("\nResults:")
    for key, value in results.items():
        if not key.endswith('_results'):
            print(f"  {key}: {value}")
