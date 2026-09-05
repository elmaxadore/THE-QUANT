"""
Global Risk Overlay Manager
Prevents over-leveraging across correlated assets and enforces daily loss limits
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
import json


class GlobalRiskManager:
    """Manages portfolio-level risk across all trading positions"""
    
    def __init__(self, config: dict):
        self.max_daily_loss_pct = config.get('max_daily_loss_pct', 0.05)
        self.max_portfolio_exposure = config.get('max_portfolio_exposure', 0.20)
        self.correlation_threshold = config.get('correlation_threshold', 0.8)
        self.correlation_lookback = config.get('correlation_lookback', 60)
        
        self.daily_pnl = 0.0
        self.initial_capital = 100000.0
        self.position_history = []
        
    def calculate_correlation_matrix(self, returns_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate rolling correlation matrix for all assets"""
        return returns_df.tail(self.correlation_lookback).corr()
    
    def get_correlated_clusters(self, corr_matrix: pd.DataFrame) -> List[List[str]]:
        """Identify clusters of highly correlated assets"""
        assets = corr_matrix.columns.tolist()
        visited = set()
        clusters = []
        
        for asset in assets:
            if asset in visited:
                continue
            
            cluster = [asset]
            visited.add(asset)
            
            for other_asset in assets:
                if other_asset != asset and other_asset not in visited:
                    corr = corr_matrix.loc[asset, other_asset]
                    if abs(corr) > self.correlation_threshold:
                        cluster.append(other_asset)
                        visited.add(other_asset)
            
            clusters.append(cluster)
        
        return clusters
    
    def adjust_positions_for_correlation(
        self, 
        signals: Dict[str, float], 
        returns_df: pd.DataFrame
    ) -> Dict[str, float]:
        """
        Reduce position sizes for highly correlated assets to prevent over-exposure
        
        Args:
            signals: Dictionary of asset -> raw signal strength (-1 to 1)
            returns_df: Historical returns for correlation calculation
            
        Returns:
            Adjusted signals with correlation-based position sizing
        """
        if len(signals) < 2:
            return signals
        
        # Calculate correlation matrix
        corr_matrix = self.calculate_correlation_matrix(returns_df)
        clusters = self.get_correlated_clusters(corr_matrix)
        
        adjusted_signals = signals.copy()
        
        # For each cluster of correlated assets, reduce individual positions
        for cluster in clusters:
            if len(cluster) > 1:
                # Count active signals in this cluster
                active_signals = [s for s in cluster if abs(signals.get(s, 0)) > 0.1]
                
                if len(active_signals) > 1:
                    # Reduce each position by sqrt(n) to maintain similar total exposure
                    reduction_factor = 1.0 / np.sqrt(len(active_signals))
                    
                    for asset in active_signals:
                        adjusted_signals[asset] = signals[asset] * reduction_factor
        
        return adjusted_signals
    
    def check_daily_loss_limit(self, current_pnl: float) -> bool:
        """Check if daily loss limit has been breached"""
        daily_loss_pct = abs(min(0, current_pnl)) / self.initial_capital
        
        if daily_loss_pct >= self.max_daily_loss_pct:
            print(f"⚠️  DAILY LOSS LIMIT BREACHED: {daily_loss_pct:.2%}")
            return True
        
        return False
    
    def get_max_position_size(self, asset: str, base_size: float, 
                             correlation_penalty: float = 1.0) -> float:
        """
        Calculate maximum allowed position size considering all risk factors
        
        Args:
            asset: Asset symbol
            base_size: Base position size from strategy
            correlation_penalty: Reduction factor from correlation analysis
            
        Returns:
            Maximum allowed position size
        """
        # Apply correlation penalty
        adjusted_size = base_size * correlation_penalty
        
        # Check portfolio exposure limit
        current_exposure = sum(abs(p) for p in self.position_history)
        max_additional = (self.max_portfolio_exposure * self.initial_capital) - current_exposure
        
        if max_additional <= 0:
            return 0.0
        
        return min(adjusted_size, max_additional)
    
    def update_position(self, asset: str, size: float, pnl: float):
        """Record a new position for tracking"""
        self.position_history.append({
            'asset': asset,
            'size': size,
            'pnl': pnl
        })
        self.daily_pnl += pnl
    
    def reset_daily(self):
        """Reset daily tracking variables"""
        self.daily_pnl = 0.0
        self.position_history = []
    
    def generate_risk_report(self) -> dict:
        """Generate comprehensive risk report"""
        return {
            'daily_pnl': self.daily_pnl,
            'daily_pnl_pct': self.daily_pnl / self.initial_capital,
            'max_daily_loss_pct': self.max_daily_loss_pct,
            'loss_limit_breached': self.check_daily_loss_limit(self.daily_pnl),
            'current_exposure': sum(abs(p['size']) for p in self.position_history),
            'max_portfolio_exposure': self.max_portfolio_exposure,
            'num_positions': len(self.position_history)
        }


# Example usage
if __name__ == "__main__":
    config = {
        'max_daily_loss_pct': 0.05,
        'max_portfolio_exposure': 0.20,
        'correlation_threshold': 0.8,
        'correlation_lookback': 60
    }
    
    risk_mgr = GlobalRiskManager(config)
    
    # Simulate some returns data
    np.random.seed(42)
    dates = pd.date_range('2024-01-01', periods=100, freq='D')
    returns_df = pd.DataFrame({
        'BTCUSDT': np.random.randn(100) * 0.03,
        'ETHUSDT': np.random.randn(100) * 0.04,
        'EURUSD': np.random.randn(100) * 0.01
    }, index=dates)
    
    # Make ETH returns correlated with BTC
    returns_df['ETHUSDT'] = 0.7 * returns_df['BTCUSDT'] + 0.3 * np.random.randn(100) * 0.03
    
    signals = {'BTCUSDT': 0.8, 'ETHUSDT': 0.7, 'EURUSD': 0.5}
    
    print("Original signals:", signals)
    adjusted = risk_mgr.adjust_positions_for_correlation(signals, returns_df)
    print("Adjusted signals:", adjusted)
    
    report = risk_mgr.generate_risk_report()
    print("\nRisk Report:", json.dumps(report, indent=2))
