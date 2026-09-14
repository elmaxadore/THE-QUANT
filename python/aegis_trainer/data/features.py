"""
Feature engineering for AEGIS hedged pairs strategy.
Implements single-asset features and pair-specific spread features.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class FeatureEngine:
    """
    Computes technical indicators and pair features for model training.
    
    Features include:
    - Individual asset features (RSI, ATR, KAMA, Hurst, etc.)
    - Pair spread features (price_ratio, log_spread, zscore, velocity)
    - Differential features (ΔRSI, ΔATR, Δmomentum)
    - Regime features (volatility regime, correlation regime)
    - Time features (session, day-of-week, hour)
    """
    
    def __init__(self, lookback_periods: Dict[str, int] = None):
        self.lookback = lookback_periods or {
            'rsi': 14,
            'atr': 14,
            'kama': 20,
            'hurs': 20,
            'corr': 20,
            'vol': 20
        }
    
    def compute_single_asset_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Compute individual asset technical features."""
        df = df.copy()
        
        # Returns
        df['returns'] = df['close'].pct_change()
        df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
        
        # RSI
        df['rsi'] = self._compute_rsi(df['close'], self.lookback['rsi'])
        
        # ATR
        df['atr'] = self._compute_atr(df, self.lookback['atr'])
        
        # KAMA (Kaufman Adaptive Moving Average)
        df['kama'] = self._compute_kama(df['close'], self.lookback['kama'])
        df['kama_slope'] = df['kama'].diff()
        
        # Volatility (rolling std)
        df['volatility'] = df['returns'].rolling(self.lookback['vol']).std()
        
        # Hurst exponent (simplified rolling estimate)
        df['hurst'] = df['close'].rolling(self.lookback['hurs']).apply(
            self._simple_hurst, raw=False
        )
        
        # Momentum (ROC)
        df['momentum_10'] = (df['close'] / df['close'].shift(10) - 1) * 100
        
        # Volume features
        if 'volume' in df.columns:
            df['volume_ma'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_ma']
        
        # Lag features
        for lag in [1, 2, 3]:
            df[f'returns_lag{lag}'] = df['returns'].shift(lag)
            df[f'rsi_lag{lag}'] = df['rsi'].shift(lag)
        
        return df
    
    def compute_pair_features(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame,
        symbol_a: str,
        symbol_b: str
    ) -> pd.DataFrame:
        """
        Compute pair-specific features for two assets.
        
        Asset A is the high-volatility leg, B is the low-volatility hedge.
        """
        # Align dataframes on timestamp
        merged = pd.merge(
            df_a[['timestamp', 'close', 'high', 'low', 'returns']],
            df_b[['timestamp', 'close', 'high', 'low', 'returns']],
            on='timestamp',
            suffixes=('_a', '_b')
        )
        
        # Price ratio and log spread
        merged['price_ratio'] = merged['close_a'] / merged['close_b']
        merged['log_spread'] = np.log(merged['close_a']) - np.log(merged['close_b'])
        
        # Spread z-score (20-day rolling)
        rolled_mean = merged['log_spread'].rolling(20).mean()
        rolled_std = merged['log_spread'].rolling(20).std()
        merged['spread_zscore'] = (merged['log_spread'] - rolled_mean) / rolled_std
        
        # Spread velocity
        merged['spread_velocity'] = merged['spread_zscore'].diff()
        
        # Volatility ratio
        vol_a = df_a['close'].rolling(14).std()
        vol_b = df_b['close'].rolling(14).std()
        merged['vol_ratio'] = vol_a / vol_b
        
        # Rolling correlation (20-day)
        merged['correlation_20'] = merged['returns_a'].rolling(20).corr(merged['returns_b'])
        
        # Cointegration residual (simplified OLS)
        merged['cointegration_residual'] = self._compute_cointegration_residual(
            merged['log_spread']
        )
        
        # Differential RSI
        rsi_a = self._compute_rsi(df_a['close'], 14)
        rsi_b = self._compute_rsi(df_b['close'], 14)
        rsi_diff = pd.concat([rsi_a, rsi_b], axis=1)
        merged['differential_rsi'] = rsi_diff.iloc[:, 0] - rsi_diff.iloc[:, 1]
        
        # Differential momentum
        mom_a = (df_a['close'] / df_a['close'].shift(10) - 1) * 100
        mom_b = (df_b['close'] / df_b['close'].shift(10) - 1) * 100
        mom_diff = pd.concat([mom_a, mom_b], axis=1)
        merged['differential_momentum'] = mom_diff.iloc[:, 0] - mom_diff.iloc[:, 1]
        
        # Add time features
        merged = self._add_time_features(merged)
        
        return merged
    
    def _compute_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Compute RSI indicator."""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))
    
    def _compute_atr(self, df: pd.DataFrame, period: int = 14) -> pd.Series:
        """Compute Average True Range."""
        high_low = df['high'] - df['low']
        high_close = np.abs(df['high'] - df['close'].shift())
        low_close = np.abs(df['low'] - df['close'].shift())
        ranges = pd.concat([high_low, high_close, low_close], axis=1)
        true_range = np.max(ranges, axis=1)
        return true_range.rolling(period).mean()
    
    def _compute_kama(self, prices: pd.Series, period: int = 20) -> pd.Series:
        """Compute Kaufman Adaptive Moving Average."""
        change = (prices - prices.shift(period)).abs()
        volatility = prices.diff().abs().rolling(period).sum()
        er = change / volatility.replace(0, np.nan)
        
        sc = (er * (2/3 - 2/31) + 2/31) ** 2
        kama = pd.Series(index=prices.index, dtype=float)
        kama.iloc[period] = prices.iloc[period]
        
        for i in range(period + 1, len(prices)):
            kama.iloc[i] = kama.iloc[i-1] + sc.iloc[i] * (prices.iloc[i] - kama.iloc[i-1])
        
        return kama
    
    def _simple_hurst(self, prices: pd.Series) -> float:
        """Simple Hurst exponent estimate."""
        if len(prices) < 10:
            return 0.5
        
        lags = range(2, min(20, len(prices)//2))
        tau = [np.std(np.subtract(prices[lag:], prices[:-lag])) for lag in lags]
        
        try:
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0]
        except:
            return 0.5
    
    def _compute_cointegration_residual(self, log_spread: pd.Series) -> pd.Series:
        """Compute cointegration residual from log spread."""
        # Simplified: just use deviation from rolling mean
        rolling_mean = log_spread.rolling(20).mean()
        return log_spread - rolling_mean
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time-based features."""
        df = df.copy()
        
        if 'timestamp' in df.columns:
            ts = pd.to_datetime(df['timestamp'])
            df['hour'] = ts.dt.hour
            df['day_of_week'] = ts.dt.dayofweek
            df['month'] = ts.dt.month
            
            # Sin/cos encoding for cyclical features
            df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
            df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
            df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
            df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Session flag (Asian, London, NY)
        def get_session(hour):
            if 0 <= hour < 8:
                return 0  # Asian
            elif 8 <= hour < 16:
                return 1  # London
            else:
                return 2  # NY
        
        if 'hour' in df.columns:
            df['session'] = df['hour'].apply(get_session)
        
        return df
    
    def build_feature_matrix(
        self,
        pair_data: pd.DataFrame,
        single_features_a: pd.DataFrame,
        single_features_b: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Build final feature matrix combining all features.
        """
        # Merge single asset features
        features = pair_data.copy()
        
        # Add individual features with suffixes
        for col in ['rsi', 'atr', 'volatility', 'hurst', 'momentum_10']:
            if col in single_features_a.columns:
                features[f'{col}_a'] = single_features_a[col].values
            if col in single_features_b.columns:
                features[f'{col}_b'] = single_features_b[col].values
        
        # Drop NaN rows
        features = features.dropna()
        
        return features
