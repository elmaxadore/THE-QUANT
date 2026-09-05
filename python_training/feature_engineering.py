"""
Feature Engineering Module for Trading Models
Implements technical indicators, statistical features, and lag features
"""

import pandas as pd
import numpy as np
from typing import List, Tuple
import pandas_ta as ta


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Add comprehensive technical indicators to dataframe."""
    
    # Make a copy to avoid modifying original
    data = df.copy()
    
    # Ensure numeric types
    for col in ['open', 'high', 'low', 'close', 'volume']:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    # Drop any rows with NaN from conversion
    initial_len = len(data)
    data = data.dropna(subset=['open', 'high', 'low', 'close'])
    if len(data) < initial_len:
        print(f"Dropped {initial_len - len(data)} rows with invalid OHLC data")
    
    # Process each symbol separately to avoid timestamp issues
    if 'symbol' in data.columns:
        result_dfs = []
        for sym in data['symbol'].unique():
            sym_data = data[data['symbol'] == sym].copy()
            sym_data = sym_data.sort_values('timestamp' if 'timestamp' in sym_data.columns else data.index.name or 0)
            sym_result = _compute_indicators_for_symbol(sym_data)
            result_dfs.append(sym_result)
        data = pd.concat(result_dfs, ignore_index=True)
    else:
        data = _compute_indicators_for_symbol(data)
    
    print(f"Added {len([c for c in data.columns if c not in df.columns])} technical indicator features")
    
    return data


def _compute_indicators_for_symbol(data: pd.DataFrame) -> pd.DataFrame:
    """Compute technical indicators for a single symbol's data."""
    
    # Set timestamp as index if available
    if 'timestamp' in data.columns:
        data = data.set_index('timestamp')
    
    # Trend Indicators
    data['sma_20'] = ta.sma(data['close'], length=20)
    data['sma_50'] = ta.sma(data['close'], length=50)
    data['ema_12'] = ta.ema(data['close'], length=12)
    data['ema_26'] = ta.ema(data['close'], length=26)
    
    # MACD
    macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
    data['macd'] = macd[f'MACD_12_26_9']
    data['macd_signal'] = macd[f'MACDs_12_26_9']
    data['macd_hist'] = macd[f'MACDh_12_26_9']
    
    # RSI
    data['rsi_14'] = ta.rsi(data['close'], length=14)
    data['rsi_7'] = ta.rsi(data['close'], length=7)
    
    # Bollinger Bands - use dynamic column names based on pandas_ta version
    bbands = ta.bbands(data['close'], length=20, std=2)
    bb_cols = bbands.columns.tolist()
    bbu_col = [c for c in bb_cols if c.startswith('BBU')][0]
    bbl_col = [c for c in bb_cols if c.startswith('BBL')][0]
    bbm_col = [c for c in bb_cols if c.startswith('BBM')][0]
    
    data['bb_upper'] = bbands[bbu_col]
    data['bb_lower'] = bbands[bbl_col]
    data['bb_middle'] = bbands[bbm_col]
    data['bb_width'] = (data['bb_upper'] - data['bb_lower']) / data['bb_middle']
    data['bb_pct'] = (data['close'] - data['bb_lower']) / (data['bb_upper'] - data['bb_lower'])
    
    # ATR (Volatility)
    data['atr_14'] = ta.atr(data['high'], data['low'], data['close'], length=14)
    data['atr_7'] = ta.atr(data['high'], data['low'], data['close'], length=7)
    
    # ADX (Trend Strength)
    adx = ta.adx(data['high'], data['low'], data['close'], length=14)
    adx_cols = adx.columns.tolist()
    adx_col = [c for c in adx_cols if c.startswith('ADX')][0]
    dmp_col = [c for c in adx_cols if c.startswith('DMP')][0]
    dmn_col = [c for c in adx_cols if c.startswith('DMN')][0]
    
    data['adx'] = adx[adx_col]
    data['plus_di'] = adx[dmp_col]
    data['minus_di'] = adx[dmn_col]
    
    # Stochastic Oscillator
    stoch = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3)
    stoch_cols = stoch.columns.tolist()
    stochk_col = [c for c in stoch_cols if c.startswith('STOCHk')][0]
    stochd_col = [c for c in stoch_cols if c.startswith('STOCHd')][0]
    
    data['stoch_k'] = stoch[stochk_col]
    data['stoch_d'] = stoch[stochd_col]
    
    # CCI
    data['cci_20'] = ta.cci(data['high'], data['low'], data['close'], length=20)
    
    # Williams %R
    data['willr_14'] = ta.willr(data['high'], data['low'], data['close'], length=14)
    
    # Momentum
    data['mom_10'] = ta.mom(data['close'], length=10)
    data['roc_10'] = ta.roc(data['close'], length=10)
    
    # Volume indicators
    data['obv'] = ta.obv(data['close'], data['volume'])
    data['volume_sma_20'] = ta.sma(data['volume'], length=20)
    data['volume_ratio'] = data['volume'] / data['volume_sma_20'].replace(0, np.nan)
    
    # Price position features
    data['hl_range'] = data['high'] - data['low']
    data['oc_range'] = data['close'] - data['open']
    data['hl_pct'] = data['hl_range'] / data['close']
    data['oc_pct'] = data['oc_range'] / data['close']
    
    # Normalized ATR (volatility relative to price)
    data['natr'] = data['atr_14'] / data['close'] * 100
    
    # Reset index if we set it
    if 'timestamp' in data.columns or data.index.name == 'timestamp':
        data = data.reset_index()
    
    return data


def add_lag_features(df: pd.DataFrame, lags: List[int] = [1, 2, 3, 5, 10]) -> pd.DataFrame:
    """Add lagged features for time series modeling."""
    
    data = df.copy()
    lag_cols = []
    
    for lag in lags:
        # Lagged returns
        data[f'return_lag_{lag}'] = data['close'].pct_change().shift(lag)
        
        # Lagged volatility
        data[f'volatility_lag_{lag}'] = data['hl_pct'].shift(lag)
        
        # Lagged RSI
        if 'rsi_14' in data.columns:
            data[f'rsi_lag_{lag}'] = data['rsi_14'].shift(lag)
        
        # Lagged volume ratio
        if 'volume_ratio' in data.columns:
            data[f'vol_ratio_lag_{lag}'] = data['volume_ratio'].shift(lag)
        
        lag_cols.extend([f'return_lag_{lag}', f'volatility_lag_{lag}'])
        if 'rsi_14' in data.columns:
            lag_cols.append(f'rsi_lag_{lag}')
        if 'volume_ratio' in data.columns:
            lag_cols.append(f'vol_ratio_lag_{lag}')
    
    print(f"Added {len(lag_cols)} lag features")
    
    return data


def add_statistical_features(df: pd.DataFrame, windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Add rolling statistical features."""
    
    data = df.copy()
    stat_cols = []
    
    for window in windows:
        # Rolling mean and std of returns
        returns = data['close'].pct_change()
        data[f'return_mean_{window}'] = returns.rolling(window=window).mean()
        data[f'return_std_{window}'] = returns.rolling(window=window).std()
        
        # Rolling skewness and kurtosis
        if window >= 10:  # Need enough data points
            data[f'return_skew_{window}'] = returns.rolling(window=window).skew()
            data[f'return_kurt_{window}'] = returns.rolling(window=window).kurt()
        
        # Rolling min/max of price
        data[f'price_min_{window}'] = data['close'].rolling(window=window).min()
        data[f'price_max_{window}'] = data['close'].rolling(window=window).max()
        data[f'price_position_{window}'] = (data['close'] - data[f'price_min_{window}']) / \
                                            (data[f'price_max_{window}'] - data[f'price_min_{window}'])
        
        stat_cols.extend([
            f'return_mean_{window}', f'return_std_{window}',
            f'price_min_{window}', f'price_max_{window}', f'price_position_{window}'
        ])
        
        if window >= 10:
            stat_cols.extend([f'return_skew_{window}', f'return_kurt_{window}'])
    
    print(f"Added {len(stat_cols)} statistical features")
    
    return data


def add_target_variable(df: pd.DataFrame, horizon: int = 5, threshold: float = 0.001) -> pd.DataFrame:
    """
    Add target variable for classification.
    
    Target: 
      1 = price increases by > threshold% in next 'horizon' periods
      0 = price decreases by > threshold% in next 'horizon' periods
      -1 = neutral (between -threshold and +threshold)
    """
    
    data = df.copy()
    
    # Forward return
    data['forward_return'] = data['close'].shift(-horizon) / data['close'] - 1
    
    # Classification target
    data['target'] = 0
    data.loc[data['forward_return'] > threshold, 'target'] = 1
    data.loc[data['forward_return'] < -threshold, 'target'] = -1
    
    # For regression, use raw forward return
    data['target_regression'] = data['forward_return']
    
    print(f"Added target variable with {horizon}-period horizon and {threshold*100:.2f}% threshold")
    print(f"Target distribution: Up={len(data[data['target']==1])}, Neutral={len(data[data['target']==0])}, Down={len(data[data['target']==-1])}")
    
    return data


def prepare_features(df: pd.DataFrame, 
                    use_lags: bool = True,
                    use_stats: bool = True,
                    horizon: int = 5,
                    threshold: float = 0.001) -> pd.DataFrame:
    """
    Complete feature engineering pipeline.
    
    Args:
        df: Input dataframe with OHLCV data
        use_lags: Whether to add lag features
        use_stats: Whether to add statistical features
        horizon: Prediction horizon in periods
        threshold: Threshold for classification target
    
    Returns:
        DataFrame with all features
    """
    
    print("Starting feature engineering...")
    
    # Add technical indicators
    data = add_technical_indicators(df)
    
    # Add lag features
    if use_lags:
        data = add_lag_features(data, lags=[1, 2, 3, 5, 10])
    
    # Add statistical features
    if use_stats:
        data = add_statistical_features(data, windows=[5, 10, 20])
    
    # Add target variable
    data = add_target_variable(data, horizon=horizon, threshold=threshold)
    
    # Drop rows with NaN values (from lagging and rolling calculations)
    initial_len = len(data)
    data = data.dropna()
    dropped = initial_len - len(data)
    print(f"\nDropped {dropped} rows with NaN values ({dropped/initial_len*100:.1f}%)")
    
    print(f"\nFinal dataset: {len(data)} rows, {len(data.columns)} columns")
    
    return data


def get_feature_columns(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Separate feature columns from non-feature columns.
    
    Returns:
        Tuple of (feature_columns, exclude_columns)
    """
    
    exclude = ['symbol', 'asset_class', 'target', 'target_regression', 'forward_return',
               'timestamp', 'open', 'high', 'low', 'close', 'volume']
    
    features = [col for col in df.columns if col not in exclude]
    
    print(f"Identified {len(features)} feature columns")
    
    return features, exclude


if __name__ == "__main__":
    # Test with sample data
    from data_ingestion import get_market_data
    
    symbols = [
        {"symbol": "EURUSD", "type": "forex", "timeframe": "1h"}
    ]
    
    print("Fetching data...")
    data = get_market_data(symbols, days=365)
    
    print("\nEngineering features...")
    featured = prepare_features(data, horizon=5, threshold=0.001)
    
    features, excluded = get_feature_columns(featured)
    print(f"\nFeatures ({len(features)}): {features[:10]}...")
    print(f"Excluded ({len(excluded)}): {excluded}")
