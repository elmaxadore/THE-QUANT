"""
Multi-timeframe data loader for AEGIS Trainer.
Fetches and aligns OHLCV data from MT5 or CSV sources.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

logger = logging.getLogger(__name__)


class DataLoader:
    """
    Loads historical market data for multiple symbols and timeframes.
    
    Supports:
    - MT5 integration (live trading)
    - CSV file loading (backtesting/training)
    - Multi-timeframe alignment (M5, H1, D1)
    """
    
    def __init__(self, data_source: str = "csv", data_dir: str = "../data"):
        self.data_source = data_source
        self.data_dir = data_dir
        self.cache: Dict[str, pd.DataFrame] = {}
        
    def load_symbol_data(
        self,
        symbol: str,
        timeframe: str = "M5",
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        bars: int = 10000
    ) -> pd.DataFrame:
        """
        Load OHLCV data for a single symbol.
        
        Args:
            symbol: Trading symbol (e.g., "EURUSD")
            timeframe: Timeframe ("M5", "H1", "D1")
            start_date: Start date (optional)
            end_date: End date (optional)
            bars: Number of bars to load
            
        Returns:
            DataFrame with columns: timestamp, open, high, low, close, volume
        """
        cache_key = f"{symbol}_{timeframe}"
        
        if cache_key in self.cache:
            df = self.cache[cache_key].copy()
        else:
            # Try to load from CSV first
            csv_path = f"{self.data_dir}/{symbol}_{timeframe}.csv"
            try:
                df = pd.read_csv(csv_path, parse_dates=['timestamp'])
                logger.info(f"Loaded {len(df)} bars for {symbol} {timeframe}")
            except FileNotFoundError:
                # Generate synthetic data for testing
                logger.warning(f"CSV not found, generating synthetic data for {symbol}")
                df = self._generate_synthetic_data(symbol, timeframe, bars)
            
            self.cache[cache_key] = df
        
        # Apply date filters
        if start_date:
            df = df[df['timestamp'] >= start_date]
        if end_date:
            df = df[df['timestamp'] <= end_date]
        
        # Limit to most recent bars if needed
        if len(df) > bars:
            df = df.tail(bars).reset_index(drop=True)
        
        return df
    
    def load_multi_timeframe(
        self,
        symbol: str,
        timeframes: List[str] = ["M5", "H1", "D1"],
        bars: int = 10000
    ) -> Dict[str, pd.DataFrame]:
        """Load data for multiple timeframes."""
        result = {}
        for tf in timeframes:
            result[tf] = self.load_symbol_data(symbol, tf, bars=bars)
        return result
    
    def load_universe_data(
        self,
        symbols: List[str],
        timeframe: str = "M5",
        bars: int = 10000
    ) -> Dict[str, pd.DataFrame]:
        """Load data for all symbols in universe."""
        result = {}
        for symbol in symbols:
            result[symbol] = self.load_symbol_data(symbol, timeframe, bars=bars)
        return result
    
    def _generate_synthetic_data(
        self,
        symbol: str,
        timeframe: str,
        bars: int
    ) -> pd.DataFrame:
        """Generate synthetic OHLCV data for testing."""
        np.random.seed(42)
        
        # Base price by symbol
        base_prices = {
            "EURUSD": 1.1000, "GBPUSD": 1.2700, "USDJPY": 150.00,
            "AUDUSD": 0.6500, "NZDUSD": 0.6100, "USDCAD": 1.3600,
            "USDCHF": 0.8900, "EURGBP": 0.8600,
            "XAUUSD": 2000.0, "XAGUSD": 24.0,
            "US30": 38000.0, "US100": 17000.0
        }
        
        base_price = base_prices.get(symbol, 1.0)
        
        # Volatility by timeframe
        tf_vol = {"M5": 0.0005, "H1": 0.002, "D1": 0.01}
        vol = tf_vol.get(timeframe, 0.001)
        
        # Generate returns
        returns = np.random.normal(0, vol, bars)
        
        # Generate price series
        prices = base_price * np.cumprod(1 + returns)
        
        # Generate OHLCV
        data = []
        for i in range(bars):
            open_price = prices[i] * (1 + np.random.uniform(-vol/2, vol/2))
            high_price = max(open_price, prices[i]) * (1 + np.random.uniform(0, vol/2))
            low_price = min(open_price, prices[i]) * (1 - np.random.uniform(0, vol/2))
            close_price = prices[i]
            volume = np.random.uniform(100, 1000)
            
            data.append({
                'timestamp': datetime.now() - timedelta(minutes=bars-i),
                'open': open_price,
                'high': high_price,
                'low': low_price,
                'close': close_price,
                'volume': volume
            })
        
        df = pd.DataFrame(data)
        return df
    
    def align_timeframes(
        self,
        fast_df: pd.DataFrame,
        slow_df: pd.DataFrame,
        fast_tf: str = "M5",
        slow_tf: str = "H1"
    ) -> pd.DataFrame:
        """
        Align slower timeframe data to faster timeframe index.
        Forward-fill slower data to match fast index.
        """
        # Set timestamp as index
        fast_df = fast_df.set_index('timestamp')
        slow_df = slow_df.set_index('timestamp')
        
        # Resample slow to fast frequency, forward fill
        aligned = slow_df.reindex(fast_df.index, method='ffill')
        
        # Merge
        result = fast_df.join(aligned, rsuffix=f'_{slow_tf}')
        return result.reset_index()
