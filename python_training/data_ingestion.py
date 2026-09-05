"""
Real Market Data Ingestion Module
Supports: Forex, Indices (via yfinance), Crypto (via CCXT)
"""

import pandas as pd
import numpy as np
import yfinance as yf
import ccxt
from datetime import datetime, timedelta
import time
from typing import List, Dict, Optional
import os

class RealDataFetcher:
    """Fetches real historical market data from free sources."""
    
    def __init__(self):
        self.crypto_exchange = ccxt.binance()  # Free, no API key needed for public data
        
    def fetch_forex_data(self, symbol: str, timeframe: str = '1h', days: int = 365) -> pd.DataFrame:
        """
        Fetch Forex data via yfinance.
        Symbols: EURUSD=X, GBPUSD=X, USDJPY=X, etc.
        """
        try:
            ticker = yf.Ticker(f"{symbol}=X")
            end = datetime.now()
            start = end - timedelta(days=days)
            
            df = ticker.history(start=start, end=end, interval=timeframe)
            
            if df.empty:
                print(f"Warning: No data found for {symbol}")
                return pd.DataFrame()
            
            # Standardize columns
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            df['symbol'] = symbol
            df['asset_class'] = 'forex'
            df.index.name = 'timestamp'
            
            print(f"Fetched {len(df)} candles for {symbol} (Forex)")
            return df[['open', 'high', 'low', 'close', 'volume', 'symbol', 'asset_class']]
            
        except Exception as e:
            print(f"Error fetching forex data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_indices_data(self, symbol: str, timeframe: str = '1h', days: int = 365) -> pd.DataFrame:
        """
        Fetch Indices data via yfinance.
        Symbols: ^GSPC (S&P 500), ^DJI (Dow Jones), ^IXIC (NASDAQ), etc.
        Note: yfinance indices don't have volume sometimes
        """
        try:
            ticker = yf.Ticker(symbol)
            end = datetime.now()
            start = end - timedelta(days=days)
            
            df = ticker.history(start=start, end=end, interval=timeframe)
            
            if df.empty:
                print(f"Warning: No data found for {symbol}")
                return pd.DataFrame()
            
            # Standardize columns
            df = df.rename(columns={
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Fill missing volume with 0 or estimate
            if 'volume' not in df.columns or df['volume'].isna().all():
                df['volume'] = 0
            
            df['symbol'] = symbol
            df['asset_class'] = 'indices'
            df.index.name = 'timestamp'
            
            print(f"Fetched {len(df)} candles for {symbol} (Indices)")
            return df[['open', 'high', 'low', 'close', 'volume', 'symbol', 'asset_class']]
            
        except Exception as e:
            print(f"Error fetching indices data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_crypto_data(self, symbol: str, timeframe: str = '1h', days: int = 365) -> pd.DataFrame:
        """
        Fetch Crypto data via CCXT (Binance).
        Symbols: BTC/USDT, ETH/USDT, etc.
        """
        try:
            # Convert timeframe to CCXT format
            tf_map = {'1m': '1m', '5m': '5m', '15m': '15m', '1h': '1h', '4h': '4h', '1d': '1d'}
            ccxt_tf = tf_map.get(timeframe, '1h')
            
            since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
            
            # Fetch OHLCV data
            ohlcv = self.crypto_exchange.fetch_ohlcv(symbol, timeframe=ccxt_tf, since=since, limit=5000)
            
            if not ohlcv:
                print(f"Warning: No data found for {symbol}")
                return pd.DataFrame()
            
            df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
            df.set_index('timestamp', inplace=True)
            
            df['symbol'] = symbol
            df['asset_class'] = 'crypto'
            
            print(f"Fetched {len(df)} candles for {symbol} (Crypto)")
            return df[['open', 'high', 'low', 'close', 'volume', 'symbol', 'asset_class']]
            
        except Exception as e:
            print(f"Error fetching crypto data for {symbol}: {e}")
            return pd.DataFrame()
    
    def fetch_deriv_synthetic_data(self, symbol: str, timeframe: str = '1h', days: int = 365) -> Optional[pd.DataFrame]:
        """
        Fetch Deriv Synthetic Indices data via WebSocket.
        Requires DERIV_API_TOKEN environment variable.
        Symbols: R_10, R_25, R_50, R_75, R_100 (Volatility indices)
        
        NOTE: This requires a valid Deriv API token in environment variables.
        If not available, returns None and logs a warning.
        """
        api_token = os.getenv('DERIV_API_TOKEN')
        
        if not api_token:
            print(f"Warning: DERIV_API_TOKEN not set. Skipping Deriv data for {symbol}.")
            print("To fetch Deriv synthetic data, export your token:")
            print("  export DERIV_API_TOKEN='your_token_here'")
            return None
        
        # Import websockets only if needed
        try:
            import websocket
            import json
            import ssl
        except ImportError:
            print("websocket-client not installed. Run: pip install websocket-client")
            return None
        
        # Deriv WebSocket endpoint
        ws_url = f"wss://ws.binaryws.com/websockets/v3?app_id=1089&token={api_token}"
        
        candles = []
        done = False
        
        def on_message(ws, message):
            nonlocal done
            data = json.loads(message)
            
            if 'error' in data:
                print(f"Deriv API Error: {data['error']['message']}")
                done = True
                return
            
            if 'ohlc' in data:
                candle = data['ohlc']
                candles.append({
                    'timestamp': pd.to_datetime(int(candle['open_time']), unit='s'),
                    'open': float(candle['open']),
                    'high': float(candle['high']),
                    'low': float(candle['low']),
                    'close': float(candle['close']),
                    'volume': float(candle.get('volume', 0))
                })
            
            if 'msg_type' in data and data['msg_type'] == 'ohlc' and 'subscription' not in data:
                done = True
                ws.close()
        
        def on_error(ws, error):
            print(f"WebSocket Error: {error}")
        
        def on_close(ws, close_status_code, close_msg):
            pass
        
        def on_open(ws):
            # Request historical candles
            ws.send(json.dumps({
                "ticks_history": symbol,
                "adjust_start_time": 1,
                "count": 5000,
                "end": "latest",
                "start": 1,
                "style": "candles",
                "granularity": int(timeframe.replace('h', '')) * 3600 if 'h' in timeframe else int(timeframe.replace('m', '')) * 60
            }))
        
        try:
            ws = websocket.WebSocketApp(
                ws_url,
                on_open=on_open,
                on_message=on_message,
                on_error=on_error,
                on_close=on_close
            )
            
            ws.run_forever(sslopt={"cert_reqs": ssl.CERT_NONE})
            
            if not candles:
                print(f"No data received for {symbol} from Deriv")
                return None
            
            df = pd.DataFrame(candles)
            df.set_index('timestamp', inplace=True)
            df['symbol'] = symbol
            df['asset_class'] = 'deriv_synthetic'
            
            print(f"Fetched {len(df)} candles for {symbol} (Deriv Synthetic)")
            return df[['open', 'high', 'low', 'close', 'volume', 'symbol', 'asset_class']]
            
        except Exception as e:
            print(f"Error fetching Deriv data for {symbol}: {e}")
            return None


def get_market_data(symbols: List[Dict], days: int = 365) -> pd.DataFrame:
    """
    Fetch data for multiple symbols across different asset classes.
    
    Args:
        symbols: List of dicts with 'symbol', 'type' (forex/indices/crypto/deriv), 'timeframe'
        days: Number of days of historical data
    
    Returns:
        Combined DataFrame with all symbols
    """
    fetcher = RealDataFetcher()
    all_data = []
    
    for sym_info in symbols:
        symbol = sym_info['symbol']
        sym_type = sym_info.get('type', 'forex')
        timeframe = sym_info.get('timeframe', '1h')
        
        if sym_type == 'forex':
            df = fetcher.fetch_forex_data(symbol, timeframe, days)
        elif sym_type == 'indices':
            df = fetcher.fetch_indices_data(symbol, timeframe, days)
        elif sym_type == 'crypto':
            df = fetcher.fetch_crypto_data(symbol, timeframe, days)
        elif sym_type == 'deriv':
            df = fetcher.fetch_deriv_synthetic_data(symbol, timeframe, days)
            if df is None:
                continue  # Skip if Deriv token not available
        else:
            print(f"Unknown asset type: {sym_type}")
            continue
        
        if not df.empty:
            # Reset index to ensure unique timestamps per symbol
            df = df.reset_index()
            # Add a unique identifier combining timestamp and symbol
            df['unique_ts'] = df.index
            all_data.append(df)
    
    if not all_data:
        raise ValueError("No data fetched for any symbol")
    
    combined = pd.concat(all_data, ignore_index=True)
    print(f"\nTotal data: {len(combined)} candles across {len(all_data)} symbols")
    
    return combined


if __name__ == "__main__":
    # Example usage
    symbols_to_fetch = [
        {"symbol": "EURUSD", "type": "forex", "timeframe": "1h"},
        {"symbol": "GBPUSD", "type": "forex", "timeframe": "1h"},
        {"symbol": "BTC/USDT", "type": "crypto", "timeframe": "1h"},
        {"symbol": "^GSPC", "type": "indices", "timeframe": "1h"},
        # {"symbol": "R_100", "type": "deriv", "timeframe": "1h"}  # Uncomment if you have DERIV_API_TOKEN
    ]
    
    data = get_market_data(symbols_to_fetch, days=365)
    print("\nData summary:")
    print(data.groupby(['symbol', 'asset_class']).size())
    print("\nSample data:")
    print(data.head())
