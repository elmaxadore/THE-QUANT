#!/usr/bin/env python3
import sys
sys.path.insert(0, 'python_training')
from data_ingestion import get_market_data

# Test with correct symbol format
symbols = [{'type': 'forex', 'symbol': 'EURUSD'}]
df = get_market_data(symbols, days=100)
print('Forex data shape:', df.shape if df is not None else 'None')
if df is not None:
    print(df.head())
    print(df.tail())
