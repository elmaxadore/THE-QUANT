# THE QUANT - Model Training Guide

## Overview

This training pipeline uses **real market data** (not simulated) to train ML models for THE QUANT trading system. It implements rigorous walk-forward validation to prevent overfitting.

## Data Sources

The system fetches real historical data from multiple sources:

| Asset Class | Source | Examples | API Key Required |
|-------------|--------|----------|------------------|
| Forex | yfinance | EURUSD, GBPUSD, USDJPY | No |
| Indices | yfinance | ^GSPC, ^DJI, ^IXIC | No |
| Crypto | CCXT (Binance) | BTC/USDT, ETH/USDT | No |
| Deriv Synthetic | Deriv WebSocket | R_10, R_100, BOOM1000 | Yes (optional) |

## Quick Start

### Basic Training (No API Keys Needed)

```bash
cd python_training

# Train with default settings (Forex + Crypto + Indices)
python train_real_models.py --model gbdt --days 365
```

### Training with Deriv Synthetic Data

If you want to include Deriv synthetic indices:

```bash
# Option 1: Command line argument
python train_real_models.py --model gbdt --days 365 --deriv-token YOUR_TOKEN

# Option 2: Environment variable (recommended)
export DERIV_API_TOKEN='your_token_here'
python train_real_models.py --model gbdt --days 365
```

⚠️ **Security Note**: Never commit API tokens to git. Always use environment variables.

## Command Line Options

```bash
python train_real_models.py [OPTIONS]

Options:
  --config PATH       Path to JSON config file
  --model TYPE        Model type: gbdt, mlp, or rf (default: gbdt)
  --deriv-token TOK   Deriv API token (or set DERIV_API_TOKEN env var)
  --output-dir DIR    Output directory for models (default: ../rust_trading_system/models)
  --days NUM          Days of historical data (default: 730)
  --help              Show this help message
```

## Configuration File

Create a JSON config file for custom settings:

```json
{
  "symbols": [
    {"symbol": "EURUSD", "type": "forex", "timeframe": "1h"},
    {"symbol": "BTC/USDT", "type": "crypto", "timeframe": "1h"},
    {"symbol": "^GSPC", "type": "indices", "timeframe": "1d"},
    {"symbol": "R_100", "type": "deriv", "timeframe": "1h"}
  ],
  "data_days": 730,
  "feature_params": {
    "use_lags": true,
    "use_stats": true,
    "horizon": 5,
    "threshold": 0.001
  },
  "validation": {
    "train_window": 500,
    "test_window": 100,
    "step": 50
  }
}
```

Then run:
```bash
python train_real_models.py --config my_config.json
```

## Output Files

After training completes, the following files are generated in `rust_trading_system/models/`:

- `GradientBoostingClassifier.onnx` - Model in ONNX format for Rust inference
- `gbdt_model.joblib` - Python model backup
- `gbdt_scaler.joblib` - Feature scaler
- `training_metadata.json` - Training details, features, and metrics

## Understanding the Results

### Walk-Forward Validation

The system uses walk-forward validation with multiple splits:
- **Train Window**: Historical data used for training (default: 500 periods)
- **Test Window**: Future data used for testing (default: 100 periods)  
- **Step**: How far to move forward each iteration (default: 50 periods)

This provides realistic performance estimates without look-ahead bias.

### Key Metrics

- **Sharpe Ratio**: Risk-adjusted returns (>1.0 is good)
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profit / Gross loss (>1.5 is good)
- **Accuracy**: Prediction accuracy on test set

### Interpreting Results

```
Walk-Forward Results Summary:
  avg_sharpe_ratio: 1.25        # Good risk-adjusted returns
  avg_max_drawdown: -0.08       # 8% maximum drawdown
  avg_win_rate: 0.55            # 55% winning trades
  avg_profit_factor: 1.8        # $1.80 profit per $1 loss
  test_accuracy: 0.52           # 52% prediction accuracy
```

## Feature Engineering

The pipeline creates 73 features automatically:

### Technical Indicators (33 features)
- Moving averages (SMA, EMA)
- MACD, RSI, Stochastic
- Bollinger Bands
- ATR, ADX, CCI
- Momentum indicators

### Lag Features (20 features)
- Lagged returns (1, 2, 3, 5, 10 periods)
- Lagged volatility
- Lagged RSI
- Lagged volume ratios

### Statistical Features (19 features)
- Rolling mean/std
- Rolling skewness/kurtosis
- Price position in range

## Preventing Overfitting

This pipeline implements several anti-overfitting measures:

1. **Walk-Forward Validation**: Tests on unseen future data
2. **Temporal Split**: Train/test split respects time order
3. **Transaction Costs**: Includes realistic trading costs (0.01% + slippage)
4. **Feature Standardization**: All features scaled to zero mean, unit variance
5. **Cross-Validation**: Grid search with 3-fold CV during hyperparameter tuning

## Retraining Schedule

Recommended retraining frequencies:

| Market Condition | Frequency | Data Window |
|-----------------|-----------|-------------|
| Normal | Weekly | 365 days |
| High Volatility | Daily | 180 days |
| Model Degradation | Immediate | 90-365 days |

Retrain when:
- Sharpe ratio drops below 0.5 in live trading
- Win rate falls below 45%
- Market regime changes significantly

## Troubleshooting

### "No data fetched"
- Check internet connection
- Verify symbol names (e.g., "EURUSD" not "EUR/USD")
- For Deriv data, ensure token is valid

### "KeyError" in feature engineering
- Usually caused by insufficient data points
- Increase `--days` parameter or reduce indicator periods

### Poor backtest results
- This is normal for random/dummy models
- Real models should show Sharpe > 0.5
- Consider adjusting horizon/threshold parameters

### ONNX export fails
- Install: `pip install skl2onnx`
- Some sklearn models may not be fully supported

## Next Steps

1. Review `training_metadata.json` for detailed results
2. Models are automatically placed in `rust_trading_system/models/`
3. The Rust system will load the ONNX model automatically
4. Monitor live performance and retrain as needed

## Security Best Practices

✅ **DO**:
- Use environment variables for API tokens
- Revoke tokens after training if not needed
- Keep tokens out of git repositories
- Use separate tokens for development/production

❌ **DON'T**:
- Hardcode tokens in scripts
- Commit tokens to version control
- Share tokens in chat/logs
- Use production tokens in development

## Support

For issues or questions:
1. Check the training logs for error messages
2. Verify data is being fetched correctly
3. Ensure sufficient historical data is available
4. Review walk-forward results for model quality
