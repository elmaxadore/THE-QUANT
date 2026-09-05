#!/bin/bash
# THE QUANT v4.1 - Train on All Major Assets
# This script trains models on all major asset classes in batches

set -e

echo "=========================================="
echo "THE QUANT v4.1 - Multi-Asset Training"
echo "=========================================="
echo ""

cd /workspace/python

# Install dependencies if needed
echo "[*] Checking dependencies..."
pip install -q yfinance ccxt scikit-learn torch onnx skl2onnx 2>/dev/null || true

# Create results directory
mkdir -p ../results
mkdir -p ../models/strategies

echo ""
echo "=========================================="
echo "Starting Training on All Major Assets"
echo "=========================================="
echo ""

# Run the complete pipeline with all major assets
# This will train on: 
# - 10 Forex pairs
# - 8 Crypto pairs  
# - 7 Indices/Commodities
# Total: 25 assets

python complete_training_pipeline.py --all_major --start_date 2023-01-01 --end_date 2024-12-01 --n_regimes 4

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Results saved to: /workspace/results/"
echo "Models saved to: /workspace/models/"
echo ""
echo "Next steps:"
echo "1. Review /workspace/results/training_results.json"
echo "2. Check individual model files in /workspace/models/strategies/"
echo "3. Deploy models to Rust trading engine"
