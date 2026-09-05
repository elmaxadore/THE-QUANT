"""
Main Training Pipeline for THE QUANT Trading System
Uses real market data, walk-forward validation, and exports to ONNX
"""

import os
import sys
import json
import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV
import joblib

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from data_ingestion import get_market_data
from feature_engineering import prepare_features, get_feature_columns
from backtesting import WalkForwardValidator, run_walk_forward_backtest


def load_config(config_path: str = None) -> dict:
    """Load training configuration."""
    
    default_config = {
        "symbols": [
            {"symbol": "EURUSD", "type": "forex", "timeframe": "1h"},
            {"symbol": "GBPUSD", "type": "forex", "timeframe": "1h"},
            {"symbol": "BTC/USDT", "type": "crypto", "timeframe": "1h"},
            {"symbol": "^GSPC", "type": "indices", "timeframe": "1d"}
        ],
        "data_days": 730,  # 2 years of data
        "feature_params": {
            "use_lags": True,
            "use_stats": True,
            "horizon": 5,
            "threshold": 0.001
        },
        "validation": {
            "train_window": 500,
            "test_window": 100,
            "step": 50
        },
        "models": {
            "gbdt": {
                "enabled": True,
                "params": {
                    "n_estimators": [100, 200],
                    "max_depth": [3, 5, 7],
                    "learning_rate": [0.01, 0.1]
                }
            },
            "mlp": {
                "enabled": False,
                "params": {
                    "hidden_layer_sizes": [(50,), (100,), (50, 25)],
                    "alpha": [0.0001, 0.001],
                    "learning_rate": ['constant', 'adaptive']
                }
            }
        },
        "export": {
            "onnx": True,
            "save_dir": "../rust_trading_system/models"
        }
    }
    
    if config_path and os.path.exists(config_path):
        with open(config_path, 'r') as f:
            user_config = json.load(f)
            # Merge configs
            for key in user_config:
                if isinstance(user_config[key], dict) and key in default_config:
                    default_config[key].update(user_config[key])
                else:
                    default_config[key] = user_config[key]
    
    return default_config


def train_model(X_train: np.ndarray, y_train: np.ndarray, 
                X_val: np.ndarray, y_val: np.ndarray,
                model_type: str = 'gbdt',
                param_grid: dict = None) -> tuple:
    """
    Train and tune a model.
    
    Returns:
        Tuple of (best_model, best_params, best_score)
    """
    
    print(f"\nTraining {model_type.upper()} model...")
    
    if model_type == 'gbdt':
        base_model = GradientBoostingClassifier(random_state=42)
        default_params = {
            'n_estimators': [100],
            'max_depth': [5],
            'learning_rate': [0.1]
        }
    elif model_type == 'mlp':
        base_model = MLPClassifier(random_state=42, max_iter=500, early_stopping=True)
        default_params = {
            'hidden_layer_sizes': [(50,)],
            'alpha': [0.0001],
            'learning_rate': ['constant']
        }
    elif model_type == 'rf':
        base_model = RandomForestClassifier(random_state=42)
        default_params = {
            'n_estimators': [100],
            'max_depth': [10],
            'min_samples_split': [2]
        }
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    params = param_grid if param_grid else default_params
    
    # Grid search with cross-validation
    grid_search = GridSearchCV(
        base_model,
        params,
        cv=3,
        scoring='f1_macro',
        n_jobs=-1,
        verbose=1
    )
    
    grid_search.fit(X_train, y_train)
    
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    
    # Evaluate on validation set
    val_score = best_model.score(X_val, y_val)
    
    print(f"Best parameters: {best_params}")
    print(f"CV Score (F1): {best_score:.4f}")
    print(f"Validation Score: {val_score:.4f}")
    
    return best_model, best_params, best_score


def export_to_onnx(model, feature_names: list, model_path: str):
    """Export trained model to ONNX format."""
    
    try:
        import skl2onnx
        from skl2onnx import convert_sklearn
        from skl2onnx.common.data_types import FloatTensorType
        
        # Convert to ONNX
        initial_type = [('float_input', FloatTensorType([None, len(feature_names)]))]
        
        onnx_model = convert_sklearn(model, initial_types=initial_type)
        
        # Save
        onnx_path = os.path.join(model_path, f"{model.__class__.__name__}.onnx")
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        
        print(f"Model exported to ONNX: {onnx_path}")
        return onnx_path
        
    except ImportError:
        print("Warning: skl2onnx not installed. Skipping ONNX export.")
        print("Install with: pip install skl2onnx")
        return None
    except Exception as e:
        print(f"Error exporting to ONNX: {e}")
        return None


def save_metadata(metadata: dict, model_path: str):
    """Save training metadata."""
    
    metadata_path = os.path.join(model_path, "training_metadata.json")
    
    # Convert numpy types to Python types for JSON serialization
    def convert_numpy(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {str(k): convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif pd.isna(obj) if isinstance(obj, (float, np.floating)) else False:
            return None
        return obj
    
    metadata_clean = convert_numpy(metadata)
    
    with open(metadata_path, 'w') as f:
        json.dump(metadata_clean, f, indent=2, default=str)
    
    print(f"Metadata saved to: {metadata_path}")


def main():
    parser = argparse.ArgumentParser(description='Train trading models on real market data')
    parser.add_argument('--config', type=str, help='Path to config file')
    parser.add_argument('--model', type=str, default='gbdt', choices=['gbdt', 'mlp', 'rf'],
                       help='Model type to train')
    parser.add_argument('--deriv-token', type=str, help='Deriv API token (or set DERIV_API_TOKEN env var)')
    parser.add_argument('--output-dir', type=str, default='../rust_trading_system/models',
                       help='Output directory for models')
    parser.add_argument('--days', type=int, default=730, help='Days of historical data')
    
    args = parser.parse_args()
    
    # Set Deriv token if provided
    if args.deriv_token:
        os.environ['DERIV_API_TOKEN'] = args.deriv_token
        print("Deriv API token set from command line")
    elif os.getenv('DERIV_API_TOKEN'):
        print("Deriv API token found in environment")
    else:
        print("No Deriv API token provided - will skip Deriv synthetic data")
    
    # Load configuration
    config = load_config(args.config)
    config['data_days'] = args.days
    
    print("\n" + "="*60)
    print("THE QUANT - Model Training Pipeline")
    print("="*60)
    print(f"\nConfiguration:")
    print(f"  Symbols: {len(config['symbols'])} assets")
    print(f"  Historical data: {config['data_days']} days")
    print(f"  Model: {args.model.upper()}")
    print(f"  Horizon: {config['feature_params']['horizon']} periods")
    
    # Step 1: Fetch real market data
    print("\n" + "-"*60)
    print("Step 1: Fetching Real Market Data")
    print("-"*60)
    
    try:
        data = get_market_data(config['symbols'], days=config['data_days'])
        
        if data.empty:
            print("ERROR: No data fetched. Check your symbols and network connection.")
            sys.exit(1)
        
        print(f"\nData summary:")
        print(data.groupby(['symbol', 'asset_class']).size().to_string())
        
    except Exception as e:
        print(f"ERROR fetching data: {e}")
        sys.exit(1)
    
    # Step 2: Feature Engineering
    print("\n" + "-"*60)
    print("Step 2: Feature Engineering")
    print("-"*60)
    
    featured_data = prepare_features(
        data,
        use_lags=config['feature_params']['use_lags'],
        use_stats=config['feature_params']['use_stats'],
        horizon=config['feature_params']['horizon'],
        threshold=config['feature_params']['threshold']
    )
    
    feature_cols, exclude_cols = get_feature_columns(featured_data)
    print(f"\nUsing {len(feature_cols)} features")
    
    # Prepare training data
    X = featured_data[feature_cols].values
    y = featured_data['target'].values
    prices = featured_data['close']
    
    # Handle any remaining NaN values
    valid_mask = ~np.isnan(y) & ~np.any(np.isnan(X), axis=1)
    X = X[valid_mask]
    y = y[valid_mask]
    prices = prices.iloc[valid_mask].reset_index(drop=True)
    
    print(f"Final dataset: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"Target distribution: {dict(zip(*np.unique(y, return_counts=True)))}")
    
    # Step 3: Train-Test Split (temporal)
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print(f"\nTrain size: {len(X_train)}, Test size: {len(X_test)}")
    
    # Step 4: Walk-Forward Validation
    print("\n" + "-"*60)
    print("Step 4: Walk-Forward Validation")
    print("-"*60)
    
    validator = WalkForwardValidator(
        train_window=config['validation']['train_window'],
        test_window=config['validation']['test_window'],
        step=config['validation']['step']
    )
    
    def create_model():
        if args.model == 'gbdt':
            return GradientBoostingClassifier(
                n_estimators=100,
                max_depth=5,
                learning_rate=0.1,
                random_state=42
            )
        elif args.model == 'mlp':
            return MLPClassifier(
                hidden_layer_sizes=(50,),
                alpha=0.0001,
                max_iter=500,
                early_stopping=True,
                random_state=42
            )
        elif args.model == 'rf':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
    
    wf_results = run_walk_forward_backtest(
        create_model,
        X_train_scaled, y_train, prices.iloc[:len(X_train_scaled)],
        validator
    )
    
    print("\nWalk-Forward Results Summary:")
    for key, value in wf_results.items():
        if not key.endswith('_results') and isinstance(value, (int, float)):
            if 'avg' in key or 'std' in key:
                print(f"  {key}: {value:.4f}" if isinstance(value, float) else f"  {key}: {value}")
    
    # Step 5: Final Model Training
    print("\n" + "-"*60)
    print("Step 5: Final Model Training on Full Training Set")
    print("-"*60)
    
    final_model, best_params, cv_score = train_model(
        X_train_scaled, y_train,
        X_test_scaled, y_test,
        model_type=args.model
    )
    
    # Evaluate on test set
    test_accuracy = final_model.score(X_test_scaled, y_test)
    print(f"\nTest Accuracy: {test_accuracy:.4f}")
    
    # Step 6: Export Models
    print("\n" + "-"*60)
    print("Step 6: Exporting Models")
    print("-"*60)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save scaler
    scaler_path = output_dir / f"{args.model}_scaler.joblib"
    joblib.dump(scaler, scaler_path)
    print(f"Scaler saved to: {scaler_path}")
    
    # Save model
    model_path = output_dir / f"{args.model}_model.joblib"
    joblib.dump(final_model, model_path)
    print(f"Model saved to: {model_path}")
    
    # Export to ONNX
    if config['export']['onnx']:
        export_to_onnx(final_model, feature_cols, str(output_dir))
    
    # Save metadata
    metadata = {
        'timestamp': datetime.now().isoformat(),
        'model_type': args.model,
        'feature_count': len(feature_cols),
        'features': feature_cols,
        'training_samples': len(X_train),
        'test_samples': len(X_test),
        'test_accuracy': test_accuracy,
        'cv_score': cv_score,
        'best_params': best_params,
        'walk_forward_results': {k: v for k, v in wf_results.items() if not k.endswith('_results')},
        'config': config,
        'symbols_trained': [s['symbol'] for s in config['symbols']]
    }
    
    save_metadata(metadata, str(output_dir))
    
    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"\nModels saved to: {output_dir.absolute()}")
    print(f"\nNext steps:")
    print(f"  1. Review training_metadata.json for detailed results")
    print(f"  2. Copy .onnx files to rust_trading_system/models/")
    print(f"  3. Update model configuration in Rust code if needed")
    print(f"  4. Run backtests in production environment")


if __name__ == "__main__":
    main()
