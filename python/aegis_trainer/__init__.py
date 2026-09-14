"""
AEGIS Trainer - Hedged Pairs Training Pipeline for Blue Guardian 5K Account
============================================================================

This module implements the complete training pipeline for the AEGIS hedged pairs
strategy, including:
- Pair universe construction with correlation/cointegration filtering
- LightGBM + MLP ensemble training for directional probability
- Risk simulation enforcing Guardian Shield and daily caps
- ONNX export for Rust deployment

Version: 4.2 Hercules
Account: Blue Guardian Instant $5K
"""

__version__ = "4.2.0"
__author__ = "THE QUANT Team"

from .config import BlueGuardianConfig, PairUniverse
from .data.loader import DataLoader
from .data.features import FeatureEngine
from .data.pairs import PairConstructor
from .models.lightgbm_trainer import LightGBMTrainer
from .models.mlp_trainer import MLPTrainer
from .models.calibration import CalibrationManager
from .models.ensemble import EnsembleModel
from .risk.sizing import PositionSizer
from .risk.shield import GuardianShield
from .risk.consistency import ConsistencyTracker
from .backtest.engine import PairBacktester
from .backtest.metrics import PerformanceMetrics
from .export.onnx_export import ONNXExporter
from .export.manifest import ModelManifest

__all__ = [
    "BlueGuardianConfig",
    "PairUniverse",
    "DataLoader",
    "FeatureEngine",
    "PairConstructor",
    "LightGBMTrainer",
    "MLPTrainer",
    "CalibrationManager",
    "EnsembleModel",
    "PositionSizer",
    "GuardianShield",
    "ConsistencyTracker",
    "PairBacktester",
    "PerformanceMetrics",
    "ONNXExporter",
    "ModelManifest",
]
