"""
Blue Guardian 5K Account Configuration and Pair Universe Definition
===================================================================

This module defines all account constraints, risk parameters, and eligible
trading pairs for the AEGIS hedged pairs strategy.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Tuple
from enum import Enum


class AssetClass(Enum):
    FOREX_MAJOR = "forex_major"
    FOREX_MINOR = "forex_minor"
    METALS = "metals"
    INDICES = "indices"


@dataclass
class BlueGuardianConfig:
    """
    Blue Guardian Instant 5K Account Rules - NON-NEGOTIABLE CONSTRAINTS
    
    These values are enforced at both Python simulation and Rust execution layers.
    """
    
    # Account Basics
    account_balance: float = 5000.00
    currency: str = "USD"
    
    # Drawdown Limits (Section 1.1)
    max_total_drawdown_pct: float = 0.10  # 10% = $500 hard stop
    max_total_drawdown_usd: float = 500.00
    
    # Guardian Shield (Per-Trade Protection)
    max_loss_per_trade_usd: float = 50.00  # Hard cap per trade
    shield_strike_warning: float = 35.00   # Warning threshold
    max_shield_strikes: int = 2            # Strike 2 = account halted
    
    # Daily Loss Limits
    daily_loss_limit_usd: float = 150.00   # Cumulative realized loss per day
    daily_loss_buffer_usd: float = 10.00   # Soft cap at $140
    
    # Lifetime Loss Floor
    lifetime_loss_floor_usd: float = 250.00  # Pause & manual reset required
    lifetime_loss_buffer_usd: float = 10.00  # Soft cap at $240
    
    # Consistency Rules
    max_daily_profit_concentration_pct: float = 0.15  # No day > 15% of total profit
    max_lot_size_cv: float = 0.40                     # Coefficient of variation limit
    
    # Payout & Extraction
    payout_cap_usd: float = 250.00           # Lifetime maximum payout
    target_daily_profit_usd: float = 25.00   # Optimal extraction rate
    acceptable_daily_band: Tuple[float, float] = (20.00, 30.00)
    daily_profit_hard_cap_usd: float = 35.00  # Prevents 15% concentration breach
    daily_profit_soft_floor_usd: float = 15.00  # Below = size up signal
    
    # Risk Budget
    risk_per_trade_pct: float = 0.005  # 0.5% of equity = $25 base
    risk_per_trade_usd: float = 25.00
    
    # Conviction Thresholds
    min_bias_to_trade: float = 0.20      # |bias| must exceed this
    min_pair_quality_score: float = 0.60 # Minimum composite score
    
    # Extraction Curve Parameters
    extraction_ahead_reduction: float = 0.40   # Reduce size 40% if >20% ahead
    extraction_behind_confidence_boost: float = 0.85  # Min confidence if <20% behind
    extraction_recovery_confidence: float = 0.90      # Min confidence if <40% behind
    
    # Leverage by Asset Class
    leverage_forex: int = 100
    leverage_metals: int = 20
    leverage_indices: int = 100
    
    # Time Constraints
    max_holding_hours: float = 8.0         # Time stop to avoid swap
    day_start_hour_utc: int = 0            # Daily reset at 00:00 UTC
    day_end_hour_utc: int = 23             # Daily reset at 23:59 UTC
    
    # Scaling Rules (Disabled for extraction focus)
    auto_scaling_enabled: bool = False     # Prefer payout over scale-up
    scale_up_profit_threshold_usd: float = 500.00
    scale_up_target_balance: float = 10000.00
    
    # Cost Filters
    max_spread_ratio: float = 0.30         # Spread < 0.3× expected daily range
    min_edge_after_cost: float = 0.20      # Cost < 20% of expected gross profit
    
    # Margin Requirements
    min_free_margin_ratio: float = 2.00    # Free margin > 200% of required
    
    # Correlation Exposure
    max_correlated_exposure: float = 0.70  # Don't hold pairs with corr > 0.7
    
    # Spread Filter
    spread_filter_multiplier: float = 1.50 # Defer if spread > 1.5× average
    
    # Swap Adjustment
    swap_size_reduction: float = 0.20      # Reduce 20% if negative swap


@dataclass
class SymbolSpec:
    """Specification for a single trading symbol."""
    symbol: str
    asset_class: AssetClass
    contract_size: float
    pip_value: float
    typical_spread_pips: float
    commission_per_lot: float = 0.0
    is_available: bool = True


@dataclass
class PairUniverse:
    """
    Eligible pair universe for Blue Guardian 5K account.
    
    Pair selection criteria (Section 2.1):
    1. Rolling 20-day Pearson correlation |ρ| > 0.70
    2. Cointegration test (ADF p < 0.05) - optional but preferred
    3. Volatility differential σ_high / σ_low > 1.3
    4. Average spread cost < 0.3× expected daily range
    5. Both symbols available on Blue Guardian
    6. NOT same underlying
    """
    
    # Base symbols with specifications
    symbols: Dict[str, SymbolSpec] = field(default_factory=dict)
    
    # Pre-defined valid pairs (to be validated dynamically)
    candidate_pairs: List[Tuple[str, str]] = field(default_factory=list)
    
    # Maximum active pairs to monitor
    max_active_pairs: int = 5
    
    def __post_init__(self):
        """Initialize the default symbol universe."""
        self._init_symbols()
        self._init_candidate_pairs()
    
    def _init_symbols(self):
        """Initialize available symbols."""
        # Forex Majors
        self.symbols["EURUSD"] = SymbolSpec("EURUSD", AssetClass.FOREX_MAJOR, 100000, 10.0, 0.8)
        self.symbols["GBPUSD"] = SymbolSpec("GBPUSD", AssetClass.FOREX_MAJOR, 100000, 10.0, 1.0)
        self.symbols["USDJPY"] = SymbolSpec("USDJPY", AssetClass.FOREX_MAJOR, 100000, 7.5, 0.7)
        self.symbols["AUDUSD"] = SymbolSpec("AUDUSD", AssetClass.FOREX_MAJOR, 100000, 10.0, 0.9)
        self.symbols["NZDUSD"] = SymbolSpec("NZDUSD", AssetClass.FOREX_MAJOR, 100000, 10.0, 1.0)
        self.symbols["USDCAD"] = SymbolSpec("USDCAD", AssetClass.FOREX_MAJOR, 100000, 7.5, 0.9)
        self.symbols["USDCHF"] = SymbolSpec("USDCHF", AssetClass.FOREX_MAJOR, 100000, 7.5, 0.8)
        
        # Forex Minors
        self.symbols["EURGBP"] = SymbolSpec("EURGBP", AssetClass.FOREX_MINOR, 100000, 10.0, 1.2)
        
        # Metals
        self.symbols["XAUUSD"] = SymbolSpec("XAUUSD", AssetClass.METALS, 100, 1.0, 15.0)
        self.symbols["XAGUSD"] = SymbolSpec("XAGUSD", AssetClass.METALS, 5000, 1.0, 25.0)
        
        # Indices
        self.symbols["US30"] = SymbolSpec("US30", AssetClass.INDICES, 10, 1.0, 2.0)
        self.symbols["US100"] = SymbolSpec("US100", AssetClass.INDICES, 1, 1.0, 3.0)
    
    def _init_candidate_pairs(self):
        """Initialize candidate pairs based on correlation clusters."""
        # Metals cluster
        self.candidate_pairs.append(("XAUUSD", "XAGUSD"))
        
        # Forex commodity cluster
        self.candidate_pairs.append(("AUDUSD", "NZDUSD"))
        
        # Indices cluster
        self.candidate_pairs.append(("US100", "US30"))
        
        # Major FX pairs (marginal vol diff but high correlation)
        self.candidate_pairs.append(("GBPUSD", "EURUSD"))
        self.candidate_pairs.append(("EURUSD", "USDCHF"))  # Negative correlation play
        
    def get_symbol_spec(self, symbol: str) -> SymbolSpec:
        """Get specification for a symbol."""
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not in universe")
        return self.symbols[symbol]
    
    def is_symbol_available(self, symbol: str) -> bool:
        """Check if symbol is available for trading."""
        return symbol in self.symbols and self.symbols[symbol].is_available
    
    def get_leverage(self, symbol: str) -> int:
        """Get leverage for a symbol based on asset class."""
        if symbol not in self.symbols:
            raise ValueError(f"Symbol {symbol} not in universe")
        
        spec = self.symbols[symbol]
        if spec.asset_class == AssetClass.FOREX_MAJOR or spec.asset_class == AssetClass.FOREX_MINOR:
            return 100
        elif spec.asset_class == AssetClass.METALS:
            return 20
        elif spec.asset_class == AssetClass.INDICES:
            return 100
        else:
            return 100  # Default


# Global instances
DEFAULT_CONFIG = BlueGuardianConfig()
DEFAULT_UNIVERSE = PairUniverse()
