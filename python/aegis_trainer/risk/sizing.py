"""
Position sizing formula implementation (Section 2.3).
Volatility-normalized notional exposure with directional skew.
"""

import numpy as np
from typing import Dict, Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class PositionSize:
    """Result of position sizing calculation."""
    symbol_a: str
    symbol_b: str
    lots_a: float
    lots_b: float
    notional_a: float
    notional_b: float
    direction_bias: float
    expected_risk_usd: float
    max_worst_loss: float
    is_valid: bool
    rejection_reason: Optional[str] = None


class PositionSizer:
    """
    Computes position sizes for hedged pairs using the formula from Section 2.3.
    """
    
    def __init__(self, account_balance: float = 5000.0, risk_per_trade_pct: float = 0.005):
        self.account_balance = account_balance
        self.risk_per_trade_pct = risk_per_trade_pct
        self.risk_per_trade_usd = account_balance * risk_per_trade_pct
    
    def compute_position_size(
        self,
        symbol_a: str,
        symbol_b: str,
        price_a: float,
        price_b: float,
        vol_a: float,
        vol_b: float,
        correlation: float,
        probability_up: float,
        contract_size_a: float = 100000,
        contract_size_b: float = 100000,
        pip_value_a: float = 10.0,
        pip_value_b: float = 10.0,
        sl_pips_a: float = 10.0,
        sl_pips_b: float = 8.0,
        spread_cost_a: float = 0.0,
        spread_cost_b: float = 0.0,
        consistency_score: float = 1.0
    ) -> PositionSize:
        """
        Compute position sizes following Section 2.3 formula.
        
        Steps:
        1. Directional bias coefficient
        2. Volatility-normalized notional exposure
        3. Directional skew application
        4. Convert to lots
        5. Cost adjustment & edge check
        6. Guardian Shield verification
        7. Consistency-aware cap
        """
        
        # STEP 1: Directional bias coefficient
        bias = 2 * probability_up - 1  # Maps [0,1] to [-1,1]
        
        # Conviction threshold
        if abs(bias) < 0.20:
            return PositionSize(
                symbol_a=symbol_a, symbol_b=symbol_b,
                lots_a=0, lots_b=0, notional_a=0, notional_b=0,
                direction_bias=bias, expected_risk_usd=0, max_worst_loss=0,
                is_valid=False, rejection_reason="Insufficient conviction (|bias| < 0.20)"
            )
        
        # STEP 2: Volatility-normalized notional exposure
        vol_ratio = vol_a / vol_b if vol_b > 0 else 1.0
        
        # Weights: long volatile, short hedge
        w_a = 1.0
        w_b = -vol_ratio
        
        # Portfolio variance for $1 invested
        port_var = (w_a**2 * vol_a**2 + w_b**2 * vol_b**2 + 
                   2 * w_a * w_b * correlation * vol_a * vol_b)
        port_vol = np.sqrt(max(0, port_var))
        
        # Scale to RiskPerTrade
        if port_vol > 0:
            scale = self.risk_per_trade_usd / port_vol
        else:
            scale = self.risk_per_trade_usd
        
        notional_a_base = scale * w_a
        notional_b_base = scale * w_b
        
        # STEP 3: Directional skew application
        skew_factor = 1.0 + abs(bias) * 0.5  # Max 1.25x skew
        
        if bias > 0:
            notional_a = notional_a_base * skew_factor
            notional_b = notional_b_base * (2.0 - skew_factor)
        elif bias < 0:
            notional_a = notional_a_base * (2.0 - skew_factor)
            notional_b = notional_b_base * skew_factor
        else:
            notional_a = notional_a_base
            notional_b = notional_b_base
        
        # MAX SKEW check
        net_exposure = (abs(notional_a) + abs(notional_b)) / 2
        base_exposure = (abs(notional_a_base) + abs(notional_b_base)) / 2
        if net_exposure > 1.5 * base_exposure:
            # Cap at 1.5x
            cap_ratio = 1.5 * base_exposure / net_exposure
            notional_a *= cap_ratio
            notional_b *= cap_ratio
        
        # STEP 4: Convert to lots
        lots_a = notional_a / (contract_size_a * price_a) if price_a > 0 else 0
        lots_b = abs(notional_b) / (contract_size_b * price_b) if price_b > 0 else 0
        
        # Round to broker precision (0.01)
        lots_a = round(lots_a, 2)
        lots_b = round(lots_b, 2)
        
        # Recalculate actual notional
        actual_notional_a = lots_a * contract_size_a * price_a
        actual_notional_b = lots_b * contract_size_b * price_b
        
        # STEP 5: Cost adjustment & expected edge
        total_entry_cost = spread_cost_a + spread_cost_b
        
        # Expected move (simplified)
        expected_move_a = vol_a * price_a * bias
        expected_move_b = vol_b * price_b * bias
        
        expected_gross_pnl = (lots_a * expected_move_a * pip_value_a - 
                             lots_b * expected_move_b * pip_value_b)
        
        # Edge check: cost must be < 20% of expected gross profit
        if abs(expected_gross_pnl) > 0 and total_entry_cost > 0.20 * abs(expected_gross_pnl):
            return PositionSize(
                symbol_a=symbol_a, symbol_b=symbol_b,
                lots_a=0, lots_b=0, notional_a=0, notional_b=0,
                direction_bias=bias, expected_risk_usd=0, max_worst_loss=0,
                is_valid=False, rejection_reason=f"Too expensive: cost ${total_entry_cost:.2f} > 20% of expected ${expected_gross_pnl:.2f}"
            )
        
        # STEP 6: Guardian Shield verification
        worst_case_loss = (lots_a * sl_pips_a * pip_value_a + 
                          lots_b * sl_pips_b * pip_value_b)
        
        if worst_case_loss >= 50.00:
            # Reduce size proportionally
            reduction = 45.00 / worst_case_loss
            lots_a = round(lots_a * reduction, 2)
            lots_b = round(lots_b * reduction, 2)
            worst_case_loss = (lots_a * sl_pips_a * pip_value_a + 
                              lots_b * sl_pips_b * pip_value_b)
        
        # STEP 7: Consistency-aware cap
        consistency_discount = consistency_score ** 2
        lots_a = round(lots_a * consistency_discount, 2)
        lots_b = round(lots_b * consistency_discount, 2)
        
        # Final risk calculation
        final_risk = (lots_a * sl_pips_a * pip_value_a + 
                     lots_b * sl_pips_b * pip_value_b)
        
        return PositionSize(
            symbol_a=symbol_a,
            symbol_b=symbol_b,
            lots_a=max(0, lots_a),
            lots_b=max(0, lots_b),
            notional_a=actual_notional_a,
            notional_b=actual_notional_b,
            direction_bias=bias,
            expected_risk_usd=final_risk,
            max_worst_loss=worst_case_loss,
            is_valid=True
        )
