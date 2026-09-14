"""
Consistency rule tracking - profit concentration and lot size CV.
"""

from dataclasses import dataclass, field
from typing import Dict, List
from datetime import date
import numpy as np
import logging

logger = logging.getLogger(__name__)


@dataclass
class ConsistencyState:
    """Current consistency tracking state."""
    daily_profits: Dict[str, float] = field(default_factory=dict)
    total_profit: float = 0.0
    lot_sizes: List[float] = field(default_factory=list)
    max_daily_concentration: float = 0.0


class ConsistencyTracker:
    """
    Tracks Blue Guardian consistency rules:
    - No single day may contribute >15% of total profit
    - Lot size coefficient of variation (CV) < 0.40
    """
    
    def __init__(self, max_concentration_pct: float = 0.15, max_lot_cv: float = 0.40):
        self.max_concentration_pct = max_concentration_pct
        self.max_lot_cv = max_lot_cv
        self.state = ConsistencyState()
    
    def record_day_profit(self, trade_date: date, profit: float):
        """Record daily profit for concentration tracking."""
        date_str = trade_date.isoformat()
        self.state.daily_profits[date_str] = self.state.daily_profits.get(date_str, 0) + profit
        self.state.total_profit += profit
        
        # Update max concentration
        if self.state.total_profit > 0:
            current_concentration = abs(profit) / self.state.total_profit
            self.state.max_daily_concentration = max(
                self.state.max_daily_concentration,
                current_concentration
            )
    
    def record_lot_size(self, lots: float):
        """Record lot size for CV calculation."""
        self.state.lot_sizes.append(lots)
    
    def check_consistency(self, projected_profit: float = 0, projected_lots: float = 0) -> tuple[bool, str]:
        """
        Check if current state meets consistency requirements.
        
        Returns: (is_consistent, reason)
        """
        # Check profit concentration
        if self.state.total_profit + projected_profit > 0:
            projected_total = self.state.total_profit + projected_profit
            for day, profit in self.state.daily_profits.items():
                concentration = abs(profit) / projected_total
                if concentration > self.max_concentration_pct:
                    return False, f"Day {day} contributes {concentration:.1%} > 15% limit"
        
        # Check lot size CV
        if len(self.state.lot_sizes) >= 2:
            cv = np.std(self.state.lot_sizes) / np.mean(self.state.lot_sizes) if np.mean(self.state.lot_sizes) > 0 else 0
            if cv > self.max_lot_cv:
                return False, f"Lot size CV {cv:.2f} > {self.max_lot_cv} limit"
        
        # Projected lot CV
        if projected_lots > 0 and len(self.state.lot_sizes) >= 1:
            projected_lot_sizes = self.state.lot_sizes + [projected_lots]
            projected_cv = np.std(projected_lot_sizes) / np.mean(projected_lot_sizes)
            if projected_cv > self.max_lot_cv - 0.02:  # 0.02 buffer
                return False, f"Projected lot CV {projected_cv:.2f} would exceed limit"
        
        return True, "Consistent"
    
    def get_status(self) -> dict:
        """Get current consistency status."""
        cv = 0
        if len(self.state.lot_sizes) >= 2:
            cv = np.std(self.state.lot_sizes) / np.mean(self.state.lot_sizes)
        
        return {
            'total_profit': self.state.total_profit,
            'max_daily_concentration': self.state.max_daily_concentration,
            'lot_size_cv': cv,
            'trading_days': len(self.state.daily_profits),
            'is_within_limits': cv <= self.max_lot_cv and self.state.max_daily_concentration <= self.max_concentration_pct
        }
