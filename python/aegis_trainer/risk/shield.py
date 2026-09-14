"""
Guardian Shield implementation - $50/trade, $150/day hard limits.
"""

from dataclasses import dataclass, field
from typing import Optional, List
from datetime import datetime, date
import logging

logger = logging.getLogger(__name__)


@dataclass
class ShieldState:
    """Current state of Guardian Shield."""
    strikes: int = 0
    today_realized_loss: float = 0.0
    lifetime_realized_loss: float = 0.0
    today_trades: int = 0
    is_halted: bool = False
    halt_reason: Optional[str] = None
    last_trade_time: Optional[datetime] = None


class GuardianShield:
    """
    Enforces Blue Guardian risk limits at simulation and execution layers.
    
    Rules:
    - Max $50 loss per trade (hard cap)
    - Strike 1: Warning + halve position size
    - Strike 2: Account HALTED
    - Daily loss limit: $150 (soft cap at $140)
    - Lifetime loss floor: $250 (soft cap at $240)
    """
    
    def __init__(
        self,
        max_loss_per_trade: float = 50.0,
        daily_loss_limit: float = 150.0,
        lifetime_loss_floor: float = 250.0,
        strike_warning_threshold: float = 35.0
    ):
        self.max_loss_per_trade = max_loss_per_trade
        self.daily_loss_limit = daily_loss_limit
        self.lifetime_loss_floor = lifetime_loss_floor
        self.strike_warning_threshold = strike_warning_threshold
        
        self.state = ShieldState()
        self.trade_history: List[dict] = []
    
    def pre_trade_check(self, estimated_loss: float) -> tuple[bool, Optional[str]]:
        """
        Pre-flight check before allowing a trade.
        
        Returns: (allowed, rejection_reason)
        """
        if self.state.is_halted:
            return False, f"Account halted: {self.state.halt_reason}"
        
        # Check estimated loss vs Guardian Shield
        if estimated_loss >= self.max_loss_per_trade:
            return False, f"Estimated loss ${estimated_loss:.2f} exceeds Guardian Shield ${self.max_loss_per_trade:.2f}"
        
        # Warning threshold
        if estimated_loss >= self.strike_warning_threshold:
            logger.warning(f"Trade approaching shield: ${estimated_loss:.2f}")
        
        # Daily loss check
        projected_daily = self.state.today_realized_loss + estimated_loss
        if projected_daily >= self.daily_loss_limit - 10:  # $10 buffer
            return False, f"Projected daily loss ${projected_daily:.2f} exceeds soft cap"
        
        # Lifetime loss check
        projected_lifetime = self.state.lifetime_realized_loss + estimated_loss
        if projected_lifetime >= self.lifetime_loss_floor - 10:  # $10 buffer
            return False, f"Projected lifetime loss ${projected_lifetime:.2f} exceeds soft cap"
        
        return True, None
    
    def record_trade(self, realized_pnl: float, trade_info: dict):
        """Record a completed trade and update shield state."""
        self.state.today_trades += 1
        self.state.last_trade_time = datetime.now()
        
        if realized_pnl < 0:
            loss = abs(realized_pnl)
            
            # Update losses
            self.state.today_realized_loss += loss
            self.state.lifetime_realized_loss += loss
            
            # Check for strike
            if loss >= self.strike_warning_threshold:
                self.state.strikes += 1
                logger.warning(f"Guardian Shield Strike {self.state.strikes}: Loss ${loss:.2f}")
                
                if self.state.strikes >= 2:
                    self.state.is_halted = True
                    self.state.halt_reason = "Two Guardian Shield strikes exceeded"
                    logger.error("ACCOUNT HALTED: Two strikes exceeded")
            
            # Check daily limit breach
            if self.state.today_realized_loss >= self.daily_loss_limit:
                self.state.is_halted = True
                self.state.halt_reason = "Daily loss limit breached"
                logger.error("ACCOUNT HALTED: Daily loss limit breached")
            
            # Check lifetime limit breach
            if self.state.lifetime_realized_loss >= self.lifetime_loss_floor:
                self.state.is_halted = True
                self.state.halt_reason = "Lifetime loss floor breached - manual reset required"
                logger.error("ACCOUNT HALTED: Lifetime loss floor breached")
        
        # Record trade
        self.trade_history.append({
            'timestamp': datetime.now(),
            'pnl': realized_pnl,
            **trade_info
        })
    
    def reset_daily(self):
        """Reset daily counters (called at 00:00 UTC)."""
        self.state.today_realized_loss = 0.0
        self.state.today_trades = 0
        logger.info("Daily counters reset")
    
    def get_status(self) -> dict:
        """Get current shield status."""
        return {
            'strikes': self.state.strikes,
            'today_loss': self.state.today_realized_loss,
            'lifetime_loss': self.state.lifetime_realized_loss,
            'is_halted': self.state.is_halted,
            'halt_reason': self.state.halt_reason,
            'remaining_daily': self.daily_loss_limit - self.state.today_realized_loss,
            'remaining_lifetime': self.lifetime_loss_floor - self.state.lifetime_realized_loss
        }
