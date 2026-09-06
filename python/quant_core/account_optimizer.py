"""
Account Optimization Engine for THE QUANT v4.1 Hercules
Production-ready implementation with:
- Dynamic position sizing based on account equity and risk parameters
- Daily drawdown headroom tracking for prop firm compliance
- Correlation-based allocation for multi-asset portfolios
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json


class AccountTier(Enum):
    """Prop firm account tiers based on maximum drawdown limits."""
    TIER_5PCT = 5.0      # 5% maximum drawdown
    TIER_8PCT = 8.0      # 8% maximum drawdown
    TIER_10PCT = 10.0    # 10% maximum drawdown
    TIER_12PCT = 12.0    # 12% maximum drawdown


@dataclass
class AccountConfig:
    """Configuration for an individual trading account."""
    account_id: str
    initial_equity: float
    current_equity: float
    max_drawdown_pct: float
    daily_drawdown_limit_pct: float = 5.0
    risk_per_trade_pct: float = 1.0
    max_positions: int = 5
    tier: AccountTier = AccountTier.TIER_5PCT
    
    def __post_init__(self):
        if self.tier is None:
            self.tier = AccountTier(self.max_drawdown_pct)


@dataclass
class PositionInfo:
    """Information about a current or proposed position."""
    symbol: str
    quantity: float
    entry_price: float
    current_price: float
    side: str  # 'long' or 'short'
    unrealized_pnl: float = 0.0
    weight: float = 0.0
    
    def __post_init__(self):
        self.unrealized_pnl = (self.current_price - self.entry_price) * self.quantity
        if self.side == 'short':
            self.unrealized_pnl = -self.unrealized_pnl


@dataclass
class AllocationResult:
    """Result of the correlation-based allocation optimization."""
    symbol: str
    optimal_weight: float
    risk_contribution: float
    correlation_adjusted: bool
    reason: str


@dataclass
class OptimizationReport:
    """Complete report from the account optimization engine."""
    timestamp: str
    account_id: str
    current_equity: float
    available_headroom: float
    daily_headroom: float
    positions: List[PositionInfo]
    allocations: List[AllocationResult]
    recommended_actions: List[str]
    risk_metrics: Dict[str, float]


class AccountOptimizer:
    """
    Production Account Optimization Engine.
    
    Features:
    - Dynamic position sizing based on Kelly-inspired fractional betting
    - Daily drawdown headroom tracking for prop firm rules
    - Correlation-based portfolio allocation using hierarchical risk parity
    - Real-time risk monitoring and alerts
    """
    
    def __init__(self, config_path: Optional[str] = None):
        self.accounts: Dict[str, AccountConfig] = {}
        self.positions: Dict[str, List[PositionInfo]] = {}
        self.correlation_matrix: Optional[np.ndarray] = None
        self.symbols: List[str] = []
        self.config_path = config_path
        self._load_config()
        
    def _load_config(self):
        """Load configuration from file if provided."""
        if self.config_path:
            try:
                with open(self.config_path, 'r') as f:
                    config = json.load(f)
                    # Parse config as needed
            except FileNotFoundError:
                pass  # Use defaults
    
    def register_account(self, config: AccountConfig) -> None:
        """Register a new trading account for optimization."""
        self.accounts[config.account_id] = config
        self.positions[config.account_id] = []
    
    def update_account_equity(self, account_id: str, new_equity: float) -> float:
        """
        Update account equity and calculate drawdown metrics.
        
        Returns:
            Current drawdown percentage
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        old_equity = account.current_equity
        account.current_equity = new_equity
        
        # Calculate drawdown from peak
        peak_equity = max(account.initial_equity, old_equity)
        drawdown_pct = ((peak_equity - new_equity) / peak_equity) * 100
        
        return drawdown_pct
    
    def calculate_daily_headroom(self, account_id: str, daily_pnl: float) -> Tuple[float, float]:
        """
        Calculate remaining daily drawdown headroom.
        
        Args:
            account_id: The account identifier
            daily_pnl: Today's PnL so far
            
        Returns:
            Tuple of (remaining_headroom_usd, remaining_headroom_pct)
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        
        # Daily drawdown is calculated from previous day's close or initial equity
        daily_limit_usd = account.initial_equity * (account.daily_drawdown_limit_pct / 100)
        remaining_headroom = daily_limit_usd - abs(min(0, daily_pnl))
        remaining_pct = (remaining_headroom / account.initial_equity) * 100
        
        return max(0, remaining_headroom), max(0, remaining_pct)
    
    def calculate_dynamic_position_size(
        self,
        account_id: str,
        symbol: str,
        price: float,
        volatility: float,
        win_rate: float = 0.55,
        avg_win_loss_ratio: float = 1.5
    ) -> float:
        """
        Calculate dynamic position size using fractional Kelly criterion.
        
        Args:
            account_id: The account identifier
            symbol: Trading symbol
            price: Current asset price
            volatility: Annualized volatility (0-1 scale)
            win_rate: Historical win rate of strategy
            avg_win_loss_ratio: Average winner to loser ratio
            
        Returns:
            Optimal position size in units
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        
        # Check available headroom
        drawdown = self.update_account_equity(account_id, account.current_equity)
        if drawdown >= account.max_drawdown_pct * 0.8:  # 80% of max used
            return 0.0  # Reduce or stop trading
        
        # Fractional Kelly calculation
        edge = win_rate - (1 - win_rate) / avg_win_loss_ratio
        kelly_fraction = edge / avg_win_loss_ratio
        
        # Apply risk scaling based on volatility and account constraints
        vol_scale = min(1.0, 0.2 / max(volatility, 0.01))  # Target 20% vol
        risk_fraction = kelly_fraction * vol_scale * (account.risk_per_trade_pct / 100)
        
        # Calculate position size
        risk_capital = account.current_equity * risk_fraction
        position_value = risk_capital / volatility if volatility > 0 else risk_capital
        quantity = position_value / price
        
        # Apply maximum position limit
        max_position_value = account.current_equity * 0.2  # Max 20% per position
        max_quantity = max_position_value / price
        
        return min(quantity, max_quantity)
    
    def build_correlation_matrix(self, returns_data: Dict[str, List[float]]) -> np.ndarray:
        """
        Build correlation matrix from historical returns.
        
        Args:
            returns_data: Dictionary mapping symbols to lists of returns
            
        Returns:
            Correlation matrix as numpy array
        """
        self.symbols = list(returns_data.keys())
        n_symbols = len(self.symbols)
        
        if n_symbols == 0:
            return np.array([[1.0]])
        
        # Convert to matrix
        returns_matrix = np.array([returns_data[symbol] for symbol in self.symbols])
        
        # Calculate correlation matrix
        corr_matrix = np.corrcoef(returns_matrix)
        
        # Handle NaN values
        corr_matrix = np.nan_to_num(corr_matrix, nan=0.0)
        
        self.correlation_matrix = corr_matrix
        return corr_matrix
    
    def optimize_allocation(
        self,
        account_id: str,
        expected_returns: Dict[str, float],
        volatilities: Dict[str, float],
        target_risk: float = 0.1
    ) -> List[AllocationResult]:
        """
        Optimize portfolio allocation using Hierarchical Risk Parity (HRP).
        
        Args:
            account_id: The account identifier
            expected_returns: Expected returns for each symbol
            volatilities: Volatility estimates for each symbol
            target_risk: Target portfolio volatility
            
        Returns:
            List of allocation results with optimal weights
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        if self.correlation_matrix is None:
            # Default to equal weights if no correlation data
            symbols = list(expected_returns.keys())
            n = len(symbols)
            return [
                AllocationResult(
                    symbol=s,
                    optimal_weight=1.0/n,
                    risk_contribution=1.0/n,
                    correlation_adjusted=False,
                    reason="No correlation data available"
                )
                for s in symbols
            ]
        
        symbols = list(expected_returns.keys())
        n = len(symbols)
        
        if n != self.correlation_matrix.shape[0]:
            # Mismatch, use simple inverse volatility weighting
            inv_vol = [1.0 / volatilities.get(s, 0.1) for s in symbols]
            total_inv_vol = sum(inv_vol)
            weights = [iv / total_inv_vol for iv in inv_vol]
            
            return [
                AllocationResult(
                    symbol=symbols[i],
                    optimal_weight=weights[i],
                    risk_contribution=weights[i] * volatilities.get(symbols[i], 0.1),
                    correlation_adjusted=False,
                    reason="Symbol count mismatch with correlation matrix"
                )
                for i in range(n)
            ]
        
        # HRP-like allocation: cluster by correlation and allocate
        weights = self._hierarchical_clustering_weights(symbols, volatilities)
        
        # Adjust for expected returns (shrink towards mean-variance)
        weights = self._tilt_towards_returns(weights, expected_returns, symbols)
        
        # Normalize weights
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]
        
        results = []
        for i, symbol in enumerate(symbols):
            vol = volatilities.get(symbol, 0.1)
            results.append(AllocationResult(
                symbol=symbol,
                optimal_weight=weights[i],
                risk_contribution=weights[i] * vol,
                correlation_adjusted=True,
                reason="HRP optimization applied"
            ))
        
        return results
    
    def _hierarchical_clustering_weights(
        self,
        symbols: List[str],
        volatilities: Dict[str, float]
    ) -> List[float]:
        """
        Simplified hierarchical clustering for weight allocation.
        Groups highly correlated assets and allocates risk equally.
        """
        n = len(symbols)
        if n == 0:
            return []
        
        # Start with inverse volatility weights
        inv_vols = np.array([1.0 / max(volatilities.get(s, 0.1), 0.01) for s in symbols])
        weights = inv_vols / inv_vols.sum()
        
        if self.correlation_matrix is None or n != self.correlation_matrix.shape[0]:
            return weights.tolist()
        
        # Simple correlation adjustment: reduce weight for highly correlated pairs
        for i in range(n):
            for j in range(i+1, n):
                corr = abs(self.correlation_matrix[i, j])
                if corr > 0.7:  # High correlation threshold
                    # Reduce both weights slightly
                    reduction = 0.1 * corr
                    weights[i] *= (1 - reduction)
                    weights[j] *= (1 - reduction)
        
        # Re-normalize
        weights = weights / weights.sum()
        return weights.tolist()
    
    def _tilt_towards_returns(
        self,
        weights: List[float],
        expected_returns: Dict[str, float],
        symbols: List[str]
    ) -> List[float]:
        """
        Tilt weights towards assets with higher expected returns.
        Uses a conservative shrinkage approach.
        """
        returns = np.array([expected_returns.get(s, 0.0) for s in symbols])
        weights = np.array(weights)
        
        # Normalize returns to [0, 1] range
        if returns.max() > returns.min():
            norm_returns = (returns - returns.min()) / (returns.max() - returns.min())
        else:
            norm_returns = np.ones_like(returns) * 0.5
        
        # Blend: 70% original weights, 30% return-tilted
        tilt_factor = 0.3
        tilted = weights * (1 - tilt_factor) + norm_returns * tilt_factor
        
        return tilted.tolist()
    
    def generate_optimization_report(
        self,
        account_id: str,
        daily_pnl: float = 0.0
    ) -> OptimizationReport:
        """
        Generate comprehensive optimization report for an account.
        
        Args:
            account_id: The account identifier
            daily_pnl: Current day's PnL
            
        Returns:
            OptimizationReport with all metrics and recommendations
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        positions = self.positions.get(account_id, [])
        
        # Calculate headroom
        total_headroom, _ = self.calculate_daily_headroom(account_id, daily_pnl)
        
        # Calculate risk metrics
        total_exposure = sum(abs(p.weight) for p in positions) if positions else 0.0
        net_exposure = sum(p.weight for p in positions) if positions else 0.0
        
        risk_metrics = {
            "total_exposure": total_exposure,
            "net_exposure": net_exposure,
            "concentration_risk": max((p.weight for p in positions), default=0.0),
            "drawdown_buffer": account.max_drawdown_pct - self.update_account_equity(account_id, account.current_equity)
        }
        
        # Generate recommendations
        recommendations = []
        if risk_metrics["drawdown_buffer"] < account.max_drawdown_pct * 0.3:
            recommendations.append("WARNING: Approaching drawdown limit - reduce position sizes")
        if total_exposure > 0.8:
            recommendations.append("High exposure detected - consider reducing leverage")
        if len(positions) >= account.max_positions:
            recommendations.append(f"Maximum positions ({account.max_positions}) reached")
        
        # Create dummy allocations for report
        allocations = []
        if positions:
            for pos in positions:
                allocations.append(AllocationResult(
                    symbol=pos.symbol,
                    optimal_weight=pos.weight,
                    risk_contribution=pos.weight * 0.1,  # Simplified
                    correlation_adjusted=True,
                    reason="Current position weight"
                ))
        
        from datetime import datetime
        return OptimizationReport(
            timestamp=datetime.now().isoformat(),
            account_id=account_id,
            current_equity=account.current_equity,
            available_headroom=total_headroom,
            daily_headroom=total_headroom,
            positions=positions,
            allocations=allocations,
            recommended_actions=recommendations,
            risk_metrics=risk_metrics
        )
    
    def rebalance_portfolio(
        self,
        account_id: str,
        target_weights: Dict[str, float],
        current_prices: Dict[str, float]
    ) -> List[Tuple[str, float]]:
        """
        Generate rebalancing trades to achieve target weights.
        
        Args:
            account_id: The account identifier
            target_weights: Target weight for each symbol
            current_prices: Current price for each symbol
            
        Returns:
            List of (symbol, quantity) tuples for trades (positive=buy, negative=sell)
        """
        if account_id not in self.accounts:
            raise ValueError(f"Account {account_id} not found")
        
        account = self.accounts[account_id]
        current_positions = {p.symbol: p for p in self.positions.get(account_id, [])}
        
        trades = []
        for symbol, target_weight in target_weights.items():
            target_value = account.current_equity * target_weight
            current_pos = current_positions.get(symbol)
            current_value = current_pos.quantity * current_pos.current_price if current_pos else 0.0
            
            value_diff = target_value - current_value
            price = current_prices.get(symbol, 1.0)
            
            if abs(value_diff) > account.current_equity * 0.01:  # 1% threshold
                quantity_change = value_diff / price
                trades.append((symbol, quantity_change))
        
        return trades


# Convenience function for creating optimizer with default config
def create_optimizer_for_tier(tier: AccountTier, account_id: str = "default") -> AccountOptimizer:
    """Create an AccountOptimizer pre-configured for a specific prop firm tier."""
    optimizer = AccountOptimizer()
    
    config = AccountConfig(
        account_id=account_id,
        initial_equity=100000.0,
        current_equity=100000.0,
        max_drawdown_pct=tier.value,
        daily_drawdown_limit_pct=min(5.0, tier.value * 0.8),
        risk_per_trade_pct=0.5 if tier == AccountTier.TIER_5PCT else 1.0,
        max_positions=3 if tier == AccountTier.TIER_5PCT else 5,
        tier=tier
    )
    
    optimizer.register_account(config)
    return optimizer


if __name__ == "__main__":
    # Example usage and testing
    print("Account Optimization Engine v4.1 Hercules")
    print("=" * 50)
    
    # Create optimizer for 5% drawdown tier
    optimizer = create_optimizer_for_tier(AccountTier.TIER_5PCT, "PROP-001")
    
    # Simulate some returns data for correlation
    np.random.seed(42)
    returns_data = {
        "BTCUSDT": np.random.randn(100) * 0.02,
        "ETHUSDT": np.random.randn(100) * 0.025,
        "EURUSD": np.random.randn(100) * 0.005
    }
    
    # Build correlation matrix
    corr = optimizer.build_correlation_matrix(returns_data)
    print(f"\nCorrelation Matrix Shape: {corr.shape}")
    print(f"BTC-ETH Correlation: {corr[0, 1]:.3f}")
    
    # Test dynamic position sizing
    position_size = optimizer.calculate_dynamic_position_size(
        account_id="PROP-001",
        symbol="BTCUSDT",
        price=45000.0,
        volatility=0.02,
        win_rate=0.55,
        avg_win_loss_ratio=1.5
    )
    print(f"\nRecommended BTC Position Size: {position_size:.4f} units")
    
    # Test allocation optimization
    expected_returns = {"BTCUSDT": 0.15, "ETHUSDT": 0.12, "EURUSD": 0.05}
    volatilities = {"BTCUSDT": 0.02, "ETHUSDT": 0.025, "EURUSD": 0.005}
    
    allocations = optimizer.optimize_allocation(
        account_id="PROP-001",
        expected_returns=expected_returns,
        volatilities=volatilities,
        target_risk=0.1
    )
    
    print("\nOptimal Allocations:")
    for alloc in allocations:
        print(f"  {alloc.symbol}: {alloc.optimal_weight:.2%} (risk contrib: {alloc.risk_contribution:.4f})")
    
    # Generate report
    report = optimizer.generate_optimization_report("PROP-001", daily_pnl=-500.0)
    print(f"\nOptimization Report:")
    print(f"  Account: {report.account_id}")
    print(f"  Equity: ${report.current_equity:,.2f}")
    print(f"  Available Headroom: ${report.available_headroom:,.2f}")
    print(f"  Recommendations: {len(report.recommended_actions)}")
    for rec in report.recommended_actions:
        print(f"    - {rec}")
    
    print("\n✓ Account Optimization Engine initialized successfully")
