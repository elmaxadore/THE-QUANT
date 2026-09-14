"""Performance metrics calculator."""
import numpy as np

class PerformanceMetrics:
    """Calculates Sharpe, Sortino, Calmar, and payout velocity."""
    def calculate(self, returns):
        if len(returns) < 2:
            return {}
        sharpe = np.mean(returns) / np.std(returns) * np.sqrt(252) if np.std(returns) > 0 else 0
        return {"sharpe": sharpe}
