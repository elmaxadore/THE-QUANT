"""
Pair construction, correlation analysis, and cointegration testing.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy import stats
import logging

logger = logging.getLogger(__name__)


class PairConstructor:
    """
    Constructs and validates trading pairs based on Section 2.1 criteria.
    """
    
    def __init__(self, min_correlation: float = 0.70, min_vol_ratio: float = 1.3):
        self.min_correlation = min_correlation
        self.min_vol_ratio = min_vol_ratio
    
    def compute_pair_score(
        self,
        correlation: float,
        vol_ratio: float,
        spread_cost_ratio: float,
        cointegration_p: Optional[float] = None,
        liquidity_score: float = 0.8
    ) -> float:
        """
        Compute composite pair quality score (0.0-1.0).
        
        score = 0.30×|ρ| + 0.25×(σ_diff/σ_max) + 0.20×(1/spread_cost_ratio) +
                0.15×cointegration_strength + 0.10×liquidity_score
        """
        # Correlation component (0.30 weight)
        corr_score = abs(correlation)
        
        # Volatility differential component (0.25 weight)
        vol_diff = max(0, (vol_ratio - 1) / 2)  # Normalize to 0-1
        vol_score = min(1.0, vol_diff)
        
        # Spread cost component (0.20 weight)
        if spread_cost_ratio > 0:
            spread_score = min(1.0, 1.0 / spread_cost_ratio)
        else:
            spread_score = 1.0
        
        # Cointegration component (0.15 weight)
        if cointegration_p is not None and cointegration_p < 0.05:
            coint_score = 1.0 - cointegration_p / 0.05
        else:
            coint_score = 0.5  # Neutral if not tested
        
        # Liquidity component (0.10 weight)
        liq_score = liquidity_score
        
        # Composite score
        score = (
            0.30 * corr_score +
            0.25 * vol_score +
            0.20 * spread_score +
            0.15 * coint_score +
            0.10 * liq_score
        )
        
        return min(1.0, max(0.0, score))
    
    def select_best_pairs(
        self,
        candidate_pairs: List[Tuple[str, str]],
        price_data: Dict[str, pd.DataFrame],
        max_pairs: int = 5
    ) -> List[Dict]:
        """Select top N pairs based on quality score."""
        scored_pairs = []
        
        for sym_a, sym_b in candidate_pairs:
            if sym_a not in price_data or sym_b not in price_data:
                continue
            
            df_a = price_data[sym_a]
            df_b = price_data[sym_b]
            
            # Compute metrics
            metrics = self._compute_pair_metrics(df_a, df_b)
            
            if metrics['correlation'] < self.min_correlation:
                continue
            if metrics['vol_ratio'] < self.min_vol_ratio:
                continue
            
            # Compute score
            score = self.compute_pair_score(
                correlation=metrics['correlation'],
                vol_ratio=metrics['vol_ratio'],
                spread_cost_ratio=metrics.get('spread_cost_ratio', 0.5),
                cointegration_p=metrics.get('cointegration_p'),
                liquidity_score=0.8
            )
            
            if score >= 0.60:  # Minimum threshold
                scored_pairs.append({
                    'symbol_a': sym_a,
                    'symbol_b': sym_b,
                    'score': score,
                    **metrics
                })
        
        # Sort by score and return top N
        scored_pairs.sort(key=lambda x: x['score'], reverse=True)
        return scored_pairs[:max_pairs]
    
    def _compute_pair_metrics(
        self,
        df_a: pd.DataFrame,
        df_b: pd.DataFrame
    ) -> Dict:
        """Compute correlation, vol ratio, and cointegration for a pair."""
        # Align data
        merged = pd.merge(
            df_a[['timestamp', 'close']],
            df_b[['timestamp', 'close']],
            on='timestamp',
            suffixes=('_a', '_b')
        )
        
        if len(merged) < 30:
            return {'correlation': 0, 'vol_ratio': 1}
        
        # Returns
        ret_a = merged['close_a'].pct_change().dropna()
        ret_b = merged['close_b'].pct_change().dropna()
        
        # Correlation
        correlation = ret_a.corr(ret_b)
        
        # Volatility ratio
        vol_a = ret_a.std()
        vol_b = ret_b.std()
        vol_ratio = max(vol_a, vol_b) / min(vol_a, vol_b) if min(vol_a, vol_b) > 0 else 1
        
        # Cointegration test (simplified ADF on spread)
        log_spread = np.log(merged['close_a']) - np.log(merged['close_b'])
        coint_p = self._adf_test(log_spread.dropna())
        
        return {
            'correlation': correlation,
            'vol_ratio': vol_ratio,
            'cointegration_p': coint_p
        }
    
    def _adf_test(self, series: pd.Series) -> float:
        """Simplified ADF test returning p-value."""
        try:
            from statsmodels.tsa.stattools import adfuller
            result = adfuller(series, maxlag=10, autolag='AIC')
            return result[1]  # p-value
        except ImportError:
            # Fallback: use simple mean reversion test
            mean = series.mean()
            std = series.std()
            z = abs(series.iloc[-1] - mean) / std if std > 0 else 0
            return 0.5 if z < 1 else 0.1
