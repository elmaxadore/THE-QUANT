"""Ensemble model combining LightGBM and MLP."""
import logging
logger = logging.getLogger(__name__)

class EnsembleModel:
    """Weighted ensemble of GBDT + MLP with stacking."""
    def __init__(self, lgbm_weight=0.6, mlp_weight=0.4):
        self.lgbm_weight = lgbm_weight
        self.mlp_weight = mlp_weight
    
    def predict(self, lgbm_prob, mlp_prob):
        return self.lgbm_weight * lgbm_prob + self.mlp_weight * mlp_prob
