"""Model manifest with metadata."""
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ModelManifest:
    """Metadata for exported models."""
    training_date: str
    feature_hash: str
    val_auc: float
    val_brier: float
    
    @classmethod
    def create(cls, **kwargs):
        return cls(training_date=datetime.now().isoformat(), feature_hash="", val_auc=0.0, val_brier=0.0, **kwargs)
