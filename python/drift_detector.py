"""
Drift Detection System
Monitors data distribution changes to detect when models become stale
Uses Population Stability Index (PSI) for robust drift detection
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
import json
from scipy.stats import chi2


class DriftDetector:
    """Detects data drift using PSI and statistical tests"""
    
    def __init__(self, config: dict):
        self.psi_threshold_warning = config.get('psi_threshold_warning', 0.1)
        self.psi_threshold_critical = config.get('psi_threshold_critical', 0.25)
        self.buckets = config.get('psi_buckets', 10)
        self.lookback_period = config.get('lookback_period', 252)
        
        self.reference_data = {}
        self.drift_history = []
        
    def calculate_psi(self, expected: np.ndarray, actual: np.ndarray, 
                     buckets: int = None) -> float:
        """
        Calculate Population Stability Index between two distributions
        
        Args:
            expected: Reference distribution (training data)
            actual: Current distribution (live data)
            buckets: Number of bins for discretization
            
        Returns:
            PSI value (higher = more drift)
        """
        if buckets is None:
            buckets = self.buckets
            
        # Handle edge cases
        if len(expected) == 0 or len(actual) == 0:
            return 0.0
            
        # Create bins based on reference data
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        breakpoints = np.unique(breakpoints)
        
        # Ensure we have at least 2 bins
        if len(breakpoints) < 2:
            return 0.0
            
        # Cut data into bins
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        
        # Convert to percentages with smoothing to avoid division by zero
        expected_percents = (expected_counts + 1) / (len(expected) + buckets)
        actual_percents = (actual_counts + 1) / (len(actual) + buckets)
        
        # Calculate PSI
        psi_values = (actual_percents - expected_percents) * np.log(actual_percents / expected_percents)
        psi = np.sum(psi_values)
        
        return psi
    
    def set_reference_data(self, feature_name: str, data: np.ndarray):
        """Store reference distribution for a feature"""
        self.reference_data[feature_name] = data.copy()
    
    def check_feature_drift(self, feature_name: str, current_data: np.ndarray) -> dict:
        """
        Check drift for a single feature
        
        Returns:
            Dictionary with PSI value and status
        """
        if feature_name not in self.reference_data:
            return {
                'feature': feature_name,
                'psi': None,
                'status': 'NO_REFERENCE',
                'message': 'No reference data available'
            }
        
        psi = self.calculate_psi(self.reference_data[feature_name], current_data)
        
        if psi >= self.psi_threshold_critical:
            status = 'CRITICAL_DRIFT'
            message = f'Critical drift detected (PSI={psi:.3f})'
        elif psi >= self.psi_threshold_warning:
            status = 'WARNING_DRIFT'
            message = f'Moderate drift detected (PSI={psi:.3f})'
        else:
            status = 'STABLE'
            message = f'No significant drift (PSI={psi:.3f})'
        
        result = {
            'feature': feature_name,
            'psi': psi,
            'status': status,
            'message': message
        }
        
        self.drift_history.append(result)
        return result
    
    def check_portfolio_drift(self, current_features: Dict[str, np.ndarray]) -> dict:
        """
        Check drift across all features in the portfolio
        
        Returns:
            Comprehensive drift report
        """
        results = {}
        critical_count = 0
        warning_count = 0
        
        for feature_name, data in current_features.items():
            result = self.check_feature_drift(feature_name, data)
            results[feature_name] = result
            
            if result['status'] == 'CRITICAL_DRIFT':
                critical_count += 1
            elif result['status'] == 'WARNING_DRIFT':
                warning_count += 1
        
        # Overall assessment
        if critical_count > 0:
            overall_status = 'CRITICAL'
            recommendation = 'IMMEDIATE_MODEL_RETRAIN_REQUIRED'
        elif warning_count > len(current_features) * 0.3:  # >30% features drifting
            overall_status = 'WARNING'
            recommendation = 'SCHEDULE_MODEL_RETRAIN'
        else:
            overall_status = 'HEALTHY'
            recommendation = 'CONTINUE_MONITORING'
        
        return {
            'timestamp': pd.Timestamp.now().isoformat(),
            'features_checked': len(results),
            'critical_drift_count': critical_count,
            'warning_drift_count': warning_count,
            'overall_status': overall_status,
            'recommendation': recommendation,
            'details': results
        }
    
    def get_model_validity_score(self, drift_report: dict) -> float:
        """
        Calculate a validity score (0-1) based on drift report
        
        Returns:
            Score where 1.0 = fully valid, 0.0 = completely invalid
        """
        if drift_report['overall_status'] == 'CRITICAL':
            return 0.2
        elif drift_report['overall_status'] == 'WARNING':
            # Reduce score based on number of drifting features
            warning_ratio = drift_report['warning_drift_count'] / max(1, drift_report['features_checked'])
            return max(0.5, 1.0 - warning_ratio)
        else:
            return 1.0
    
    def export_drift_report(self, filename: str = 'drift_report.json'):
        """Export drift history to JSON file"""
        with open(filename, 'w') as f:
            json.dump({
                'configuration': {
                    'psi_threshold_warning': self.psi_threshold_warning,
                    'psi_threshold_critical': self.psi_threshold_critical,
                    'buckets': self.buckets
                },
                'history': self.drift_history[-100:],  # Last 100 checks
                'reference_features': list(self.reference_data.keys())
            }, f, indent=2, default=str)
        
        print(f"✓ Drift report exported to {filename}")


# Example usage
if __name__ == "__main__":
    config = {
        'psi_threshold_warning': 0.1,
        'psi_threshold_critical': 0.25,
        'psi_buckets': 10,
        'lookback_period': 252
    }
    
    detector = DriftDetector(config)
    
    # Simulate reference data (training period)
    np.random.seed(42)
    reference_data = np.random.normal(0, 1, 1000)
    
    # Set reference
    detector.set_reference_data('feature_1', reference_data)
    
    # Test 1: Stable data (should pass)
    stable_data = np.random.normal(0, 1, 100)
    result1 = detector.check_feature_drift('feature_1', stable_data)
    print("Test 1 - Stable:", result1)
    
    # Test 2: Drifted data (mean shift)
    drifted_data = np.random.normal(1.5, 1, 100)
    result2 = detector.check_feature_drift('feature_1', drifted_data)
    print("\nTest 2 - Drifted:", result2)
    
    # Test 3: Portfolio check
    current_features = {
        'feature_1': drifted_data,
        'feature_2': np.random.normal(0, 1, 100)
    }
    
    detector.set_reference_data('feature_2', np.random.normal(0, 1, 1000))
    portfolio_report = detector.check_portfolio_drift(current_features)
    
    print("\nPortfolio Drift Report:")
    print(json.dumps(portfolio_report, indent=2))
    
    validity = detector.get_model_validity_score(portfolio_report)
    print(f"\nModel Validity Score: {validity:.2f}")
    
    detector.export_drift_report()
