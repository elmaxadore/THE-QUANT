#!/usr/bin/env python3
"""
Unsupervised Learning Module for Market Regime Detection and Insights
======================================================================

This module applies unsupervised learning techniques to:
1. Detect market regimes (trending, ranging, volatile, calm)
2. Extract latent features using autoencoders
3. Cluster similar market conditions
4. Identify anomalies and outliers
5. Provide actionable insights for strategy adaptation

Methods implemented:
- K-Means clustering for regime detection
- Gaussian Mixture Models (GMM) for probabilistic regime assignment
- Autoencoders for dimensionality reduction and feature extraction
- Isolation Forest for anomaly detection
- t-SNE/UMAP for visualization
"""

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans, DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.ensemble import IsolationForest
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import joblib

# Try to import optional dependencies
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Autoencoder features disabled.")

try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    print("Warning: UMAP not available. Using t-SNE for visualization only.")

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

# Import from training module
try:
    from python_training.data_ingestion import get_market_data
    from python_training.feature_engineering import prepare_features, get_feature_columns
except ImportError:
    # Fallback imports
    sys.path.insert(0, str(Path(__file__).parent.parent / 'python_training'))
    from data_ingestion import get_market_data
    from feature_engineering import prepare_features, get_feature_columns


def fetch_market_data(symbol: str, start_date: str, end_date: str) -> Optional[pd.DataFrame]:
    """
    Fetch market data for a single symbol between dates.
    
    Args:
        symbol: Symbol to fetch (e.g., 'EURUSD', 'BTCUSDT')
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        
    Returns:
        DataFrame with OHLCV data or None if failed
    """
    from datetime import timedelta
    
    # Calculate days between dates
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    days = max((end - start).days, 100)  # At least 100 days
    
    # Map symbol to appropriate format
    if '=' in symbol or '-' in symbol:
        # Already has suffix, try to clean it
        clean_symbol = symbol.replace('=X', '').replace('=F', '').replace('-USD', 'USDT').replace('-', '')
        if 'USDT' in clean_symbol.upper():
            symbol_dict = {'type': 'crypto', 'symbol': clean_symbol}
        else:
            symbol_dict = {'type': 'forex', 'symbol': clean_symbol}
    elif symbol.upper().endswith('USDT'):
        # Crypto USDT pair
        symbol_dict = {'type': 'crypto', 'symbol': symbol}
    elif len(symbol) >= 6 and symbol.upper() in ['EURUSD', 'GBPUSD', 'USDJPY', 'AUDUSD', 'USDCAD', 'USDCHF', 'NZDUSD']:
        # Major forex pairs
        symbol_dict = {'type': 'forex', 'symbol': symbol.upper()}
    else:
        # Try crypto first, then forex
        if any(x in symbol.upper() for x in ['BTC', 'ETH', 'XRP', 'LTC', 'ADA', 'SOL']):
            symbol_dict = {'type': 'crypto', 'symbol': symbol.upper() if not symbol.upper().endswith('USDT') else symbol.upper()}
        else:
            symbol_dict = {'type': 'forex', 'symbol': symbol.upper()}
    
    print(f"  Fetching {symbol_dict['type']} data for {symbol_dict['symbol']} ({days} days)...")
    
    try:
        df = get_market_data([symbol_dict], days=days)
        if df is not None and len(df) > 0:
            # Filter to date range - handle different index types
            if isinstance(df.index[0], str):
                df.index = pd.to_datetime(df.index)
            mask = (df.index >= pd.to_datetime(start_date)) & (df.index <= pd.to_datetime(end_date))
            df = df[mask]
            return df if len(df) > 0 else None
    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        # Return unfiltered data if filtering fails
        try:
            df = get_market_data([symbol_dict], days=days)
            if df is not None and len(df) > 0:
                return df
        except:
            pass
    
    return None


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """
    Create features from raw price data.
    
    Args:
        df: DataFrame with OHLCV data
        
    Returns:
        Tuple of (feature DataFrame, list of feature names)
    """
    df_features = prepare_features(df.copy())
    
    # Get feature columns
    feature_cols, _ = get_feature_columns(df_features)
    
    # Drop rows with NaN
    df_features = df_features.dropna()
    
    return df_features[feature_cols], feature_cols


class MarketRegimeDetector:
    """
    Detects market regimes using unsupervised learning techniques.
    
    Regimes typically include:
    - Low volatility / ranging
    - High volatility / trending up
    - High volatility / trending down
    - Extreme volatility / crisis
    """
    
    def __init__(self, n_regimes: int = 4, method: str = 'gmm', random_state: int = 42):
        """
        Initialize the regime detector.
        
        Args:
            n_regimes: Number of market regimes to detect
            method: Clustering method ('kmeans', 'gmm', 'dbscan')
            random_state: Random seed for reproducibility
        """
        self.n_regimes = n_regimes
        self.method = method
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.model = None
        self.regime_labels = None
        self.regime_stats = None
        
    def fit(self, features: np.ndarray) -> 'MarketRegimeDetector':
        """
        Fit the regime detection model.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            
        Returns:
            Self for method chaining
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Choose and fit model
        if self.method == 'kmeans':
            self.model = KMeans(
                n_clusters=self.n_regimes,
                random_state=self.random_state,
                n_init=10,
                max_iter=300
            )
        elif self.method == 'gmm':
            self.model = GaussianMixture(
                n_components=self.n_regimes,
                random_state=self.random_state,
                n_init=3,
                max_iter=200,
                covariance_type='full'
            )
        elif self.method == 'dbscan':
            self.model = DBSCAN(
                eps=0.5,
                min_samples=10,
                metric='euclidean'
            )
            # DBSCAN doesn't use n_regimes
            self.regime_labels = self.model.fit_predict(features_scaled)
            return self
        else:
            raise ValueError(f"Unknown method: {self.method}")
        
        # Fit model
        self.model.fit(features_scaled)
        self.regime_labels = self.model.predict(features_scaled)
        
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """Predict regime labels for new data."""
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def get_regime_probabilities(self, features: np.ndarray) -> np.ndarray:
        """Get regime probabilities (only for GMM)."""
        if self.method != 'gmm':
            raise ValueError("Probabilities only available for GMM method")
        features_scaled = self.scaler.transform(features)
        return self.model.predict_proba(features_scaled)
    
    def analyze_regimes(self, returns: np.ndarray, volatility: np.ndarray, 
                       timestamps: Optional[pd.Series] = None) -> Dict[str, Any]:
        """
        Analyze characteristics of each detected regime.
        
        Args:
            returns: Array of returns
            volatility: Array of volatility measures
            timestamps: Optional timestamps for temporal analysis
            
        Returns:
            Dictionary with regime statistics and insights
        """
        regime_stats = {}
        
        for regime_id in range(self.n_regimes):
            mask = self.regime_labels == regime_id
            if np.sum(mask) == 0:
                continue
                
            regime_returns = returns[mask]
            regime_volatility = volatility[mask]
            
            stats = {
                'regime_id': int(regime_id),
                'n_samples': int(np.sum(mask)),
                'percentage': float(np.sum(mask) / len(self.regime_labels) * 100),
                'mean_return': float(np.mean(regime_returns)),
                'std_return': float(np.std(regime_returns)),
                'mean_volatility': float(np.mean(regime_volatility)),
                'std_volatility': float(np.std(regime_volatility)),
                'skewness': float(pd.Series(regime_returns).skew()),
                'kurtosis': float(pd.Series(regime_returns).kurtosis()),
                'sharpe_ratio': float(np.mean(regime_returns) / (np.std(regime_returns) + 1e-8)) * np.sqrt(252),
            }
            
            # Classify regime type based on statistics
            regime_type = self._classify_regime_type(stats)
            stats['regime_type'] = regime_type
            
            # Temporal analysis if timestamps provided
            if timestamps is not None:
                regime_timestamps = timestamps[mask]
                stats['first_occurrence'] = str(regime_timestamps.min())
                stats['last_occurrence'] = str(regime_timestamps.max())
                
            regime_stats[f'regime_{regime_id}'] = stats
        
        # Overall clustering quality metrics
        if len(np.unique(self.regime_labels)) > 1:
            # Use the same features that were used for clustering
            try:
                silhouette = silhouette_score(features_scaled, self.regime_labels)
                calinski = calinski_harabasz_score(features_scaled, self.regime_labels)
                regime_stats['clustering_quality'] = {
                    'silhouette_score': float(silhouette),
                    'calinski_harabasz_score': float(calinski)
                }
            except Exception as e:
                print(f"Could not compute clustering quality metrics: {e}")
                pass
        
        self.regime_stats = regime_stats
        return regime_stats
    
    def _classify_regime_type(self, stats: Dict) -> str:
        """Classify regime type based on statistical properties."""
        mean_ret = abs(stats['mean_return'])
        std_ret = stats['std_return']
        mean_vol = stats['mean_volatility']
        
        # Simple heuristic classification
        if mean_vol < 0.005:  # Low volatility
            if mean_ret < 0.0005:
                return "CALM_RANGING"
            else:
                return "CALM_TRENDING"
        elif mean_vol >= 0.005 and mean_vol < 0.015:  # Medium volatility
            if stats['mean_return'] > 0:
                return "MODERATE_BULL"
            else:
                return "MODERATE_BEAR"
        else:  # High volatility
            if stats['kurtosis'] > 3:  # Fat tails
                return "HIGH_VOL_EXTREME"
            elif stats['mean_return'] > 0:
                return "HIGH_VOL_BULL"
            else:
                return "HIGH_VOL_BEAR"
        
        return "UNKNOWN"


class DeepFeatureExtractor:
    """
    Autoencoder-based feature extraction for dimensionality reduction.
    
    Learns compressed representations of market data that capture
    essential patterns while filtering out noise.
    """
    
    def __init__(self, input_dim: int, latent_dim: int = 8, hidden_dims: List[int] = None):
        """
        Initialize the autoencoder.
        
        Args:
            input_dim: Dimension of input features
            latent_dim: Dimension of latent representation
            hidden_dims: List of hidden layer dimensions
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch required for DeepFeatureExtractor")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims or [64, 32, 16]
        
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.scaler = StandardScaler()
        
    def _build_autoencoder(self) -> nn.Module:
        """Build the autoencoder network."""
        layers = []
        dims = [self.input_dim] + self.hidden_dims
        
        # Encoder
        for i in range(len(dims) - 1):
            layers.extend([
                nn.Linear(dims[i], dims[i+1]),
                nn.BatchNorm1d(dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
        
        # Latent space
        layers.append(nn.Linear(dims[-1], self.latent_dim))
        self.encoder = nn.Sequential(*layers)
        
        # Decoder (mirror of encoder)
        decoder_layers = []
        decoder_dims = [self.latent_dim] + self.hidden_dims[::-1] + [self.input_dim]
        
        for i in range(len(decoder_dims) - 2):
            decoder_layers.extend([
                nn.Linear(decoder_dims[i], decoder_dims[i+1]),
                nn.BatchNorm1d(decoder_dims[i+1]),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
        
        decoder_layers.append(nn.Linear(decoder_dims[-2], decoder_dims[-1]))
        self.decoder = nn.Sequential(*decoder_layers)
        
        # Full autoencoder
        self.autoencoder = nn.Sequential(self.encoder, self.decoder)
        
        return self.autoencoder
    
    def fit(self, features: np.ndarray, epochs: int = 100, batch_size: int = 64, 
            learning_rate: float = 0.001, verbose: bool = True) -> 'DeepFeatureExtractor':
        """
        Train the autoencoder.
        
        Args:
            features: Input feature matrix
            epochs: Number of training epochs
            batch_size: Batch size for training
            learning_rate: Learning rate
            verbose: Print training progress
            
        Returns:
            Self for method chaining
        """
        # Scale features
        features_scaled = self.scaler.fit_transform(features)
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(features_scaled)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Build model
        self._build_autoencoder()
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = torch.optim.Adam(self.autoencoder.parameters(), lr=learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        # Training loop
        self.autoencoder.train()
        best_loss = float('inf')
        patience_counter = 0
        
        for epoch in range(epochs):
            total_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                output = self.autoencoder(batch_x)
                loss = criterion(output, batch_x)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(dataloader)
            scheduler.step(avg_loss)
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}")
            
            # Early stopping
            if avg_loss < best_loss:
                best_loss = avg_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 20:
                    if verbose:
                        print(f"Early stopping at epoch {epoch+1}")
                    break
        
        return self
    
    def extract_features(self, features: np.ndarray) -> np.ndarray:
        """
        Extract latent features using the trained encoder.
        
        Args:
            features: Input feature matrix
            
        Returns:
            Latent feature representations
        """
        features_scaled = self.scaler.transform(features)
        X_tensor = torch.FloatTensor(features_scaled)
        
        self.encoder.eval()
        with torch.no_grad():
            latent_features = self.encoder(X_tensor).numpy()
        
        return latent_features


class AnomalyDetector:
    """
    Detects anomalies and outliers in market data using Isolation Forest.
    
    Useful for identifying:
    - Flash crashes
    - Unusual volume spikes
    - Market manipulation events
    - Black swan events
    """
    
    def __init__(self, contamination: float = 0.05, random_state: int = 42):
        """
        Initialize the anomaly detector.
        
        Args:
            contamination: Expected proportion of anomalies in the dataset
            random_state: Random seed
        """
        self.contamination = contamination
        self.random_state = random_state
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100,
            max_samples='auto'
        )
        self.scaler = StandardScaler()
        
    def fit(self, features: np.ndarray) -> 'AnomalyDetector':
        """Fit the anomaly detector."""
        features_scaled = self.scaler.fit_transform(features)
        self.model.fit(features_scaled)
        return self
    
    def predict(self, features: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (-1 for anomaly, 1 for normal).
        
        Returns:
            Array of predictions (-1 or 1)
        """
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)
    
    def anomaly_scores(self, features: np.ndarray) -> np.ndarray:
        """
        Get continuous anomaly scores (lower = more anomalous).
        
        Returns:
            Array of anomaly scores
        """
        features_scaled = self.scaler.transform(features)
        return self.model.score_samples(features_scaled)
    
    def get_anomaly_insights(self, features: np.ndarray, 
                            df_original: pd.DataFrame) -> Dict[str, Any]:
        """
        Get detailed insights about detected anomalies.
        
        Args:
            features: Feature matrix
            df_original: Original DataFrame with timestamps and prices
            
        Returns:
            Dictionary with anomaly insights
        """
        predictions = self.predict(features)
        scores = self.anomaly_scores(features)
        
        anomaly_mask = predictions == -1
        n_anomalies = np.sum(anomaly_mask)
        
        insights = {
            'total_samples': len(features),
            'n_anomalies': int(n_anomalies),
            'anomaly_percentage': float(n_anomalies / len(features) * 100),
            'anomaly_indices': np.where(anomaly_mask)[0].tolist()[:50],  # First 50
            'mean_anomaly_score': float(np.mean(scores[anomaly_mask])) if n_anomalies > 0 else None,
            'mean_normal_score': float(np.mean(scores[~anomaly_mask])),
        }
        
        # Add details about top anomalies
        if n_anomalies > 0:
            anomaly_df = df_original.iloc[np.where(anomaly_mask)[0]].head(10)
            insights['top_anomalies'] = anomaly_df.to_dict('records')
        
        return insights


def visualize_regimes(features: np.ndarray, regime_labels: np.ndarray, 
                     returns: np.ndarray, volatility: np.ndarray,
                     save_path: Optional[str] = None) -> Dict:
    """
    Create visualizations of market regimes.
    
    Generates:
    - 2D projection (t-SNE/UMAP) colored by regime
    - Time series of regime changes
    - Regime distribution pie chart
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.gridspec import GridSpec
    
    # Reduce to 2D for visualization
    pca = PCA(n_components=2)
    features_2d = pca.fit_transform(features[:, :50])  # Use first 50 features
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(2, 2, figure=fig)
    
    # Plot 1: 2D projection
    ax1 = fig.add_subplot(gs[0, 0])
    scatter = ax1.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=regime_labels, cmap='viridis', alpha=0.6, s=10)
    ax1.set_title('Market Regimes (PCA Projection)', fontsize=14)
    ax1.set_xlabel('PC1')
    ax1.set_ylabel('PC2')
    plt.colorbar(scatter, ax=ax1, label='Regime ID')
    
    # Plot 2: Regime time series
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.plot(regime_labels, linewidth=0.5)
    ax2.set_title('Regime Changes Over Time', fontsize=14)
    ax2.set_xlabel('Time')
    ax2.set_ylabel('Regime ID')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Return distribution by regime
    ax3 = fig.add_subplot(gs[1, 0])
    unique_regimes = np.unique(regime_labels)
    for regime_id in unique_regimes:
        mask = regime_labels == regime_id
        ax3.hist(returns[mask], bins=30, alpha=0.6, label=f'Regime {regime_id}', density=True)
    ax3.set_title('Return Distribution by Regime', fontsize=14)
    ax3.set_xlabel('Return')
    ax3.set_ylabel('Density')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Volatility by regime
    ax4 = fig.add_subplot(gs[1, 1])
    regime_vols = [volatility[regime_labels == i].mean() for i in unique_regimes]
    ax4.bar(unique_regimes, regime_vols, color='coral', alpha=0.7)
    ax4.set_title('Average Volatility by Regime', fontsize=14)
    ax4.set_xlabel('Regime ID')
    ax4.set_ylabel('Mean Volatility')
    ax4.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Visualization saved to {save_path}")
    
    plt.close()
    
    return {'pca_explained_variance': pca.explained_variance_ratio_.tolist()}


def run_unsupervised_analysis(symbols: List[str], start_date: str, end_date: str,
                             n_regimes: int = 4, use_autoencoder: bool = False,
                             output_dir: str = 'unsupervised_results') -> Dict[str, Any]:
    """
    Run complete unsupervised learning analysis pipeline.
    
    Args:
        symbols: List of symbols to analyze
        start_date: Start date for data fetching
        end_date: End date for data fetching
        n_regimes: Number of market regimes to detect
        use_autoencoder: Whether to use autoencoder for feature extraction
        output_dir: Directory to save results
        
    Returns:
        Dictionary with all analysis results
    """
    print("=" * 70)
    print("UNSUPERVISED LEARNING MARKET ANALYSIS")
    print("=" * 70)
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    results = {
        'symbols': symbols,
        'analysis_date': datetime.now().isoformat(),
        'parameters': {
            'n_regimes': n_regimes,
            'use_autoencoder': use_autoencoder,
            'start_date': start_date,
            'end_date': end_date
        },
        'symbols_analysis': {}
    }
    
    for symbol in symbols:
        print(f"\n{'='*60}")
        print(f"Analyzing: {symbol}")
        print('='*60)
        
        # Fetch data
        print("\n[1/6] Fetching market data...")
        df = fetch_market_data(symbol, start_date, end_date)
        if df is None or len(df) < 100:
            print(f"Insufficient data for {symbol}. Skipping...")
            continue
        
        print(f"Retrieved {len(df)} data points from {df.index[0]} to {df.index[-1]}")
        
        # Preprocess features
        print("\n[2/6] Engineering features...")
        df_features, feature_names = preprocess_features(df)
        features = df_features.values
        
        # Calculate returns and volatility from original data
        if 'returns' not in df.columns:
            df['returns'] = df['close'].pct_change().fillna(0)
        if 'volatility_20d' not in df.columns:
            df['volatility_20d'] = df['returns'].rolling(20).std().fillna(df['returns'].std())
        
        returns = df['returns'].values[-len(features):]  # Align with features
        volatility = df['volatility_20d'].values[-len(features):]  # Align with features
        
        print(f"Created {features.shape[1]} features")
        
        # Optional: Autoencoder feature extraction
        latent_features = None
        if use_autoencoder and TORCH_AVAILABLE:
            print("\n[3/6] Training autoencoder for feature extraction...")
            try:
                extractor = DeepFeatureExtractor(
                    input_dim=features.shape[1],
                    latent_dim=min(16, features.shape[1] // 4)
                )
                extractor.fit(features, epochs=50, verbose=True)
                latent_features = extractor.extract_features(features)
                print(f"Extracted {latent_features.shape[1]} latent features")
                
                # Save autoencoder
                torch.save({
                    'encoder_state_dict': extractor.encoder.state_dict(),
                    'input_dim': extractor.input_dim,
                    'latent_dim': extractor.latent_dim,
                    'scaler_mean': extractor.scaler.mean_,
                    'scaler_scale': extractor.scaler.scale_
                }, output_path / f'{symbol.replace("/", "_")}_autoencoder.pt')
                
                # Use latent features for regime detection
                features_for_clustering = latent_features
            except Exception as e:
                print(f"Autoencoder failed: {e}. Using original features.")
                features_for_clustering = features
        else:
            features_for_clustering = features
        
        # Regime detection
        print(f"\n[{'4' if not use_autoencoder else '5'}/6] Detecting {n_regimes} market regimes...")
        detector = MarketRegimeDetector(n_regimes=n_regimes, method='gmm')
        detector.fit(features_for_clustering)
        
        # Analyze regimes
        regime_insights = detector.analyze_regimes(returns, volatility, df_features.index)
        
        print("\nRegime Analysis Results:")
        print("-" * 60)
        for regime_key, regime_data in regime_insights.items():
            if regime_key.startswith('regime_'):
                print(f"\n{regime_data['regime_type']} (ID: {regime_data['regime_id']})")
                print(f"  Occurrence: {regime_data['percentage']:.1f}% of time")
                print(f"  Mean Return: {regime_data['mean_return']*100:.3f}%")
                print(f"  Mean Volatility: {regime_data['mean_volatility']*100:.3f}%")
                print(f"  Sharpe Ratio: {regime_data['sharpe_ratio']:.2f}")
        
        # Anomaly detection
        print(f"\n[{'5' if not use_autoencoder else '6'}/6] Detecting anomalies...")
        anomaly_detector = AnomalyDetector(contamination=0.03)
        anomaly_detector.fit(features_for_clustering)
        anomaly_insights = anomaly_detector.get_anomaly_insights(features_for_clustering, df_features)
        
        print(f"\nDetected {anomaly_insights['n_anomalies']} anomalies ({anomaly_insights['anomaly_percentage']:.2f}%)")
        
        # Visualization
        print(f"\n[{'6' if not use_autoencoder else '7'}/6] Creating visualizations...")
        viz_info = visualize_regimes(
            features_for_clustering,
            detector.regime_labels,
            returns,
            volatility,
            save_path=str(output_path / f'{symbol.replace("/", "_")}_regimes.png')
        )
        
        # Store results
        results['symbols_analysis'][symbol] = {
            'data_points': len(df),
            'n_features': features.shape[1],
            'regime_insights': regime_insights,
            'anomaly_insights': anomaly_insights,
            'visualization_info': viz_info,
            'regime_labels': detector.regime_labels.tolist(),
            'feature_names': feature_names
        }
        
        # Save regime labels for trading system integration
        regime_df = pd.DataFrame({
            'timestamp': df_features.index,
            'regime': detector.regime_labels,
            'regime_type': [regime_insights.get(f'regime_{r}', {}).get('regime_type', 'UNKNOWN') 
                           for r in detector.regime_labels]
        })
        regime_df.to_csv(output_path / f'{symbol.replace("/", "_")}_regimes.csv', index=False)
        print(f"Regime labels saved to {output_path / f'{symbol.replace("/", "_")}_regimes.csv'}")
    
    # Save comprehensive results
    results_file = output_path / 'unsupervised_analysis_results.json'
    with open(results_file, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        def convert(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert(i) for i in obj]
            return obj
        
        json.dump(convert(results), f, indent=2, default=str)
    
    print(f"\n{'='*70}")
    print("ANALYSIS COMPLETE")
    print(f"Results saved to: {output_path}")
    print(f"Summary: {results_file}")
    print("="*70)
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Unsupervised Learning for Market Regime Detection',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python unsupervised_insights.py --symbols EURUSD=X BTC-USD --n_regimes 4
  python unsupervised_insights.py --symbols GC=F CL=F --n_regimes 3 --use_autoencoder
  python unsupervised_insights.py --symbols EURUSD=X --start_date 2020-01-01 --n_regimes 5
        """
    )
    
    parser.add_argument('--symbols', nargs='+', required=True,
                       help='Symbols to analyze (e.g., EURUSD=X BTC-USD)')
    parser.add_argument('--start_date', type=str, default='2020-01-01',
                       help='Start date for data fetching (YYYY-MM-DD)')
    parser.add_argument('--end_date', type=str, default=None,
                       help='End date for data fetching (YYYY-MM-DD)')
    parser.add_argument('--n_regimes', type=int, default=4,
                       help='Number of market regimes to detect')
    parser.add_argument('--use_autoencoder', action='store_true',
                       help='Use autoencoder for feature extraction')
    parser.add_argument('--output_dir', type=str, default='unsupervised_results',
                       help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set end date to today if not specified
    if args.end_date is None:
        args.end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Run analysis
    results = run_unsupervised_analysis(
        symbols=args.symbols,
        start_date=args.start_date,
        end_date=args.end_date,
        n_regimes=args.n_regimes,
        use_autoencoder=args.use_autoencoder,
        output_dir=args.output_dir
    )
    
    # Print strategic insights
    print("\n\n" + "="*70)
    print("STRATEGIC INSIGHTS & RECOMMENDATIONS")
    print("="*70)
    
    for symbol, analysis in results['symbols_analysis'].items():
        print(f"\n{symbol}:")
        regime_insights = analysis['regime_insights']
        
        # Find best and worst regimes
        regimes_list = [v for k, v in regime_insights.items() if k.startswith('regime_')]
        if not regimes_list:
            continue
            
        best_regime = max(regimes_list, key=lambda x: x['sharpe_ratio'])
        worst_regime = min(regimes_list, key=lambda x: x['sharpe_ratio'])
        
        print(f"  ✓ Best regime: {best_regime['regime_type']} (Sharpe: {best_regime['sharpe_ratio']:.2f})")
        print(f"    - Active {best_regime['percentage']:.1f}% of the time")
        print(f"    - Consider: Increase position size during this regime")
        
        print(f"  ⚠ Worst regime: {worst_regime['regime_type']} (Sharpe: {worst_regime['sharpe_ratio']:.2f})")
        print(f"    - Active {worst_regime['percentage']:.1f}% of the time")
        print(f"    - Consider: Reduce exposure or switch to hedging strategy")
        
        # Anomaly insights
        anomaly_pct = analysis['anomaly_insights']['anomaly_percentage']
        if anomaly_pct > 5:
            print(f"  ⚡ High anomaly rate ({anomaly_pct:.1f}%): Market shows unusual behavior")
            print(f"    - Consider: Wider stops, reduced leverage")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
