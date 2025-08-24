"""
    Collection of scalers for Neural Networks.
"""

import numpy as np
from sklearn.preprocessing import StandardScaler

class MomentumAwareScaler:
    """
    Scaler that handles momentum separately from other features.
    """
    
    def __init__(self, momentum_feature_idx: int, scale_momentum: bool = False):
        """
        Args:
            momentum_feature_idx: Index of momentum feature in the input
            scale_momentum: Whether to scale momentum (usually False for gating)
        """
        self.momentum_feature_idx = momentum_feature_idx
        self.scale_momentum = scale_momentum
        self.feature_scaler = StandardScaler()
        self.momentum_scaler = StandardScaler() if scale_momentum else None
        
    def fit(self, X):
        """Fit the scalers to the data."""
        # Get all features except momentum
        non_momentum_indices = [i for i in range(X.shape[1]) if i != self.momentum_feature_idx]
        X_features = X[:, non_momentum_indices]
        
        # Fit feature scaler
        self.feature_scaler.fit(X_features)
        
        # Optionally fit momentum scaler
        if self.scale_momentum and self.momentum_scaler is not None:
            momentum = X[:, self.momentum_feature_idx].reshape(-1, 1)
            self.momentum_scaler.fit(momentum)
            
        return self
    
    def transform(self, X):
        """Transform the data."""
        X_transformed = np.zeros_like(X)
        
        # Transform non-momentum features
        non_momentum_indices = [i for i in range(X.shape[1]) if i != self.momentum_feature_idx]
        X_features = X[:, non_momentum_indices]
        X_features_scaled = self.feature_scaler.transform(X_features)
        
        # Place scaled features back
        for i, orig_idx in enumerate(non_momentum_indices):
            X_transformed[:, orig_idx] = X_features_scaled[:, i]
        
        # Handle momentum
        momentum = X[:, self.momentum_feature_idx]
        if self.scale_momentum and self.momentum_scaler is not None:
            momentum = self.momentum_scaler.transform(momentum.reshape(-1, 1)).flatten()
        
        X_transformed[:, self.momentum_feature_idx] = momentum
        
        return X_transformed
    
    def fit_transform(self, X):
        """Fit and transform in one step."""
        return self.fit(X).transform(X)
