"""
Configuration and utility functions for particle identification neural network.
"""

from typing import List, Dict
import numpy as np
import pandas as pd

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.data_config import DataConfig


def add_physics_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add physics-motivated features to the dataset."""
    df_new = df.copy()
    
    # Transverse momentum
    df_new['fPt'] = df_new['fPAbs'] * df_new['fCosL']
    
    # Cluster size statistics across layers
    cluster_cols = [f'fItsClusterSizeL{i}' for i in range(7)]
    cluster_data = df_new[cluster_cols].values
    
    # Shower shape variables
    df_new['fClusterSizeStd'] = np.std(cluster_data, axis=1)
    df_new['fClusterSizeSkew'] = pd.DataFrame(cluster_data).skew(axis=1).values
    
    # Layer ratios (early vs late layers)
    early_layers = df_new[['fItsClusterSizeL0', 'fItsClusterSizeL1', 'fItsClusterSizeL2']].mean(axis=1)
    late_layers = df_new[['fItsClusterSizeL4', 'fItsClusterSizeL5', 'fItsClusterSizeL6']].mean(axis=1)
    df_new['fEarlyLateRatio'] = early_layers / (late_layers + 1e-8)  # Avoid division by zero
    
    # Total energy deposition
    df_new['fTotalClusterSize'] = df_new[cluster_cols].sum(axis=1)
    
    return df_new

def get_feature_columns(data_config: DataConfig) -> List[str]:
    """Get list of feature columns based on configuration."""
    base_features = [
        'fPAbs',
        'fEta', 'fPhi', 'fCosL', 
        'fItsClusterSizeL0', 'fItsClusterSizeL1', 'fItsClusterSizeL2', 
        'fItsClusterSizeL3', 'fItsClusterSizeL4', 'fItsClusterSizeL5',
        'fItsClusterSizeL6', 
        'fMeanItsClSize', 'fClSizeCosL'
    ]
    
    additional_features = []
    
    if data_config.add_pt:
        additional_features.append('fPt')
    
    if data_config.add_shower_shape:
        additional_features.extend([
            'fClusterSizeStd', 'fClusterSizeSkew', 'fTotalClusterSize'
        ])
    
    if data_config.add_layer_ratios:
        additional_features.append('fEarlyLateRatio')
    
    return base_features + additional_features

def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced dataset."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

