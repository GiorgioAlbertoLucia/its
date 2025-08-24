"""
Configuration and utility functions for particle identification neural network.
"""

from typing import List, Dict
import numpy as np
import pandas as pd
from particle import Particle

from torchic.physics.ITS import expected_cluster_size, sigma_its

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from utils.data_config import DataConfig
from utils.pid_routine import PARTICLE_ID, PDG_CODE


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
    df_new['fClusterSizeRange'] = np.ptp(cluster_data, axis=1)  # Range (max - min)
    df_new['fClusterSizeSkew'] = pd.DataFrame(cluster_data).skew(axis=1).values
    
    # Layer ratios (early vs late layers)
    early_layers = df_new[['fItsClusterSizeL0', 'fItsClusterSizeL1', 'fItsClusterSizeL2']].mean(axis=1)
    late_layers = df_new[['fItsClusterSizeL4', 'fItsClusterSizeL5', 'fItsClusterSizeL6']].mean(axis=1)
    df_new['fEarlyLateRatio'] = early_layers / (late_layers + 1e-8)  # Avoid division by zero
    
    # Total energy deposition
    df_new['fTotalClusterSize'] = df_new[cluster_cols].sum(axis=1)
    
    return df_new

def add_parametrised_features(df: pd.DataFrame, particle_ids:List[int]) -> pd.DataFrame:
    """
        Add expected cluster size based on the bethe-bloch parametrisation.
        3-parameter Bethe-Bloch (with parameters obtained fitting K for Z=1, fitting He3 for Z=2).
        Exp = [0] / (beta gamma)^[1] + [2]

        Parameters:
        df (pd.DataFrame): DataFrame containing particle data.
        particle_ids (List[int]]): List of PARTICLE_IDs to compute expected cluster sizes for.
    """
    df_new = df.copy()
    
    # Bethe-Bloch parameters for K and He3
    bethe_bloch_params = {
        'Z=1': {0: 0.9883, 1: 1.8940, 2: 1.9502},
        'Z=2': {0: 2.1718, 1: 1.8724, 2: 4.6988}
    }
    resolution_params = {
        'Z=1': {0: 0.2083, 1: -0.3125, 2: 1.3427},
        'Z=2': {0: 0.1466, 1: -0.0246, 2: 0.}
    }

    name_to_id_map = {v: k for k, v in PARTICLE_ID.items()}
    particle_names = [name_to_id_map[p] for p in particle_ids if p in name_to_id_map]
    
    for particle in particle_names:
        charge = 'Z=2' if particle == 'He' else 'Z=1'
        pid_params = (*bethe_bloch_params[charge].values(), *resolution_params[charge].values())
        df_new[f'fExpectedClusterSize{particle}'] = expected_cluster_size(
            df_new['fPAbs'] / Particle.from_pdgid(PDG_CODE[particle]).mass,
            pid_params
        )
        df_new[f'fSigmaIts{particle}'] = sigma_its(
            df_new['fPAbs'] / Particle.from_pdgid(PDG_CODE[particle]).mass,
            pid_params, particle=particle
        )
        df_new[f'fNSigmaIts{particle}'] = (
            (df_new[f'fClSizeCosL'] - df_new[f'fExpectedClusterSize{particle}']) /
            df_new[f'fSigmaIts{particle}']
        )

    return df_new


def get_feature_columns(data_config: DataConfig, particles:List[int] = None) -> List[str]:
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
            'fClusterSizeStd', 'fClusterSizeSkew', 'fTotalClusterSize', 'fClusterSizeRange'
        ])
    
    if data_config.add_layer_ratios:
        additional_features.append('fEarlyLateRatio')

    if data_config.add_expected_cluster_size:
        name_to_id_map = {v: k for k, v in PARTICLE_ID.items()}
        particle_names = [name_to_id_map[p] for p in particles if p in name_to_id_map]
        for particle in particle_names:
            additional_features.append(f'fExpectedClusterSize{particle}')
            additional_features.append(f'fSigmaIts{particle}')
            additional_features.append(f'fNSigmaIts{particle}')
    
    return base_features + additional_features

def calculate_class_weights(y: np.ndarray) -> Dict[int, float]:
    """Calculate class weights for imbalanced dataset."""
    from sklearn.utils.class_weight import compute_class_weight
    
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    return dict(zip(classes, weights))

