"""Clean data processing pipeline."""
from copy import deepcopy
import numpy as np
import pandas as pd
from typing import Tuple, List

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import DataLoader

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import ExperimentConfig
from core.nn.scaler import MomentumAwareScaler
from utils.pid_routine import PARTICLE_ID, PDG_CODE

from particle import Particle
from torchic import Dataset
from torchic.physics.ITS import unpack_cluster_sizes, average_cluster_size, expected_cluster_size, sigma_its

class DataProcessor:
    """Handles all data loading and preprocessing."""
    
    def __init__(self, config: ExperimentConfig, df: pd.DataFrame = None):
        self.config = config
        self.df = df
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        if self.config.scaler == 'momentum_aware':
            self.scaler = MomentumAwareScaler(momentum_feature_idx=self.config.momentum_feature_idx)
        self.feature_columns: List[str] = []
    
    def load_raw_data(self, input_files:List[str], tree_names:List[str], 
                      folder_name:str='DF*', columns:List[str]=['fP', 'fEta', 'fPhi', 'fItsClusterSize', 'fPartID']) \
                        -> pd.DataFrame:
        """Load data from ROOT files."""

        datasets = []

        for tree_name in tree_names:
            
            datasets.append(Dataset.from_root(input_files,
                tree_name=tree_name,
                folder_name=folder_name,
                columns=columns
            ))

        dataset = datasets[0].concat(datasets[1:], axis=1)
        self.df = deepcopy(dataset.data)
    
    def downsample_data(self) -> None:
        """Downsample data to a fraction for faster processing."""
        if self.config.data_fraction < 1.0:
            self.df = self.df.sample(frac=self.config.data_fraction, random_state=42)
            print(f"Downsampled dataset to {len(self.df)} samples")

    def engineer_features(self) -> None:
        """Add physics-motivated features."""

        self.add_basic_features()

        if self.config.add_statistics_features:
            self.add_statistics_features()

        if self.config.add_parametrised_features:
            self.add_parametrised_features()
        
    def add_basic_features(self) -> None:
            
        self.df['fCosL'] = 1 / np.cosh(self.df['fEta'])
        self.df['fPAbs'] = np.abs(self.df['fP'])
        self.df['fPt'] = self.df['fPAbs'] * self.df['fCosL']

        print("Unpacking cluster sizes...")
        np_unpack_cluster_sizes = np.vectorize(unpack_cluster_sizes)
        for layer in range(7):
            self.df[f'fItsClusterSizeL{layer}'] = np_unpack_cluster_sizes(self.df['fItsClusterSize'], layer)

        #self.df.loc[self.df['fPartID'] == PARTICLE_ID['He'], 'fP'] *= 2

        self.df['fMeanItsClSize'], self.df['fNHitsIts'] = average_cluster_size(self.df['fItsClusterSize'])
        self.df['fClSizeCosL'] = self.df['fMeanItsClSize'] * self.df['fCosL']

        self.df.query(f'fNHitsIts >= {self.config.min_hits_required}', inplace=True)
        self.df.query(f'fPAbs < {self.config.max_p_required}', inplace=True)
        print(f"Filtered to tracks with {self.config.min_hits_required} hits: {len(self.df)} samples")
    
    def add_parametrised_features(self) -> None:
        """
            Add expected cluster size based on the bethe-bloch parametrisation.
            3-parameter Bethe-Bloch (with parameters obtained fitting K for Z=1, fitting He3 for Z=2).
            Exp = [0] / (beta gamma)^[1] + [2]

            Parameters:
            df (pd.DataFrame): DataFrame containing particle data.
            particle_ids (List[int]]): List of PARTICLE_IDs to compute expected cluster sizes for.
        """

        # Bethe-Bloch parameters for K and He3
        bethe_bloch_params = {
            'Z=1': {0: 0.9883, 1: 1.8940, 2: 1.9502},
            'Z=2': {0: 2.1718, 1: 1.8724, 2: 4.6988}
        }
        resolution_params = {
            'Z=1': {0: 0.2083, 1: -0.3125, 2: 1.3427},
            'Z=2': {0: 0.1466, 1: -0.0246, 2: 0.}
        }

        particle_ids = self.df['fPartID'].unique()
        id_to_name = {v: k for k, v in PARTICLE_ID.items()}
        particle_names = [id_to_name[p] for p in particle_ids if p in id_to_name]

        for particle in particle_names:
            charge = 'Z=2' if particle == 'He' else 'Z=1'
            charge_float = 2.0 if particle == 'He' else 1.0
            momentum = self.df['fPAbs'] * charge_float
            
            pid_params = list(bethe_bloch_params[charge].values()) + list(resolution_params[charge].values())
            self.df[f'fBetaGamma{particle}'] = momentum / (Particle.from_pdgid(PDG_CODE[particle]).mass / 1_000)
            self.df[f'fExpectedClusterSize{particle}'] = expected_cluster_size(
                self.df[f'fBetaGamma{particle}'],
                pid_params
            )
            self.df[f'fSigmaIts{particle}'] = sigma_its(
                self.df[f'fBetaGamma{particle}'],
                pid_params, particle=particle
            )
            self.df[f'fNSigmaIts{particle}'] = (
                (self.df[f'fClSizeCosL'] - self.df[f'fExpectedClusterSize{particle}']) /
                self.df[f'fSigmaIts{particle}']
            )
    
    def add_statistics_features(self) -> None:
        """Add features that give statistical insight to the dataset."""

        cluster_cols = [f'fItsClusterSizeL{i}' for i in range(7)]
        cluster_data = self.df[cluster_cols].values

        self.df['fClusterSizeStd'] = np.std(cluster_data, axis=1)
        self.df['fClusterSizeRange'] = np.ptp(cluster_data, axis=1)  # Range (max - min)
        self.df['fClusterSizeSkew'] = pd.DataFrame(cluster_data).skew(axis=1).values

        early_layers = self.df[['fItsClusterSizeL0', 'fItsClusterSizeL1', 'fItsClusterSizeL2']].mean(axis=1)
        late_layers = self.df[['fItsClusterSizeL4', 'fItsClusterSizeL5', 'fItsClusterSizeL6']].mean(axis=1)
        self.df['fEarlyLateRatio'] = early_layers / (late_layers + 1e-8)  # Avoid division by zero

        self.df['fTotalClusterSize'] = self.df[cluster_cols].sum(axis=1)

    def select_clean_data(self) -> None:

        self.df.query(f'fNSigmaItsHe > -1.5 or fPartID != {PARTICLE_ID["He"]}', inplace=True)

    def prepare_features(self) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features and labels for training."""
        self.feature_columns = self._get_feature_columns()
        X = self.df[self.feature_columns].values.astype(np.float32)
        y = self.label_encoder.fit_transform(self.df['fPartID'].values)
        
        X = self.scaler.fit_transform(X)
        return X, y
    
    def _get_feature_columns(self) -> List[str]:
        """Get list of feature columns based on config."""


        if self.config.feature_columns is not None:
            return self.config.feature_columns

        base_features = [
            'fPAbs', 'fEta', 'fPhi', 'fCosL', 'fPt',
            'fItsClusterSizeL0', 'fItsClusterSizeL1', 
            'fItsClusterSizeL2',
            'fItsClusterSizeL3', 'fItsClusterSizeL4', 'fItsClusterSizeL5',
            'fItsClusterSizeL6', 'fMeanItsClSize', 'fClSizeCosL'
        ]
        
        additional_features = []
        if self.config.add_statistics_features:
            additional_features.extend([
                'fClusterSizeStd', 'fClusterSizeSkew', 'fTotalClusterSize', 'fClusterSizeRange'
            ])
        
        parametrised_features = []
        if self.config.add_parametrised_features:
            particle_ids = self.df['fPartID'].unique()
            id_to_name_map = {v: k for k, v in PARTICLE_ID.items()}
            particle_names = [id_to_name_map[p] for p in particle_ids if p in id_to_name_map]
            for particle in particle_names:
                parametrised_features.append(f'fExpectedClusterSize{particle}')
                parametrised_features.append(f'fSigmaIts{particle}')
                parametrised_features.append(f'fNSigmaIts{particle}')

        return base_features + additional_features + parametrised_features
    
    def balance_classes(self) -> None:

        class_counts = self.df['fPartID'].value_counts()
        min_count = class_counts.min()
        print(f"Balancing classes to {min_count} samples each")
        
        balanced_dfs = []
        for class_id in class_counts.index:
            class_df = self.df[self.df['fPartID'] == class_id].sample(n=min_count, random_state=42)
            balanced_dfs.append(class_df)
        self.df = pd.concat(balanced_dfs, ignore_index=True)
        
        print(f"Balanced dataset size: {len(self.df)} samples")
    
    def get_class_weights(self, y: np.ndarray) -> dict:
        """Calculate class weights for imbalanced dataset."""
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        class_weights = compute_class_weight('balanced', classes=classes, y=y)
        return dict(zip(classes, class_weights))
    
class ParticleDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
def torch_data_preparation(X, y, test_val_size:float=0.2, batch_size:int=128):
    """
    Prepares the data for training by separating features and target labels.

    Parameters:
    X (np.ndarray): Feature data.
    y (np.ndarray): Target labels.

    Returns:
    train_loader (DataLoader): DataLoader for the training set.
    test_loader (DataLoader): DataLoader for the test set.
    label_encoder (LabelEncoder): LabelEncoder fitted to the target labels.
    """

    print('Preparing data for training.')

    X_train, X_test_val, y_train, y_test_val = train_test_split(X, y, test_size=test_val_size, random_state=42)
    X_test, X_val, y_test, y_val = train_test_split(X_test_val, y_test_val, test_size=0.5, random_state=42)

    train_dataset = ParticleDataset(X_train, y_train)
    val_dataset = ParticleDataset(X_val, y_val)
    test_dataset = ParticleDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)

    return train_loader, val_loader, test_loader

    