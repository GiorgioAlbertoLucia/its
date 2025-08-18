"""
Configuration and utility functions for particle identification neural network.
"""

from dataclasses import dataclass
from typing import Dict, List
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))


@dataclass
class TrainingConfig:
    """Configuration for neural network training."""
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 50
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4
    
    # Data splits
    test_size: float = 0.2
    val_size: float = 0.2
    
    # Model architecture
    hidden_dims: List[int] = None
    dropout_rate: float = 0.15
    use_batch_norm: bool = True

    model_type: str = 'PidFCNN'  # Type of model to use, e.g., 'PidFCNN', 'MomentumGatedPID'
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]

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

    return train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader
