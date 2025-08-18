"""
Momentum-Gated PID neural network that adapts its processing based on momentum regimes.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional
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

class ResidualBlock(nn.Module):
    """Residual block with batch normalization and dropout."""
    
    def __init__(self, dim: int, dropout_rate: float = 0.1):
        super().__init__()
        self.linear = nn.Linear(dim, dim)
        self.bn = nn.BatchNorm1d(dim)
        self.dropout = nn.Dropout(dropout_rate)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn(self.linear(x)))
        x = self.dropout(x)
        return x + residual  # Residual connection

class MomentumRegimeBranch(nn.Module):
    """Individual branch for processing a specific momentum regime."""
    
    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 128, 
                 dropout_rate: float = 0.15, use_batch_norm: bool = True):
        super().__init__()
        
        self.branch = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_dim, output_dim),
            nn.BatchNorm1d(output_dim) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
    
    def forward(self, x):
        return self.branch(x)

class MomentumGatedPID(nn.Module):
    """
    Momentum-Gated PID network that uses different processing pathways 
    for different momentum regimes.
    """
    
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int,
                 momentum_feature_idx: int = 0,  # Index of momentum in input features
                 dropout_rate: float = 0.15,
                 use_batch_norm: bool = True,
                 momentum_boundaries: List[float] = None):
        super().__init__()
        
        # Store configuration
        self.input_dim = input_dim
        self.num_classes = num_classes
        self.momentum_feature_idx = momentum_feature_idx
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Default momentum boundaries (in GeV/c) based on ALICE plot
        if momentum_boundaries is None:
            self.momentum_boundaries = [0.0, 1.0, 3.0, 10.0]  # Low, Medium, High regimes
        else:
            self.momentum_boundaries = momentum_boundaries
        
        self.num_regimes = len(self.momentum_boundaries) - 1
        
        # Momentum gate network - determines which regime weights to use
        self.momentum_gate = nn.Sequential(
            nn.Linear(1, 32),  # Just momentum as input
            nn.ReLU(),
            nn.Dropout(dropout_rate * 0.5),  # Less dropout for gating
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, self.num_regimes),
            nn.Softmax(dim=1)  # Outputs weights for each regime
        )
        
        # Regime-specific feature extraction branches
        self.regime_branches = nn.ModuleList()
        
        # Low momentum branch (0-1 GeV/c): Best discrimination, larger network
        self.regime_branches.append(
            MomentumRegimeBranch(input_dim, 128, hidden_dim=256, 
                               dropout_rate=dropout_rate, use_batch_norm=use_batch_norm)
        )
        
        # Medium momentum branch (1-3 GeV/c): Moderate discrimination
        self.regime_branches.append(
            MomentumRegimeBranch(input_dim, 64, hidden_dim=128,
                               dropout_rate=dropout_rate, use_batch_norm=use_batch_norm)
        )
        
        # High momentum branch (3+ GeV/c): Poor discrimination, smaller network
        self.regime_branches.append(
            MomentumRegimeBranch(input_dim, 32, hidden_dim=64,
                               dropout_rate=dropout_rate, use_batch_norm=use_batch_norm)
        )
        
        # Feature fusion layer
        total_branch_output_dim = 128 + 64 + 32  # Sum of all branch outputs
        self.feature_fusion = nn.Sequential(
            nn.Linear(total_branch_output_dim, 128),
            nn.BatchNorm1d(128) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, 64),
            nn.BatchNorm1d(64) if use_batch_norm else nn.Identity(),
            nn.ReLU(),
            nn.Dropout(dropout_rate)
        )
        
        # Final classifier
        self.classifier = nn.Linear(64, num_classes)
        
        # Initialize weights
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights using Xavier initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def _get_momentum_regime_weights(self, momentum: torch.Tensor) -> torch.Tensor:
        """
        Compute soft weights for momentum regimes using a smooth gating function.
        This is more robust than hard binning.
        """
        # Use the learned gating network
        momentum_input = momentum.unsqueeze(1)  # Shape: (batch_size, 1)
        regime_weights = self.momentum_gate(momentum_input)  # Shape: (batch_size, num_regimes)
        
        return regime_weights
    
    def forward(self, x):
        """
        Forward pass through momentum-gated network.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               Assumes momentum is at index self.momentum_feature_idx
        
        Returns:
            logits: Output tensor of shape (batch_size, num_classes)
        """
        batch_size = x.size(0)
        
        # Extract momentum
        momentum = x[:, self.momentum_feature_idx]  # Shape: (batch_size,)
        
        # Get regime weights from gating network
        regime_weights = self._get_momentum_regime_weights(momentum)  # Shape: (batch_size, num_regimes)
        
        # Process through all regime branches
        branch_outputs = []
        for branch in self.regime_branches:
            branch_output = branch(x)  # Each branch outputs different dimensions
            branch_outputs.append(branch_output)
        
        # Weight each branch output by its regime weight
        weighted_outputs = []
        for i, branch_output in enumerate(branch_outputs):
            # Broadcast regime weight to match branch output dimensions
            weight = regime_weights[:, i:i+1]  # Shape: (batch_size, 1)
            weighted_output = branch_output * weight  # Broadcasting
            weighted_outputs.append(weighted_output)
        
        # Concatenate all weighted branch outputs
        combined_features = torch.cat(weighted_outputs, dim=1)  # Shape: (batch_size, total_branch_dims)
        
        # Feature fusion
        fused_features = self.feature_fusion(combined_features)
        
        # Final classification
        logits = self.classifier(fused_features)
        
        return logits
    
    def predict_proba(self, x):
        """Return class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities
    
    def get_regime_weights(self, x):
        """
        Get the momentum regime weights for analysis.
        Useful for understanding which regimes the model is using.
        """
        with torch.no_grad():
            momentum = x[:, self.momentum_feature_idx]
            regime_weights = self._get_momentum_regime_weights(momentum)
        return regime_weights
    
    def analyze_momentum_gating(self, momentum_range=None):
        """
        Analyze how the gating function behaves across momentum range.
        Useful for debugging and understanding the model.
        """
        if momentum_range is None:
            momentum_range = torch.linspace(0.1, 5.0, 100)
        
        # Create dummy input with varying momentum
        dummy_input = torch.zeros(len(momentum_range), self.input_dim)
        dummy_input[:, self.momentum_feature_idx] = momentum_range
        
        regime_weights = self.get_regime_weights(dummy_input)
        
        return momentum_range.numpy(), regime_weights.numpy()

class EarlyStopping:
    """Early stopping utility to prevent overfitting."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def __call__(self, val_loss: float) -> bool:
        """Check if training should be stopped."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            self.should_stop = True
            
        return self.should_stop

# Utility function to visualize momentum gating behavior
def plot_momentum_gating(model, output_path='momentum_gating.pdf'):
    """Plot how the momentum gating function behaves."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    momentum_range, regime_weights = model.analyze_momentum_gating()
    
    plt.figure(figsize=(10, 6))
    
    regime_names = ['Low (0-1 GeV/c)', 'Medium (1-3 GeV/c)', 'High (3+ GeV/c)']
    colors = ['blue', 'green', 'red']
    
    for i in range(regime_weights.shape[1]):
        plt.plot(momentum_range, regime_weights[:, i], 
                label=f'{regime_names[i]}', color=colors[i], linewidth=2)
    
    plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Regime boundaries')
    plt.axvline(x=3.0, color='gray', linestyle='--', alpha=0.7)
    
    plt.xlabel('Momentum (GeV/c)')
    plt.ylabel('Regime Weight')
    plt.title('Momentum Gating Function')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.xlim(0, 5)
    plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return momentum_range, regime_weights