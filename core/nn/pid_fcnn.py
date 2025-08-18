"""
Improved neural network model for particle identification with better architecture.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

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

class PidFCNN(nn.Module):
    """Improved fully connected neural network for particle identification."""
    
    def __init__(self, 
                 input_dim: int, 
                 num_classes: int,
                 hidden_dims: List[int] = None,
                 dropout_rate: float = 0.15,
                 use_batch_norm: bool = True,
                 use_residual: bool = True):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 128, 64]
            
        self.use_batch_norm = use_batch_norm
        self.use_residual = use_residual
        
        # Input layer
        self.input_layer = nn.Linear(input_dim, hidden_dims[0])
        self.input_bn = nn.BatchNorm1d(hidden_dims[0]) if use_batch_norm else nn.Identity()
        self.input_dropout = nn.Dropout(dropout_rate)
        
        # Hidden layers
        self.hidden_layers = nn.ModuleList()
        self.hidden_bns = nn.ModuleList()
        self.hidden_dropouts = nn.ModuleList()
        self.residual_blocks = nn.ModuleList()
        
        for i in range(len(hidden_dims) - 1):
            # Standard layer
            self.hidden_layers.append(nn.Linear(hidden_dims[i], hidden_dims[i+1]))
            if use_batch_norm:
                self.hidden_bns.append(nn.BatchNorm1d(hidden_dims[i+1]))
            else:
                self.hidden_bns.append(nn.Identity())
            self.hidden_dropouts.append(nn.Dropout(dropout_rate))
            
            # Optional residual block (only if dimensions match)
            if use_residual and hidden_dims[i] == hidden_dims[i+1]:
                self.residual_blocks.append(ResidualBlock(hidden_dims[i+1], dropout_rate))
            else:
                self.residual_blocks.append(nn.Identity())
        
        # Output layer
        self.output_layer = nn.Linear(hidden_dims[-1], num_classes)
        
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
    
    def forward(self, x):
        # Input layer
        x = F.relu(self.input_bn(self.input_layer(x)))
        x = self.input_dropout(x)
        
        # Hidden layers with optional residual connections
        for layer, bn, dropout, res_block in zip(
            self.hidden_layers, self.hidden_bns, self.hidden_dropouts, self.residual_blocks
        ):
            x = F.relu(bn(layer(x)))
            x = dropout(x)
            x = res_block(x)  # Apply residual block if applicable
        
        # Output layer (no activation - CrossEntropyLoss handles this)
        x = self.output_layer(x)
        return x
    
    def predict_proba(self, x):
        """Return class probabilities."""
        with torch.no_grad():
            logits = self.forward(x)
            probabilities = F.softmax(logits, dim=1)
        return probabilities

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
