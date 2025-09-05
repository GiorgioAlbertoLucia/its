from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import ExperimentConfig
from core.bdt.models import BaseBDTModel

@dataclass
class TrainingState:
    """Container for BDT training state - similar to NN TrainingState."""
    train_losses: List[float] = None
    val_losses: List[float] = None
    train_accuracies: List[float] = None  # Can be computed if needed
    val_accuracies: List[float] = None
    best_val_loss: float = float('inf')
    best_iteration: int = 0
    
    def __post_init__(self):
        if self.train_losses is None:
            self.train_losses = []
        if self.val_losses is None:
            self.val_losses = []
        if self.train_accuracies is None:
            self.train_accuracies = []
        if self.val_accuracies is None:
            self.val_accuracies = []

class Trainer:
    """BDT trainer following the same pattern as NN Trainer."""
    
    def __init__(self, model: BaseBDTModel, config: ExperimentConfig, 
                 class_weights: Optional[Dict[int, float]] = None):
        self.model = model
        self.config = config
        self.class_weights = class_weights
        self.logger = logging.getLogger(__name__)
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray,
             X_val: np.ndarray, y_val: np.ndarray) -> TrainingState:
        """Train the BDT model."""
        
        self.logger.info("Starting BDT training")
        self.logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        # Combine train and validation for model's internal splitting
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        
        # Train the model
        training_result = self.model.fit(
            X_combined, y_combined, 
            val_size=len(X_val) / len(X_combined)
        )
        
        # Create training state from results
        state = TrainingState()
        state.train_losses = training_result.get('train_losses', [])
        state.val_losses = training_result.get('val_losses', [])
        state.best_iteration = training_result.get('best_iteration', 0)
        
        if state.val_losses:
            state.best_val_loss = min(state.val_losses)
        
        self.logger.info("BDT training completed")
        return state
