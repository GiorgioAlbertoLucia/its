from dataclasses import dataclass
from typing import Dict, Optional, List
import numpy as np
import logging

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import ExperimentConfig
from core.bdt.models import BaseBDTModel
from core.bdt.optimizer import BDTOptunaOptimizer

@dataclass
class TrainingState:
    """Container for BDT training state - similar to NN TrainingState."""
    train_losses: List[float] = None
    val_losses: List[float] = None
    train_accuracies: List[float] = None  # Can be computed if needed
    val_accuracies: List[float] = None
    best_val_loss: float = float('inf')
    best_iteration: int = 0

    # optuna
    optimization_summary: Optional[Dict] = None
    best_hyperparameters: Optional[Dict] = None
    
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

    def _train_with_optimization(self, 
                                X_train: np.ndarray, 
                                y_train: np.ndarray,
                                X_val: np.ndarray, 
                                y_val: np.ndarray) -> TrainingState:
        """Train model with Optuna hyperparameter optimization."""
        
        self.logger.info("Starting hyperparameter optimization with Optuna")
        
        self.optimizer = BDTOptunaOptimizer(
            config=self.config
        )
        
        study = self.optimizer.optimize(X_train, y_train, X_val, y_val)
        self.optimized_model = self.optimizer.get_best_model()
        self.logger.info("Training optimized model on full training data")
        
        X_combined = np.vstack([X_train, X_val])
        y_combined = np.hstack([y_train, y_val])
        
        training_result = self.optimized_model.fit(
            X_combined, y_combined, 
            val_size=len(X_val) / len(X_combined)
        )
        
        state = TrainingState()
        state.train_losses = training_result.get('train_losses', [])
        state.val_losses = training_result.get('val_losses', [])
        state.best_iteration = training_result.get('best_iteration', 0)
        
        if state.val_losses:
            state.best_val_loss = min(state.val_losses)
        
        state.optimization_summary = self.optimizer.get_optimization_summary()
        state.best_hyperparameters = study.best_params
        self.model = self.optimized_model
        
        return state
    
    def _train_without_optimization(self, 
                                   X_train: np.ndarray, 
                                   y_train: np.ndarray,
                                   X_val: np.ndarray, 
                                   y_val: np.ndarray) -> TrainingState:
        """Train model without optimization (original behavior)."""
        
        self.logger.info("Training BDT without hyperparameter optimization")
        
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
        
        return state
    
    def train(self, 
             X_train: np.ndarray, 
             y_train: np.ndarray,
             X_val: np.ndarray, 
             y_val: np.ndarray) -> TrainingState:
        """
        Train the BDT model with or without Optuna optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels  
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            TrainingState with training history and optimization info
        """
        
        self.logger.info(f"Training samples: {len(X_train)}, Validation samples: {len(X_val)}")
        
        if self.config.use_optuna:
            state = self._train_with_optimization(X_train, y_train, X_val, y_val)
            self.logger.info("Training with optimization completed")
            
            if self.config.save_output_optuna:
                study_path = self.config.output_dir / f"optuna_study_{self.config.output_file_suffix}"
                self.optimizer.save_study(study_path)
                
                # Save optimization plots
                try:
                    plot_path = self.config.output_dir / f"optuna_plots_{self.config.output_file_suffix}.png"
                    self.optimizer.plot_optimization_history(plot_path)
                except Exception as e:
                    self.logger.warning(f"Could not save optimization plots: {e}")

        else:
            state = self._train_without_optimization(X_train, y_train, X_val, y_val)
            self.logger.info("Training without optimization completed")
        
        return state
    