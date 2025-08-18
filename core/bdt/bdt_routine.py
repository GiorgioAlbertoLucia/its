"""
Simple Momentum-Aware BDT implementation following the neural network patterns.
"""

from dataclasses import dataclass
import numpy as np
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

from ROOT import TFile
from torchic.core.histogram import AxisSpec, build_TH2

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from core.bdt.momentum_aware_bdt import MomentumAwareBDT

@dataclass
class BDTConfig:
    """
    Configuration for BDT training.
    """
    learning_rate: float = 0.1
    n_estimators: int = 100
    max_depth: int = 6
    min_child_weight: int = 1
    gamma: float = 0.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    scale_pos_weight: Optional[float] = None
    test_size: float = 0.2

class BDTRoutine:
    """
    Training routine for BDT models, similar to NNRoutine.
    """
    
    def __init__(self, model: MomentumAwareBDT, class_weights: Optional[Dict] = None):
        self.model = model
        self.class_weights = class_weights
        self.logger = logging.getLogger(__name__)
        
        # Training history (will be populated during training)
        self.training_history = {}
    
    def run_training_loop(self, 
                         X: np.ndarray, 
                         y: np.ndarray,
                         val_size: float = 0.2,
                         **kwargs) -> Dict:
        """
        Run complete training loop similar to neural network routine.
        
        Returns:
            Dictionary with training history and metrics
        """
        
        self.logger.info("Starting BDT training")
        self.logger.info(f"Training samples: {len(X)}")
        self.logger.info(f"Features: {X.shape[1]}")
        self.logger.info(f"Classes: {len(np.unique(y))}")
        
        # Train the model
        training_history = self.model.fit(X, y, val_size=val_size)
        self.training_history = training_history
        
        self.logger.info("BDT training completed")
        
        return training_history
    
    def get_model_scores(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get model scores compatible with neural network interface.
        
        Returns:
            all_scores: Shape (N, num_classes) - probabilities for each class
            all_labels: Shape (N,) - placeholder (empty array since no labels provided)
            all_features: Shape (N, num_features) - input features
        """
        
        all_scores = self.model.predict_proba(X)
        all_labels = np.array([])  # Empty since we don't have labels
        all_features = X
        
        return all_scores, all_labels, all_features
    
    def plot_model_scores(self, 
                          X: np.ndarray,
                          class_names: List[str],
                          output_file: TFile,
                          momentum_idx: Optional[int] = None) -> None:
        """
        Plot model scores for each class as a function of momentum.
        Mirrors the neural network plotting function exactly.
        
        Args:
            X: Input features  
            class_names: List of class names for labeling
            output_file: ROOT TFile object for saving
            momentum_idx: Index of momentum (uses model's if None)
        """
        
        if momentum_idx is None:
            momentum_idx = self.model.momentum_feature_idx
        
        all_scores, _, all_features = self.get_model_scores(X)
        momentum = all_features[:, momentum_idx]
        
        print(f'{all_scores.shape=}')
        
        output_file.cd()
        
        axis_spec_p = AxisSpec(50, 0, 5, 'p', '#it{p} (GeV/#it{c})')
        
        for class_idx, class_name in enumerate(class_names):
            axis_spec_score = AxisSpec(
                100, 0, 1, 'Score', 
                f'{class_name} hypothesis;#it{{p}} (GeV/#it{{c}});Probability;'
            )
            hist = build_TH2(momentum, all_scores[:, class_idx], axis_spec_p, axis_spec_score)
            hist.Write(f'bdt_{class_name}')
            
        self.logger.info(f"Created BDT score histograms for {len(class_names)} classes")

def plot_momentum_performance(model: MomentumAwareBDT, X: np.ndarray, y: np.ndarray,
                            output_path: str = 'bdt_momentum_performance.pdf'):
    """
    Plot BDT performance as function of momentum.
    Similar to plot_momentum_gating for neural networks.
    """
    import matplotlib.pyplot as plt
    
    momentum_centers, accuracies, confidences = model.analyze_momentum_dependence(X, y)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accuracy vs momentum
    valid_mask = ~np.isnan(accuracies)
    ax1.plot(momentum_centers[valid_mask], accuracies[valid_mask], 'bo-', linewidth=2)
    ax1.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Regime boundaries')
    ax1.axvline(x=3.0, color='gray', linestyle='--', alpha=0.7)
    ax1.set_xlabel('Momentum (GeV/c)')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('BDT Accuracy vs Momentum')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    
    # Confidence vs momentum
    valid_mask = ~np.isnan(confidences)
    ax2.plot(momentum_centers[valid_mask], confidences[valid_mask], 'ro-', linewidth=2)
    ax2.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Regime boundaries')
    ax2.axvline(x=3.0, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Momentum (GeV/c)')
    ax2.set_ylabel('Average Max Probability')
    ax2.set_title('BDT Confidence vs Momentum')
    ax2.grid(True, alpha=0.3)
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.show()
    
    return momentum_centers, accuracies, confidences