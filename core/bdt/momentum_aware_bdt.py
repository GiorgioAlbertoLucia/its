"""
Simple Momentum-Aware BDT implementation following the neural network patterns.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import logging
from typing import Dict, List, Tuple, Optional

from ROOT import TFile
from torchic.core.histogram import AxisSpec, build_TH2

class MomentumAwareBDT:
    """
    Simple momentum-aware BDT using single XGBoost model with enhanced features.
    Follows similar structure to the MomentumGatedPID neural network.
    """
    
    def __init__(self, 
                 momentum_feature_idx: int = 0,
                 xgb_params: Optional[Dict] = None):
        """
        Args:
            momentum_feature_idx: Index of momentum feature in input
            xgb_params: XGBoost hyperparameters
        """
        
        self.momentum_feature_idx = momentum_feature_idx
        
        if xgb_params is None:
            self.xgb_params = {
                'objective': 'multi:softprob',
                'eval_metric': 'mlogloss',
                'max_depth': 6,
                'learning_rate': 0.3,
                'n_estimators': 300,
                'subsample': 0.8,
                'colsample_bytree': 0.8,
                'min_child_weight': 3,
                'reg_alpha': 0.1,
                'reg_lambda': 1.0,
                'random_state': 42,
                'n_jobs': 1,
                'nthread': 1,
                'verbosity': 0,
                'early_stopping_rounds': 20,
            }
        else:
            self.xgb_params = xgb_params
        
        self.model = None
        self.classes_ = None
        self.feature_names_ = None
        self.logger = logging.getLogger(__name__)
    
    def _create_momentum_features(self, X: np.ndarray) -> np.ndarray:
        """
        Create momentum-enhanced features similar to regime processing in NN.
        """
        X_enhanced = X.copy()
        momentum = X[:, self.momentum_feature_idx]
        
        # Log momentum (helps with wide momentum range)
        log_momentum = np.log(momentum + 1e-6)
        X_enhanced = np.column_stack([X_enhanced, log_momentum])
        
        # Momentum squared (relativistic effects)
        momentum_sq = momentum ** 2
        X_enhanced = np.column_stack([X_enhanced, momentum_sq])
        
        # Momentum regime indicators (soft boundaries like NN gating)
        # Low momentum indicator (Gaussian-like activation)
        low_p_indicator = np.exp(-((momentum - 0.5) ** 2) / (2 * 0.3 ** 2))
        X_enhanced = np.column_stack([X_enhanced, low_p_indicator])
        
        # Medium momentum indicator
        med_p_indicator = np.exp(-((momentum - 2.0) ** 2) / (2 * 0.8 ** 2))
        X_enhanced = np.column_stack([X_enhanced, med_p_indicator])
        
        # High momentum indicator
        high_p_indicator = 1.0 / (1.0 + np.exp(-2 * (momentum - 3.0)))  # Sigmoid
        X_enhanced = np.column_stack([X_enhanced, high_p_indicator])
        
        # Interaction terms: multiply original features by momentum
        for i in range(X.shape[1]):
            if i != self.momentum_feature_idx:
                interaction = X[:, i] * momentum
                X_enhanced = np.column_stack([X_enhanced, interaction])
        
        return X_enhanced
    
    def fit(self, X: np.ndarray, y: np.ndarray, 
            val_size: float = 0.2,
            sample_weight: Optional[np.ndarray] = None) -> Dict:
        """
        Fit the BDT model with validation monitoring.
        Returns training history similar to neural network.
        """
        self.classes_ = np.unique(y)
        n_classes = len(self.classes_)
        
        X_enhanced = self._create_momentum_features(X)
        X_train, X_val, y_train, y_val = train_test_split(
            X_enhanced, y, test_size=val_size, random_state=42, stratify=y
        )
        
        if sample_weight is None:
            class_weights = compute_class_weight('balanced', classes=self.classes_, y=y_train)
            sample_weight_train = np.array([class_weights[cls] for cls in y_train])
        else:
            sample_weight_train, _ = train_test_split(
                sample_weight, test_size=val_size, random_state=42, stratify=y
            )
        
        self.model = xgb.XGBClassifier(**self.xgb_params, num_class=n_classes)
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False,
            callbacks=[xgb.callback.EvaluationMonitor(show_stdv=False)]
        )
        
        evals_result = self.model.evals_result()
        train_losses = evals_result['validation_0']['mlogloss'] # Training loss
        val_losses = evals_result['validation_1']['mlogloss'] # Validation loss
        
        train_accuracies = []
        val_accuracies = []
        
        for epoch in range(len(train_losses)):
            temp_model = xgb.XGBClassifier(**self.xgb_params)
            temp_model.set_params(n_estimators=epoch+1)
            temp_model.fit(X_train, y_train, sample_weight=sample_weight_train, verbose=False,
                           eval_set=[(X_train, y_train), (X_val, y_val)])
            
            train_pred = temp_model.predict(X_train)
            val_pred = temp_model.predict(X_val)
            
            train_acc = 100 * accuracy_score(y_train, train_pred)
            val_acc = 100 * accuracy_score(y_val, val_pred)
            
            train_accuracies.append(train_acc)
            val_accuracies.append(val_acc)
        
        self.logger.info(f"Training completed. Best iteration: {self.model.best_iteration}")
        self.logger.info(f"Final validation loss: {val_losses[-1]:.4f}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_accuracies': train_accuracies,
            'val_accuracies': val_accuracies,
            'best_iteration': self.model.best_iteration
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Return class probabilities - equivalent to get_model_scores in NN.
        
        Returns:
            probabilities: Shape (N, num_classes) - probabilities for each class
        """
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        X_enhanced = self._create_momentum_features(X)
        return self.model.predict_proba(X_enhanced)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""

        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def get_model_scores(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model scores similar to neural network implementation.
        
        Returns:
            all_scores: Shape (N, num_classes) - probabilities for each class
            momentum: Shape (N,) - momentum values for plotting
        """

        scores = self.predict_proba(X)
        momentum = X[:, self.momentum_feature_idx]
        
        return scores, momentum
    
    def plot_model_scores(self, 
                          X: np.ndarray,
                          class_names: List[str],
                          output_file: TFile) -> None:
        """
        Plot model scores for each class as a function of momentum using ROOT.
        Mirrors the neural network plotting function.
        
        Args:
            X: Input features
            class_names: List of class names for labeling
            output_file: ROOT TFile object for saving histograms
        """
        
        all_scores, momentum = self.get_model_scores(X)
        self.logger.info(f"Plotting BDT scores. Shape: {all_scores.shape}")
        
        output_file.cd()
        axis_spec_p = AxisSpec(50, 0, 5, 'p', '#it{p} (GeV/#it{c})')
        
        for class_idx, class_name in enumerate(class_names):
            axis_spec_score = AxisSpec(
                100, 0, 1, 'Score', 
                f'{class_name} BDT Score;#it{{p}} (GeV/#it{{c}});Probability'
            )
            
            hist = build_TH2(
                momentum, 
                all_scores[:, class_idx], 
                axis_spec_p, 
                axis_spec_score
            )
            
            hist_name = f'bdt_scores_{class_name}'
            hist.SetName(hist_name)
            hist.SetTitle(f'BDT {class_name} Hypothesis Score vs Momentum')
            hist.Write(hist_name)
            
            self.logger.info(f"Created histogram: {hist_name}")
    
    def analyze_momentum_dependence(self, X: np.ndarray, y: np.ndarray, 
                                  momentum_range: Tuple[float, float] = (0.1, 5.0),
                                  n_bins: int = 20) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Analyze how BDT performance varies with momentum.
        Similar to momentum gating analysis in neural network.
        
        Returns:
            momentum_centers: Bin centers for momentum
            accuracies: Accuracy in each momentum bin
            confidences: Average max probability in each bin
        """
        
        momentum = X[:, self.momentum_feature_idx]
        predictions = self.predict(X)
        probabilities = self.predict_proba(X)
        
        momentum_bins = np.linspace(momentum_range[0], momentum_range[1], n_bins + 1)
        momentum_centers = (momentum_bins[:-1] + momentum_bins[1:]) / 2
        
        accuracies = []
        confidences = []
        sample_counts = []
        
        for i in range(len(momentum_bins) - 1):
            bin_mask = ((momentum >= momentum_bins[i]) & 
                       (momentum < momentum_bins[i+1]))
            
            n_samples = np.sum(bin_mask)
            sample_counts.append(n_samples)
            
            if n_samples > 10:
                bin_acc = accuracy_score(y[bin_mask], predictions[bin_mask])
                bin_conf = np.mean(np.max(probabilities[bin_mask], axis=1))
                accuracies.append(bin_acc)
                confidences.append(bin_conf)
            else:
                accuracies.append(np.nan)
                confidences.append(np.nan)
        
        self.logger.info(f"Momentum analysis completed across {n_bins} bins")
        self.logger.info(f"Sample distribution: min={min(sample_counts)}, max={max(sample_counts)}")
        
        return momentum_centers, np.array(accuracies), np.array(confidences)
    
    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature importance analysis.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        importance = self.model.feature_importances_
        
        if feature_names is None:

            n_original = len(importance) - 3 - len(importance)//2 
            feature_names = []
            for i in range(n_original):
                feature_names.append(f'original_feature_{i}')
            
            feature_names.extend([
                'log_momentum', 'momentum_squared', 
                'low_p_indicator', 'med_p_indicator', 'high_p_indicator'
            ])
            
            for i in range(n_original):
                if i != self.momentum_feature_idx:
                    feature_names.append(f'momentum_x_feature_{i}')
        
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df

class EarlyStopping:
    """Early stopping utility for BDT training."""
    
    def __init__(self, patience: int = 20, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.best_score = float('inf')
        self.patience_counter = 0
        self.should_stop = False
    
    def __call__(self, val_score: float) -> bool:
        """Check if training should be stopped."""
        if val_score < self.best_score - self.min_delta:
            self.best_score = val_score
            self.patience_counter = 0
        else:
            self.patience_counter += 1
            
        if self.patience_counter >= self.patience:
            self.should_stop = True
            
        return self.should_stop
    