"""
Regime-Specific BDT Ensemble implementation for momentum-dependent classification.
Each momentum bin gets its own specialized BDT model.
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score
import logging
from typing import Dict, List, Tuple, Optional, Union
import pickle
import os

from ROOT import TFile
from torchic.core.histogram import AxisSpec, build_TH2

class RegimeSpecificBDT:
    """
    Single BDT model for a specific momentum regime.
    Simplified version without momentum features since it operates in narrow range.
    """
    
    def __init__(self, 
                 momentum_range: Tuple[float, float],
                 xgb_params: Optional[Dict] = None):
        """
        Args:
            momentum_range: (min_momentum, max_momentum) for this regime
            xgb_params: XGBoost hyperparameters
        """
        
        self.momentum_range = momentum_range
        
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
        self.n_samples_trained = 0
        self.logger = logging.getLogger(__name__)
    
    def _filter_momentum_range(self, X: np.ndarray, y: np.ndarray, 
                              momentum_feature_idx: int,
                              sample_weight: Optional[np.ndarray] = None
                              ) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
        """Filter data to momentum range for this regime."""
        momentum = X[:, momentum_feature_idx]
        mask = ((momentum >= self.momentum_range[0]) & 
                (momentum < self.momentum_range[1]))
        
        X_filtered = X[mask]
        y_filtered = y[mask]
        
        sample_weight_filtered = None
        if sample_weight is not None:
            sample_weight_filtered = sample_weight[mask]
        
        return X_filtered, y_filtered, sample_weight_filtered
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            momentum_feature_idx: int,
            val_size: float = 0.2,
            sample_weight: Optional[np.ndarray] = None,
            min_samples: int = 50) -> Dict:
        """
        Fit the regime-specific BDT model.
        
        Args:
            X: Input features
            y: Target labels
            momentum_feature_idx: Index of momentum feature
            val_size: Validation split ratio
            sample_weight: Sample weights
            min_samples: Minimum samples needed to train model
            
        Returns:
            Training history dictionary
        """
        
        # Filter data to momentum range
        X_regime, y_regime, weight_regime = self._filter_momentum_range(
            X, y, momentum_feature_idx, sample_weight
        )
        
        self.n_samples_trained = len(X_regime)
        
        if self.n_samples_trained < min_samples:
            self.logger.warning(
                f"Regime {self.momentum_range}: Only {self.n_samples_trained} samples, "
                f"minimum required: {min_samples}. Skipping training."
            )
            return {
                'trained': False,
                'n_samples': self.n_samples_trained,
                'reason': 'insufficient_samples'
            }
        
        self.classes_ = np.unique(y_regime)
        n_classes = len(self.classes_)
        
        # Remove momentum feature since we're in a specific regime
        feature_mask = np.ones(X_regime.shape[1], dtype=bool)
        feature_mask[momentum_feature_idx] = False
        X_regime_no_p = X_regime[:, feature_mask]
        
        # Split into train/validation
        if len(X_regime_no_p) > 10:  # Need enough for validation split
            X_train, X_val, y_train, y_val = train_test_split(
                X_regime_no_p, y_regime, test_size=val_size, 
                random_state=42, stratify=y_regime
            )
            
            if weight_regime is not None:
                weight_train, weight_val = train_test_split(
                    weight_regime, test_size=val_size, 
                    random_state=42, stratify=y_regime
                )
            else:
                # Compute class weights for this regime
                class_weights = compute_class_weight('balanced', 
                                                   classes=self.classes_, 
                                                   y=y_train)
                weight_train = np.array([class_weights[cls] for cls in y_train])
                weight_val = None
        else:
            # Too few samples for validation split
            X_train, X_val = X_regime_no_p, None
            y_train, y_val = y_regime, None
            weight_train = weight_regime
            weight_val = None
        
        # Train model
        self.model = xgb.XGBClassifier(**self.xgb_params, num_class=n_classes)
        
        eval_set = [(X_train, y_train)]
        eval_names = ['train']
        if X_val is not None:
            eval_set.append((X_val, y_val))
            eval_names.append('val')
        
        self.model.fit(
            X_train, y_train,
            sample_weight=weight_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Get training history
        evals_result = self.model.evals_result()
        train_losses = evals_result['validation_0']['mlogloss']
        val_losses = evals_result.get('validation_1', {}).get('mlogloss', [])
        
        # Calculate accuracies
        train_pred = self.model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        
        val_acc = None
        if X_val is not None:
            val_pred = self.model.predict(X_val)
            val_acc = accuracy_score(y_val, val_pred)
        
        self.logger.info(
            f"Regime {self.momentum_range}: Trained on {self.n_samples_trained} samples. "
            f"Train acc: {train_acc:.3f}, Val acc: {val_acc:.3f if val_acc else 'N/A'}"
        )
        
        return {
            'trained': True,
            'n_samples': self.n_samples_trained,
            'train_losses': train_losses,
            'val_losses': val_losses,
            'train_acc': train_acc,
            'val_acc': val_acc,
            'best_iteration': self.model.best_iteration
        }
    
    def predict_proba(self, X: np.ndarray, momentum_feature_idx: int) -> np.ndarray:
        """Predict probabilities for samples in this momentum regime."""
        if self.model is None:
            raise ValueError(f"Regime {self.momentum_range} model not trained yet.")
        
        # Remove momentum feature
        feature_mask = np.ones(X.shape[1], dtype=bool)
        feature_mask[momentum_feature_idx] = False
        X_no_p = X[:, feature_mask]
        
        return self.model.predict_proba(X_no_p)
    
    def predict(self, X: np.ndarray, momentum_feature_idx: int) -> np.ndarray:
        """Predict class labels for samples in this momentum regime."""
        probabilities = self.predict_proba(X, momentum_feature_idx)
        return np.argmax(probabilities, axis=1)


class RegimeSpecificBDTEnsemble:
    """
    Ensemble of regime-specific BDT models operating in different momentum ranges.
    """
    
    def __init__(self, 
                 momentum_bins: Optional[List[float]] = None,
                 momentum_feature_idx: int = 0,
                 xgb_params: Optional[Dict] = None,
                 overlap_strategy: str = 'average',
                 min_samples_per_regime: int = 50):
        """
        Args:
            momentum_bins: Bin edges for momentum regimes
            momentum_feature_idx: Index of momentum feature in input
            xgb_params: XGBoost parameters for all regime models
            overlap_strategy: How to handle predictions in overlapping regions
            min_samples_per_regime: Minimum samples needed to train a regime model
        """
        
        if momentum_bins is None:
            momentum_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0]
        
        self.momentum_bins = sorted(momentum_bins)
        self.momentum_feature_idx = momentum_feature_idx
        self.xgb_params = xgb_params
        self.overlap_strategy = overlap_strategy
        self.min_samples_per_regime = min_samples_per_regime
        
        # Create regime models
        self.regime_models = {}
        self.momentum_ranges = []
        
        for i in range(len(self.momentum_bins) - 1):
            momentum_range = (self.momentum_bins[i], self.momentum_bins[i + 1])
            self.momentum_ranges.append(momentum_range)
            
            regime_model = RegimeSpecificBDT(
                momentum_range=momentum_range,
                xgb_params=xgb_params
            )
            self.regime_models[momentum_range] = regime_model
        
        self.classes_ = None
        self.trained_regimes = []
        self.logger = logging.getLogger(__name__)
        
        self.logger.info(f"Created ensemble with {len(self.regime_models)} regime models")
        self.logger.info(f"Momentum ranges: {self.momentum_ranges}")
    
    def fit(self, X: np.ndarray, y: np.ndarray,
            val_size: float = 0.2,
            sample_weight: Optional[np.ndarray] = None) -> Dict:
        """
        Fit all regime-specific models.
        
        Returns:
            Dictionary with training results for each regime
        """
        
        self.classes_ = np.unique(y)
        training_results = {}
        self.trained_regimes = []
        
        self.logger.info("Starting regime-specific training...")
        
        for momentum_range, regime_model in self.regime_models.items():
            self.logger.info(f"Training regime {momentum_range}...")
            
            result = regime_model.fit(
                X, y,
                momentum_feature_idx=self.momentum_feature_idx,
                val_size=val_size,
                sample_weight=sample_weight,
                min_samples=self.min_samples_per_regime
            )
            
            training_results[momentum_range] = result
            
            if result.get('trained', False):
                self.trained_regimes.append(momentum_range)
        
        self.logger.info(f"Training completed. {len(self.trained_regimes)} out of "
                        f"{len(self.regime_models)} regimes trained successfully.")
        
        return training_results
    
    def _get_regime_for_momentum(self, momentum_val: float) -> List[Tuple[float, float]]:
        """Get the regime(s) that should handle a given momentum value."""
        applicable_regimes = []
        
        for momentum_range in self.momentum_ranges:
            min_p, max_p = momentum_range
            if min_p <= momentum_val < max_p:
                applicable_regimes.append(momentum_range)
        
        # Handle edge case for maximum momentum
        if momentum_val >= self.momentum_bins[-1] and self.momentum_ranges:
            applicable_regimes.append(self.momentum_ranges[-1])
        
        # Filter to only trained regimes
        applicable_regimes = [r for r in applicable_regimes if r in self.trained_regimes]
        
        return applicable_regimes
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """
        Predict class probabilities using appropriate regime models.
        
        Returns:
            probabilities: Shape (N, num_classes) - probabilities for each class
        """
        
        if not self.trained_regimes:
            raise ValueError("No regime models trained yet. Call fit() first.")
        
        n_samples = X.shape[0]
        n_classes = len(self.classes_)
        momentum = X[:, self.momentum_feature_idx]
        
        # Initialize prediction arrays
        all_probabilities = np.zeros((n_samples, n_classes))
        prediction_counts = np.zeros(n_samples)
        
        # Get predictions from each applicable regime
        for sample_idx in range(n_samples):
            momentum_val = momentum[sample_idx]
            applicable_regimes = self._get_regime_for_momentum(momentum_val)
            
            if not applicable_regimes:
                # Use nearest trained regime
                distances = [
                    min(abs(momentum_val - r[0]), abs(momentum_val - r[1])) 
                    for r in self.trained_regimes
                ]
                nearest_regime_idx = np.argmin(distances)
                applicable_regimes = [self.trained_regimes[nearest_regime_idx]]
            
            # Accumulate predictions from applicable regimes
            sample_probs = np.zeros(n_classes)
            regime_count = 0
            
            for regime_range in applicable_regimes:
                if regime_range in self.trained_regimes:
                    regime_model = self.regime_models[regime_range]
                    regime_probs = regime_model.predict_proba(
                        X[sample_idx:sample_idx+1], 
                        self.momentum_feature_idx
                    )[0]  # Get first (and only) prediction
                    
                    sample_probs += regime_probs
                    regime_count += 1
            
            # Average predictions from multiple regimes
            if regime_count > 0:
                all_probabilities[sample_idx] = sample_probs / regime_count
                prediction_counts[sample_idx] = regime_count
        
        # Check for samples without predictions
        no_prediction_mask = prediction_counts == 0
        if np.any(no_prediction_mask):
            self.logger.warning(f"{np.sum(no_prediction_mask)} samples have no regime predictions")
            # Use uniform probabilities for these samples
            all_probabilities[no_prediction_mask] = 1.0 / n_classes
        
        return all_probabilities
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    
    def get_model_scores(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get model scores similar to single BDT implementation.
        
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
        Plot ensemble model scores for each class as a function of momentum.
        """
        
        all_scores, momentum = self.get_model_scores(X)
        self.logger.info(f"Plotting ensemble BDT scores. Shape: {all_scores.shape}")
        
        output_file.cd()
        axis_spec_p = AxisSpec(50, 0, 5, 'p', '#it{p} (GeV/#it{c})')
        
        for class_idx, class_name in enumerate(class_names):
            axis_spec_score = AxisSpec(
                100, 0, 1, 'Score', 
                f'{class_name} Ensemble BDT Score;#it{{p}} (GeV/#it{{c}});Probability'
            )
            
            hist = build_TH2(
                momentum, 
                all_scores[:, class_idx], 
                axis_spec_p, 
                axis_spec_score
            )
            
            hist_name = f'ensemble_bdt_scores_{class_name}'
            hist.SetName(hist_name)
            hist.SetTitle(f'Ensemble BDT {class_name} Hypothesis Score vs Momentum')
            hist.Write(hist_name)
            
            self.logger.info(f"Created histogram: {hist_name}")
    
    def analyze_regime_performance(self, X: np.ndarray, y: np.ndarray) -> pd.DataFrame:
        """
        Analyze performance of each regime model on test data.
        
        Returns:
            DataFrame with performance metrics for each regime
        """
        
        momentum = X[:, self.momentum_feature_idx]
        results = []
        
        for momentum_range in self.trained_regimes:
            regime_model = self.regime_models[momentum_range]
            
            # Filter data to this regime
            mask = ((momentum >= momentum_range[0]) & 
                   (momentum < momentum_range[1]))
            
            if not np.any(mask):
                continue
            
            X_regime = X[mask]
            y_regime = y[mask]
            
            # Get predictions
            regime_probs = regime_model.predict_proba(X_regime, self.momentum_feature_idx)
            regime_preds = np.argmax(regime_probs, axis=1)
            
            # Calculate metrics
            accuracy = accuracy_score(y_regime, regime_preds)
            confidence = np.mean(np.max(regime_probs, axis=1))
            n_samples = len(y_regime)
            
            results.append({
                'momentum_range': f"{momentum_range[0]:.1f}-{momentum_range[1]:.1f}",
                'momentum_min': momentum_range[0],
                'momentum_max': momentum_range[1],
                'n_samples': n_samples,
                'accuracy': accuracy,
                'mean_confidence': confidence,
                'n_estimators': regime_model.model.best_iteration if regime_model.model else 0
            })
        
        return pd.DataFrame(results)
    
    def save_ensemble(self, filepath: str) -> None:
        """Save the entire ensemble to disk."""
        ensemble_data = {
            'momentum_bins': self.momentum_bins,
            'momentum_feature_idx': self.momentum_feature_idx,
            'xgb_params': self.xgb_params,
            'overlap_strategy': self.overlap_strategy,
            'min_samples_per_regime': self.min_samples_per_regime,
            'classes_': self.classes_,
            'trained_regimes': self.trained_regimes,
            'regime_models': {}
        }
        
        # Save each trained regime model
        for regime_range in self.trained_regimes:
            regime_model = self.regime_models[regime_range]
            ensemble_data['regime_models'][regime_range] = regime_model
        
        with open(filepath, 'wb') as f:
            pickle.dump(ensemble_data, f)
        
        self.logger.info(f"Ensemble saved to {filepath}")
    
    @classmethod
    def load_ensemble(cls, filepath: str) -> 'RegimeSpecificBDTEnsemble':
        """Load ensemble from disk."""
        with open(filepath, 'rb') as f:
            ensemble_data = pickle.load(f)
        
        # Recreate ensemble
        ensemble = cls(
            momentum_bins=ensemble_data['momentum_bins'],
            momentum_feature_idx=ensemble_data['momentum_feature_idx'],
            xgb_params=ensemble_data['xgb_params'],
            overlap_strategy=ensemble_data['overlap_strategy'],
            min_samples_per_regime=ensemble_data['min_samples_per_regime']
        )
        
        # Restore trained models
        ensemble.classes_ = ensemble_data['classes_']
        ensemble.trained_regimes = ensemble_data['trained_regimes']
        
        for regime_range, regime_model in ensemble_data['regime_models'].items():
            ensemble.regime_models[regime_range] = regime_model
        
        return ensemble
    
    def get_regime_summary(self) -> pd.DataFrame:
        """Get summary of all regime models."""
        summary_data = []
        
        for momentum_range in self.momentum_ranges:
            regime_model = self.regime_models[momentum_range]
            is_trained = momentum_range in self.trained_regimes
            
            summary_data.append({
                'momentum_range': f"{momentum_range[0]:.1f}-{momentum_range[1]:.1f}",
                'momentum_min': momentum_range[0],
                'momentum_max': momentum_range[1],
                'is_trained': is_trained,
                'n_samples_trained': regime_model.n_samples_trained if is_trained else 0,
                'best_iteration': (regime_model.model.best_iteration 
                                 if is_trained and regime_model.model else 0)
            })
        
        return pd.DataFrame(summary_data)