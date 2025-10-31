"""Optuna hyperparameter optimization for BDT models."""

import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import numpy as np
import logging
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import pickle
import json
from datetime import datetime

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))

from core.bdt.models import BaseBDTModel, MomentumAwareBDT, MomentumEnsembleBDT
from core.config import ExperimentConfig
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, log_loss

class BDTOptunaOptimizer:
    """Hyperparameter optimization for BDT models using Optuna."""
    
    def __init__(self, 
                 config: ExperimentConfig):
        """
        Initialize the optimizer.
        
        Args:
            model_type: Type of model ('momentum_aware' or 'momentum_ensemble')
            config: Experiment configuration
            n_trials: Number of optimization trials
            cv_folds: Number of cross-validation folds
            timeout: Maximum optimization time in seconds
            study_name: Name for the Optuna study
        """
        self.config = config
        self.model_type = self.config.model_type
        self.n_trials = self.config.optuna_trials
        self.cv_folds = self.config.optuna_cv_folds

        self.timeout = self.config.optuna_timeout
        self.study_name = f"bdt_{self.model_type}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.logger = logging.getLogger(__name__)
        self.study = None
        
        # Data placeholders - will be set during optimization
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        
    def _suggest_xgb_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest XGBoost hyperparameters."""
        return {
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000, step=50),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 1.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 1.0, log=True),
            'gamma': trial.suggest_float('gamma', 1e-8, 1.0, log=True),
            
            # Fixed parameters
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'random_state': 42,
            'n_jobs': -1,
            'early_stopping_rounds': 20,
        }
    
    def _suggest_ensemble_params(self, trial: optuna.Trial) -> Dict[str, Any]:
        """Suggest ensemble-specific parameters."""
        # Number of momentum bins
        n_bins = trial.suggest_int('n_momentum_bins', 5, 15)
        
        # Strategy for bin edges
        bin_strategy = trial.suggest_categorical('bin_strategy', ['uniform', 'quantile', 'custom'])
        
        if bin_strategy == 'uniform':
            # Uniform spacing in momentum
            momentum_bins = np.linspace(0.1, 5.0, n_bins).tolist()
        elif bin_strategy == 'quantile':
            # Quantile-based bins (will be computed from data)
            momentum_bins = None  # Will be set later
        else:  # custom
            # Custom bins with more density in low momentum region
            momentum_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0][:n_bins]
        
        min_samples_per_region = trial.suggest_int('min_samples_per_region', 30, 200)
        
        return {
            'momentum_bins': momentum_bins,
            'min_samples_per_region': min_samples_per_region,
            'bin_strategy': bin_strategy,
            'n_momentum_bins': n_bins
        }
    
    def _compute_quantile_bins(self, momentum: np.ndarray, n_bins: int) -> List[float]:
        """Compute quantile-based momentum bins."""
        quantiles = np.linspace(0, 1, n_bins)
        bins = np.quantile(momentum, quantiles)
        
        # Ensure minimum and maximum values
        bins[0] = max(bins[0], 0.1)
        bins[-1] = min(bins[-1], 5.0)
        
        # Remove duplicates and ensure monotonic
        bins = np.unique(bins)
        if len(bins) < n_bins:
            # Fill missing bins with linear interpolation
            bins = np.linspace(bins[0], bins[-1], n_bins)
        
        return bins.tolist()
    
    def _create_model_with_params(self, trial: optuna.Trial) -> BaseBDTModel:
        """Create a model with suggested hyperparameters."""
        xgb_params = self._suggest_xgb_params(trial)
        
        if self.model_type == 'MomentumAwareBDT':
            return MomentumAwareBDT(
                momentum_feature_idx=self.config.momentum_feature_idx,
                xgb_params=xgb_params
            )
        
        elif self.model_type == 'MomentumEnsembledBDT':
            ensemble_params = self._suggest_ensemble_params(trial)
            
            # Handle quantile bins
            if ensemble_params['bin_strategy'] == 'quantile':
                momentum = self.X_train[:, self.config.momentum_feature_idx]
                momentum_bins = self._compute_quantile_bins(momentum, ensemble_params['n_momentum_bins'])
                ensemble_params['momentum_bins'] = momentum_bins
            
            return MomentumEnsembleBDT(
                momentum_feature_idx=self.config.momentum_feature_idx,
                momentum_bins=ensemble_params['momentum_bins'],
                xgb_params=xgb_params,
                min_samples_per_region=ensemble_params['min_samples_per_region']
            )
        
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _objective(self, trial: optuna.Trial) -> float:
        """Optuna objective function."""
        try:
            model = self._create_model_with_params(trial)
            
            # Use cross-validation for robust evaluation
            cv = StratifiedKFold(n_splits=self.cv_folds, shuffle=True, random_state=42)
            
            # Custom cross-validation since our models need special handling
            scores = []
            for fold_idx, (train_idx, val_idx) in enumerate(cv.split(self.X_train, self.y_train)):
                X_fold_train, X_fold_val = self.X_train[train_idx], self.X_train[val_idx]
                y_fold_train, y_fold_val = self.y_train[train_idx], self.y_train[val_idx]
                
                # Train model on fold
                model_fold = self._create_model_with_params(trial)  # Fresh model for each fold
                model_fold.fit(X_fold_train, y_fold_train, val_size=0.0)  # No internal validation
                
                # Predict on validation fold
                y_pred_proba = model_fold.predict_proba(X_fold_val)
                
                # Calculate log loss (lower is better)
                fold_score = log_loss(y_fold_val, y_pred_proba)
                scores.append(fold_score)
                
                # Report intermediate value for pruning
                trial.report(fold_score, fold_idx)
                
                # Handle pruning
                if trial.should_prune():
                    raise optuna.TrialPruned()
            
            # Return mean CV score
            mean_score = np.mean(scores)
            return mean_score
            
        except Exception as e:
            self.logger.error(f"Trial failed with error: {e}")
            # Return a high loss value for failed trials
            return float('inf')
    
    def optimize(self, 
                 X_train: np.ndarray, 
                 y_train: np.ndarray, 
                 X_val: np.ndarray, 
                 y_val: np.ndarray) -> optuna.Study:
        """
        Run hyperparameter optimization.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features  
            y_val: Validation labels
            
        Returns:
            Completed Optuna study
        """
        self.X_train = X_train
        self.y_train = y_train
        self.X_val = X_val
        self.y_val = y_val
        
        # Create study
        sampler = TPESampler(seed=42)
        pruner = MedianPruner(n_startup_trials=5, n_warmup_steps=10, interval_steps=1)
        
        self.study = optuna.create_study(
            direction='minimize',  # Minimize log loss
            sampler=sampler,
            pruner=pruner,
            study_name=self.study_name
        )
        
        self.logger.info(f"Starting hyperparameter optimization with {self.n_trials} trials")
        self.logger.info(f"Model type: {self.model_type}")
        self.logger.info(f"Cross-validation folds: {self.cv_folds}")
        
        # Optimize
        self.study.optimize(
            self._objective, 
            n_trials=self.n_trials, 
            timeout=self.timeout,
            show_progress_bar=True
        )
        
        # Log results
        self.logger.info("Optimization completed!")
        self.logger.info(f"Best trial: {self.study.best_trial.number}")
        self.logger.info(f"Best value (log loss): {self.study.best_value:.6f}")
        self.logger.info("Best parameters:")
        for key, value in self.study.best_params.items():
            self.logger.info(f"  {key}: {value}")
        
        return self.study
    
    def get_best_model(self) -> BaseBDTModel:
        """Get a model with the best hyperparameters."""
        if self.study is None:
            raise ValueError("No optimization has been run yet. Call optimize() first.")
        
        # Create a dummy trial with best parameters
        class DummyTrial:
            def __init__(self, params):
                self.params = params
            
            def suggest_int(self, name, *args, **kwargs):
                return self.params[name]
            
            def suggest_float(self, name, *args, **kwargs):
                return self.params[name]
            
            def suggest_categorical(self, name, *args, **kwargs):
                return self.params[name]
        
        dummy_trial = DummyTrial(self.study.best_params)
        return self._create_model_with_params(dummy_trial)
    
    def save_study(self, filepath: Path) -> None:
        """Save the study to disk."""
        if self.study is None:
            raise ValueError("No study to save. Run optimization first.")
        
        study_data = {
            'study_name': self.study_name,
            'model_type': self.model_type,
            'best_params': self.study.best_params,
            'best_value': self.study.best_value,
            'n_trials': len(self.study.trials),
            'trials': []
        }
        
        # Save trial information
        for trial in self.study.trials:
            trial_data = {
                'number': trial.number,
                'value': trial.value,
                'params': trial.params,
                'state': trial.state.name,
                'duration': trial.duration.total_seconds() if trial.duration else None
            }
            study_data['trials'].append(trial_data)
        
        # Save as JSON
        with open(filepath.with_suffix('.json'), 'w') as f:
            json.dump(study_data, f, indent=2)
        
        # Save study object as pickle
        with open(filepath.with_suffix('.pkl'), 'wb') as f:
            pickle.dump(self.study, f)
        
        self.logger.info(f"Study saved to {filepath}")
    
    def load_study(self, filepath: Path) -> optuna.Study:
        """Load a study from disk."""
        with open(filepath.with_suffix('.pkl'), 'rb') as f:
            self.study = pickle.load(f)
        
        self.logger.info(f"Study loaded from {filepath}")
        return self.study
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get a summary of the optimization results."""
        if self.study is None:
            raise ValueError("No study available. Run optimization first.")
        
        # Calculate statistics
        all_values = [trial.value for trial in self.study.trials if trial.value is not None]
        completed_trials = [trial for trial in self.study.trials if trial.state.name == 'COMPLETE']
        pruned_trials = [trial for trial in self.study.trials if trial.state.name == 'PRUNED']
        failed_trials = [trial for trial in self.study.trials if trial.state.name == 'FAIL']
        
        return {
            'study_name': self.study_name,
            'model_type': self.model_type,
            'total_trials': len(self.study.trials),
            'completed_trials': len(completed_trials),
            'pruned_trials': len(pruned_trials),
            'failed_trials': len(failed_trials),
            'best_trial_number': self.study.best_trial.number,
            'best_value': self.study.best_value,
            'best_params': self.study.best_params,
            'value_statistics': {
                'mean': np.mean(all_values) if all_values else None,
                'std': np.std(all_values) if all_values else None,
                'min': np.min(all_values) if all_values else None,
                'max': np.max(all_values) if all_values else None
            }
        }
    
    def plot_optimization_history(self, save_path: Optional[Path] = None):
        """Plot optimization history."""
        if self.study is None:
            raise ValueError("No study available. Run optimization first.")
        
        try:
            import matplotlib.pyplot as plt
            from optuna.visualization import plot_optimization_history, plot_param_importances
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
            
            # Optimization history
            plot_optimization_history(self.study).show()
            
            # Parameter importances
            plot_param_importances(self.study).show()
            
            if save_path:
                plt.tight_layout()
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
                plt.close()
                self.logger.info(f"Optimization plots saved to {save_path}")
            else:
                plt.show()
                
        except ImportError:
            self.logger.warning("Matplotlib not available. Skipping plots.")
        except Exception as e:
            self.logger.error(f"Error creating plots: {e}")
            