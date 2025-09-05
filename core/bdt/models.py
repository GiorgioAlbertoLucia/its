import numpy as np
import xgboost as xgb

from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import train_test_split

from typing import Dict, List, Tuple, Optional
import logging

class BaseBDTModel:
    """Base class for BDT models with common interface."""
    
    def __init__(self, momentum_feature_idx: int = 0):
        self.momentum_feature_idx = momentum_feature_idx
        self.model = None
        self.classes_ = None
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def fit(self, X: np.ndarray, y: np.ndarray, **kwargs) -> Dict:
        """Fit the model. Must be implemented by subclasses."""
        raise NotImplementedError
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        raise NotImplementedError
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return predicted class labels."""
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)

class MomentumAwareBDT(BaseBDTModel):
    """Single XGBoost model with momentum-enhanced features."""
    
    def __init__(self, momentum_feature_idx: int = 0, xgb_params: Optional[Dict] = None):
        super().__init__(momentum_feature_idx)
        self.xgb_params = xgb_params or self._default_params()
    
    def _default_params(self) -> Dict:
        return {
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
            'n_jobs': -1,
            'early_stopping_rounds': 20,
        }
    
    def _create_momentum_features(self, X: np.ndarray) -> np.ndarray:
        """Create momentum-enhanced features."""
        X_enhanced = X.copy()
        momentum = X[:, self.momentum_feature_idx]
        
        # Momentum-based features
        log_momentum = np.log(momentum + 1e-6)
        momentum_sq = momentum ** 2
        
        # Regime indicators (soft boundaries)
        low_p_indicator = np.exp(-((momentum - 0.5) ** 2) / (2 * 0.3 ** 2))
        med_p_indicator = np.exp(-((momentum - 2.0) ** 2) / (2 * 0.8 ** 2))
        high_p_indicator = 1.0 / (1.0 + np.exp(-2 * (momentum - 3.0)))
        
        # Add new features
        X_enhanced = np.column_stack([
            X_enhanced, log_momentum, momentum_sq,
            low_p_indicator, med_p_indicator, high_p_indicator
        ])
        
        # Interaction terms
        for i in range(X.shape[1]):
            if i != self.momentum_feature_idx:
                interaction = X[:, i] * momentum
                X_enhanced = np.column_stack([X_enhanced, interaction])
        
        return X_enhanced
    
    def fit(self, X: np.ndarray, y: np.ndarray, val_size: float = 0.2, 
            sample_weight: Optional[np.ndarray] = None) -> Dict:
        """Fit the momentum-aware BDT."""
        from sklearn.model_selection import train_test_split
        
        self.classes_ = np.unique(y)
        X_enhanced = self._create_momentum_features(X)
        
        X_train, X_val, y_train, y_val = train_test_split(
            X_enhanced, y, test_size=val_size, random_state=42, stratify=y
        )
        
        # Handle sample weights
        if sample_weight is None:
            class_weights = compute_class_weight('balanced', classes=self.classes_, y=y_train)
            sample_weight_train = np.array([class_weights[cls] for cls in y_train])
        else:
            sample_weight_train, _ = train_test_split(
                sample_weight, test_size=val_size, random_state=42, stratify=y
            )
        
        # Train model
        self.model = xgb.XGBClassifier(**self.xgb_params, num_class=len(self.classes_))
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight_train,
            eval_set=[(X_train, y_train), (X_val, y_val)],
            verbose=False
        )
        
        # Return training history
        evals_result = self.model.evals_result()
        return {
            'train_losses': evals_result['validation_0']['mlogloss'],
            'val_losses': evals_result['validation_1']['mlogloss'],
            'best_iteration': self.model.best_iteration
        }
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities."""
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        X_enhanced = self._create_momentum_features(X)
        return self.model.predict_proba(X_enhanced)

class MomentumRegionBDT:
    """Individual BDT for a specific momentum region."""
    
    def __init__(self, momentum_range: Tuple[float, float], n_classes: int, xgb_params: Optional[Dict] = None):
        self.momentum_range = momentum_range
        self.xgb_params = xgb_params or self._default_params()
        self.model = None
        self.classes_ = None
        self.n_original_classes = n_classes # Number of original classes (the number of classes in a specific region can be less)
        self.original_classes_ = None       # Store original class labels
        self.class_mapping_ = None          # Map original -> consecutive
        self.inverse_mapping_ = None        # Map consecutive -> original
        self.n_samples_trained = 0
        
    def _default_params(self) -> Dict:
        return {
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
            'n_jobs': -1,
            'early_stopping_rounds': 20,
        }
    
    def _create_region_features(self, X: np.ndarray, momentum_idx: int) -> np.ndarray:
        """Create features optimized for this momentum region."""
        X_enhanced = X.copy()
        momentum = X[:, momentum_idx]
        
        # Normalize momentum within the region
        p_min, p_max = self.momentum_range
        momentum_norm = (momentum - p_min) / (p_max - p_min)
        momentum_norm = np.clip(momentum_norm, 0, 1)
        
        # Region-specific transformations
        log_momentum_norm = np.log(momentum_norm + 1e-6)
        momentum_norm_sq = momentum_norm ** 2
        
        # Higher-order momentum features
        momentum_cube = momentum ** 3
        momentum_sqrt = np.sqrt(momentum)
        
        # Add enhanced features
        X_enhanced = np.column_stack([
            X_enhanced, momentum_norm, log_momentum_norm, momentum_norm_sq,
            momentum_cube, momentum_sqrt
        ])
        
        # Interaction terms with normalized momentum
        for i in range(X.shape[1]):
            if i != momentum_idx:
                interaction = X[:, i] * momentum_norm
                X_enhanced = np.column_stack([X_enhanced, interaction])
        
        return X_enhanced
    
    def _create_label_mapping(self, y: np.ndarray) -> np.ndarray:
        """Create mapping from original labels to consecutive integers starting from 0."""
        self.original_classes_ = np.unique(y)
        self.classes_ = np.arange(len(self.original_classes_))  # [0, 1, 2, ...]
        
        # Create mappings
        self.class_mapping_ = {orig: new for new, orig in enumerate(self.original_classes_)}
        self.inverse_mapping_ = {new: orig for new, orig in enumerate(self.original_classes_)}
        
        # Transform labels to consecutive integers
        y_mapped = np.array([self.class_mapping_[label] for label in y])
        return y_mapped
    
    def fits_region(self, momentum: np.ndarray) -> np.ndarray:
        """Check which samples belong to this momentum region."""
        p_min, p_max = self.momentum_range
        return (momentum >= p_min) & (momentum < p_max)
    
    def fit(self, X: np.ndarray, y: np.ndarray, momentum_idx: int,
            val_size: float = 0.2, sample_weight: Optional[np.ndarray] = None,
            min_samples: int = 50) -> Dict:
        """Fit the BDT for this momentum region."""
        from sklearn.model_selection import train_test_split
        
        momentum = X[:, momentum_idx]
        region_mask = self.fits_region(momentum)
        
        if np.sum(region_mask) < min_samples:
            self.logger.warning(f"Region {self.momentum_range} has only {np.sum(region_mask)} samples, skipping training")
            return {'trained': False, 'n_samples': np.sum(region_mask), 'reason': 'insufficient_samples'}
        
        # Filter data for this region
        X_region = X[region_mask]
        y_region = y[region_mask]
        self.classes_ = np.unique(y_region)
        self.n_samples_trained = len(X_region)
        
        if len(self.classes_) < 2:
            self.logger.warning(f"Region {self.momentum_range} has only one class, skipping training")
            return {'trained': False, 'n_samples': self.n_samples_trained, 'n_classes': len(self.classes_), 'reason': 'single_class'}
        
        # Check if we can do stratified splitting
        class_counts = np.bincount(y_region)
        min_class_count = np.min(class_counts[class_counts > 0])
        
        can_stratify = min_class_count >= 2 and len(X_region) >= 4
        
        actual_val_size = val_size
        if len(X_region) < 10:
            actual_val_size = 0.0  # No validation split for very small datasets
        elif not can_stratify and len(X_region) < 20:
            actual_val_size = min(0.1, val_size)  # Smaller validation set
        
        #X_region = self._create_region_features(X_region, momentum_idx) # enhanced features
        y_region_mapped = self._create_label_mapping(y_region)
        
        # Split data with appropriate strategy
        if actual_val_size == 0.0:
            # No validation split
            X_train, X_val = X_region, X_region[:0]  # Empty validation set
            y_train, y_val = y_region_mapped, y_region_mapped[:0]
        else:
            try:
                X_train, X_val, y_train, y_val = train_test_split(
                    X_region, y_region_mapped, test_size=actual_val_size, random_state=42, 
                    stratify=y_region_mapped if can_stratify else None
                )
            except ValueError as e:
                self.logger.warning(f"Stratified split failed for region {self.momentum_range}, using random split: {e}")
                X_train, X_val, y_train, y_val = train_test_split(
                    X_region, y_region_mapped, test_size=actual_val_size, random_state=42
                )
        
        # Handle sample weights
        if sample_weight is not None:
            sample_weight_region = sample_weight[region_mask]
            if actual_val_size > 0.0:
                try:
                    sample_weight_train, _ = train_test_split(
                        sample_weight_region, test_size=actual_val_size, random_state=42,
                        stratify=y_region_mapped if can_stratify else None
                    )
                except ValueError:
                    sample_weight_train, _ = train_test_split(
                        sample_weight_region, test_size=actual_val_size, random_state=42
                    )
            else:
                sample_weight_train = sample_weight_region
        else:
            # Use mapped classes for weight computation - y_train is already mapped to [0,1,2,...]
            class_weights = compute_class_weight('balanced', classes=self.classes_, y=y_train)
            class_weight_dict = dict(zip(self.classes_, class_weights))
            sample_weight_train = np.array([class_weight_dict[cls] for cls in y_train])
        
        # Train model
        self.model = xgb.XGBClassifier(**self.xgb_params, num_class=len(self.classes_))
        
        # Prepare eval_set
        eval_set = [(X_train, y_train)]
        if len(X_val) > 0:
            eval_set.append((X_val, y_val))
        
        self.model.fit(
            X_train, y_train,
            sample_weight=sample_weight_train,
            eval_set=eval_set,
            verbose=False
        )
        
        # Return training history
        evals_result = self.model.evals_result()
        result = {
            'trained': True,
            'n_samples': self.n_samples_trained,
            'n_classes': len(self.classes_),
            'val_size_used': actual_val_size,
            'stratified': can_stratify,
            'train_losses': evals_result['validation_0']['mlogloss'],
            'best_iteration': self.model.best_iteration
        }
        
        if len(X_val) > 0:
            result['val_losses'] = evals_result['validation_1']['mlogloss']
        
        return result
    
    def predict_proba(self, X: np.ndarray, momentum_idx: int) -> np.ndarray:
        """Return class probabilities for samples in this region."""
        if self.model is None:
            raise ValueError(f"Model for region {self.momentum_range} not trained yet.")
        
        #X_enhanced = self._create_region_features(X, momentum_idx)
        proba_mapped = self.model.predict_proba(X)
        
        # Map back to original class labels
        proba_original = np.zeros((X.shape[0], self.n_original_classes))
        for mapped_idx, orig_class in self.inverse_mapping_.items():
            proba_original[:, orig_class] = proba_mapped[:, mapped_idx]
        
        return proba_original


class MomentumEnsembleBDT(BaseBDTModel):
    """Ensemble of BDTs trained on different momentum regions."""
    
    def __init__(self, momentum_feature_idx: int = 0, 
                 momentum_bins: Optional[List[float]] = None,
                 xgb_params: Optional[Dict] = None,
                 min_samples_per_region: int = 50):
        super().__init__(momentum_feature_idx)
        
        if momentum_bins is None:
            momentum_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0]
        
        self.momentum_bins = momentum_bins
        self.xgb_params = xgb_params
        self.min_samples_per_region = min_samples_per_region
        self.regional_models = {}
        self.momentum_ranges = self._create_momentum_ranges()
        self.fallback_model = None  # For regions with insufficient data
        self.original_classes_ = None  # Store original class labels
        self.class_mapping_ = None     # Map original -> consecutive  
        self.inverse_mapping_ = None   # Map consecutive -> original
        
    def _create_global_label_mapping(self, y: np.ndarray) -> np.ndarray:
        """Create global mapping from original labels to consecutive integers."""
        self.original_classes_ = np.unique(y)
        self.classes_ = np.arange(len(self.original_classes_))  # [0, 1, 2, ...]
        
        # Create mappings
        self.class_mapping_ = {orig: new for new, orig in enumerate(self.original_classes_)}
        self.inverse_mapping_ = {new: orig for new, orig in enumerate(self.original_classes_)}
        
        # Transform labels to consecutive integers
        y_mapped = np.array([self.class_mapping_[label] for label in y])
        return y_mapped
    
    def _create_momentum_ranges(self) -> List[Tuple[float, float]]:
        """Create momentum ranges from bins."""
        ranges = []
        for i in range(len(self.momentum_bins) - 1):
            ranges.append((self.momentum_bins[i], self.momentum_bins[i + 1]))
        return ranges
    
    def _train_fallback_model(self, X: np.ndarray, y_mapped: np.ndarray, 
                             sample_weight: Optional[np.ndarray] = None) -> None:
        """Train a fallback model using all data for regions with insufficient samples."""
        self.logger.info("Training fallback model for regions with insufficient data")
        
        fallback_params = self.xgb_params.copy() if self.xgb_params else {}
        fallback_params.update({
            'objective': 'multi:softprob',
            'eval_metric': 'mlogloss',
            'max_depth': 4,  # Simpler model for fallback
            'learning_rate': 0.3,
            'n_estimators': 100,
            'random_state': 42,
            'n_jobs': -1
        })
        
        if sample_weight is None:
            class_weights = compute_class_weight('balanced', classes=self.classes_, y=y_mapped)
            class_weight_dict = dict(zip(self.classes_, class_weights))
            sample_weight = np.array([class_weight_dict[cls] for cls in y_mapped])
        
        self.fallback_model = xgb.XGBClassifier(**fallback_params, num_class=len(self.classes_))
        self.fallback_model.fit(X, y_mapped, sample_weight=sample_weight)
    
    def fit(self, X: np.ndarray, y: np.ndarray, val_size: float = 0.2,
            sample_weight: Optional[np.ndarray] = None) -> Dict:
        """Fit ensemble of momentum-region BDTs."""
        
        self.classes_ = np.unique(y)
        momentum = X[:, self.momentum_feature_idx]
        
        training_results = {}
        trained_regions = []
        failed_regions = []
        
        self.logger.info(f"Training ensemble with {len(self.momentum_ranges)} momentum regions")
        
        # Train models for each momentum region
        for i, momentum_range in enumerate(self.momentum_ranges):
            self.logger.info(f"Training region {i+1}/{len(self.momentum_ranges)}: {momentum_range}")
            
            regional_model = MomentumRegionBDT(momentum_range, len(self.classes_), self.xgb_params)
            
            result = regional_model.fit(
                X, y, self.momentum_feature_idx, val_size, 
                sample_weight, self.min_samples_per_region
            )
            
            training_results[f'region_{i}_{momentum_range}'] = result
            
            if result.get('trained', False):
                self.regional_models[momentum_range] = regional_model
                trained_regions.append(momentum_range)
                self.logger.info(f"Successfully trained region {momentum_range} with {result['n_samples']} samples")
            else:
                failed_regions.append(momentum_range)
                self.logger.warning(f"Failed to train region {momentum_range}: {result}")
        
        # Train fallback model if needed
        if failed_regions:
            self._train_fallback_model(X, y, sample_weight)
        
        # Summary statistics
        total_samples = len(X)
        samples_in_trained_regions = sum(
            np.sum((momentum >= r[0]) & (momentum < r[1])) 
            for r in trained_regions
        )
        
        training_results['summary'] = {
            'total_regions': len(self.momentum_ranges),
            'trained_regions': len(trained_regions),
            'failed_regions': len(failed_regions),
            'total_samples': total_samples,
            'samples_in_trained_regions': samples_in_trained_regions,
            'coverage': samples_in_trained_regions / total_samples,
            'has_fallback': self.fallback_model is not None
        }
        
        self.logger.info(f"Ensemble training complete: {len(trained_regions)}/{len(self.momentum_ranges)} regions trained")
        self.logger.info(f"Coverage: {training_results['summary']['coverage']:.3f}")
        
        return training_results
    
    def _get_model_for_momentum(self, momentum_val: float) -> Tuple[Optional[MomentumRegionBDT], bool]:
        """Get the appropriate model for a given momentum value."""
        for momentum_range, model in self.regional_models.items():
            if momentum_range[0] <= momentum_val < momentum_range[1]:
                return model, False
        
        # Handle edge case for maximum momentum
        if momentum_val >= self.momentum_bins[-1]:
            last_range = (self.momentum_bins[-2], self.momentum_bins[-1])
            if last_range in self.regional_models:
                return self.regional_models[last_range], False
        
        # Use fallback model
        return None, True
    
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Return class probabilities using ensemble of regional models."""
        if not self.regional_models and self.fallback_model is None:
            raise ValueError("No models trained yet. Call fit() first.")
        
        momentum = X[:, self.momentum_feature_idx]
        probabilities = np.zeros((len(X), len(self.classes_)))
        
        # Track which samples use fallback
        fallback_mask = np.zeros(len(X), dtype=bool)
        
        # Get predictions for each sample
        for i, momentum_val in enumerate(momentum):
            model, use_fallback = self._get_model_for_momentum(momentum_val)
            
            if use_fallback or model is None:
                if self.fallback_model is not None:
                    probabilities[i] = self.fallback_model.predict_proba(X[i:i+1])[0]
                    fallback_mask[i] = True
                else:
                    # If no fallback, use uniform distribution
                    probabilities[i] = np.ones(len(self.classes_)) / len(self.classes_)
                    fallback_mask[i] = True
            else:
                probabilities[i] = model.predict_proba(X[i:i+1], self.momentum_feature_idx)[0]
        
        if np.any(fallback_mask):
            self.logger.debug(f"Used fallback model for {np.sum(fallback_mask)}/{len(X)} samples")
        
        return probabilities
    
    def get_model_info(self) -> Dict:
        """Get information about the trained models."""
        info = {
            'momentum_bins': self.momentum_bins,
            'momentum_ranges': self.momentum_ranges,
            'trained_regions': list(self.regional_models.keys()),
            'has_fallback': self.fallback_model is not None,
            'regional_sample_counts': {}
        }
        
        for momentum_range, model in self.regional_models.items():
            info['regional_sample_counts'][str(momentum_range)] = model.n_samples_trained
        
        return info
    