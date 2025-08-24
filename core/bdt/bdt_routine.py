"""
Simple Momentum-Aware BDT implementation following the neural network patterns.
"""

from dataclasses import dataclass
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple, Optional
from pathlib import Path

import shap
import os
from ROOT import TFile, TH1F, kBlue, kRed, kGreen, kMagenta
from torchic.core.histogram import AxisSpec, build_TH2

from sklearn.metrics import accuracy_score, roc_curve, auc

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
    n_jobs: int = 30

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
                          X: np.ndarray, y: np.ndarray,
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
        
        all_scores, __, all_features = self.get_model_scores(X)
        momentum = all_features[:, momentum_idx]
        print(f'{all_scores.shape=}')
        
        output_file.cd()
        
        axis_spec_p = AxisSpec(50, 0, 5, 'p', '#it{p} (GeV/#it{c})')
        
        for species_idx, species_name in enumerate(class_names):
            species_mask = (y == species_idx)
            n_particles = np.sum(species_mask)
            
            print(f'Processing {species_name}: {n_particles} particles')
            if n_particles == 0:
                self.logger.warning(f"No particles found for {species_name}, skipping histogram")
                continue

            momentum_species = momentum[species_mask]
            all_scores_species = all_scores[species_mask]
            dir_species = output_file.mkdir(species_name)
            dir_species.cd()
            
            for hypothesis_idx, hypothesis_name in enumerate(class_names):

                axis_spec_score = AxisSpec(
                    100, 0, 1, 'Score', 
                    f'{hypothesis_name} hypothesis;#it{{p}} (GeV/#it{{c}});Probability;'
                )
                hist = build_TH2(momentum_species, all_scores_species[:, hypothesis_idx], axis_spec_p, axis_spec_score)
                hist.Write(f'bdt_{hypothesis_name}')
                del hist
            
        self.logger.info(f"Created BDT score histograms for {len(class_names)} classes")

    def get_feature_importance(self, feature_names: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Get feature importance analysis.
        """
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        importance = self.model.feature_importances_
        
        if feature_names is None:
            # Generate names for enhanced features
            n_original = len(importance) - 3 - len(importance)//2  # Approximate original count
            feature_names = []
            for i in range(n_original):
                feature_names.append(f'original_feature_{i}')
            
            # Add names for engineered features
            feature_names.extend([
                'log_momentum', 'momentum_squared', 
                'low_p_indicator', 'med_p_indicator', 'high_p_indicator'
            ])
            
            # Add interaction term names
            for i in range(n_original):
                if i != self.momentum_feature_idx:
                    feature_names.append(f'momentum_x_feature_{i}')
        
        importance_df = pd.DataFrame({
            'feature': feature_names[:len(importance)],
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def analyze_feature_importance_shap(self,
                                       X: np.ndarray, 
                                       y: np.ndarray,
                                       feature_names: List[str],
                                       class_names: List[str],
                                       output_file: TFile,
                                       max_samples: int = 1000,
                                       background_samples: int = 100) -> Dict:
        """
        Analyze feature importance using SHAP for BDT.
        Mirrors the neural network SHAP implementation.
        
        Args:
            X: Input features
            y: True labels
            feature_names: List of feature names
            class_names: List of class names
            output_file: ROOT TFile object for saving plots
            max_samples: Maximum number of samples to analyze
            background_samples: Number of background samples for explainer
            
        Returns:
            Dictionary containing SHAP values and analysis results
        """
        
        if self.model is None:
            raise ValueError("Model not trained yet. Call fit() first.")
        
        print(f"SHAP analysis on {X.shape[0]} samples with {X.shape[1]} features")
        
        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
            y_sample = y[indices]
        else:
            X_sample = X
            y_sample = y
        
        if len(X) > background_samples:
            bg_indices = np.random.choice(len(X), background_samples, replace=False)
            X_background = X[bg_indices]
        else:
            X_background = X
        
        print(f"Using {len(X_sample)} samples for SHAP analysis")
        print(f"Using {len(X_background)} background samples")
        
        print("Creating SHAP explainer...")
        explainer = shap.TreeExplainer(self.model.model, X_background)
        
        print("Calculating SHAP values...")
        shap_values = explainer.shap_values(X_sample, check_additivity=False)
        
        if isinstance(shap_values, list): # Multi-class case
            shap_values_array = np.stack(shap_values, axis=-1)
        else: # Binary case
            shap_values_array = shap_values[..., np.newaxis]
        
        df = pd.DataFrame(X_sample, columns=feature_names)
        df['true_class'] = y_sample
        df['true_class_name'] = [class_names[int(i)] for i in y_sample]
        predictions = self.model.predict_proba(X_sample)
        predicted_classes = np.argmax(predictions, axis=1)
        df['predicted_class'] = predicted_classes
        df['predicted_class_name'] = [class_names[int(i)] for i in predicted_classes]
        
        plot_dir = "../output/bdt/shap_plots"
        
        results = {
            'shap_values': shap_values_array,
            'feature_names': feature_names,
            'original_feature_names': feature_names,
            'class_names': class_names,
            'sample_data': df,
            'predictions': predictions,
            'plots': {}
        }
        
        # Create summary bar plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            # Multi-class case
            shap.summary_plot(shap_values, X_sample, 
                             feature_names=feature_names,
                             class_names=class_names, 
                             plot_type="bar", show=False)
        else:
            # Binary case
            shap.summary_plot(shap_values, X_sample,
                             feature_names=feature_names,
                             plot_type="bar", show=False)
        
        plt.tight_layout()
        bar_plot_path = os.path.join(plot_dir, "bdt_shap_bar.pdf")
        plt.savefig(bar_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        results['plots']['bar'] = bar_plot_path
        
        # Create detailed summary plot
        plt.figure(figsize=(12, 8))
        if isinstance(shap_values, list):
            # For multi-class, show first class or average
            shap.summary_plot(shap_values[0], X_sample,
                             feature_names=feature_names, show=False)
        else:
            shap.summary_plot(shap_values, X_sample,
                             feature_names=feature_names, show=False)
        
        plt.tight_layout()
        summary_plot_path = os.path.join(plot_dir, "bdt_shap_summary.pdf")
        plt.savefig(summary_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        results['plots']['summary'] = summary_plot_path
        
        output_file.cd()
        colors = [kBlue, kRed, kGreen, kMagenta]
        
        for class_idx, class_name in enumerate(class_names):
            if isinstance(shap_values, list):
                class_shap_values = shap_values[class_idx]
            else:
                class_shap_values = shap_values
            
            mean_abs_shap = np.mean(np.abs(class_shap_values), axis=0)
            
            hist_name = f'h_bdt_shap_importance_{class_name}'
            hist_title = f'BDT Feature Importance ({class_name});Feature;Mean |SHAP Value|'
            hist = TH1F(hist_name, hist_title, len(feature_names), 
                       0, len(feature_names))
            
            for feat_idx, (feat_name, importance) in enumerate(zip(feature_names, mean_abs_shap)):
                hist.SetBinContent(feat_idx + 1, importance)
                hist.GetXaxis().SetBinLabel(feat_idx + 1, feat_name)
            
            color = colors[class_idx % len(colors)]
            hist.SetLineColor(color)
            hist.SetLineWidth(2)
            hist.Write()
            
            results[f'root_hist_{class_name}'] = hist
        
        print(f"\nBDT SHAP analysis complete!")
        print(f"Plots saved in: {plot_dir}/")
        print("ROOT histograms saved in output file")
        
        return results
    
    def plot_roc_curves_momentum_bins(self,
                                     X: np.ndarray,
                                     y: np.ndarray,
                                     class_names: List[str],
                                     output_file: TFile,
                                     momentum_bins: Optional[List[float]] = None,
                                     plot_dir: str = "../output/bdt/roc_plots") -> Dict:
        """
        Create ROC curves for each class in different momentum bins.
        
        Args:
            X: Input features
            y: True labels  
            class_names: List of class names
            output_file: ROOT TFile for saving histograms
            momentum_bins: List of momentum bin edges
            plot_dir: Directory for saving plots
            
        Returns:
            Dictionary with ROC analysis results
        """
        
        if momentum_bins is None:
            momentum_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0]  # Default bins
        
        Path(plot_dir).mkdir(parents=True, exist_ok=True)
        
        momentum = X[:, self.model.momentum_feature_idx]
        probabilities = self.model.predict_proba(X)
        
        results = {
            'momentum_bins': momentum_bins,
            'class_names': class_names,
            'bin_results': {},
            'plots': {}
        }
        
        for bin_idx in range(len(momentum_bins) - 1):
            bin_min = momentum_bins[bin_idx]
            bin_max = momentum_bins[bin_idx + 1]
            bin_name = f"{bin_min:.1f}-{bin_max:.1f}GeV"
            
            bin_mask = (momentum >= bin_min) & (momentum < bin_max)
            
            if np.sum(bin_mask) < 50:  # Skip bins with insufficient statistics
                print(f"Skipping momentum bin {bin_name} - insufficient statistics ({np.sum(bin_mask)} samples)")
                continue
            
            y_bin = y[bin_mask]
            prob_bin = probabilities[bin_mask]
            
            print(f"\nProcessing momentum bin {bin_name}: {np.sum(bin_mask)} samples")
            
            bin_results = {
                'bin_range': (bin_min, bin_max),
                'n_samples': np.sum(bin_mask),
                'class_rocs': {}
            }
            
            plt.figure(figsize=(10, 8))
            colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
            
            for class_idx, class_name in enumerate(class_names):
                y_true_binary = (y_bin == class_idx).astype(int)
                y_score = prob_bin[:, class_idx]
                
                if np.sum(y_true_binary) == 0:
                    print(f"  Skipping {class_name} - no positive samples in this bin")
                    continue
                
                fpr, tpr, thresholds = roc_curve(y_true_binary, y_score)
                roc_auc = auc(fpr, tpr)
                
                color = colors[class_idx % len(colors)]
                plt.plot(fpr, tpr, color=color, lw=2,
                        label=f'{class_name} (AUC = {roc_auc:.3f})')
                
                bin_results['class_rocs'][class_name] = {
                    'fpr': fpr,
                    'tpr': tpr,
                    'thresholds': thresholds,
                    'auc': roc_auc,
                    'n_positive': np.sum(y_true_binary)
                }
            
            plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Classifier')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title(f'BDT ROC Curves - Momentum {bin_name}')
            plt.legend(loc="lower right")
            plt.grid(True, alpha=0.3)
            
            plot_path = os.path.join(plot_dir, f"roc_curves_{bin_name.replace('-', '_')}.pdf")
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            results['bin_results'][bin_name] = bin_results
            results['plots'][bin_name] = plot_path
            
            print(f"  ROC plot saved: {plot_path}")
        
        self._plot_auc_vs_momentum(results, plot_dir)
        self._save_roc_histograms(results, output_file)
        
        print(f"\nROC analysis complete! Plots saved in: {plot_dir}/")
        
        return results
    
    def _plot_auc_vs_momentum(self, roc_results: Dict, plot_dir: str):
        """Plot AUC vs momentum for each class."""
        
        plt.figure(figsize=(10, 6))
        colors = ['blue', 'red', 'green', 'orange', 'purple', 'brown']
        
        class_names = roc_results['class_names']
        momentum_bins = roc_results['momentum_bins']
        
        for class_idx, class_name in enumerate(class_names):
            bin_centers = []
            aucs = []
            
            for bin_name, bin_result in roc_results['bin_results'].items():
                if class_name in bin_result['class_rocs']:
                    bin_min, bin_max = bin_result['bin_range']
                    bin_center = (bin_min + bin_max) / 2
                    auc_value = bin_result['class_rocs'][class_name]['auc']
                    
                    bin_centers.append(bin_center)
                    aucs.append(auc_value)
            
            if bin_centers:  # Only plot if we have data
                color = colors[class_idx % len(colors)]
                plt.plot(bin_centers, aucs, 'o-', color=color, 
                        label=f'{class_name}', linewidth=2, markersize=6)
        
        plt.axvline(x=1.0, color='gray', linestyle='--', alpha=0.7, label='Regime boundaries')
        plt.axvline(x=3.0, color='gray', linestyle='--', alpha=0.7)
        
        plt.xlabel('Momentum (GeV/c)')
        plt.ylabel('AUC (Area Under ROC Curve)')
        plt.title('BDT Classification Performance vs Momentum')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.ylim([0.5, 1.0])
        
        plot_path = os.path.join(plot_dir, "auc_vs_momentum.pdf")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"AUC summary plot saved: {plot_path}")
    
    def _save_roc_histograms(self, roc_results: Dict, output_file: TFile):
        """Save ROC analysis results as ROOT histograms."""
        
        output_file.cd()
        
        roc_dir = output_file.mkdir("bdt_roc_analysis")
        roc_dir.cd()
        
        class_names = roc_results['class_names']
        colors = [kBlue, kRed, kGreen, kMagenta]
        
        momentum_centers = []
        bin_names = list(roc_results['bin_results'].keys())
        
        for bin_name in bin_names:
            bin_result = roc_results['bin_results'][bin_name]
            bin_min, bin_max = bin_result['bin_range']
            momentum_centers.append((bin_min + bin_max) / 2)
        
        if momentum_centers:  # Only create if we have data
            # Create histogram for AUC vs momentum for each class
            for class_idx, class_name in enumerate(class_names):
                hist_name = f'h_auc_vs_momentum_{class_name}'
                hist_title = f'BDT AUC vs Momentum ({class_name});Momentum (GeV/c);AUC'
                hist = TH1F(hist_name, hist_title, len(momentum_centers), 
                           min(momentum_centers) - 0.1, max(momentum_centers) + 0.1)
                
                bin_idx = 1
                for bin_name in bin_names:
                    bin_result = roc_results['bin_results'][bin_name]
                    if class_name in bin_result['class_rocs']:
                        auc_value = bin_result['class_rocs'][class_name]['auc']
                        hist.SetBinContent(bin_idx, auc_value)
                    bin_idx += 1
                
                color = colors[class_idx % len(colors)]
                hist.SetLineColor(color)
                hist.SetMarkerColor(color)
                hist.SetLineWidth(2)
                hist.Write()
    
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

