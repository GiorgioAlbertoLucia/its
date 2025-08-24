"""Visualization utilities."""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict

import torch
import shap

from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

from torchic.core.histogram import AxisSpec, build_TH2

from ROOT import TGraph, TH1F, kBlue

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.nn.evaluator import EvaluationResult

class Plotter:
    """Handles all plotting operations."""
    
    def __init__(self, output_dir: Path):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_training_history(self, train_losses: List[float], 
                            val_losses: List[float],
                            train_accs: List[float], 
                            val_accs: List[float]):
        """Plot training history."""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(1, len(train_losses) + 1)
        
        ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, train_accs, 'b-', label='Training Accuracy')
        ax2.plot(epochs, val_accs, 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(self.output_dir / 'training_history.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_confusion_matrix(self, labels: np.ndarray, 
                            predictions: np.ndarray, class_names: List[str]):
        """Plot confusion matrix."""
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))

        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(self.output_dir / 'confusion_matrix.pdf', 
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def plot_model_scores(self,
                          test_results: EvaluationResult,
                          class_names: List[str],
                          output_file,
                          momentum_idx: int) -> None:
        """
        Plot model scores for each class as a function of momentum.
        
        Args:
            data_loader: Either a Dataset or DataLoader
            class_names: List of class names for labeling
            output_file: ROOT TFile object for saving
            momentum_key: Key for momentum in dataset (if using dict-like dataset)
        """
        
        momentum = test_results.features[:, momentum_idx]
        axis_spec_p = AxisSpec(50, 0, 5, 'p', '#it{p} (GeV/#it{c})')
        
        output_file.cd()
        
        for species_idx, species_name in enumerate(class_names):

            species_mask = (test_results.labels == species_idx)
            n_particles = np.sum(species_mask)

            print(f'\nProcessing {species_name}: {n_particles} particles')
            if n_particles == 0:
                print(f'  Warning: No particles found for species {species_name}')
                continue

            momentum_species = momentum[species_mask]
            all_scores_species = test_results.probabilities[species_mask]
            dir_species = output_file.mkdir(species_name)
            dir_species.cd()

            for hypothesis_idx, hypothesis_name in enumerate(class_names):

                axis_spec_score = AxisSpec(100, 0, 1, 'Score', f'{hypothesis_name} hypothesis; #it{{p}} (GeV/#it{{c}};Probability;')
                hist = build_TH2(momentum_species, all_scores_species[:, hypothesis_idx], axis_spec_p, axis_spec_score,
                                 name=f'score_{hypothesis_name}_hypothesis',
                                 title=f'{hypothesis_name} hypothesis; #it{{p}} (GeV/#it{{c}}); Probability;')
                hist.Write(f'hist_{hypothesis_name}')
                del hist

    def plot_roc_curves_per_class(self,
                                 test_results: EvaluationResult,
                                 class_names: List[str],
                                 output_file) -> Dict:
        """
        Plot overall ROC curves for each class (one-vs-rest).
        
        Args:
            data_loader: DataLoader containing test data
            class_names: List of class names
            output_file: ROOT TFile object for saving ROOT objects
            output_dir: Directory for saving matplotlib plots
            
        Returns:
            Dictionary containing ROC curve data and AUC values
        """

        n_classes = len(class_names)
        y_bin = label_binarize(test_results.labels, classes=range(n_classes))
        
        output_file.cd()
        root_dir = output_file.mkdir('ROC')
        root_dir.cd()
        
        momentum_bin_edges = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2., 
                              3., 5., 10.]

        for class_idx, class_name in enumerate(class_names):

            y_true = y_bin[:, class_idx]
            y_score = test_results.probabilities[:, class_idx]

            class_dir = root_dir.mkdir(class_name)
            class_dir.cd()

            for momentum_bin in range(len(momentum_bin_edges) - 1):
                
                mask = (test_results.features[:, 0] >= momentum_bin_edges[momentum_bin]) & \
                            (test_results.features[:, 0] < momentum_bin_edges[momentum_bin + 1])
                
                if not np.any(mask):
                    print(f"  No samples found in momentum bin {momentum_bin + 1}")
                    continue
                
                y_true_bin = y_true[mask]
                y_score_bin = y_score[mask]

                # Calculate ROC curve
                fpr, tpr, thresholds = roc_curve(y_true_bin, y_score_bin)
                roc_auc = auc(fpr, tpr)

                n_points = len(fpr)
                graph = TGraph(n_points)
                for i in range(n_points):
                    graph.SetPoint(i, fpr[i], tpr[i])
                graph.SetName(f'roc_curve_{class_name}_bin_{momentum_bin}')
                graph.SetTitle(f'ROC Curve {class_name} (p={momentum_bin_edges[momentum_bin]}-{momentum_bin_edges[momentum_bin + 1]} GeV/c);False Positive Rate;True Positive Rate')
                graph.SetLineColor(kBlue + class_idx)
                graph.SetLineWidth(2)
                graph.Write()

    def plot_efficiency_purity_curves_momentum_bins(self,
                                                   test_results: EvaluationResult,
                                                   class_names: List[str],
                                                   momentum_idx: int,
                                                   output_file,
                                                   momentum_bins: List[float] = None) -> Dict:
        """
        Plot efficiency vs purity curves for each class in different momentum bins.
        
        Efficiency = True Positive / (True Positive + False Negative) = Recall
        Purity = True Positive / (True Positive + False Positive) = Precision
        
        Args:
            test_results: EvaluationResult containing test data
            class_names: List of class names
            momentum_idx: Index of momentum feature in input features
            output_file: ROOT TFile object for saving ROOT objects
            output_dir: Directory for saving matplotlib plots
            momentum_bins: List of momentum bin edges
            
        Returns:
            Dictionary containing efficiency-purity curve data
        """
        
        if momentum_bins is None:
            momentum_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0, 10]
        
        momentum = test_results.features[:, momentum_idx]
        output_dir = output_file.mkdir('EfficiencyPurity')
        
        for class_idx, class_name in enumerate(class_names):
            
            print(f"\nProcessing efficiency-purity curves for {class_name}...")            

            class_dir = output_dir.mkdir(f'{class_name}')
            class_dir.cd()
            
            y_true_class = (test_results.labels == class_idx).astype(int)
            y_scores_class = test_results.probabilities[:, class_idx]
            
            for bin_idx in range(len(momentum_bins) - 1):
                
                bin_min = momentum_bins[bin_idx]
                bin_max = momentum_bins[bin_idx + 1]
                bin_mask = (momentum >= bin_min) & (momentum < bin_max)
                
                if np.sum(bin_mask) < 10:  # Skip bins with too few samples
                    print(f"  Skipping momentum bin [{bin_min:.1f}, {bin_max:.1f}) GeV/c: only {np.sum(bin_mask)} samples")
                    continue
                
                y_true_bin = y_true_class[bin_mask]
                y_score_bin = y_scores_class[bin_mask]
                
                if np.sum(y_true_bin) == 0:
                    print(f"  Skipping momentum bin [{bin_min:.1f}, {bin_max:.1f}) GeV/c: no positive samples")
                    continue
                
                thresholds = np.linspace(0, 1, 101)  # 101 points from 0 to 1
                efficiencies = []
                purities = []
                
                for threshold in thresholds:

                    y_pred_bin = (y_score_bin >= threshold).astype(int)
                    
                    tp = np.sum((y_true_bin == 1) & (y_pred_bin == 1))
                    fp = np.sum((y_true_bin == 0) & (y_pred_bin == 1))
                    fn = np.sum((y_true_bin == 1) & (y_pred_bin == 0))
                    
                    efficiency = tp / (tp + fn) if (tp + fn) > 0 else 0.0
                    purity = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
                    
                    efficiencies.append(efficiency)
                    purities.append(purity)
                
                efficiencies = np.array(efficiencies)
                purities = np.array(purities)
                
                n_points = len(efficiencies)
                graph = TGraph(n_points)
                for i in range(n_points):
                    graph.SetPoint(i, efficiencies[i], purities[i])
                
                graph.SetName(f'efficiency_purity_curve_bin_{bin_idx}')
                graph.SetTitle(f'Efficiency vs Purity p=[{bin_min:.1f},{bin_max:.1f}) GeV/c;Efficiency;Purity')
                graph.SetLineColor(kBlue + bin_idx % 6)
                graph.SetLineWidth(2)
                graph.SetMarkerColor(kBlue + bin_idx % 6)
                graph.SetMarkerStyle(20)
                graph.SetMarkerSize(0.5)
                graph.Write()
        
        print(f"\nEfficiency-Purity curve analysis complete!")
        print("ROOT graphs saved in output file")

    def analyze_feature_importance_shap(self,
                                        model,
                                        test_results: EvaluationResult,
                                        feature_names: List[str],
                                        class_names_dict: Dict[int, str],
                                        output_file,
                                        max_samples: int = 1000,
                                        background_samples: int = 100) -> Dict:
        """
        Analyze feature importance using SHAP (SHapley Additive exPlanations).

        Args:
            data_loader: DataLoader containing data for analysis
            feature_names: List of feature names (e.g., ['momentum', 'dE/dx', 'cluster_size', ...])
            class_names: List of class names (e.g., ['electron', 'pion', 'kaon', 'proton'])
            output_file: ROOT TFile object for saving plots
            max_samples: Maximum number of samples to analyze (for computational efficiency)
            background_samples: Number of background samples for SHAP explainer

        Returns:
            Dictionary containing SHAP values and analysis results
        """

        model.eval()
        X = test_results.features
        print(f"SHAP analysis on {X.shape[0]} samples with {X.shape[1]} features")

        if len(X) > max_samples:
            indices = np.random.choice(len(X), max_samples, replace=False)
            X_sample = X[indices]
            y_sample = test_results.labels[indices]
        else:
            X_sample = X
            y_sample = test_results.labels

        if len(X) > background_samples:
            bg_indices = np.random.choice(len(X), background_samples, replace=False)
            X_background = X[bg_indices]
        else:
            X_background = X

        print(f"Using {len(X_sample)} samples for SHAP analysis")
        print(f"Using {len(X_background)} background samples")

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        def _shap_model_predict_wrapper(x: np.ndarray) -> np.ndarray:
            """Wrapper function for SHAP that returns probabilities."""
            x_tensor = torch.FloatTensor(x).to(device)
            with torch.no_grad():
                outputs = model(x_tensor)
                probabilities = torch.softmax(outputs, dim=1)
            return probabilities.cpu().numpy()

        print("Creating SHAP explainer...")
        explainer = shap.Explainer(_shap_model_predict_wrapper, X_background)

        print("Calculating SHAP values...")
        shap_values = explainer(X_sample)

        df = pd.DataFrame(X_sample, columns=feature_names)
        df['true_class'] = y_sample
        df['true_class_name'] = [class_names_dict[int(i)] for i in y_sample]
        predictions = _shap_model_predict_wrapper(X_sample)
        predicted_classes = np.argmax(predictions, axis=1)
        df['predicted_class'] = predicted_classes
        df['predicted_class_name'] = [class_names_dict[int(i)] for i in predicted_classes]

        plot_dir = str(self.output_dir)+'/shap_plots'
        results = {
            'shap_values': shap_values,
            'feature_names': feature_names,
            'class_names': class_names_dict,
            'sample_data': df,
            'predictions': predictions,
            'plots': {}
        }

        plt.figure(figsize=(12, 8))
        shap.summary_plot(shap_values, X_sample, feature_names=feature_names,
                         class_names=list(class_names_dict.values()), plot_type="bar", show=False)
        plt.tight_layout()
        plt.savefig(plot_dir+'/shap_bar.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        results['plots']['bar'] = plot_dir+'/shap_bar.pdf'

        shap_dir = output_file.mkdir('SHAP_analysis')
        shap_dir.cd()
        for class_idx, class_name in class_names_dict.items():
            class_shap_values = shap_values.values[:, :, class_idx]
            mean_abs_shap = np.mean(np.abs(class_shap_values), axis=0)

            hist_name = f'h_shap_importance_{class_name}'
            hist_title = f'Feature Importance ({class_name});Feature;Mean |SHAP Value|'
            hist = TH1F(hist_name, hist_title, len(feature_names), 0, len(feature_names))

            for feat_idx, (feat_name, importance) in enumerate(zip(feature_names, mean_abs_shap)):
                hist.SetBinContent(feat_idx + 1, importance)
                hist.GetXaxis().SetBinLabel(feat_idx + 1, feat_name)

            hist.SetLineColor(kBlue)
            hist.SetLineWidth(2)
            hist.Write()

            results[f'root_hist_{class_name}'] = hist

        print(f"\nSHAP analysis complete!")
        print(f"Plots saved in: {plot_dir}/")
        print("ROOT histograms saved in output file")

        return results         
    