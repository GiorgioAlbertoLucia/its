"""
Improved training routine with proper validation, metrics, and monitoring.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import classification_report, confusion_matrix

from ROOT import TFile
from torchic.core.histogram import AxisSpec, build_TH2

class NNRoutine:
    """Improved neural network training routine with validation and detailed metrics."""
    
    def __init__(self, model: nn.Module, class_weights: Optional[Dict] = None):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        # Training history
        self.train_losses = []
        self.val_losses = []
        self.train_accuracies = []
        self.val_accuracies = []
        
        # Setup loss function with class weights
        if class_weights:
            weights = torch.FloatTensor(list(class_weights.values())).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        else:
            self.criterion = nn.CrossEntropyLoss()
            
        self.logger = logging.getLogger(__name__)
        
    def train_epoch(self, train_loader: DataLoader, optimizer: optim.Optimizer) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for batch_idx, (inputs, labels) in enumerate(train_loader):
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item() * inputs.size(0)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(train_loader.dataset)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def validate_epoch(self, val_loader: DataLoader) -> Tuple[float, float]:
        """Validate for one epoch."""
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                loss = self.criterion(outputs, labels)
                
                total_loss += loss.item() * inputs.size(0)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_loss = total_loss / len(val_loader.dataset)
        accuracy = 100 * correct / total
        
        return avg_loss, accuracy
    
    def run_training_loop(self, 
                         train_loader: DataLoader, 
                         val_loader: DataLoader,
                         learning_rate: float = 1e-3,
                         num_epochs: int = 50,
                         early_stopping_patience: int = 10,
                         weight_decay: float = 1e-4) -> Dict:
        """Run complete training loop with validation and early stopping."""
        
        optimizer = optim.Adam(self.model.parameters(), 
                              lr=learning_rate, 
                              weight_decay=weight_decay)
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 
                                                        mode='min', 
                                                        factor=0.5, 
                                                        patience=5)
        
        from core.nn.pid_fcnn import EarlyStopping
        early_stopping = EarlyStopping(patience=early_stopping_patience)
        
        best_val_loss = float('inf')
        best_model_state = None
        
        self.logger.info(f"Starting training on {self.device}")
        self.logger.info(f"Training samples: {len(train_loader.dataset)}")
        self.logger.info(f"Validation samples: {len(val_loader.dataset)}")
        
        for epoch in range(num_epochs):

            train_loss, train_acc = self.train_epoch(train_loader, optimizer)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]['lr']
            
            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.train_accuracies.append(train_acc)
            self.val_accuracies.append(val_acc)
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
            
            self.logger.info(f"Epoch {epoch+1}/{num_epochs}: "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                           f"LR: {current_lr:.6f}")
            
            if early_stopping(val_loss):
                self.logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if best_model_state:
            self.model.load_state_dict(best_model_state)
            self.logger.info(f"Loaded best model with validation loss: {best_val_loss:.4f}")
        
        return {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'train_accuracies': self.train_accuracies,
            'val_accuracies': self.val_accuracies,
            'best_val_loss': best_val_loss
        }
    
    def get_model_scores(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Get model scores for each class hypothesis on the provided data.
        
        Returns:
            all_scores: Shape (N, num_classes) - probabilities for each class
            all_labels: Shape (N,) - true labels
            all_features: Shape (N, num_features) - input features (needed for momentum)
        """
        
        self.model.eval()
        all_scores = []
        all_labels = []
        all_features = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                if inputs.dim() == 1:
                    inputs = inputs.unsqueeze(0)  # Add batch dimension if missing
                
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                
                all_scores.append(probabilities.cpu().numpy())
                all_labels.append(labels.cpu().numpy())
                all_features.append(inputs.cpu().numpy())
        
        all_scores = np.concatenate(all_scores, axis=0)
        all_labels = np.concatenate(all_labels, axis=0)
        all_features = np.concatenate(all_features, axis=0)
        
        return all_scores, all_labels, all_features

    def evaluate_model(self, test_loader: DataLoader, label_encoder=None) -> Dict:
        """Comprehensive model evaluation."""

        self.model.eval()
        
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, labels in test_loader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        all_predictions = np.array(all_predictions)
        all_labels = np.array(all_labels)
        all_probabilities = np.array(all_probabilities)
        
        accuracy = 100 * np.sum(all_predictions == all_labels) / len(all_labels)
        
        if label_encoder:
            target_names = label_encoder.classes_
        else:
            target_names = None
            
        class_report = classification_report(all_labels, all_predictions, 
                                           target_names=target_names, 
                                           output_dict=True)
        
        return {
            'accuracy': accuracy,
            'predictions': all_predictions,
            'labels': all_labels,
            'probabilities': all_probabilities,
            'classification_report': class_report
        }
    
    def plot_training_history(self, output_dir: str):
        """Plot training and validation metrics."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        fig, ((ax1, ax2)) = plt.subplots(1, 2, figsize=(12, 5))
        
        epochs = range(1, len(self.train_losses) + 1)
        ax1.plot(epochs, self.train_losses, 'b-', label='Training Loss')
        ax1.plot(epochs, self.val_losses, 'r-', label='Validation Loss')
        ax1.set_title('Model Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(epochs, self.train_accuracies, 'b-', label='Training Accuracy')
        ax2.plot(epochs, self.val_accuracies, 'r-', label='Validation Accuracy')
        ax2.set_title('Model Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.savefig(output_dir / 'training_history.pdf', dpi=300, bbox_inches='tight')
        plt.close()
        
    def plot_confusion_matrix(self, labels: np.ndarray, predictions: np.ndarray, 
                            label_encoder=None, output_dir: str = '.'):
        """Plot confusion matrix."""

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        cm = confusion_matrix(labels, predictions)
        
        plt.figure(figsize=(8, 6))
        if label_encoder:
            class_names = label_encoder.classes_
        else:
            class_names = [f'Class {i}' for i in range(len(np.unique(labels)))]
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(output_dir / 'confusion_matrix.pdf', dpi=300, bbox_inches='tight')
        plt.close()

    def plot_model_scores(self, 
                          data_loader: DataLoader,
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
        
        all_scores, __, all_features = self.get_model_scores(data_loader)
        momentum = all_features[:, momentum_idx]
        print(f'{all_scores.shape=}')
        
        output_file.cd()
        
        axis_spec_p = AxisSpec(50, 0, 5, 'p', '#it{p} (GeV/#it{c})')
        
        for class_idx, class_name in enumerate(class_names):
            print(f'{all_scores[:, class_idx].shape=}')
            axis_spec_score = AxisSpec(100, 0, 1, 'Score', f'{class_name} hypothesis; #it{{p}} (GeV/#it{{c}};Probability;')
            hist = build_TH2(momentum, all_scores[:, class_idx], axis_spec_p, axis_spec_score)
            hist.Write(f'hist_{class_name}')
            
        
        