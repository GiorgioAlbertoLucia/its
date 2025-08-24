"""Focused training logic."""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from typing import Dict, Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import ExperimentConfig
from core.nn.early_stopping import EarlyStopping

@dataclass
class TrainingState:
    """Container for training state."""
    train_losses: List[float] = None
    val_losses: List[float] = None
    train_accuracies: List[float] = None
    val_accuracies: List[float] = None
    best_val_loss: float = float('inf')
    best_model_state: Optional[Dict] = None
    epoch: int = 0
    
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
    """Handles model training logic only."""
    
    def __init__(self, model: nn.Module, config: ExperimentConfig, class_weights: Optional[Dict[int, float]] = None):

        self.model = model
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
        self.criterion = nn.CrossEntropyLoss()
        if class_weights is not None and self.config.use_class_weights:
            weights = torch.tensor([class_weights[i] for i in range(len(class_weights))], dtype=torch.float32).to(self.device)
            self.criterion = nn.CrossEntropyLoss(weight=weights)
        self.optimizer = optim.Adam(
            model.parameters(), 
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=5
        )
    
    def train_epoch(self, train_loader: DataLoader) -> Tuple[float, float]:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(self.device), labels.to(self.device)
            
            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            loss = self.criterion(outputs, labels)
            loss.backward()
            self.optimizer.step()
            
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
    
    def train(self, train_loader: DataLoader, val_loader: DataLoader) -> TrainingState:
        """Complete training loop."""
        state = TrainingState()
        early_stopping = EarlyStopping(patience=self.config.early_stopping_patience)
        
        for epoch in range(self.config.num_epochs):
            state.epoch = epoch
            
            train_loss, train_acc = self.train_epoch(train_loader)
            val_loss, val_acc = self.validate_epoch(val_loader)
            
            state.train_losses.append(train_loss)
            state.val_losses.append(val_loss)
            state.train_accuracies.append(train_acc)
            state.val_accuracies.append(val_acc)
            
            self.scheduler.step(val_loss)
            current_lr = self.optimizer.param_groups[0]['lr']
            
            if val_loss < state.best_val_loss:
                state.best_val_loss = val_loss
                state.best_model_state = self.model.state_dict().copy()
            
            print(f"Epoch {epoch+1}/{self.config.num_epochs}: "
                           f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                           f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}% | "
                           f"LR: {current_lr:.6f}")
            
            if early_stopping(val_loss):
                print(f"Early stopping triggered at epoch {epoch+1}")
                break
        
        if state.best_model_state:
            self.model.load_state_dict(state.best_model_state)
        
        return state