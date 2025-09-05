"""Model evaluation and metrics."""
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from typing import Tuple

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.evaluation_result import EvaluationResult

class Evaluator:
    """Handles model evaluation."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    def predict(self, data_loader: DataLoader) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Get model predictions."""
        self.model.eval()
        all_predictions = []
        all_labels = []
        all_probabilities = []
        
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs = inputs.to(self.device)
                outputs = self.model(inputs)
                probabilities = torch.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_predictions.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probabilities.extend(probabilities.cpu().numpy())
        
        return (np.array(all_predictions), np.array(all_labels), 
                np.array(all_probabilities))
    
    def evaluate(self, data_loader: DataLoader, return_features:bool = False) -> EvaluationResult:
        """Complete evaluation."""
        predictions, labels, probabilities = self.predict(data_loader)
        accuracy = 100 * np.sum(predictions == labels) / len(labels)

        features = None
        if return_features:
            features = data_loader.dataset.X.numpy()
        
        return EvaluationResult(
            features=features,
            predictions=predictions, 
            labels=labels, 
            probabilities=probabilities, 
            accuracy=accuracy
        )
    