"""Model evaluation and metrics."""
import numpy as np
from sklearn.metrics import classification_report
from typing import Dict, Optional
from dataclasses import dataclass

@dataclass
class EvaluationResult:
    """Container for evaluation results."""
    predictions: np.ndarray
    labels: np.ndarray
    probabilities: np.ndarray
    accuracy: float
    features: Optional[np.ndarray] = None
    classification_report: Optional[Dict] = None
    
    def __post_init__(self):
        if self.classification_report is None:
            self.classification_report = classification_report(
                self.labels, self.predictions, output_dict=True
            )
            