import numpy as np

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.evaluation_result import EvaluationResult
from core.bdt.models import BaseBDTModel

class Evaluator:
    """BDT evaluator that creates compatible EvaluationResult objects."""
    
    def __init__(self, model: BaseBDTModel):
        self.model = model
    
    def evaluate(self, X: np.ndarray, y: np.ndarray, 
                return_features: bool = False) -> EvaluationResult:
        """Evaluate BDT model and return EvaluationResult compatible with NN."""
        
        probabilities = self.model.predict_proba(X)
        predictions = np.argmax(probabilities, axis=1)
        accuracy = 100 * np.sum(predictions == y) / len(y)
        
        features = X if return_features else None
        
        return EvaluationResult(
            features=features,
            predictions=predictions,
            labels=y,
            probabilities=probabilities,
            accuracy=accuracy
        )