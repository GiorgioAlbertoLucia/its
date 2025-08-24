"""Model factory."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.nn.pid_fcnn import PidFCNN
from core.nn.momentum_gated_pid import MomentumGatedPID
from core.config import ExperimentConfig

class ModelFactory:
    """Creates models based on configuration."""
    
    @staticmethod
    def create_model(config: ExperimentConfig, input_dim: int, num_classes: int):
        """Create model based on config."""
        if config.model_type == 'PidFCNN':
            return PidFCNN(
                input_dim=input_dim,
                num_classes=num_classes,
                hidden_dims=config.hidden_dims,
                dropout_rate=config.dropout_rate,
                use_batch_norm=config.use_batch_norm
            )
        elif config.model_type == 'MomentumGatedPID':
            return MomentumGatedPID(
                input_dim=input_dim,
                num_classes=num_classes,
                dropout_rate=config.dropout_rate,
                use_batch_norm=config.use_batch_norm
            )
        else:
            raise ValueError(f"Unknown model type: {config.model_type}")
