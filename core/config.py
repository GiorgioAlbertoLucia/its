"""Unified configuration management for the neural network training."""
from dataclasses import dataclass
from typing import List
from pathlib import Path

@dataclass
class ExperimentConfig:
    """Single source of truth for all experiment settings."""
    # Data settings
    input_files: List[str]
    tree_names: List[str] = None  # Names of ROOT trees to read
    folder_name: str = 'DF*'  # Folder name for data
    columns: List[str] = None  # Columns to read from ROOT files

    data_fraction: float = 1.0  # Scaling factor for faster pipelines
    balance_classes: bool = False
    min_hits_required: int = 5
    max_p_required: float = 10.0
    
    # Feature engineering
    add_statistics_features: bool = True
    add_parametrised_features: bool = True
    scaler: str = 'momentum_aware'  # Options: 'standard', 'momentum_aware'
    momentum_feature_idx: int = 0
    balance_classes: bool = True
    
    # Model settings
    model_type: str = 'PidFCNN'  # or 'MomentumGatedPID'
    hidden_dims: List[int] = None
    dropout_rate: float = 0.15
    use_batch_norm: bool = True
    
    # Training settings
    use_class_weights: bool = True
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 50
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4
    
    # Evaluation settings
    test_size: float = 0.2
    val_size: float = 0.2
    
    # Output settings
    output_dir: Path = Path("../output/nn")
    save_plots: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]
        self.output_dir = Path(self.output_dir)