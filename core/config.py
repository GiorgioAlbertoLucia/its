"""Unified configuration management for the neural network training."""
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path

@dataclass
class BDTConfig:
    """BDT-specific configuration."""
    learning_rate: float = 0.1
    n_estimators: int = 100
    max_depth: int = 6
    min_child_weight: int = 1
    gamma: float = 0.0
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    reg_alpha: float = 0.0
    reg_lambda: float = 1.0
    early_stopping_rounds: int = 20
    n_jobs: int = -1
    verbosity: int = 0
    objective: str = 'multi:softprob'  # For multi-class classification
    eval_metric: str = 'mlogloss'
    random_state: int = 42

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

    # NN-specific settings
    hidden_dims: List[int] = None
    dropout_rate: float = 0.15
    use_batch_norm: bool = True

    # BDT-specific settings
    bdt_config: Optional[BDTConfig] = None
    momentum_bins: List[float] = None  # For regime-specific BDT
    
    # Training settings
    use_class_weights: bool = True
    batch_size: int = 128
    learning_rate: float = 1e-3
    num_epochs: int = 50
    early_stopping_patience: int = 10
    weight_decay: float = 1e-4
    feature_columns: List[str] = None
    
    # Evaluation settings
    test_size: float = 0.2
    val_size: float = 0.2
    
    # Output settings
    output_dir: Path = Path("../output")
    output_file_suffix: str = ""
    save_plots: bool = True
    
    def __post_init__(self):
        if self.hidden_dims is None:
            self.hidden_dims = [256, 128, 64]
        self.output_dir = Path(self.output_dir)

    def get(self, key: str, default=None):
        return getattr(self, key, default)

    