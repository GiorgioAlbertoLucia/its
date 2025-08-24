from dataclasses import dataclass

@dataclass
class DataConfig:
    """Configuration for data processing."""
    data_fraction: float = 1.0  # Fraction of data to use
    balance_classes: bool = True
    min_hits_required: int = 7
    max_p_required: float = 10  # Maximum momentum to consider
    
    # Feature engineering
    add_pt: bool = True
    add_shower_shape: bool = True
    add_layer_ratios: bool = True
    add_expected_cluster_size: bool = False
