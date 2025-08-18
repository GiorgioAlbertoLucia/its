"""
Simple Momentum-Aware BDT implementation following the neural network patterns.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import json
from typing import Dict, List, Tuple, Optional

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from ROOT import TFile
from torchic import Dataset
from torchic.physics.ITS import unpack_cluster_sizes, average_cluster_size

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.bdt.momentum_aware_bdt import MomentumAwareBDT
from core.bdt.bdt_routine import BDTRoutine, BDTConfig, plot_momentum_performance
from core.ml_common import add_physics_features, get_feature_columns, calculate_class_weights
from utils.pid_routine import PARTICLE_ID
from utils.data_config import DataConfig
from utils.logging import setup_logging

def load_data(data_config: DataConfig) -> Dataset:
    """Load and preprocess the dataset."""
    
    logger = setup_logging()
    logger.info("Loading dataset...")
    
    dataset = Dataset.from_root([
        '/data/galucia/its_pid/LHC24_pass1_skimmed/data_04_08_2025.root',
        ],
        tree_name='O2clsttable',
        folder_name='DF*',
        columns=['fP', 'fEta', 'fPhi', 'fItsClusterSize', 'fPartID']
    )
    
    dataset.concat(Dataset.from_root([
        '/data/galucia/its_pid/LHC24_pass1_skimmed/data_04_08_2025.root'
        ],
        tree_name='O2clsttableextra',
        folder_name='DF*',
        columns=['fP', 'fEta', 'fPhi', 'fItsClusterSize', 'fPartID']),
        axis=1
    )
    
    logger.info("Unpacking cluster sizes...")
    np_unpack_cluster_sizes = np.vectorize(unpack_cluster_sizes)
    for layer in range(7):
        dataset[f'fItsClusterSizeL{layer}'] = np_unpack_cluster_sizes(dataset['fItsClusterSize'], layer)

    dataset.loc[dataset['fPartID'] == PARTICLE_ID['He'], 'fP'] *= 2
    
    dataset['fCosL'] = 1 / np.cosh(dataset['fEta'])
    dataset['fPAbs'] = np.abs(dataset['fP'])
    dataset['fClSizeCosL'], dataset['fNHitsIts'] = average_cluster_size(dataset['fItsClusterSize'])
    dataset['fMeanItsClSize'] = dataset['fClSizeCosL'] / dataset['fCosL']
    
    dataset.query(f'fNHitsIts == {data_config.min_hits_required}', inplace=True)
    dataset.query(f'fPAbs < {data_config.max_p_required}', inplace=True)
    logger.info(f"Filtered to tracks with {data_config.min_hits_required} hits: {len(dataset)} samples")
    
    return dataset

def init_configs(output_dir: Path = Path("../output/nn")):
    """Initialize data and training configurations."""

    data_config = DataConfig(
        data_fraction=0.3,
        balance_classes=True,
        add_pt=True,
        add_shower_shape=True,
        add_layer_ratios=True,
        min_hits_required=5,
        max_p_required=10.0,  # GeV/c
    )

    train_config = BDTConfig(
        learning_rate=0.1,
        n_estimators=100,
        max_depth=6,
        min_child_weight=1,
        gamma=0.0,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.0,
        reg_lambda=1.0,
        scale_pos_weight=None,  # Will be set later based on class weights
        test_size=0.2,
    )

    with open(output_dir / "data_config.json", "w") as f:
        json.dump(data_config.__dict__, f, indent=2)
    with open(output_dir / "train_config.json", "w") as f:
        json.dump(train_config.__dict__, f, indent=2)
    
    return data_config, train_config

def prepare_data(df: pd.DataFrame, data_config: DataConfig, train_config: BDTConfig)\
     -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, LabelEncoder, Dict, List[str]]:
    """Prepare data for training with feature engineering and scaling."""
    logger = setup_logging()
    
    if data_config.data_fraction < 1.0:
        df = df.sample(frac=data_config.data_fraction, random_state=42)
        logger.info(f"Sampled {data_config.data_fraction*100}% of data: {len(df)} samples")
    
    logger.info("Adding physics features...")
    df = add_physics_features(df)
    
    if data_config.balance_classes:

        class_counts = df['fPartID'].value_counts()
        min_count = class_counts.min()
        logger.info(f"Balancing classes to {min_count} samples each")
        
        balanced_dfs = []
        for class_id in class_counts.index:
            class_df = df[df['fPartID'] == class_id].sample(n=min_count, random_state=42)
            balanced_dfs.append(class_df)
        df = pd.concat(balanced_dfs, ignore_index=True)
        
        logger.info(f"Balanced dataset size: {len(df)} samples")
    
    feature_columns = get_feature_columns(data_config)
    logger.info(f"Using {len(feature_columns)} features: {feature_columns}")
    X = df[feature_columns].values.astype(np.float32)
    
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(df['fPartID'].values)
    
    logger.info(f"Classes: {dict(zip(label_encoder.classes_, range(len(label_encoder.classes_))))}")
    
    class_weights = calculate_class_weights(y)
    logger.info(f"Class weights: {class_weights}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=data_config.test_size, random_state=42)
    
    return X_train, y_train, X_test, y_test, label_encoder, class_weights, feature_columns

def main():
    """Example of how to use the BDT following NN patterns."""
    
    # Assuming you have your data in X, y format
    # X shape: (n_samples, n_features) with momentum at index 0
    # y shape: (n_samples,) with class labels

    data_config, train_config = init_configs()
    dataset = load_data(data_config)
    df = dataset.data
    
    X_train, y_train, X_test, y_test, label_encoder, class_weights, feature_columns = prepare_data(df, data_config, train_config)
    
    bdt_model = MomentumAwareBDT(momentum_feature_idx=0)
    
    bdt_routine = BDTRoutine(bdt_model, class_weights)
    training_history = bdt_routine.run_training_loop(X_train, y_train, val_size=0.2)
    scores, momentum = bdt_model.get_model_scores(X_test)
    
    output_file = TFile("../output/bdt/test_results.root", "RECREATE")
    class_names = ['Pi', 'Ka', 'Pr', 'De', 'He']
    bdt_routine.plot_model_scores(X_test, class_names, output_file)
    output_file.Close()
    
    plot_momentum_performance(bdt_model, X_test, y_test)
    
if __name__ == "__main__":
    main()
