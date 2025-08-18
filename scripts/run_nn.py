"""
Improved main script for particle identification neural network training.
"""

import numpy as np
import pandas as pd
import torch
from pathlib import Path
import json

from sklearn.preprocessing import LabelEncoder, StandardScaler

from ROOT import TFile
from torchic import Dataset
from torchic.physics.ITS import unpack_cluster_sizes, average_cluster_size

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.nn.nn_common import TrainingConfig, torch_data_preparation
from core.nn.pid_fcnn import PidFCNN
from core.nn.momentum_gated_pid import MomentumGatedPID, plot_momentum_gating, MomentumAwareScaler
from core.nn.nn_routine import NNRoutine
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

def prepare_data(df: pd.DataFrame, data_config: DataConfig, training_config: TrainingConfig):
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
    
    scaler = StandardScaler() # fcnn - non momentum aware
    if training_config.model_type == 'MomentumGatedPID':
        scaler = MomentumAwareScaler(momentum_feature_idx=0)
    X_scaled = scaler.fit_transform(X)
    
    class_weights = calculate_class_weights(y)
    logger.info(f"Class weights: {class_weights}")
    
    train_dataset, val_dataset, test_dataset, train_loader, val_loader, test_loader = torch_data_preparation(
        X_scaled, y,
        test_val_size=training_config.test_size + training_config.val_size,
        batch_size=training_config.batch_size
    )
    
    return train_dataset, val_dataset, test_dataset, \
            train_loader, val_loader, test_loader, \
            label_encoder, scaler, class_weights, feature_columns

def init_configs(output_dir: Path = Path("../output/nn")):
    """Initialize data and training configurations."""

    data_config = DataConfig(
        data_fraction=0.3,
        balance_classes=True,
        add_pt=True,
        add_shower_shape=True,
        add_layer_ratios=True
    )
    
    training_config = TrainingConfig(
        batch_size=256,
        learning_rate=1e-3,
        #num_epochs=100,
        num_epochs=1,  # Set to 1 for quick testing, change to 100 for full training
        early_stopping_patience=15,
        hidden_dims=[64, 32, 16],
        dropout_rate=0.15,
        model_type='MomentumGatedPID',  # Change to 'PidFCNN' if needed
    )

    with open(output_dir / "data_config.json", "w") as f:
        json.dump(data_config.__dict__, f, indent=2)
    with open(output_dir / "training_config.json", "w") as f:
        json.dump(training_config.__dict__, f, indent=2)
    
    return data_config, training_config

def init_model(training_config: TrainingConfig, input_dim: int, num_classes: int):
    """Initialize the model based on the training configuration."""

    if training_config.model_type == 'PidFCNN':
        model = PidFCNN(
            input_dim=input_dim,
            num_classes=num_classes,
            hidden_dims=training_config.hidden_dims,
            dropout_rate=training_config.dropout_rate,
            use_batch_norm=training_config.use_batch_norm,
            use_residual=True
        )
    elif training_config.model_type == 'MomentumGatedPID':
        model = MomentumGatedPID(
            input_dim=input_dim,
            num_classes=num_classes,
            dropout_rate=training_config.dropout_rate,
            use_batch_norm=training_config.use_batch_norm
        )
    else:
        raise ValueError(f"Unknown model type: {training_config.model_type}")
    
    return model

def train_model(logger, input_dim, num_classes, train_loader, val_loader, training_config: TrainingConfig, class_weights) -> NNRoutine:
    """Train the model using the provided data loaders and configuration."""

    logger.info(f"Creating model with input_dim={input_dim}, num_classes={num_classes}")
    model = init_model(training_config, input_dim=input_dim, num_classes=num_classes)

    nn_routine = NNRoutine(model, class_weights)
    logger.info("Starting training routine...")
    
    training_history = nn_routine.run_training_loop(
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=training_config.learning_rate,
        num_epochs=training_config.num_epochs,
        early_stopping_patience=training_config.early_stopping_patience,
        weight_decay=training_config.weight_decay
    )
    
    return nn_routine, training_history

def load_trained_model(model_path: str):
    """Load a trained model for inference."""
    checkpoint = torch.load(model_path)
    
    model = PidFCNN(**checkpoint['model_config'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    return model, checkpoint['label_encoder'], checkpoint['scaler'], checkpoint['feature_columns']

def performance_evaluation(logger, nn_routine: NNRoutine, output_dir: Path, label_encoder: LabelEncoder, test_loader) -> dict:
    
    nn_routine.plot_training_history(output_dir)
    #plot_momentum_gating(model, output_dir / 'momentum_gating_plot.png')
    
    logger.info("Evaluating on test set...")
    test_results = nn_routine.evaluate_model(test_loader, label_encoder)
    output_file = TFile(str(output_dir / 'test_results.root'), 'RECREATE')
    print(f'{test_results["classification_report"]}')
    nn_routine.plot_model_scores(test_loader, momentum_idx=0,
                                 class_names=label_encoder.classes_, output_file=output_file)
    
    logger.info(f"Test Accuracy: {test_results['accuracy']:.2f}%")
    print("\nClassification Report:")
    print("=" * 50)
    for class_name, metrics in test_results['classification_report'].items():
        if isinstance(metrics, dict):
            print(f"{class_name:>15}: Precision={metrics['precision']:.3f}, "
                  f"Recall={metrics['recall']:.3f}, F1={metrics['f1-score']:.3f}")
    
    nn_routine.plot_confusion_matrix(
        test_results['labels'], 
        test_results['predictions'],
        label_encoder,
        output_dir
    )

    return test_results

def main(do_train: bool = True):
    """Main training function."""

    logger = setup_logging()
    output_dir = Path("../output/nn")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    data_config, training_config = init_configs(output_dir)
    
    logger.info("Loading and preparing data...")
    dataset = load_data(data_config)
    df = dataset.data
    
    train_dataset, test_dataset, val_dataset, train_loader, val_loader, test_loader,\
        label_encoder, scaler, class_weights, feature_columns = prepare_data(
        df, data_config, training_config
    )
    input_dim = len(feature_columns)
    num_classes = len(label_encoder.classes_)
    
    if do_train:
        nn_routine, training_history = train_model(logger, input_dim, num_classes, train_loader, val_loader, training_config, class_weights)
    else:
        logger.info("Skipping training, loading pre-trained model...")
        model, label_encoder, scaler, feature_columns = load_trained_model(output_dir / 'trained_model.pth')
        nn_routine = NNRoutine(model, class_weights)
        training_history = None
    
    test_results = performance_evaluation(logger, nn_routine, output_dir, label_encoder, test_loader)
    
    torch.save({
        'model_state_dict': nn_routine.model.state_dict(),
        'model_config': {
            'input_dim': input_dim,
            'num_classes': num_classes,
            'hidden_dims': training_config.hidden_dims,
            'dropout_rate': training_config.dropout_rate,
            'use_batch_norm': training_config.use_batch_norm
        },
        'label_encoder': label_encoder,
        'scaler': scaler,
        'feature_columns': feature_columns,
        'training_history': training_history,
        'test_results': {k: v for k, v in test_results.items() 
                        if k not in ['predictions', 'labels', 'probabilities']}
    }, output_dir / 'trained_model.pth')
    
    np.savez(output_dir / 'predictions.npz',
             predictions=test_results['predictions'],
             labels=test_results['labels'],
             probabilities=test_results['probabilities'])
    
    logger.info(f"Training completed! Results saved to {output_dir}")
    
    return nn_routine.model, test_results

def predict_particle_id(model, scaler, feature_columns, sample_data: pd.DataFrame):
    """Predict particle ID for new data."""

    sample_data = add_physics_features(sample_data)
    X = sample_data[feature_columns].values.astype(np.float32)
    X_scaled = scaler.transform(X)
    
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    with torch.no_grad():
        probabilities = model.predict_proba(X_tensor)
        predicted_classes = torch.argmax(probabilities, dim=1)
    
    return predicted_classes.numpy(), probabilities.numpy()

if __name__ == "__main__":
    try:
        model, results = main()
        print("\n" + "="*60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print("="*60)
        print(f"Final Test Accuracy: {results['accuracy']:.2f}%")
        print(f"Results saved to: ../output/nn/")
        
    except Exception as e:
        logger = setup_logging()
        logger.error(f"Training failed with error: {str(e)}")
        raise