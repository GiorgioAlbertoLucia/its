"""Main experiment orchestrator."""
import numpy as np
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch.utils.data import DataLoader

from ROOT import TFile

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import ExperimentConfig
from core.processor import DataProcessor, torch_data_preparation
from core.nn.factory import ModelFactory
from core.nn.trainer import Trainer
from core.nn.evaluator import Evaluator
from core.nn.plotter import Plotter
from utils.pid_routine import PARTICLE_ID

class Pipeline:
    """Orchestrates the entire ML pipeline."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor(config)
        self.plotter = Plotter(config.output_dir)
        
        # Will be set during pipeline
        self.model = None
        self.trainer = None
        self.evaluator = None
    
    def run(self) -> Dict:
        """Run complete experiment pipeline."""
        results = {}
        
        self.data_processor.load_raw_data(self.config.input_files,
                                          self.config.tree_names,
                                          self.config.folder_name,
                                          self.config.columns)
        self.data_processor.downsample_data()
        self.data_processor.engineer_features()
        if self.config.balance_classes:
            self.data_processor.balance_classes()

        part_id_column = self.data_processor.df['fPartID']
        encoded_ids = self.data_processor.label_encoder.fit_transform(part_id_column.unique())
        id_to_encoded_id_map = {k: v for k, v in zip(part_id_column.unique(), encoded_ids)}
        encoded_id_to_name_map = {}
        for name, id in PARTICLE_ID.items():
            encoded_id = id_to_encoded_id_map.get(id, None)
            if encoded_id is not None:
                encoded_id_to_name_map[encoded_id] = name
        class_names = encoded_id_to_name_map.values()

        X, y = self.data_processor.prepare_features()
        class_weights = self.data_processor.get_class_weights(y)
        
        train_loader, val_loader, test_loader = self._create_data_loaders(X, y)
        
        self.model = ModelFactory.create_model(
            self.config, input_dim=X.shape[1], num_classes=len(np.unique(y))
        )
        
        self.trainer = Trainer(self.model, self.config, class_weights=class_weights)
        training_state = self.trainer.train(train_loader, val_loader)
        
        self.evaluator = Evaluator(self.model)
        test_results = self.evaluator.evaluate(test_loader, return_features=True)
        
        if self.config.save_plots:
            self.plotter.plot_training_history(
                training_state.train_losses, training_state.val_losses,
                training_state.train_accuracies, training_state.val_accuracies
            )
            self.plotter.plot_confusion_matrix(
                test_results.labels, test_results.predictions,
                class_names
            )
            output_file = TFile(str(self.config.output_dir)+'/results.root', 'RECREATE')
            self.plotter.plot_model_scores(test_results, 
                class_names, output_file,
                self.config.momentum_feature_idx
            )
            self.plotter.plot_roc_curves_per_class(
                test_results, class_names,
                output_file
            )
            self.plotter.plot_efficiency_purity_curves_momentum_bins(
                test_results, class_names,
                self.config.momentum_feature_idx, output_file
            )
            self.plotter.analyze_feature_importance_shap(
                self.model,
                test_results, self.data_processor.feature_columns,
                encoded_id_to_name_map, output_file
            )
        
        self._save_results(training_state, test_results)
        
        results = {
            'test_accuracy': test_results.accuracy,
            'training_state': training_state,
            'test_results': test_results
        }
        
        return results
    
    def _create_data_loaders(self, X, y) -> Tuple[DataLoader, DataLoader, DataLoader]:
        """Create train/val/test data loaders."""
        
        print('Preparing data loaders.')
        return torch_data_preparation(X, y,
            batch_size=self.config.batch_size,
            test_val_size=self.config.test_size + self.config.val_size
        )
    
    def _save_results(self, training_state, test_results):
        """Save experiment results."""
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': self.config,
            'feature_columns': self.data_processor.feature_columns,
            'label_encoder': self.data_processor.label_encoder,
            'scaler': self.data_processor.scaler,
        }, self.config.output_dir / 'trained_model.pth')
