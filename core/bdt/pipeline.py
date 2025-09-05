"""Main experiment orchestrator."""
from dataclasses import dataclass
import numpy as np
from pathlib import Path
from typing import Dict

from ROOT import TFile

import sys
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import ExperimentConfig
from core.processor import DataProcessor
from core.bdt.factory import ModelFactory
from core.bdt.trainer import Trainer, TrainingState
from core.bdt.evaluator import Evaluator
from core.evaluation_result import EvaluationResult
from core.plotter import Plotter
from utils.pid_routine import PARTICLE_ID

@dataclass
class DataHolder:
    """Container for holding data splits."""
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray

class Pipeline:
    """Orchestrates both NN and BDT experiments."""
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize components
        self.data_processor = DataProcessor(config)
        self.plotter = Plotter(config.output_dir, config.output_file_suffix)
        
        # Will be set during pipeline
        self.model = None
        self.trainer = None
        self.evaluator = None

    def _create_data_holder(self, X, y) -> DataHolder:

        """Create data splits for training, validation, and testing."""
        from sklearn.model_selection import train_test_split
        
        X_train_val, X_test, y_train_val, y_test = train_test_split(
            X, y, test_size=self.config.test_size, random_state=42
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_train_val, y_train_val, test_size=self.config.val_size, 
            random_state=42
        )
        
        return DataHolder(X_train, y_train, X_val, y_val, X_test, y_test)
    
    def run(self) -> Dict:
        """Run experiment pipeline for both NN and BDT."""
        results = {}
        
        # Data processing (same for both)
        self.data_processor.load_raw_data(
            self.config.input_files, self.config.tree_names,
            self.config.folder_name, self.config.columns
        )
        self.data_processor.downsample_data()
        self.data_processor.engineer_features()
        self.data_processor.select_clean_data()
        if self.config.balance_classes:
            self.data_processor.balance_classes()
        
        X, y = self.data_processor.prepare_features()
        class_weights = self.data_processor.get_class_weights(y)

        data_holder = self._create_data_holder(X, y)
        
        # Create model
        self.model = ModelFactory.create_model(
            self.config, input_dim=X.shape[1], num_classes=len(np.unique(y))
        )

        self.trainer = Trainer(self.model, self.config, class_weights)
        training_state = self.trainer.train(data_holder.X_train, data_holder.y_train,
                                           data_holder.X_val, data_holder.y_val)

        self.evaluator = Evaluator(self.model)
        test_results = self.evaluator.evaluate(data_holder.X_test, data_holder.y_test, return_features=True)

        if self.config.save_plots:
            self._create_plots(training_state, test_results)

        results = {
            'test_accuracy': test_results.accuracy,
            'training_state': training_state,
            'test_results': test_results
        }

        return results
    
    def _create_plots(self, training_state: TrainingState, test_results: EvaluationResult):
        """Create plots using existing plotter."""

        part_id_column = self.data_processor.df['fPartID']
        encoded_ids = self.data_processor.label_encoder.fit_transform(part_id_column.unique())
        id_to_encoded_id_map = {k: v for k, v in zip(part_id_column.unique(), encoded_ids)}
        
        from utils.pid_routine import PARTICLE_ID
        encoded_id_to_name_map = {}
        for name, id in PARTICLE_ID.items():
            encoded_id = id_to_encoded_id_map.get(id, None)
            if encoded_id is not None:
                encoded_id_to_name_map[encoded_id] = name
        
        class_names = list(encoded_id_to_name_map.values())
        
        #self.plotter.plot_training_history(
        #    training_state.train_losses, training_state.val_losses,
        #    training_state.train_accuracies, training_state.val_accuracies
        #)
        
        self.plotter.plot_confusion_matrix(
            test_results.labels, test_results.predictions, class_names
        )
        
        output_file = TFile(str(self.config.output_dir) + '/results'+ str(self.config.output_file_suffix) +'.root', 'RECREATE')
        self.plotter.plot_model_scores(
            test_results, class_names, output_file, self.config.momentum_feature_idx
        )
        
        self.plotter.plot_efficiency_purity_curves_momentum_bins(
            test_results, class_names, self.config.momentum_feature_idx, output_file
        )

        self.plotter.analyze_feature_importance_flexible(
            self.model, test_results, self.data_processor.feature_columns, 
            encoded_id_to_name_map, output_file
        )
        
        output_file.Close()