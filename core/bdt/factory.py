from typing import Union
import torch.nn as nn
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import ExperimentConfig
from core.bdt.models import MomentumAwareBDT, BaseBDTModel, MomentumEnsembleBDT

class ModelFactory:
    """Factory for creating both NN and BDT models."""
    
    @staticmethod
    def create_model(config: ExperimentConfig, input_dim: int, 
                    num_classes: int) -> Union[nn.Module, BaseBDTModel]:
        """Create model based on config."""
        
        if config.model_type == 'MomentumAwareBDT':
            xgb_params = {
                'objective': config.bdt_config.objective,
                'eval_metric': config.bdt_config.eval_metric,
                'learning_rate': config.bdt_config.learning_rate,
                'n_estimators': config.bdt_config.n_estimators,
                'max_depth': config.bdt_config.max_depth,
                'min_child_weight': config.bdt_config.min_child_weight,
                'gamma': config.bdt_config.gamma,
                'subsample': config.bdt_config.subsample,
                'colsample_bytree': config.bdt_config.colsample_bytree,
                'reg_alpha': config.bdt_config.reg_alpha,
                'reg_lambda': config.bdt_config.reg_lambda,
                'early_stopping_rounds': config.bdt_config.early_stopping_rounds,
                'n_jobs': config.bdt_config.n_jobs,
                'verbosity': config.bdt_config.verbosity,
                'random_state': config.bdt_config.random_state,
            }
            return MomentumAwareBDT(
                momentum_feature_idx=config.momentum_feature_idx,
                xgb_params=xgb_params
            )
        
        elif config.model_type == 'MomentumEnsembledBDT':
            xgb_params = {
                'objective': config.bdt_config.objective,
                'eval_metric': config.bdt_config.eval_metric,
                'learning_rate': config.bdt_config.learning_rate,
                'n_estimators': config.bdt_config.n_estimators,
                'max_depth': config.bdt_config.max_depth,
                'min_child_weight': config.bdt_config.min_child_weight,
                'gamma': config.bdt_config.gamma,
                'subsample': config.bdt_config.subsample,
                'colsample_bytree': config.bdt_config.colsample_bytree,
                'reg_alpha': config.bdt_config.reg_alpha,
                'reg_lambda': config.bdt_config.reg_lambda,
                'early_stopping_rounds': config.bdt_config.early_stopping_rounds,
                'n_jobs': config.bdt_config.n_jobs,
                'verbosity': config.bdt_config.verbosity,
                'random_state': config.bdt_config.random_state,
            }
            return MomentumEnsembleBDT(
                momentum_feature_idx=config.momentum_feature_idx,
                xgb_params=xgb_params,
                momentum_bins=config.momentum_bins
            )
        
        else:
            raise ValueError(f"Unsupported model type: {config.model_type}")
        