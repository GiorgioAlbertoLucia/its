"""Clean entry point."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import ExperimentConfig, BDTConfig
from core.bdt.pipeline import Pipeline

def main():
    """Simple experiment runner for both NN and BDT."""
    
    # BDT configuration
    config = ExperimentConfig(
        input_files=['/data/galucia/its_pid/LHC24_pass1_skimmed/data_04_08_2025.root'],
        tree_names=['O2clsttable'],
        folder_name='DF*',
        columns=['fP', 'fEta', 'fPhi', 'fItsClusterSize', 'fPartID'],
        
        # Data settings
        data_fraction=0.3,
        balance_classes=True,
        add_statistics_features=True,
        add_parametrised_features=True,
        
        # Model selection
        model_type='MomentumAwareBDT',  # or 'PidFCNN' for NN
        #model_type='MomentumEnsembledBDT',
        
        # BDT settings
        bdt_config=BDTConfig(
            learning_rate=0.1,
            n_estimators=100,
            max_depth=6,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            early_stopping_rounds=20,
            n_jobs=30,
            verbosity=0
        ),
        
        # Output
        output_dir=Path("../output/bdt"),
        #output_file_suffix="_bdt_ensemble",
        output_file_suffix="_momentum_aware",
        save_plots=True
    )
    
    # Run pipeline
    pipeline = Pipeline(config)
    results = pipeline.run()
    
    print(f"Experiment completed. Test accuracy: {results.get('test_accuracy', 'N/A'):.2f}%")

if __name__ == "__main__":
    main()