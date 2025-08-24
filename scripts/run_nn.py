"""Clean entry point."""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.config import ExperimentConfig
from core.nn.pipeline import Pipeline


def main():
    """Main execution function."""
    config = ExperimentConfig(
        input_files=['/data/galucia/its_pid/LHC24_pass1_skimmed/data_04_08_2025.root'],
        tree_names=['O2clsttable', 'O2clsttableextra'],
        folder_name='DF*',
        columns=['fP', 'fEta', 'fPhi', 'fItsClusterSize', 'fPartID'],
        model_type='MomentumGatedPID',
        hidden_dims=[64, 32, 16],
        data_fraction=0.3,
        balance_classes=True,
        num_epochs=100,
        output_dir=Path("../output/nn")
    )
    
    pipeline = Pipeline(config)
    results = pipeline.run()
    
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    return results

if __name__ == "__main__":
    results = main()