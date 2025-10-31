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
        tree_names=['O2clsttable'],
        folder_name='DF*',
        columns=['fP', 'fEta', 'fPhi', 'fItsClusterSize', 'fPartID'],
        model_type='MomentumGatedPID',
        hidden_dims=[64, 32, 16],
        data_fraction=0.3,
        balance_classes=True,
        num_epochs=100,
        output_dir=Path("../output/nn"),
        output_file_suffix="_trial2",
        feature_columns=['fPAbs', 'fEta', 'fPhi', #'fCosL', 'fPt',
            'fItsClusterSizeL0', 'fItsClusterSizeL1', 
            'fItsClusterSizeL2',
            'fItsClusterSizeL3', 'fItsClusterSizeL4', 'fItsClusterSizeL5',
            'fItsClusterSizeL6', 'fMeanItsClSize', 'fClSizeCosL',
            'fClusterSizeStd', 'fClusterSizeSkew', 
            'fTotalClusterSize', 
            'fClusterSizeRange',
            #'fNSigmaItsPi', 'fNSigmaItsKa', 'fNSigmaItsPr',
            #'fNSigmaItsDe', 'fNSigmaItsHe',
            #'fExpectedClusterSizePi', 'fExpectedClusterSizeKa',
            #'fExpectedClusterSizePr', 'fExpectedClusterSizeDe', 'fExpectedClusterSizeHe',
            #'fSigmaItsPi', 'fSigmaItsKa', 'fSigmaItsPr', 'fSigmaItsDe', 'fSigmaItsHe',
        ],
        min_hits_required=7,
        run_shap_analysis=True,
    )
    
    pipeline = Pipeline(config)
    results = pipeline.run()
    
    print(f"Test Accuracy: {results['test_accuracy']:.2f}%")
    return results

if __name__ == "__main__":
    results = main()