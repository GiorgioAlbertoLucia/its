'''
    QA plots for the particles used in the neural network.
'''
import pandas as pd
from typing import List

from ROOT import TFile, TDirectory

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent.parent))
from core.processor import DataProcessor
from core.config import ExperimentConfig
from utils.pid_routine import PARTICLE_ID

from histogram_archive import register_qa_histograms
from histogram_registry import HistogramRegistry

def load_data(input_files: List[str], tree_names: List[str], folder_name: str, columns: List[str]):
    
    config = ExperimentConfig(
        input_files=input_files,
        add_statistics_features=True,
        add_parametrised_features=True,
        balance_classes=False,
    )

    data_processor = DataProcessor(config)
    data_processor.load_raw_data(input_files, tree_names, folder_name, columns)
    data_processor.downsample_data()
    data_processor.engineer_features()
    if config.balance_classes:
        data_processor.balance_classes()

    return data_processor.df

def visual_single_particle(df:pd.DataFrame, particles:List[str], output_file:TDirectory):

    histogram_registry = HistogramRegistry()
    register_qa_histograms(histogram_registry)

    for particle in particles:
        pid = PARTICLE_ID[particle]
        particle_df = df[df['fPartID'] == pid]
        particle_dir = output_file.mkdir(particle)
        
        histogram_registry.draw_histogram(particle_df)
        histogram_registry.save_histograms(particle_dir)

if __name__ == "__main__":

    input_files = '/data/galucia/its_pid/LHC24_pass1_skimmed/data_04_08_2025.root'
    folder_name = 'DF*'
    tree_names = ['O2clsttable', 'O2clsttableextra']
    columns = [
        'fP', 'fEta', 'fPhi', 'fItsClusterSize', 'fPartID', 'fTpcNSigma'
    ]
    df = load_data(input_files, tree_names, folder_name, columns)

    output_path = 'output/single_particle_qa.root'
    output_file = TFile(output_path, "RECREATE")
    particles = ['Pi', 'Ka', 'Pr', 'De', 'He']
    visual_single_particle(df, particles, output_file)
    
    output_file.Close()