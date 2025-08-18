'''
    Code to run calibration of ITS and TPC parametrisations
'''

import numpy as np
from ROOT import TCanvas, gStyle
from particle import Particle

from torchic import Dataset, AxisSpec
from torchic.physics.ITS import average_cluster_size, expected_cluster_size, sigma_its

import sys
sys.path.append('..')
from utils.pid_routine import PARTICLE_ID, PDG_CODE

DATASET_COLUMN_NAMES = {
    'P':'fP',
    'Pt':'fPt',
    'Eta':'fEta',
    'Phi':'fPhi',
    'TofNSigma':'fTofNSigma',
    'Chi2TPC':'fChi2TPC',
    'PIDtracking': 'fPIDinTrk',
    'ItsClusterSize':'fItsClusterSize',
}

X_DICT = {
    'beta_gamma': {
        'axis_name': 'bg',
        'axis_title': '#beta#gamma',
        'var_name': 'fBetaGamma',
    },
    'p': {
        'axis_name': 'p',
        'axis_title': '#it{p} (GeV/c)',
        'var_name': 'fP',
    }
}

FIT_PARAMS = {
    'Z=1': {
        'mean': [0.9883, 1.894, 1.950],
        'resolution': [0.2083, -0.3125, 1.347]
    },
    'Z=2': {
        'mean': [2.172, 1.872, 4.699],
        'resolution': [0.1466, -0.0246, 0],
    }
}


def prepare_dataset(dataset: Dataset, particle: str):

    dataset['fItsClusterSize'] = np.array(dataset['fItsClusterSize'], np.uint64)
    dataset['fAvgClSizeCosLam'], dataset['fNHitsIts'] = np.zeros(dataset.shape[0], dtype=float), np.zeros(dataset.shape[0], dtype=int)
    dataset['fAvgClusterSize'], dataset['fNHitsIts'] = average_cluster_size(dataset['fItsClusterSize'])

    dataset.query('fNHitsIts > 5', inplace=True)
    dataset['fAvgClSizeCosLam'] = dataset['fAvgClusterSize'] / np.cosh(dataset['fEta'])
    dataset['fBetaGamma'] = abs(dataset['fP']) / (Particle.from_pdgid(PDG_CODE[particle]).mass / 1_000)
    dataset['fP'] = np.abs(dataset['fP'])


def visualize_nsigma(dataset, particle, pdf_file_path:str, x:str):

    x_dict = X_DICT[x]

    charge = 'Z=2' if particle == 'He' else 'Z=1'
    pid_params = (FIT_PARAMS[charge]['mean'] + FIT_PARAMS[charge]['resolution'])

    dataset['fExpClSizeCosLam'] = expected_cluster_size(dataset['fBetaGamma'], pid_params)
    if particle == 'He':    dataset['fSigmaITS'] = dataset['fExpClSizeCosLam'] * (pid_params[3] + pid_params[4]*dataset['fBetaGamma'])
    else:                   dataset['fSigmaITS'] = sigma_its(dataset['fBetaGamma'], pid_params)
    dataset['fNSigmaITS'] = (dataset['fAvgClSizeCosLam'] - dataset['fExpClSizeCosLam']) / dataset['fSigmaITS']

    axis_spec_bg = AxisSpec(50, 0, 5, x_dict['axis_name'], ';;')
    axis_spec_nsigma = AxisSpec(100, -5, 5, 'nsigma', 
                                f'{Particle.from_pdgid(PDG_CODE[particle]).latex_name.replace('\\', '#')};\
                                {x_dict["axis_title"]};n#sigma_{{ITS}};')
    h2_nsigma = dataset.build_th2('fBetaGamma', 'fNSigmaITS', axis_spec_bg, axis_spec_nsigma)

    canvas = TCanvas(f'c_{particle}_nsigma_{x}', '', 800, 600)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    canvas.SetBottomMargin(0.15)
    canvas.SetTopMargin(0.15)

    h2_nsigma.Draw('colz')
    h2_nsigma.SetTitle(f'{Particle.from_pdgid(PDG_CODE[particle]).latex_name.replace("\\", "#")};\
        {x_dict["axis_title"]};n#sigma_{{ITS}};')
    canvas.Print(pdf_file_path)

    
def main_routine(dataset: Dataset, x: str):

    pdf_file_path = f'/home/galucia/its/plots/output/LHC24_pass1_skimmed_nsigmaITS_{x}.pdf'
    blank_canvas = TCanvas('c_blank', '', 800, 600)
    blank_canvas.Print(pdf_file_path + '(')

    gStyle.SetOptStat(0)
    
    particles = ['Pi', 'Ka', 'Pr', 'De', 'He']
    for particle in particles:
        
        tmp_dataset = dataset.query(f'fPartID == {PARTICLE_ID[particle]}', inplace=False)
        #tmp_dataset.query(f'abs(fPartIDMc) == {PDG_CODE[particle]}', inplace=True)
        if particle == 'He':
            tmp_dataset['fP'] = tmp_dataset['fP'] * 2
        
        prepare_dataset(tmp_dataset, particle)
        visualize_nsigma(tmp_dataset, particle, pdf_file_path=pdf_file_path, x=x)
        del tmp_dataset
    
    blank_canvas.Print(pdf_file_path + ')')



if __name__ == '__main__':

    #infile_path = '/Users/glucia/Projects/ALICE/data/its_pid/LHC22o_pass7_minBias_small.root'
    #infile_path = '/Users/glucia/Projects/ALICE/data/its_pid/LHC24f3c.root'
    #infile_path = '/Users/glucia/Projects/ALICE/data/its_pid/LHC25a3.root'
    infile_path = '/data/galucia/its_pid/LHC24_pass1_skimmed/data_04_08_2025.root'

    folder_name = 'DF*'
    tree_names = ['O2clsttable', 'O2clsttableextra']
    columns = ['fP', 'fEta', 'fPhi', 'fItsClusterSize', 'fPartID', 'fPTPC',
               'fPIDinTrk', 'fTpcNSigma', 'fTofNSigma', 'fTofMass', 'fCosPAMother',
               'fMassMother']
    datasets = []
    
    for tree_name in tree_names:
        datasets.append(Dataset.from_root(infile_path, tree_name=tree_name, folder_name=folder_name)) #, columns=columns))
    dataset = datasets[0].concat(datasets[1], ignore_index=True)
    print(f'{dataset.shape=}\n{dataset.columns=}')

    for x in ['beta_gamma']:
        main_routine(dataset, x=x)