'''
    Code to run calibration of ITS and TPC parametrisations
'''

import numpy as np
import pandas as pd
from ROOT import TCanvas, gStyle, TGraph, kBlue, TFile, TDirectory, TLegend, TF1, \
    kOrange, kGreen, kRed, kMagenta, kYellow, kBlack, kAzure, kViolet
from particle import Particle

from torchic import Dataset, AxisSpec
from torchic.physics.ITS import average_cluster_size, expected_cluster_size, sigma_its

import sys
sys.path.append('..')
from utils.pid_routine import PARTICLE_ID, PDG_CODE, LATEX_PARTICLE

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
        'mean': [1.023, 1.963, 2.208],
        'resolution': [0.187, -1.015, 1.833]
    },
    'Z=2': {
        'mean': [2.383, 1.710, 4.951],
        'resolution': [0.133, -0.004, 0],
    }
}

def bethe_bloch_3p(xs, pars, mass):
    '''
    3-parameter Bethe-Bloch function
    pars[0], pars[1], pars[2] = calibration parameters
    '''
    x = xs[0] / mass
    return pars[0] * (1.0 / (x**pars[1])) + pars[2] 

def make_fit_func(imass, jmass):
    def fit_func(x, pars):
        bb_hyp_jparticle = bethe_bloch_3p(x, pars, jmass)
        bb_hyp_iparticle = bethe_bloch_3p(x, pars, imass)
        return (bb_hyp_jparticle - bb_hyp_iparticle)
    return fit_func


def prepare_dataset(dataset: Dataset):

    dataset['fItsClusterSize'] = np.array(dataset['fItsClusterSize'], np.uint64)
    dataset['fAvgClSizeCosLam'], dataset['fNHitsIts'] = np.zeros(dataset.shape[0], dtype=float), np.zeros(dataset.shape[0], dtype=int)
    dataset['fAvgClusterSize'], dataset['fNHitsIts'] = average_cluster_size(dataset['fItsClusterSize'], do_truncated=False)

    dataset.query('fNHitsIts > 5', inplace=True)
    dataset['fAvgClSizeCosLam'] = dataset['fAvgClusterSize'] / np.cosh(dataset['fEta'])

    dataset['fBetaGamma'] = np.zeros(dataset.shape[0], dtype=float)
    particles = [key for key, idx in PARTICLE_ID.items() if idx in dataset['fPartID'].unique()]
    for particle in particles:
        dataset.loc[dataset['fPartID'] == PARTICLE_ID[particle], 'fBetaGamma'] = \
            np.abs(dataset['fP']) / (Particle.from_pdgid(PDG_CODE[particle]).mass / 1_000)
    dataset['fP'] = np.abs(dataset['fP'])

def define_nsigma(dataset: Dataset, particle: str):

    charge = 'Z=2' if particle == 'He' else 'Z=1'
    pid_params = (FIT_PARAMS[charge]['mean'] + FIT_PARAMS[charge]['resolution'])

    # Calculate nsigma with the given mass (betagamma) hypothesis
    dataset[f'fBetaGamma{particle}'] = np.abs(dataset['fP']) / (Particle.from_pdgid(PDG_CODE[particle]).mass / 1_000)
    if charge == 'Z=2':
        dataset[f'fBetaGamma{particle}'] = dataset[f'fBetaGamma{particle}'] * 2

    dataset[f'fExpClSizeCosLam{particle}'] = expected_cluster_size(dataset[f'fBetaGamma{particle}'], pid_params)
    if particle == 'He':    dataset[f'fSigmaITS{particle}'] = dataset[f'fExpClSizeCosLam{particle}'] * (pid_params[3] + pid_params[4]*dataset[f'fBetaGamma{particle}'])
    else:                   dataset[f'fSigmaITS{particle}'] = sigma_its(dataset[f'fBetaGamma{particle}'], pid_params)
    dataset[f'fNSigmaITS{particle}'] = (dataset['fAvgClSizeCosLam'] - dataset[f'fExpClSizeCosLam{particle}']) / dataset[f'fSigmaITS{particle}']

def sample_dataset(dataset: Dataset) -> Dataset:
    '''
    Sample the dataset to have balanced number of particles
    '''
    min_count = np.inf
    particles = [key for key, idx in PARTICLE_ID.items() if idx in dataset['fPartID'].unique()]
    for particle in particles:
        count = dataset.query(f'fPartID == {PARTICLE_ID[particle]}', inplace=False).shape[0]
        if count < min_count:
            min_count = count
    print(f'Sampling dataset to have {min_count} samples for each particle species')
    
    sampled_dfs = []
    for particle in particles:
        particle_df = dataset.query(f'fPartID == {PARTICLE_ID[particle]}', inplace=False)
        sampled_dfs.append(particle_df.data.sample(n=min_count, random_state=42))
    
    return Dataset(pd.concat(sampled_dfs, ignore_index=True))

def visualize_expected_cluster_size(dataset, particle, outdir:TDirectory):
    
    particle_dir = outdir.mkdir(particle)
    particle_list = ['Pi', 'Ka', 'Pr', 'De', 'He']
    axis_spec_x = AxisSpec(50, 0, 5, 'p', '#it{p} (GeV/c);')

    for iparticle in particle_list:

        axis_spec_y = AxisSpec(100, 0, 20, 'Expected Cluster Size', f';#LT ITS Cluster Size #GT #LT cos#lambda #GT ({LATEX_PARTICLE[iparticle]});')
        h2_exp_cl_size = dataset.build_th2('fP', f'fExpClSizeCosLam{iparticle}', axis_spec_x, axis_spec_y)

        h2_exp_cl_size.Draw('colz')
        h2_exp_cl_size.SetTitle(f'{LATEX_PARTICLE[particle]} - {LATEX_PARTICLE[iparticle]} hypothesis;#it{{p}} (GeV/#it{{c}});Expected #LT ITS Cluster Size #GT #LT cos#lambda #GT ({LATEX_PARTICLE[iparticle]});')

        particle_dir.cd()
        h2_exp_cl_size.Write(f'expected_cluster_size_{iparticle}_hypothesis')

        del h2_exp_cl_size

def visualize_nsigma(dataset, particle, outdir:TDirectory, pdf_file_path:str, x:str):

    x_dict = X_DICT[x]

    particle_dir = outdir.mkdir(f'{particle}')
    particle_list = ['Pi', 'Ka', 'Pr', 'De', 'He']
    axis_spec_x = AxisSpec(50, 0, 5, x_dict['axis_name'], ';;')

    for iparticle in particle_list:

        axis_spec_nsigma = AxisSpec(100, -5, 5, 'nsigma', 
                                    f'{LATEX_PARTICLE[particle]};\
                                    {x_dict["axis_title"]};n#sigma_{{ITS}} ({LATEX_PARTICLE[iparticle]});')
        h2_nsigma = dataset.build_th2(x_dict["var_name"], f'fNSigmaITS{iparticle}', axis_spec_x, axis_spec_nsigma)

        imass = Particle.from_pdgid(PDG_CODE[iparticle]).mass / 1_000
        mass = Particle.from_pdgid(PDG_CODE[particle]).mass / 1_000
        fit_func = make_fit_func(imass, mass)
        
        func = TF1(f'fit_nsigma_its_{particle}_as_{iparticle}', fit_func, 0, 5, 3)
        if iparticle == 'He':
            func.SetParameters(*FIT_PARAMS['Z=2']['mean'])
        else:
            func.SetParameters(*FIT_PARAMS['Z=1']['mean'])
        
        func.SetLineColor(kBlack)
        func.SetLineWidth(2)
            
        canvas = TCanvas(f'c_{particle}_nsigma_{iparticle}_{x}', '', 800, 600)
        canvas.SetLeftMargin(0.15)
        canvas.SetRightMargin(0.15)
        canvas.SetBottomMargin(0.15)
        canvas.SetTopMargin(0.15)

        h2_nsigma.Draw('colz')
        h2_nsigma.SetTitle(f'{LATEX_PARTICLE[particle]};\
            {x_dict["axis_title"]};n#sigma_{{ITS}} ({LATEX_PARTICLE[iparticle]});')
        func.Draw('same l')
        canvas.Print(pdf_file_path)

        particle_dir.cd()
        h2_nsigma.Write(f'nsigma_its_{iparticle}_vs_{x}')
        del h2_nsigma, canvas

def visualize_roc_curve_nsigma(dataset: Dataset, particle: str, outfile:TDirectory, pdf_file_path:str, x:str):
    
    momentum_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0, 10]
    canvas = TCanvas(f'c_{particle}_roc_curve_nsigma_{x}', '', 800, 600)

    outdir = outfile.mkdir(f'{particle}')

    for bin_idx in range(len(momentum_bins)-1):

        thresholds = np.linspace(-5, 5, 50)
        graph = TGraph(50)
        momentum_dataset = dataset.query(f'fP >= {momentum_bins[bin_idx]} and fP < {momentum_bins[bin_idx+1]}', inplace=False)
        
        if momentum_dataset.shape[0] < 10:
            print(f'Skipping bin {bin_idx} for {particle} due to insufficient statistics ({momentum_dataset.shape[0]} samples)')
            continue
        if momentum_dataset.query(f'fPartID == {PARTICLE_ID[particle]}', inplace=False).shape[0] < 5:
            print(f'Skipping bin {bin_idx} for {particle} due to insufficient signal statistics ({momentum_dataset.query(f"fPartID == {PARTICLE_ID[particle]}", inplace=False).shape[0]} signal samples)')
            continue

        for ithreshold, threshold in enumerate(thresholds):

            # True Positive, False Positive, False Negative
            tp = momentum_dataset.query(f'abs(fNSigmaITS{particle}) < {threshold} and fPartID == {PARTICLE_ID[particle]}', inplace=False).shape[0]
            fp = momentum_dataset.query(f'abs(fNSigmaITS{particle}) < {threshold} and fPartID != {PARTICLE_ID[particle]}', inplace=False).shape[0]
            fn = momentum_dataset.query(f'abs(fNSigmaITS{particle}) >= {threshold} and fPartID == {PARTICLE_ID[particle]}', inplace=False).shape[0]

            efficiency = tp / (tp + fn) if (tp + fn) > 0 else 0
            purity = tp / (tp + fp) if (tp + fp) > 0 else (1.0 if tp == 0 else 0.0)
            graph.SetPoint(ithreshold, efficiency, purity)
        
        graph.SetTitle(f'Efficiency vs Purity for {Particle.from_pdgid(PDG_CODE[particle]).latex_name.replace("\\", "#")} - #it{{p}}=[{momentum_bins[bin_idx]}, {momentum_bins[bin_idx+1]}) GeV/#it{{c}};Efficiency;Purity')
        graph.SetLineColor(kBlue + bin_idx % 6)
        graph.SetLineWidth(2)
        graph.SetMarkerColor(kBlue + bin_idx % 6)
        graph.SetMarkerStyle(20)
        graph.SetMarkerSize(0.5)
        
        canvas.cd()
        graph.Draw('ALP')
        canvas.Print(pdf_file_path)
        canvas.Clear()

        outdir.cd()
        graph.Write(f'efficiency_purity_curve_bin_{bin_idx}')

        del graph, momentum_dataset


    
def main_routine(dataset: Dataset, x: str):

    gStyle.SetOptStat(0)

    pdf_file_path = f'/home/galucia/its/plots/output/LHC24_pass1_skimmed_nsigmaITS_{x}.pdf'
    pdf_file_path_roc = f'/home/galucia/its/plots/output/LHC24_pass1_skimmed_roc_curve_nsigmaITS_{x}.pdf'
    
    output_file = TFile(pdf_file_path_roc.replace('.pdf', '.root'), 'RECREATE')
    roc_outdir = output_file.mkdir('EfficiencyPurity')
    nsigma_outdir = output_file.mkdir('NSigmaITS')
    expected_cluster_size_outdir = output_file.mkdir('ExpectedClusterSize')

    blank_canvas = TCanvas('c_blank', '', 800, 600)
    blank_canvas.Print(pdf_file_path + '(')
    blank_canvas.Print(pdf_file_path_roc + '(')

    prepare_dataset(dataset)
    roc_dataset = sample_dataset(dataset)
    print(f'{dataset.shape=}, {roc_dataset.shape=}')
    
    particles = ['Pi', 'Ka', 'Pr', 'De', 'He']
    for particle in particles:
        define_nsigma(dataset, particle)
        define_nsigma(roc_dataset, particle)
    roc_dataset.query(f'fNSigmaITSHe > -1.5 or fPartID != {PARTICLE_ID["He"]}', inplace=True)
    dataset.query(f'fNSigmaITSHe > -1.5 or fPartID != {PARTICLE_ID["He"]}', inplace=True)
    
    for particle in particles:
        
        dataset.loc[dataset['fPartID'] == PARTICLE_ID['He'], 'fP'] = dataset['fP'] * 2
        visualize_roc_curve_nsigma(roc_dataset, particle, roc_outdir, pdf_file_path=pdf_file_path_roc, x=x)
        dataset.loc[dataset['fPartID'] == PARTICLE_ID['He'], 'fP'] = dataset['fP'] / 2
        
        tmp_dataset = dataset.query(f'fPartID == {PARTICLE_ID[particle]}', inplace=False)
        #tmp_dataset.query(f'abs(fPartIDMc) == {PDG_CODE[particle]}', inplace=True)
        if particle == 'He':
            tmp_dataset['fP'] = tmp_dataset['fP'] * 2
        
        visualize_expected_cluster_size(tmp_dataset, particle, expected_cluster_size_outdir)
        visualize_nsigma(tmp_dataset, particle, nsigma_outdir, pdf_file_path=pdf_file_path, x=x)
        del tmp_dataset
    
    blank_canvas.Print(pdf_file_path + ')')
    blank_canvas.Print(pdf_file_path_roc + ')')



if __name__ == '__main__':

    #infile_path = '/Users/glucia/Projects/ALICE/data/its_pid/LHC22o_pass7_minBias_small.root'
    #infile_path = '/Users/glucia/Projects/ALICE/data/its_pid/LHC24f3c.root'
    #infile_path = '/Users/glucia/Projects/ALICE/data/its_pid/LHC25a3.root'
    infile_path = '/data/galucia/its_pid/LHC24_pass1_skimmed/data_04_08_2025.root'

    folder_name = 'DF*'
    tree_names = ['O2clsttable', 'O2clsttableextra']
    columns = ['fP', 'fEta', 'fItsClusterSize', 'fPartID']
    datasets = []
    
    for tree_name in tree_names:
        datasets.append(Dataset.from_root(infile_path, tree_name=tree_name, folder_name=folder_name, columns=columns))
    dataset = datasets[0].concat(datasets[1], ignore_index=True)
    print(f'{dataset.shape=}\n{dataset.columns=}')

    for x in ['p']:
        main_routine(dataset, x=x)