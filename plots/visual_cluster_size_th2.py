import numpy as np
import pandas as pd
from scipy.special import erf
from ROOT import TCanvas, gStyle, TF1, \
    kOrange, kGreen, kRed, kBlack, kAzure, kViolet, TLatex
from particle import Particle

from torchic import Dataset, AxisSpec
from torchic.physics.ITS import average_cluster_size, expected_cluster_size, sigma_its

import sys
sys.path.append('..')
from utils.pid_routine import PARTICLE_ID, PDG_CODE, LATEX_PARTICLE
from utils.utils import set_root_object
from utils.plot_utils import get_alice_watermark, init_legend

TCOLOR_PARTICLE = {
    'Pi': kGreen,
    'Ka': 4,  # 
    'Pr': kRed,
    'Xi': kOrange, # Blue
    'Omega': kViolet+6,
    'De': kAzure+1,
    'He': 9, # Ocean Blue
}

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

PARTICLE_LATEX = {
    'Pi': '#pi',
    'Ka': 'K',
    'Pr': 'p',
    'De': 'd',
    'He': '^{3}He'
}

def define_nsigma(dataset: Dataset, particle: str):

    charge = 'Z=2' if particle == 'He' else 'Z=1'
    pid_params = (FIT_PARAMS[charge]['mean'] + FIT_PARAMS[charge]['resolution'])

    # Calculate nsigma with the given mass (betagamma) hypothesis
    dataset[f'fBetaGamma{particle}'] = np.abs(dataset['fP']) / (Particle.from_pdgid(PDG_CODE[particle]).mass / 1_000)
    if charge == 'Z=2':
        dataset[f'fBetaGamma{particle}'] = dataset[f'fBetaGamma{particle}'] * 2

    dataset[f'fExpClSizeCosLam{particle}'] = expected_cluster_size(dataset[f'fBetaGamma{particle}'], pid_params)
    if particle == 'He':    dataset[f'fSigmaITS{particle}'] = dataset[f'fExpClSizeCosLam{particle}'] * (pid_params[3] + pid_params[4]*dataset['fBetaGamma'])
    else:                   dataset[f'fSigmaITS{particle}'] = sigma_its(dataset[f'fBetaGamma{particle}'], pid_params)
    dataset[f'fNSigmaITS{particle}'] = (dataset['fAvgClSizeCosLam'] - dataset[f'fExpClSizeCosLam{particle}']) / dataset[f'fSigmaITS{particle}']

def visualize_cluster_size(dataset: Dataset, particle, pdf_file_path:str):

    particle_list = ['Pi', 'Ka', 'Pr', 'De', 'He']

    canvas = TCanvas(f'c', '', 800, 600)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    canvas.SetBottomMargin(0.15)

    axis_spec_x = AxisSpec(50, 0, 5, 'rigidity', ';;')
    axis_spec_cluster_size = AxisSpec(180, 0, 15, 'AvgClSizeCosLam', ';#it{p}/|#it{Z}| (GeV/#it{c});#LT ITS cluster size#GT #times #LT cos#lambda#GT;')
    h2_cluster_size = dataset.build_th2('fP', f'fAvgClSizeCosLam', axis_spec_x, axis_spec_cluster_size,
                                        title=';#it{p}/|#it{Z}| (GeV/#it{c});#LT ITS Cluster size#GT #times #LT cos#lambda#GT;')
    h2_cluster_size.GetXaxis().SetTitleSize(0.05)
    h2_cluster_size.GetYaxis().SetTitleSize(0.05)
    
    watermark = get_alice_watermark(0.5, 0.7, 0.8, 0.85)

    canvas.cd()
    h2_cluster_size.Draw('col')
    watermark.Draw('same')
    canvas.SetLogz()
    canvas.Print(pdf_file_path)

    pi_text = TLatex(0.19, 0.28, '#bf{#pi}')
    ka_text = TLatex(0.22, 0.32, '#bf{K}')
    pr_text = TLatex(0.25, 0.38, '#bf{p}')
    de_text = TLatex(0.33, 0.4, '#bf{d}')
    he_text = TLatex(0.33, 0.53, '#bf{^{3}He}')
    for text in [pi_text, ka_text, pr_text, de_text, he_text]:
        text.SetNDC()
        text.SetTextSize(0.05)
        text.SetTextColor(kBlack)
        text.Draw('same')
    canvas.Print(pdf_file_path)

    funcs, pyfuncs = [], []

    def bethe_bloch_3p(mass, particle, xs, pars):
        x = xs[0] / mass
        x = 2*x if particle == 'He' else x
        return pars[0] * (1.0 / (x**pars[1])) + pars[2]
    def factory_bethe_bloch_3p(particle):
        mass = Particle.from_pdgid(PDG_CODE[particle]).mass / 1_000
        def bethe_bloch_3p_particle(xs, pars):
            return bethe_bloch_3p(mass, particle, xs, pars)
        return bethe_bloch_3p_particle

    for iter, iparticle in enumerate(particle_list):

            pyfuncs.append(factory_bethe_bloch_3p(iparticle))
            
            func = TF1(f'3p_bethe_bloch_{iparticle}', pyfuncs[iter], 0, 5, 3)
            if iparticle == 'He':
                func.SetParameters(*FIT_PARAMS['Z=2']['mean'])
            else:
                func.SetParameters(*FIT_PARAMS['Z=1']['mean'])

            set_root_object(func, line_color=kBlack, line_width=2, line_style=2)
            funcs.append(func)

    canvas.cd()
    for func in funcs:
        func.Draw('same l')
    canvas.Print(pdf_file_path)

    canvas.Clear()
    canvas.SetLogz(0)
    
    momentum_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0, 10]
    for imomentum in range(len(momentum_bins)-1):

        dataset_momentum_bin = dataset.query(f'fP >= {momentum_bins[imomentum]} and fP < {momentum_bins[imomentum+1]}', inplace=False)
        if dataset_momentum_bin.shape[0] == 0:
            continue

        legend = init_legend(0.55, 0.65, 0.85, 0.85)

        h_particles = []
        for iparticle in particle_list:
            dataset_momentum_bin_particle:Dataset = dataset_momentum_bin.query(f'fPartID == {PARTICLE_ID[iparticle]}', inplace=False)
            h_particle = dataset_momentum_bin_particle.build_th1('fAvgClSizeCosLam', axis_spec_cluster_size,
                                                                 title=f'{momentum_bins[imomentum]} #leq |#it{{p}}|/#it{{Z}} < {momentum_bins[imomentum+1]};#LT ITS Cluster size#GT #times #LT cos#lambda#GT;')
            h_particle.SetName(f'h_{iparticle}_momentum_bin_{imomentum}')
            set_root_object(h_particle, line_color=TCOLOR_PARTICLE[iparticle], line_width=1, fill_style=3244,
                            fill_color_alpha=(TCOLOR_PARTICLE[iparticle], 0.5))
            if h_particle.GetEntries() == 0:
                continue
            h_particles.append(h_particle)
            legend.AddEntry(h_particle, PARTICLE_LATEX[iparticle], 'lf')
        
        canvas.cd()
        for ih, h in enumerate(h_particles):
            if ih == 0:
                h.DrawNormalized('hist')
                h.SetTitle(f'{momentum_bins[imomentum]} #leq |#it{{p}}|/#it{{Z}} < {momentum_bins[imomentum+1]};#LT ITS Cluster size#GT #times #LT cos#lambda#GT;')
            else:
                h.DrawNormalized('hist same')
        legend.Draw()
        canvas.Print(pdf_file_path)
        canvas.Clear()

        del h_particles, legend, dataset_momentum_bin

def bethe_bloch_3p(xs, pars, particle, particle_hypo=None):
    '''
    3-parameter Bethe-Bloch function
    pars[0], pars[1], pars[2] = calibration parameters
    '''
    mass = Particle.from_pdgid(PDG_CODE[particle]).mass / 1_000
    x = xs[0] / mass
    #x = 2*x if particle == 'He' else x
    return pars[0] * (1.0 / (x**pars[1])) + pars[2] 

def resolution_its(xs, pars, particle_hypo):
    '''
    ITS resolution function
    pars[0], pars[1], pars[2] = calibration parameters
    '''
    
    mass = Particle.from_pdgid(PDG_CODE[particle_hypo]).mass / 1_000
    x = xs[0] / mass
    #x = 2*x if particle_hypo == 'He' else x
    if particle_hypo == 'He':
        return pars[0] + pars[1]*x + pars[2]*x*x
    return pars[0] * erf((x - pars[1]) / pars[2])

def make_fit_func(iparticle, jparticle):
    def fit_func(xs, pars):
        bb_hyp_jparticle = bethe_bloch_3p(xs, pars, jparticle, iparticle)
        bb_hyp_iparticle = bethe_bloch_3p(xs, pars, iparticle, iparticle)
        res_hyp_iparticle = resolution_its(xs, [pars[3], pars[4], pars[5]], iparticle)
        return (bb_hyp_jparticle - bb_hyp_iparticle) / (res_hyp_iparticle * bb_hyp_iparticle)
    return fit_func

def visualize_nsigma(dataset: Dataset, pdf_file_path:str):

    #particle_dir = outdir.mkdir(f'{particle}')
    particle_list = ['Pi', 'Ka', 'Pr', 'De', 'He']
    axis_spec_x = AxisSpec(50, 0, 5, 'p', ';;')

    iparticle_yrange = {
        'Pi': (100, -5, 15),
        'Ka': (100, -7, 13),
        'Pr': (110, -10, 12),
        'De': (100, -9, 11),
        'He': (80, -10, 6),
    }

    for iparticle in particle_list:

        axis_spec_nsigma = AxisSpec(*iparticle_yrange[iparticle], 'nsigma', 
                                    ';#it{{p}}/#it{{Z}} (GeV/#it{{c}});n#sigma_{{ITS}} ({LATEX_PARTICLE[iparticle]});')
        h2_nsigma = dataset.build_th2('fP', f'fNSigmaITS{iparticle}', axis_spec_x, axis_spec_nsigma)

        funcs = []
        for jiter, jparticle in enumerate(particle_list):
            
            
            fit_func = make_fit_func(iparticle, jparticle)
            
            func = TF1(f'fit_nsigma_its_{jparticle}_as_{iparticle}', fit_func, 0, 5, 6)
            if iparticle == 'He':
                func.SetParameters(*(FIT_PARAMS['Z=2']['mean']+FIT_PARAMS['Z=2']['resolution']))
            else:
                func.SetParameters(*(FIT_PARAMS['Z=1']['mean']+FIT_PARAMS['Z=1']['resolution']))
            
            #evals = [func.Eval(x) for x in np.linspace(0.1, 5, 49)]
            #print(f'nsigma ITS {jparticle} as {iparticle}:', evals)

            set_root_object(func, line_color=kBlack, line_width=2, line_style=2)
            funcs.append(func)

        canvas = TCanvas(f'c_nsigma_{iparticle}', '', 800, 600)
        canvas.SetLeftMargin(0.15)
        canvas.SetRightMargin(0.15)
        canvas.SetBottomMargin(0.15)
        canvas.SetTopMargin(0.15)
        canvas.SetLogz()

        h2_nsigma.Draw('colz')
        h2_nsigma.SetTitle(f';#it{{p}}/#it{{Z}} (GeV/#it{{c}});n#sigma_{{ITS}} ({LATEX_PARTICLE[iparticle]});')
        for func in funcs:
            func.Draw('same l')
        canvas.Print(pdf_file_path)

        #particle_dir.cd()
        #h2_nsigma.Write(f'nsigma_its_{iparticle}_vs_{x}')
        del h2_nsigma, canvas, funcs

def visualise_cluster_size_he3(dataset: Dataset, pdf_file_path:str):

    axis_spec_nsigma = AxisSpec(100, -2.5, 2.5, 'nsigma', 
                                f';#it{{p}}/#it{{Z}} (GeV/#it{{c}});n#sigma_{{TPC}} (^{{3}}He);')
    axis_spec_clsize = AxisSpec(180, 0, 15, 'AvgClSizeCosLam', ';n#sigma_{{TPC}} (^{{3}}He);#LT ITS Cluster size#GT #times #LT cos#lambda#GT;')

    dataset_he3 = dataset.query(f'fPartID == {PARTICLE_ID["He"]}', inplace=False)
    h2_nsigma = dataset_he3.build_th2('fTpcNSigma', f'fAvgClSizeCosLam', axis_spec_nsigma, axis_spec_clsize,
                                    title=';n#sigma_{TPC} (^3He);#LT ITS Cluster size#GT #times #LT cos#lambda#GT;')
    
    canvas = TCanvas(f'c_he3', '', 800, 600)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    canvas.SetBottomMargin(0.15)
    canvas.SetTopMargin(0.15)
    canvas.SetLogz()

    he3_text = TLatex(0.2, 0.8, '#bf{^{3}He}')
    he3_text.SetNDC()
    he3_text.SetTextSize(0.05)
    he3_text.SetTextColor(0)

    h2_nsigma.Draw('colz')
    h2_nsigma.SetTitle(';n#sigma_{TPC} (^3He);#LT ITS Cluster size#GT #times #LT cos#lambda#GT;')
    watermark = get_alice_watermark(0.15, 0.7, 0.35, 0.85)
    watermark.Draw('same')
    he3_text.Draw('same')
    canvas.Print(pdf_file_path)
    del h2_nsigma, canvas
    
            



def main_routine(dataset: Dataset):

    gStyle.SetOptStat(0)

    pdf_file_path = f'/home/galucia/its/plots/output/LHC24_pass1_skimmed_cluster_size.pdf'
    
    blank_canvas = TCanvas('c_blank', '', 800, 600)
    blank_canvas.Print(pdf_file_path + '(')

    prepare_dataset(dataset)
    
    visualise_cluster_size_he3(dataset, pdf_file_path)
    
    particles = ['Pi', 'Ka', 'Pr', 'De', 'He']
    for particle in particles:
        define_nsigma(dataset, particle)
    dataset.query(f'fNSigmaITSHe > -1.5 or fPartID != {PARTICLE_ID["He"]}', inplace=True)

    visualize_cluster_size(dataset, particles, pdf_file_path)
    visualize_nsigma(dataset, pdf_file_path)

    blank_canvas.Print(pdf_file_path + ')')
    del dataset
    del blank_canvas

if __name__ == '__main__':
    input_files = '/data/galucia/its_pid/LHC24_pass1_skimmed/data_04_08_2025.root'
    folder_name = 'DF*'
    tree_names = ['O2clsttable', 'O2clsttableextra']
    multicols = [
        ['fP', 'fEta', 'fPhi', 'fItsClusterSize', 'fPartID'],
        ['fTpcNSigma']
    ]
    
    datasets = []
    for tree_name, columns in zip(tree_names, multicols):
        datasets.append(Dataset.from_root(input_files, tree_name, folder_name, columns))
    dataset = datasets[0].concat(datasets[1:])

    main_routine(dataset)
