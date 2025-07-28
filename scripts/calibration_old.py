'''
    Code to run calibration of ITS and TPC parametrisations.
    NOTE: This uses the old definition of the ITS cluster size (no truncated mean)!!!!!!!!
'''

import numpy as np
import pandas as pd
from ROOT import TFile, TF1, TCanvas, gInterpreter
from ROOT import RooRealVar, RooCrystalBall, RooGaussian, RooAddPdf
from torchic import Dataset, AxisSpec

import yaml

from torchic.physics.ITS import average_cluster_size, expected_cluster_size, sigma_its

import importlib.resources
try:
    with importlib.resources.path("torchic.core.RooCustomPdfs", "RooGausExp.cxx") as roo_gaus_exp_path:
        ROOGAUSEXP_PATH = str(roo_gaus_exp_path)
except ModuleNotFoundError:
    raise ImportError("The torchic package or the RooCustomPdfs module is not properly installed.")

# Include the file in ROOT's interpreter
gInterpreter.ProcessLine(f'#include "{ROOGAUSEXP_PATH}"')
from ROOT import RooGausExp

import sys
sys.path.append('..')
from utils.particles import ParticleMasses
from utils.utils import create_graph, calibration_fit_slice

CONF = {
    'Pi': {
        'bg_min': 1.5,
        'bg_max': 4,
    },
    'Ka': {
        'bg_min': 0.7,
        'bg_max': 4,
    },
    'Pr': {
        'bg_min': 0.5,
        'bg_max': 3.5,
    },
    'De': {
        'bg_min': 0.45,
        'bg_max': 1.9,
    },
    'He': {
        'bg_min': 0.7,
        'bg_max': 3.5,
    }
}

PARTICLE_ID = {
    'Un': 0,    # undefined
    'El': 1,
    'Pi': 2,
    'Ka': 3,
    'Pr': 4,
    'De': 5,
    'He': 6,
}

PDG_CODE = {
    'Pi': 211,
    'Ka': 321,
    'Pr': 2212,
    'De': 1000010020,
    'He': 1000020030,
}

def init_signal_pdf(x: RooRealVar, opt: str):
    '''
        Initialize the signal pdf.
    '''
    signal, signal_pars = None, None

    if opt == 'cb':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', 0., 15, ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.6, 0.01, 10, ''),
            'aL': RooRealVar('aL', 'aL', 1.7, 0.7, 30.),
            'nL': RooRealVar('nL', 'nL', 1.7, 0.3, 30.),
            'aR': RooRealVar('aR', 'aR', 1.7, 0.7, 30.),
            'nR': RooRealVar('nR', 'nR', 1.7, 0.3, 30.),
        }
        signal = RooCrystalBall('signal', 'signal', x, signal_pars['mean'], signal_pars['sigma'],
                                signal_pars['aL'], signal_pars['nL']) #, doubleSided=True) #
                                #signal_pars['aR'], signal_pars['nR'])
    elif opt == 'gausexp':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', 2., 15, ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.6, 0.1, 3, ''),
            'tau': RooRealVar('tau', 'tau', -4, -8, -1.),
        }
        signal = RooGausExp('signal', 'signal', x, signal_pars['mean'], signal_pars['sigma'], signal_pars['tau'])
    else:
        raise ValueError(f'Unknown signal pdf option: {opt}')
    
    return signal, signal_pars

N_ITS_LAYERS = 7
def average_cluster_size_old(cluster_sizes: pd.Series) -> tuple:
    '''
        Compute the average cluster size. A truncated mean will be used to avoid the presence of outliers.
    '''
    
    np_cluster_sizes = cluster_sizes.to_numpy()
    avg_cluster_size = np.zeros(len(np_cluster_sizes))
    max_cluster_size = 0
    n_hits = np.zeros(len(np_cluster_sizes))
    for ilayer in range(N_ITS_LAYERS):
        cluster_size_layer = np.right_shift(np_cluster_sizes, 4*ilayer) & 0b1111
        avg_cluster_size += cluster_size_layer
        n_hits += (cluster_size_layer > 0).astype(int)
        max_cluster_size = np.maximum(max_cluster_size, cluster_size_layer)
    
    avg_cluster_size = (avg_cluster_size - max_cluster_size) / (n_hits - 1)
    # avg_cluster_size /= n_hits

    return avg_cluster_size, n_hits

def calibration_routine(dataset: Dataset, outfile: TFile, params_file_path: str, particle: str):

    cfg = CONF[particle]

    dataset['fAvgClusterSize'], dataset['fNHitsIts'] = average_cluster_size_old(dataset['fItsClusterSize'])
    dataset.query('fNHitsIts > 5', inplace=True)
    dataset['fAvgClSizeCosLam'] = dataset['fAvgClusterSize'] / np.cosh(dataset['fEta'])
    dataset['fBetaGamma'] = abs(dataset['fP']) / ParticleMasses[particle]

    axis_spec_bg = AxisSpec(50, 0, 5, 'bg', ';;')
    axis_spec_clsize = AxisSpec(60, 0, 15, 'cluster_size_cal', ';#beta#gamma;#LT ITS Cluster Size #GT #times cos #LT #lambda #GT')
    axis_spec_nsigma = AxisSpec(100, -5, 5, 'nsigma', ';#beta#gamma;n#sigma_{ITS}')

    particle_dir = outfile.mkdir(particle)
    h2_clsize = dataset.build_th2('fBetaGamma', 'fAvgClSizeCosLam', axis_spec_bg, axis_spec_clsize)

    # RooFit initialization
    clsize = RooRealVar('fClSizeCosLam', '#LT Cluster size #GT #LT cos#lambda #GT', 0., 15.)
    signal, signal_pars = init_signal_pdf(clsize, 'gausexp')

    bkg_pars = {
        'bkg_mean': RooRealVar('bkg_mean', 'bkg_mean', 0., 1, ''),
        'bkg_sigma': RooRealVar('bkg_sigma', 'bkg_sigma', 0.1, 0.8, ''),
        #'rlife': RooRealVar('rlife', 'rlife', 0., 10.),
    }
    if particle == 'He':
        bkg_pars['bkg_mean'] = RooRealVar('bkg_mean', 'bkg_mean', 0., 3, '')
    bkg = RooGaussian('bkg', 'bkg', clsize, bkg_pars['bkg_mean'], bkg_pars['bkg_sigma'])

    bg_min = cfg['bg_min']
    bg_max = cfg['bg_max']

    fit_results_df = None

    bg_bin_min = h2_clsize.GetXaxis().FindBin(bg_min)
    bg_bin_max = h2_clsize.GetXaxis().FindBin(bg_max)
    for bg_bin in range(bg_bin_min, bg_bin_max+1):
        
        bg = h2_clsize.GetXaxis().GetBinCenter(bg_bin)
        bg_error = h2_clsize.GetXaxis().GetBinWidth(bg_bin) / 2.
        bg_low_edge = h2_clsize.GetXaxis().GetBinLowEdge(bg_bin)
        bg_high_edge = h2_clsize.GetXaxis().GetBinLowEdge(bg_bin+1)
        
        model = None
        if (particle == 'He'): # and bg < 2.5):
            sig_frac = RooRealVar('sig_frac', 'sig_frac', 0.5, 0., 1.)
            model = RooAddPdf('model', 'signal + bkg', [signal, bkg], [sig_frac])
        else:
            model = signal

        h_clsize = h2_clsize.ProjectionY(f'clsize_{bg:.2f}', bg_bin, bg_bin, 'e')
        frame, fit_results = calibration_fit_slice(model, h_clsize, clsize, signal_pars, bg_low_edge, bg_high_edge)
        fit_results['bg'] = np.abs(bg)
        fit_results['bg_error'] = bg_error
        if fit_results_df is None:
            fit_results_df = pd.DataFrame.from_dict([fit_results])
        else:
            fit_results_df = pd.concat([fit_results_df, pd.DataFrame.from_dict([fit_results])], ignore_index=True)

        canvas = TCanvas(f'cClSizeCosLam_{bg:.2f}', f'cClSizeCosLam_{bg:.2f}', 800, 600)
        frame.Draw()
        particle_dir.cd()
        canvas.Write()

    g_mean = create_graph(fit_results_df, 'bg', 'mean', 'bg_error', 'mean_err', 
                                f'g_mean', ';#beta#gamma;#LT ITS Cluster Size #GT #times cos #LT #lambda #GT')
    f_mean = TF1('simil_bethe_bloch_func', '[0]/x^[1] + [2]', bg_min, bg_max)
    f_mean.SetParameters(2.6, 2., 5.5)
    g_mean.Fit(f_mean, 'RMS+')
    c_mean = TCanvas('c_mean', 'c_mean', 800, 600)
    g_mean.Draw('ap')
    f_mean.Draw('same')

    g_resolution = create_graph(fit_results_df, 'bg', 'resolution', 'bg_error', 'resolution_err', 
                                f'g_resolution', ';#beta#gamma;#sigma / #mu')
    f_resolution = TF1('resolution_fit', '[0]*ROOT::Math::erf((x - [1])/[2])', bg_min, bg_max)
    f_resolution.SetParameters(0.24, -0.32, 1.53)
    if particle == 'He':    
        f_resolution.SetParameters(0.11, -0.32, 10)
        f_resolution = TF1('resolution_fit', '[0] + x*[1] + x^2*[2]', bg_min, bg_max)
        f_resolution.SetParameters(0.116, -0.01/1.7)
        f_resolution.FixParameter(2, 0)
    g_resolution.Fit(f_resolution, 'RMS+')
    c_resolution = TCanvas('c_resolution', 'c_resolution', 800, 600)
    g_resolution.Draw('ap')
    f_resolution.Draw('same')
    
    pid_params = (f_mean.GetParameter(0), f_mean.GetParameter(1), f_mean.GetParameter(2),
                  f_resolution.GetParameter(0), f_resolution.GetParameter(1), f_resolution.GetParameter(2))
    dataset['fExpClSizeCosLam'] = expected_cluster_size(dataset['fBetaGamma'], pid_params)
    if particle == 'He':    dataset['fSigmaITS'] = dataset['fExpClSizeCosLam'] * (pid_params[3] + pid_params[4]*dataset['fBetaGamma'])
    else:                   dataset['fSigmaITS'] = sigma_its(dataset['fBetaGamma'], pid_params)
    dataset['fNSigmaITS'] = (dataset['fAvgClSizeCosLam'] - dataset['fExpClSizeCosLam']) / dataset['fSigmaITS']
    h2_nsigma = dataset.build_th2('fBetaGamma', 'fNSigmaITS', axis_spec_bg, axis_spec_nsigma)

    pid_params_df = pd.DataFrame.from_dict([{'particle': particle,
                                             'kp0': pid_params[0],
                                             'kp1': pid_params[1],
                                             'kp2': pid_params[2],
                                             'res0': pid_params[3],
                                             'res1': pid_params[4],
                                             'res2': pid_params[5],}])
    pid_params_df.to_csv(params_file_path, mode='a', float_format='%.3f', header=False)

    particle_dir.cd()
    c_mean.Write()
    c_resolution.Write()
    h2_clsize.Write()
    h2_nsigma.Write()


if __name__ == '__main__':

    #infile_path = '/Users/glucia/Projects/ALICE/data/its_pid/LHC22o_pass7_minBias_small.root'
    #infile_path = '/Users/glucia/Projects/ALICE/data/its_pid/LHC24f3c.root'
    infile_path = ['/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root',
                   '/data/galucia/its_pid/pass7/LHC22o_pass7_minBias_small.root',]

    dataset = Dataset.from_root(infile_path, tree_name='O2clsttable', folder_name='DF*', columns=['fP', 'fEta', 'fItsClusterSize', 'fPartID']) #,'fPartIDMc'

    infile_path_he = ['/data/galucia/its_pid/LHC23_pass4_skimmed/LHC23_pass4_skimmed.root',]
    dataset = Dataset.concat(dataset, 
                             Dataset.from_root(infile_path_he, tree_name='O2clsttableextra', folder_name='DF*', 
                                               columns=['fP', 'fEta', 'fItsClusterSize', 'fPartID'])) #,'fPartIDMc'


    #outfile_path = f'/Users/glucia/Projects/ALICE/its/output/MC/LHC25a3_calibration.root'
    outfile_path = f'../output/data/calibration.root'
    outfile = TFile(outfile_path, 'RECREATE')
    
    #params_file_path = f'/Users/glucia/Projects/ALICE/its/output/MC/LHC25a3_calibration.csv'
    params_file_path = f'../output/data/calibration.csv'
    pid_params_df = pd.DataFrame.from_dict([{'particle': '-',
                                             'kp0': '-',
                                             'kp1': '-',
                                             'kp2': '-',
                                             'res0': '-',
                                             'res1': '-',
                                             'res2': '-',}])
    pid_params_df.to_csv(params_file_path, mode='w', float_format='%.3f')
    
    particles = ['Pi', 'Ka', 'Pr', 'De', 'He']
    for particle in particles:
        tmp_dataset = dataset.query(f'fPartID == {PARTICLE_ID[particle]}', inplace=False)
        #tmp_dataset.query(f'abs(fPartIDMc) == {PDG_CODE[particle]}', inplace=True)
        if particle == 'He':
            tmp_dataset['fP'] = tmp_dataset['fP'] * 2
        calibration_routine(tmp_dataset, outfile, params_file_path, particle)
        del tmp_dataset

    outfile.Close()