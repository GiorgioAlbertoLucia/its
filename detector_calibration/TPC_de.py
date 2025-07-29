'''
    Code to run calibration of ITS and TPC parametrisations
'''

import numpy as np
import pandas as pd
from ROOT import TFile, TCanvas, TF1
from ROOT import RooRealVar, RooCrystalBall, RooAddPdf, RooGaussian

from particle import Particle

from torchic import Dataset, AxisSpec, RooGausExp
from torchic.physics import BetheBloch, py_BetheBloch
from torchic.core.graph import create_graph

from calibration_utils import calibration_fit_slice, initialize_means_and_sigmas
import sys
sys.path.append('..')
from utils.pid_routine import standard_selections, define_variables

DATASET_COLUMN_NAMES = {
    'P': 'fTPCInnerParam',
    'Pt': 'fPt',
    'Eta': 'fEta',
    'TpcSignal': 'fTPCsignal',
    'Chi2TPC': 'fTPCchi2',
    'ItsClusterSize': 'fITSclusterSizes',
}

BETAGAMMA_MIN = 0.6
BETAGAMMA_MAX = 4.0

def init_signal_roofit(nsigma_tpc: RooRealVar, function: str = 'crystalball'):

    if function == 'crystalball':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', 200., 0., 500., ''),
            'sigma': RooRealVar('sigma', 'sigma', 10, 1000, ''),
            'aL': RooRealVar('aL', 'aL', 0.7, 30.),
            'nL': RooRealVar('nL', 'nL', 0.3, 30.),
            'aR': RooRealVar('aR', 'aR', 0.7, 30.),
            'nR': RooRealVar('nR', 'nR', 0.3, 30.),
        }
        signal = RooCrystalBall('signal', 'signal', nsigma_tpc, signal_pars['mean'], signal_pars['sigma'],
                                signal_pars['aL'], signal_pars['nL'], doubleSided=True) #
                                #signal_pars['aR'], signal_pars['nR'])

        return signal, signal_pars
    
    elif function == 'gausexp':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', 200., 0., 500., ''),
            'sigma': RooRealVar('sigma', 'sigma', 10, 1000, ''),
            'rlife': RooRealVar('rlife', 'rlife', 0., 10.),
        }
        signal = RooGausExp('signal', 'signal', nsigma_tpc, *signal_pars.values())
        return signal, signal_pars
    
    elif function == 'gaus':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', 200., 0., 500., ''),
            'sigma': RooRealVar('sigma', 'sigma', 10, 1000, ''),
        }
        signal = RooGaussian('signal', 'signal', nsigma_tpc, *signal_pars.values())
        return signal, signal_pars
    
    else:
        raise ValueError(f'Unknown function: {function}. Supported functions are "crystalball" and "gausexp".')

def init_background_roofit(nsigma_tpc: RooRealVar, function: str = 'gaus'):

    if function == 'gausexp':
        bkg_pars = {
            'mean': RooRealVar('bkg_mean', 'bkg_mean', 0., 0., 500., ''),
            'sigma': RooRealVar('bkg_sigma', 'bkg_sigma', 60, 20., 100, ''),
            'rlife': RooRealVar('bkg_rlife', 'bkg_rlife', 0., 10.),
        }
        bkg_pdf = RooGausExp('bkg', 'bkg', nsigma_tpc, *bkg_pars.values())
        return bkg_pdf, bkg_pars

    elif function == 'gaus':
        bkg_pars = {
            'mean': RooRealVar('bkg_mean', 'bkg_mean', 0., 0., 500., ''),
            'sigma': RooRealVar('bkg_sigma', 'bkg_sigma', 60, 20., 100, ''),
        }
        bkg_pdf = RooGaussian('bkg', 'bkg', nsigma_tpc, *bkg_pars.values())
        return bkg_pdf, bkg_pars
    
    else: 
        raise ValueError(f'Unknown function: {function}. Supported functions are "gausexp" and "gaus".')

def fit_parametrisation(fit_results: pd.DataFrame, sign: str, outfile: TFile):

    g_mean = create_graph(fit_results, 'bg', 'mean', 'bg_err', 'mean_err', 
                            f'g_mean_{sign}', ';#beta#gamma;#LT #mathrm{d}E/#mathrm{d}x #GT (GeV/#it{c}^{2})')
    
    f_mean = TF1('f_mean', BetheBloch, BETAGAMMA_MIN, BETAGAMMA_MAX, 5)
    f_mean.SetParameters(-136.71, 0.441, 0.2269, 1.347, 0.8035)
    g_mean.Fit(f_mean, 'RMS+')
    fit_params = [f_mean.GetParameter(i) for i in range(5)]
    
    g_resolution = create_graph(fit_results, 'bg', 'resolution', 'bg_err', 'resolution_err', 
                                f'g_resolution_{sign}', ';#beta#gamma;#sigma_{TPC} / #LT #mathrm{d}E/#mathrm{d}x #GT')

    BETAGAMMA_MIN_RES, BETAGAMMA_MAX_RES = 0., 3
    f_resolution = TF1('f_resolution', '[0]', BETAGAMMA_MIN_RES, BETAGAMMA_MAX_RES)
    f_resolution.SetParameters(0.07)
    g_resolution.Fit(f_resolution, 'RMS+')
        
    outfile.cd()
    g_mean.Write()
    g_resolution.Write()

    return fit_params, f_resolution.GetParameter(0)

def TPC_calibration(dataset: Dataset, outfile:TFile, column_names:dict=DATASET_COLUMN_NAMES):

    axis_spec_betagamma = AxisSpec(160, -8, 8, 'beta_gamma', ';#beta#gamma;dE/dx (a.u.)')
    axis_spec_tpcsignal = AxisSpec(100, 0, 500, 'tpc_signal', ';#beta#gamma;dE/dx (a.u.)')
    h2_tpc = dataset.build_th2('fBetaGamma', column_names['TpcSignal'], axis_spec_betagamma, axis_spec_tpcsignal)
    
    tpc_signal = RooRealVar('fSignalTPC', 'dE/dx (a.u.)', 0., 500.)
    signal_pdf, signal_pars = init_signal_roofit(tpc_signal, function='gaus')

    fit_params, model = None, None

    for sign in ['matter', 'antimatter']:

        if sign == 'matter':
            slice_range = [BETAGAMMA_MIN, BETAGAMMA_MAX]
        else:
            slice_range = [-BETAGAMMA_MAX, -BETAGAMMA_MIN]

        fit_results = []

        tpc_dir = outfile.mkdir(f'TPC_{sign}')

        bg_bin_min = h2_tpc.GetXaxis().FindBin(slice_range[0])
        bg_bin_max = h2_tpc.GetXaxis().FindBin(slice_range[1])
        bg_step = h2_tpc.GetXaxis().GetBinWidth(1)

        for bg_bin in range(bg_bin_min, bg_bin_max):
            
            bg = h2_tpc.GetXaxis().GetBinCenter(bg_bin)
            bg_low_edge = h2_tpc.GetXaxis().GetBinLowEdge(bg_bin)
            bg_high_edge = h2_tpc.GetXaxis().GetBinLowEdge(bg_bin+1)
            
            h_tpc = h2_tpc.ProjectionY(f'tpc_signal_{bg:.2f}', bg_bin, bg_bin, 'e')
            
            means, sigmas = initialize_means_and_sigmas(h_tpc, 1)
            signal_pars['mean'].setVal(means[0][0])
            signal_pars['sigma'].setVal(np.sqrt(float(sigmas[0])))
            model = signal_pdf

            iframe, ifit_result = calibration_fit_slice(model, h_tpc, tpc_signal, signal_pars, bg_low_edge, bg_high_edge)
            ifit_result['bg'] = np.abs(bg)
            ifit_result['bg_err'] = bg_step / 2
            fit_results.append(ifit_result)

            canvas = TCanvas(f'TPCfit_{bg:.2f}', f'#beta#gamma = {bg:.2f}', 200, 600)
            iframe.Draw()
            tpc_dir.cd()
            canvas.Write()
        fit_results_df = pd.DataFrame(fit_results)
        
        fit_params, resolution = fit_parametrisation(fit_results_df, sign, tpc_dir)

    return fit_params, resolution

def visualize_distributions_and_fit(dataset: Dataset, outfile: TFile, fit_params:list, resolution:float, column_names:dict=DATASET_COLUMN_NAMES):

    np_bethe_bloch = np.vectorize(py_BetheBloch)

    dataset['fExpTpcSignal'] = np_bethe_bloch(np.abs(dataset['fBetaGamma']), *fit_params)
    dataset['fNSigmaTPC'] = (dataset[column_names['TpcSignal']] - dataset['fExpTpcSignal']) / (dataset['fExpTpcSignal'] * resolution)

    axis_spec_betagamma = AxisSpec(160, -8, 8, 'beta_gamma', ';#beta#gamma;dE/dx (a.u.)')
    axis_spec_tpcsignal = AxisSpec(100, 0, 500, 'tpc_signal', ';#beta#gamma;#mathrm{d}E/#mathrm{d}x (a.u.)')
    axis_spec_nsigmatpc = AxisSpec(100, -5, 5, 'nsigma_tpc', ';#beta#gamma;n#sigma_{TPC}')
    axis_spec_clsize = AxisSpec(90, 0, 15., 'cl_size', ';#beta#gamma;#LT ITS cluster size (a.u.)#GT #times #LT cos#lambda#GT')

    h2_nsigmatpc = dataset.build_th2('fBetaGamma', 'fNSigmaTPC', axis_spec_betagamma, axis_spec_nsigmatpc)
    h2_exptpc = dataset.build_th2('fBetaGamma', 'fExpTpcSignal', axis_spec_betagamma, axis_spec_tpcsignal)
    h2_tpc = dataset.build_th2('fBetaGamma', column_names['TpcSignal'], axis_spec_betagamma, axis_spec_tpcsignal)
    h2_clsize = dataset.build_th2('fBetaGamma', 'fClSizeCosLam', axis_spec_betagamma, axis_spec_clsize)

    f_fit_matter = TF1('f_fit_matter', BetheBloch, BETAGAMMA_MIN, BETAGAMMA_MAX, 5)
    f_fit_matter.SetParameters(*fit_params[:5])
    def BetheBlochAntimatter(betagamma, *params):
        return BetheBloch(-betagamma, *params)
    f_fit_antimatter = TF1('f_fit_antimatter', BetheBlochAntimatter, -BETAGAMMA_MAX, -BETAGAMMA_MIN, 5)
    f_fit_antimatter.SetParameters(*fit_params[:5])

    outfile.cd()
    
    h2_tpc.Write()
    h2_nsigmatpc.Write()
    h2_exptpc.Write('exp_tpc_signal')
    h2_clsize.Write('h2PtClSizeCosLamMean')
    
    canvas = TCanvas('cNSigmaTPC', 'cNSigmaTPC', 200, 600)
    h2_tpc.Draw('colz')
    f_fit_matter.Draw('same')
    f_fit_antimatter.Draw('same')
    canvas.Write()


if __name__ == '__main__':

    
    infile_path = '/data/galucia/its_pid/calibration/de_nucleispectra_24pass4.root'
    #infile_path = '/data/galucia/lithium_local/same/LHC24as_pass1_same.root'
    folder_name = 'DF*'
    tree_name = 'O2nucleitable'
    column_names = {key: value for key, value in DATASET_COLUMN_NAMES.items()}
    dataset = Dataset.from_root(infile_path, tree_name, folder_name, columns=[col for col in column_names.values()])

    dataset.eval(f'fBetaGamma = {column_names["Pt"]}/abs({column_names["Pt"]}) * \
                 {column_names["P"]} / {Particle.from_name("D2").mass / 1_000}', 
                 inplace=True)
    
    define_variables(dataset, DATASET_COLUMN_NAMES=DATASET_COLUMN_NAMES)
    standard_selections(dataset, particle='De', DATASET_COLUMN_NAMES=DATASET_COLUMN_NAMES)
    outfile = TFile('output/TPC_de.root', 'recreate')

    fit_params, resolution = TPC_calibration(dataset, outfile, column_names)
    #fit_params = [-136.71, 0.441, 0.2269, 1.347, 0.8035]
    visualize_distributions_and_fit(dataset, outfile, fit_params, resolution, column_names)

    outfile.Close()
    