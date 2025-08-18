'''
    Code to run calibration of ITS and TPC parametrisations
'''

import numpy as np
import pandas as pd
from ROOT import TFile, TCanvas, TF1, TPaveText
from ROOT import RooRealVar, RooCrystalBall, RooAddPdf, RooGaussian, RooExponential, RooGenericPdf

from particle import Particle

from torchic import Dataset, AxisSpec
from torchic.roopdf import RooGausExp
from torchic.physics import BetheBloch
from torchic.core.graph import create_graph

import sys
sys.path.append('..')
from utils.pid_routine import standard_selections, define_variables
from utils.utils import calibration_fit_slice, initialize_means_and_covariances

DATASET_COLUMN_NAMES = {
    'P': 'fP',
    'TpcInnerParam': 'fTPCInnerParam',
    'Pt': 'fPt',
    'Eta': 'fEta',
    'TpcSignal': 'fTPCsignal',
    'TofMass': 'fTOFmass',
    'TofBeta': 'fBeta',
    'Chi2TPC': 'fTPCchi2',
    'ItsClusterSize': 'fITSclusterSizes',
    'Flags': 'fFlags',
}

PT_MIN = 0.7
PT_MAX = 5
PT_BKG_MIN = 0.7

def init_signal_roofit(nsigma_tof: RooRealVar, function: str = 'crystalball'):

    if function == 'crystalball':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', Particle.from_name('D2').mass / 1_000, 1.5, 2.5, ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.05, 0.01, 0.2, ''),
            'aL': RooRealVar('aL', 'aL', 0.7, 30.),
            'nL': RooRealVar('nL', 'nL', 0.3, 30.),
            'aR': RooRealVar('aR', 'aR', 0.7, 30.),
            'nR': RooRealVar('nR', 'nR', 0.3, 30.),
        }
        signal = RooCrystalBall('signal', 'signal', nsigma_tof, signal_pars['mean'], signal_pars['sigma'],
                                signal_pars['aL'], signal_pars['nL'], doubleSided=True) #
                                #signal_pars['aR'], signal_pars['nR'])

        return signal, signal_pars
    
    elif function == 'gausexp':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', Particle.from_name('D2').mass / 1_000, 1.5, 2.5, ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.05, 0.01, 0.2, ''),
            'rlife': RooRealVar('rlife', 'rlife', 9., 0., 10.),
        }
        signal = RooGausExp('signal', 'signal', nsigma_tof, *signal_pars.values())
        return signal, signal_pars
    
    elif function == 'gaus':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', Particle.from_name('D2').mass / 1_000, 1.5, 2.5, ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.05, 0.01, 0.2, ''),
        }
        signal = RooGaussian('signal', 'signal', nsigma_tof, *signal_pars.values())
        return signal, signal_pars
    
    else:
        raise ValueError(f'Unknown function: {function}. Supported functions are "crystalball" and "gausexp".')

def init_background_roofit(nsigma_tof: RooRealVar, function: str = 'gaus'):

    if function == 'gausexp':
        bkg_pars = {
            'mean': RooRealVar('bkg_mean', 'bkg_mean', 0., 1.5, 2.5, ''),
            'sigma': RooRealVar('bkg_sigma', 'bkg_sigma', 60, 20., 100, ''),
            'rlife': RooRealVar('bkg_rlife', 'bkg_rlife', 0., 10.),
        }
        bkg_pdf = RooGausExp('bkg', 'bkg', nsigma_tof, *bkg_pars.values())
        return bkg_pdf, bkg_pars

    elif function == 'gaus':
        bkg_pars = {
            'mean': RooRealVar('bkg_mean', 'bkg_mean', 0., 1.5, 2.5, ''),
            'sigma': RooRealVar('bkg_sigma', 'bkg_sigma', 60, 20., 100, ''),
        }
        bkg_pdf = RooGaussian('bkg', 'bkg', nsigma_tof, *bkg_pars.values())
        return bkg_pdf, bkg_pars
    
    elif function == 'exp':
        bkg_pars = {
            'offset': RooRealVar('bkg_offset', 'bkg_offset', 1., 0., 10., ''),
            'slope': RooRealVar('bkg_slope', 'bkg_slope', -0.1, -2.5, 0., ''),
        }
        #bkg_pdf = RooGenericPdf('bkg', 'bkg', f'bkg_offset + exp(bkg_slope * {nsigma_tof.GetName()})', [nsigma_tof, *bkg_pars.values()])
        bkg_pdf = RooExponential('bkg', 'bkg', nsigma_tof, bkg_pars['slope'])
        return bkg_pdf, bkg_pars
    
    else: 
        raise ValueError(f'Unknown function: {function}. Supported functions are "gausexp" and "gaus".')

def fit_parametrisation(fit_results: pd.DataFrame, sign: str, outfile: TFile):

    g_mean = create_graph(fit_results, 'pt', 'mean', 'pt_err', 'mean_err', 
                            f'g_mean_{sign}', ';#it{p}_{T} (GeV/#it{c});#LT m_{TOF} #GT (GeV/#it{c}^{2})')
    
    f_mean = TF1('f_mean', '[0]', PT_MIN, PT_MAX)
    f_mean.SetParameters(Particle.from_name('D2').mass / 1_000)  # Initial guess for the mean
    g_mean.Fit(f_mean, 'RMS+')
    
    g_resolution = create_graph(fit_results, 'pt', 'resolution', 'pt_err', 'resolution_err', 
                                f'g_resolution_{sign}', ';#it{p}_{T} (GeV/#it{c});#sigma_{m_{TOF}} / #LT m_{TOF} #GT')

    PT_RES, PT_MAX_RES = PT_MIN, 3.3
    f_resolution = TF1('f_resolution', '[0]', PT_RES, PT_MAX_RES)
    f_resolution.SetParameters(0.07)
    g_resolution.Fit(f_resolution, 'RMS+')
        
    outfile.cd()
    g_mean.Write()
    g_resolution.Write()

    return f_mean.GetParameter(0), f_resolution.GetParameter(0)

def TOF_calibration(dataset: Dataset, outfile:TFile, column_names:dict=DATASET_COLUMN_NAMES):

    axis_spec_betagamma = AxisSpec(64, -8, 8, 'pt', ';#it{p}_{T} (GeV/#it{c});m_{TOF} (GeV/#it{c}^{2})')
    axis_spec_tofmass = AxisSpec(50, 1.5, 2.5, 'tof_mass', ';#it{p}_{T} (GeV/#it{c});m_{TOF} (GeV/#it{c}^{2})')
    h2_tof = dataset.build_th2(column_names['Pt'], column_names['TofMass'], axis_spec_betagamma, axis_spec_tofmass)

    outfile.cd()
    h2_tof.Write('h2PtTofMass')
    
    tof_mass = RooRealVar(column_names['TofMass'], 'm_{TOF} (GeV/#it{c}^{2})', 1.5, 2.5)
    signal_pdf, signal_pars = init_signal_roofit(tof_mass, function='gausexp')
    bkg_pdf, bkg_pars = init_background_roofit(tof_mass, function='exp')

    mass, model = None, None

    for sign in ['matter', 'antimatter']:

        if sign == 'matter':
            slice_range = [PT_MIN, PT_MAX]
        else:
            slice_range = [-PT_MAX, -PT_MIN]

        fit_results = []

        tof_dir = outfile.mkdir(f'TOF_{sign}')

        pt_bin_min = h2_tof.GetXaxis().FindBin(slice_range[0])
        pt_bin_max = h2_tof.GetXaxis().FindBin(slice_range[1])
        pt_step = h2_tof.GetXaxis().GetBinWidth(1)

        for pt_bin in range(pt_bin_min, pt_bin_max):
            
            pt = h2_tof.GetXaxis().GetBinCenter(pt_bin)
            pt_low_edge = h2_tof.GetXaxis().GetBinLowEdge(pt_bin)
            pt_high_edge = h2_tof.GetXaxis().GetBinLowEdge(pt_bin+1)
            
            h_tof = h2_tof.ProjectionY(f'tof_mass_{pt:.2f}', pt_bin, pt_bin, 'e')

            if np.abs(pt) > PT_BKG_MIN:
                sig_frac = RooRealVar('sig_frac', 'sig_frac', 0.5, 0., 1.)
                model = RooAddPdf('model', 'model', [signal_pdf, bkg_pdf], [sig_frac])
                
            else:
                model = signal_pdf

            iframe, ifit_result = calibration_fit_slice(model, h_tof, tof_mass, signal_pars, pt_low_edge, pt_high_edge)
            chi2 = iframe.chiSquare()

            ifit_result['pt'] = np.abs(pt)
            ifit_result['pt_err'] = pt_step / 2
            fit_results.append(ifit_result)

            canvas = TCanvas(f'TOFfit_{pt:.2f}', f'#it{{p}}_{{T}} = {pt:.2f} (GeV/#{{c}}^{{2}})', 800, 600)
            text = TPaveText(0.65, 0.4, 0.88, 0.5, 'ndc')
            text.SetFillColor(0)
            text.SetBorderSize(0)
            text.AddText(f'#chi^{2} / NDF = {chi2:.2f}')

            canvas.cd()
            iframe.Draw()
            text.Draw()

            tof_dir.cd()
            canvas.Write()

        fit_results_df = pd.DataFrame(fit_results)
        
        mass, resolution = fit_parametrisation(fit_results_df, sign, tof_dir)

    return mass, resolution

def visualize_distributions_and_fit(dataset: Dataset, outfile: TFile, mass:float, resolution:float, column_names:dict=DATASET_COLUMN_NAMES):

    dataset['fExpTofMass'] = Particle.from_name('D2').mass / 1_000
    dataset['fNSigmaTOF'] = (dataset[column_names['TofMass']] - dataset['fExpTofMass']) / (dataset['fExpTofMass'] * resolution)

    axis_spec_betagamma = AxisSpec(64, -8, 8, 'pt', ';#it{p}_{T} (GeV/#it{c});m_{TOF} (GeV/#it{c}^{2})')
    axis_spec_tofmass = AxisSpec(50, 1.5, 2.5, 'tof_mass', ';#it{p}_{T} (GeV/#it{c});m_{TOF} (GeV/#it{c}^{2})')
    axis_spec_nsigmatof = AxisSpec(100, -5, 5, 'nsigma_tof', ';#it{p}_{T} (GeV/#it{c});#LT m_{TOF} #GT (GeV/#it{c}^{2})')
    axis_spec_clsize = AxisSpec(90, 0, 15., 'cl_size', ';#it{p}_{T} (GeV/#it{c});#LT ITS cluster size (a.u.)#GT #times #LT cos#lambda#GT')

    h2_nsigmatof = dataset.build_th2(column_names['Pt'], 'fNSigmaTOF', axis_spec_betagamma, axis_spec_nsigmatof)
    h2_exptof = dataset.build_th2(column_names['Pt'], 'fExpTofMass', axis_spec_betagamma, axis_spec_tofmass)
    h2_tof = dataset.build_th2(column_names['Pt'], column_names['TofMass'], axis_spec_betagamma, axis_spec_tofmass)
    h2_clsize = dataset.build_th2(column_names['Pt'], 'fClSizeCosLam', axis_spec_betagamma, axis_spec_clsize)

    f_fit_matter = TF1('f_fit_matter', '[0]', PT_MIN, PT_MAX, 5)
    f_fit_matter.SetParameters(mass)
    f_fit_antimatter = TF1('f_fit_antimatter', '[0]', -PT_MAX, -PT_MIN, 5)
    f_fit_antimatter.SetParameters(mass)

    outfile.cd()
    
    h2_nsigmatof.Write()
    h2_exptof.Write('exp_tof_mass')
    h2_clsize.Write('h2PtClSizeCosLamMean')
    
    canvas = TCanvas('cNSigmaTOF', 'cNSigmaTOF', 800, 600)
    h2_tof.Draw('colz')
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
    
    #dataset[column_names['P']] = dataset[column_names['Pt']] * np.cosh(dataset[column_names['Eta']])
    #dataset[column_names['TofMass']] = np.abs(dataset[column_names['P']]) * np.sqrt(1 / (dataset[column_names['TofBeta']] ** 2) - 1)

    dataset[column_names['TofMass']] = np.abs(dataset[column_names['TpcInnerParam']]) * np.sqrt(1 / (dataset[column_names['TofBeta']] ** 2) - 1)
    print(f'{dataset.columns=}')
    readPidTracking = lambda x: (x >> 12) & 0x1F
    readPidTracking_vectorized = np.vectorize(readPidTracking) 
    dataset['fPidTracking'] = readPidTracking_vectorized(dataset['fFlags'])
    print(f'{dataset['fPidTracking'].unique()=}')

    define_variables(dataset, DATASET_COLUMN_NAMES=DATASET_COLUMN_NAMES)
    standard_selections(dataset, particle='De', DATASET_COLUMN_NAMES=DATASET_COLUMN_NAMES)
    outfile = TFile('output/TOF_de.root', 'recreate')
    
    mass, resolution = TOF_calibration(dataset, outfile, column_names)
    #fit_params = [-136.71, 0.441, 0.2269, 1.347, 0.8035]
    visualize_distributions_and_fit(dataset, outfile, mass, resolution, column_names)

    outfile.Close()
    