'''
    Code to run calibration of ITS and TPC parametrisations
'''

from typing import List
import numpy as np
import pandas as pd
from ROOT import TFile, TCanvas, TF1, TMath, TLine, kDashed, TDirectory, TPaveText, TLegend
from ROOT import RooRealVar, RooCrystalBall, RooGaussian, RooAddPdf, RooGenericPdf, RooArgusBG,\
    RooChebychev, RooHistPdf, RooDataHist, RooFit, RooArgSet

from torchic.roopdf import RooGausExp
from torchic.core.graph import create_graph
from torchic.utils.terminal_colors import TerminalColors as tc
from torchic.utils.root import set_root_object

from particle import Particle

import sys
sys.path.append('..')
from utils.pid_routine import PDG_CODE
from utils.utils import calibration_fit_slice, initialize_means_and_covariances

CONF = {

    'Lambda': {
        'x_min_fit': 0.7,
        'x_max_fit': 5,
        'x_nbins': 120,
        #'y_min': 1.085,
        #'y_max': 1.145,
        'y_min': 1.095,
        'y_max': 1.135,
        'y_min_lazy': 1.102,
        'y_max_lazy': 1.132,
        'invariant_mass_peak': 1.1157,
        'invariant_mass_cut': 0.06,
    },
    'Omega': {
        'x_min_fit': 1.1,
        'x_max_fit': 5,
        'x_nbins': 60,
        'y_min': 1.65,
        'y_max': 1.695,
        'invariant_mass_peak': 1.6725,
        'invariant_mass_cut': 0.02,
    },
}

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

def init_signal_roofit(clsize: RooRealVar, function: str = 'crystalball', particle: str = 'Lambda'):

    if function == 'crystalball':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', 1.115, 1.10, 1.13),
            'sigma': RooRealVar('sigma', 'sigma', 0.003, 0.001, 0.02),
            'aL': RooRealVar('aL', 'aL', 1.5, 0.5, 5.0),
            'nL': RooRealVar('nL', 'nL', 2.0, 0.5, 15.0),
            'aR': RooRealVar('aR', 'aR', 1.5, 0.5, 5.),
            'nR': RooRealVar('nR', 'nR', 2.0, 0.5, 10.),
        }
        if particle == 'Omega':
            signal_pars['mean'].setVal(1.672)
            signal_pars['mean'].setRange(1.66, 1.685)
        signal = RooCrystalBall('signal', 'signal', clsize, signal_pars['mean'], signal_pars['sigma'],
                                signal_pars['aL'], signal_pars['nL'], #doubleSided=True)
                                signal_pars['aR'], signal_pars['nR'])

        return signal, signal_pars
    
    elif function == 'gausexp':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', 0., 2, ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.0001, 10, ''),
            'rlife': RooRealVar('rlife', 'rlife', 2., 0., 10.),
        }
        signal = RooGausExp('signal', 'signal', clsize, *signal_pars.values())
        return signal, signal_pars
    
    elif function == 'gaus':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', 0., 2., ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.0001, 10, ''),
        }
        signal = RooGaussian('signal', 'signal', clsize, *signal_pars.values())
        return signal, signal_pars
    
    else:
        raise ValueError(f'Unknown function: {function}. Supported functions are "crystalball" and "gausexp".')

def init_background_roofit(x: RooRealVar, function: str = 'gaus', particle: str = 'Lambda'):
    """
    Initialize background model in RooFit.
    Parameters
    ----------
    x : RooRealVar
        Variable for the background model (e.g., invariant mass)
    function : str
        Background function type. Supported: 'gausexp', 'gaus', 'exp', 'pol1', 'pol2', 'argus', 'cheb2', 'cheb1', 'cheb3', 'erf'
    """

    if function == 'gausexp':
        bkg_pars = {
            'mean': RooRealVar('bkg_mean', 'bkg_mean', 0., 1, ''),
            'sigma': RooRealVar('bkg_sigma', 'bkg_sigma', 0.1, 0.8, ''),
            'rlife': RooRealVar('bkg_rlife', 'rlife', 2., 0., 10.),
        }
        if particle == 'He':
            bkg_pars['mean'] = RooRealVar('bkg_mean', 'bkg_mean', 0., 3, '')
        bkg = RooGausExp('bkg', 'bkg', x, *bkg_pars.values())
        return bkg, bkg_pars
    
    elif function == 'gaus':
        bkg_pars = {
            'mean': RooRealVar('bkg_mean', 'bkg_mean', 0., 1, ''),
            'sigma': RooRealVar('bkg_sigma', 'bkg_sigma', 0.1, 0.8, ''),
        }
        if particle == 'He':
            bkg_pars['mean'] = RooRealVar('bkg_mean', 'bkg_mean', 0., 3, '')
        bkg = RooGaussian('bkg', 'bkg', x, bkg_pars['mean'], bkg_pars['sigma'])

        return bkg, bkg_pars
    
    elif function == 'exp':
        if particle == 'Lambda':
            bkg_pars = {
                'offset': RooRealVar('bkg_offset', 'bkg_offset', 5., 0., 20.),
                'slope': RooRealVar('bkg_slope', 'bkg_slope', -10., -100., 100),
            }
        elif particle == 'Omega':
            bkg_pars = {
                'offset': RooRealVar('bkg_offset', 'bkg_offset', 5., 0., 20.),
                'slope': RooRealVar('bkg_slope', 'bkg_slope', -5., -30., -0.1),
            }
        else:
            # Default values
            bkg_pars = {
                'offset': RooRealVar('bkg_offset', 'bkg_offset', 5., 0., 20.),
                'slope': RooRealVar('bkg_slope', 'bkg_slope', -5., -30., -0.1),
            }

        bkg = RooGenericPdf('bkg', 'bkg', f'exp(bkg_offset + bkg_slope * {x.GetName()})', [x, *bkg_pars.values()])
        return bkg, bkg_pars
    
    elif function == 'pol1':
        bkg_pars = {
            'p0': RooRealVar('bkg_p0', 'bkg_p0', 100, 20., 1e5),
            'p1': RooRealVar('bkg_p1', 'bkg_p1', 0.1, -10., 10.),
        }
        bkg = RooGenericPdf('bkg', 'bkg', f'bkg_p0 + bkg_p1 * {x.GetName()}', [x, *bkg_pars.values()])
        return bkg, bkg_pars
    
    elif function == 'pol2':
        bkg_pars = {
            'p0': RooRealVar('bkg_p0', 'bkg_p0', 0.1, -10., 10.),
            'p1': RooRealVar('bkg_p1', 'bkg_p1', 0.1, -10., 10.),
            'p2': RooRealVar('bkg_p2', 'bkg_p2', -0.1, -10., 10.),
        }
        bkg = RooGenericPdf('bkg', 'bkg', f'bkg_p0 + bkg_p1 * {x.GetName()} + bkg_p2 * {x.GetName()}^2', [x, *bkg_pars.values()])
        return bkg, bkg_pars
    
    elif function == 'argus':
        bkg_pars = {
            'm0': RooRealVar('bkg_m0', 'bkg_m0', 1.1, 0., 2.),
            'chi': RooRealVar('bkg_chi', 'bkg_chi', 0.5, 0., 6.),
            'p': RooRealVar('bkg_p', 'bkg_p', 0.5),
        }
        bkg_pars['p'].setConstant(True)
        #if particle == 'Omega':
        #    bkg_pars['m0'] = RooRealVar('bkg_m0', 'bkg_m0', 2., 1.7, 3.)
        bkg = RooArgusBG('bkg', 'bkg', x, *bkg_pars.values())
        return bkg, bkg_pars
    
    elif function == 'cheb2':
        bkg_pars = {
            'p0': RooRealVar('bkg_p0', 'bkg_p0', 0.28, 0.1, 2.),
            'p1': RooRealVar('bkg_p1', 'bkg_p1', 0.28, 0., 1.),
            'p2': RooRealVar('bkg_p2', 'bkg_p2', 0.86, 0, 1.),
        }
        bkg = RooChebychev('bkg', 'bkg', x, list(bkg_pars.values()))
        return bkg, bkg_pars
    
    elif function == 'cheb1':
        bkg_pars = {
            'p0': RooRealVar('bkg_p0', 'bkg_p0', 1., 0.1, 10.),
            'p1': RooRealVar('bkg_p1', 'bkg_p1', 0., -2., 2.),
        }
        bkg = RooChebychev('bkg', 'bkg', x, list(bkg_pars.values()))
        return bkg, bkg_pars
    
    elif function == 'cheb3':
        bkg_pars = {
            'p0': RooRealVar('bkg_p0', 'bkg_p0', 1., 0.1, 10.),
            'p1': RooRealVar('bkg_p1', 'bkg_p1', 0., -2., 2.),
            'p2': RooRealVar('bkg_p2', 'bkg_p2', 0., -1., 1.),
            'p3': RooRealVar('bkg_p3', 'bkg_p3', 0., -1., 1.),
        }
        bkg = RooChebychev('bkg', 'bkg', x, list(bkg_pars.values()))
        return bkg, bkg_pars

    else:
        raise ValueError(f'Unknown function: {function}. Supported functions are "gausexp" and "gaus".')

# SIGNAL INITIALISATION FROM MC

H2_SIG_TEMPLATE = {
    'Lambda': None,
    'Omega': None
}

def load_template_signal(particle: str):

    infile = TFile.Open(f'/data/galucia/its_pid/LHC24_pass1_skimmed/analysis_results_mc_19_09_2025.root ', 'READ')
    h_name = 'massLambdaMc' if particle == 'Lambda' else 'massOmegaMc'
    H2_SIG_TEMPLATE[particle] = infile.Get(f'lf-tree-creator-cluster-studies/LFTreeCreator/{h_name}')
    H2_SIG_TEMPLATE[particle].SetDirectory(0)
    infile.Close()

def init_signal_from_template(particle: str, ibin: int, xvar: RooRealVar, signal_pdf, signal_pars, outfile, pt_edges: List[float]): 

    if H2_SIG_TEMPLATE[particle] is None:
        load_template_signal(particle)
    
    h2_sig = H2_SIG_TEMPLATE[particle]
    h_sig = h2_sig.ProjectionY(f'h_sig_{particle}_{ibin}', ibin, ibin, 'e')
    dh_sig = RooDataHist(f'data_sig_{particle}_{ibin}', f'data_sig_{particle}_{ibin}', [xvar], Import=h_sig)
    signal_pdf.fitTo(dh_sig, PrintLevel=-1)

    frame = xvar.frame(Title=f'{pt_edges[0]:.2f} < #it{{p}}_{{T}} < {pt_edges[1]:.2f} GeV/#it{{c}}')
    dh_sig.plotOn(frame, RooFit.Name('data'))
    signal_pdf.plotOn(frame, RooFit.Name('model'), LineColor=2)
    signal_pdf.paramOn(frame)

    signal_pars['sigma'].setConstant(True)
    if 'aL' in signal_pars.keys():  signal_pars['aL'].setConstant(True)
    if 'nL' in signal_pars.keys():  signal_pars['nL'].setConstant(True)
    if 'aR'in signal_pars.keys():   signal_pars['aR'].setConstant(True)
    if 'nR' in signal_pars.keys():  signal_pars['nR'].setConstant(True)

    canvas = TCanvas(f'cInvMassSignal_{ibin:.2f}', f'cInvMassSignal_{ibin:.2f}', 800, 600)
    frame.Draw()
    outfile.cd()
    canvas.Write()

    

# BACKGROUND INITIALISATION

H2_BKG_TEMPLATE = {
    'Lambda': None,
    'Omega': None
}

def initialize_exponential_from_data(h_invmass, invmass_var, particle):
    """Initialize exponential parameters from histogram tail regions"""
    
    # Get the mass range
    x_min = invmass_var.getMin()
    x_max = invmass_var.getMax()
    
    # Find bins in the tail regions (avoiding the peak)
    if particle == 'Lambda':
        # Use sidebands: 1.09-1.105 and 1.125-1.15
        left_bin_min = h_invmass.FindBin(1.09)
        left_bin_max = h_invmass.FindBin(1.105)
        right_bin_min = h_invmass.FindBin(1.125)
        right_bin_max = h_invmass.FindBin(1.15)
    elif particle == 'Omega':
        # Use sidebands: 1.65-1.665 and 1.680-1.695
        left_bin_min = h_invmass.FindBin(1.65)
        left_bin_max = h_invmass.FindBin(1.665)
        right_bin_min = h_invmass.FindBin(1.680)
        right_bin_max = h_invmass.FindBin(1.695)
    
    # Get average counts in sideband regions
    left_counts = sum(h_invmass.GetBinContent(i) for i in range(left_bin_min, left_bin_max+1))
    right_counts = sum(h_invmass.GetBinContent(i) for i in range(right_bin_min, right_bin_max+1))
    left_bins = left_bin_max - left_bin_min + 1
    right_bins = right_bin_max - right_bin_min + 1
    
    if left_bins > 0 and right_bins > 0:
        left_avg = left_counts / left_bins
        right_avg = right_counts / right_bins
        left_x = (h_invmass.GetBinCenter(left_bin_min) + h_invmass.GetBinCenter(left_bin_max)) / 2
        right_x = (h_invmass.GetBinCenter(right_bin_min) + h_invmass.GetBinCenter(right_bin_max)) / 2
        
        # Calculate slope: slope = ln(N2/N1) / (x2-x1)
        if left_avg > 0 and right_avg > 0:
            slope_estimate = np.log(right_avg / left_avg) / (right_x - left_x)
            offset_estimate = np.log(left_avg) - slope_estimate * left_x
        else:
            slope_estimate = -10.0
            offset_estimate = 5.0
    else:
        slope_estimate = -10.0
        offset_estimate = 5.0
    
    return slope_estimate, offset_estimate

def load_template_background(particle: str, mode: str = 'mixing', outfile = None):
    if mode == 'mixing':
        return load_template_background_from_mixing(particle, outfile)
    elif mode == 'mc':
        return load_template_background_from_mc(particle, outfile)
    else:
        raise ValueError('Invalid value for mode. Accepted values are "mixing", "mc"')

def load_template_background_from_mixing(particle: str, outfile = None):

    infile = TFile.Open(f'/home/galucia/its/single_particle/output/v0_cascade_mixing.root', 'READ')
    H2_BKG_TEMPLATE[particle] = infile.Get(f'h2PInvariantMassRotation')
    #H2_BKG_TEMPLATE[particle] = infile.Get(f'h2PInvariantMassLikeSign')
    H2_BKG_TEMPLATE[particle].SetDirectory(0)
    infile.Close()

    if outfile:
        outfile.cd()
        H2_BKG_TEMPLATE[particle].Write('h2Bkg')

def load_template_background_from_mc(particle: str, outfile = None):

    infile = TFile.Open(f'/data/galucia/its_pid/LHC24_pass1_skimmed/analysis_results_mc_19_09_2025.root ', 'READ')
    h_name_signal = 'massLambdaMc' if particle == 'Lambda' else 'massOmegaMc'
    h_name_total = 'massLambda' if particle == 'Lambda' else 'massOmega'
    h2_signal = infile.Get(f'lf-tree-creator-cluster-studies/LFTreeCreator/{h_name_signal}')
    h2_signal.RebinY()
    h2_total = infile.Get(f'lf-tree-creator-cluster-studies/LFTreeCreator/{h_name_total}')
    h2_total.RebinY()
    h2_total.Add(h2_signal, -1.)
    H2_BKG_TEMPLATE[particle] = h2_total
    H2_BKG_TEMPLATE[particle].SetDirectory(0)
    infile.Close()

    if outfile:
        outfile.cd()
        H2_BKG_TEMPLATE[particle].Write('h2Bkg')

def get_template_background(particle: str, ibin: int, xvar: RooRealVar, mode: str = 'mixing', outfile: TDirectory = None):

    if H2_BKG_TEMPLATE[particle] is None:
        load_template_background(particle, mode, outfile)
    
    h2_bkg = H2_BKG_TEMPLATE[particle]
    h_bkg = h2_bkg.ProjectionY(f'h_bkg_{particle}_{ibin}', ibin, ibin, 'e')
    dh_bkg = RooDataHist(f'data_bkg_{particle}_{ibin}', f'data_bkg_{particle}_{ibin}', [xvar], Import=h_bkg)
    return RooHistPdf(f'bkg_{particle}_{ibin}', f'bkg_{particle}_{ibin}', [xvar], dh_bkg), h_bkg, dh_bkg

# CHEBYSHEV INITIALISATION

def fit_with_sidebands(h_data, invmass, bkg, bkg_pars, sig_window):
    """
    Perform a two-step RooFit:
      1. Fit sidebands with background model
      2. Constrain background parameters in full signal+background fit
    
    Parameters
    ----------
    data : RooDataSet
        Full dataset
    invmass : RooRealVar
        Mass variable
    bkg : RooAbsPdf
        Background model
    bkg_pars : dict
        Dictionary of background parameters (RooRealVar)
    sig_window : tuple
        Signal window (min, max) to exclude from sideband fit
    """

    data = RooDataHist(h_data.GetName(), h_data.GetTitle(), [invmass], Import=h_data)
    sbcut = f"{invmass.GetName()} < {sig_window[0]} || {invmass.GetName()} > {sig_window[1]}"
    sidebands = data.reduce(sbcut)

    bkg.fitTo(sidebands, RooFit.Save())

    for par in bkg_pars.values():
        if par.isConstant():  # skip unused params
            continue
        #par.setRange(par.getVal() - par.getError(), par.getVal() + par.getError())
        par.setConstant(True)



#######

def visualize_fit_results(fit_results_df, particle, particle_dir):

    latex_particle = Particle.from_pdgid(PDG_CODE[particle]).latex_name
    g_purity = create_graph(fit_results_df, 'x', 'purity', 'x_error', 'purity_err', 
                            f'g_purity', f'{latex_particle};#it{{p}} (GeV/c);Purity')
    g_purity.SetMarkerStyle(20)
    g_purity.SetMarkerColor(797)
    g_purity.SetLineColor(797)

    g_chi2 = create_graph(fit_results_df, 'x', 'chi2_ndf', 0, 0,
                            f'g_chi2', f'{latex_particle};#it{{p}} (GeV/c);#chi^{{2}}/NDF')
    g_chi2.SetMarkerStyle(20)
    g_chi2.SetMarkerColor(598)
    g_chi2.SetLineColor(598)
    
    particle_dir.cd()
    g_purity.Write()
    g_chi2.Write()
    
def fit_routine(h2_invariant_mass, outfile: TFile, particle: str): 

    cfg = CONF[particle]
    outdir_fits = outfile.mkdir('fits')
    outdir_prefits = outfile.mkdir('prefits')

    invmass = RooRealVar('fInvariantMass', '#it{m}_{#Lambda}' if particle == 'Lambda' else '#it{m}_{#Omega}', cfg['y_min'], cfg['y_max'], 'GeV/c^{2}')
    signal_func = 'crystalball' if particle in ['Lambda'] else 'gausexp'
    #signal_func = 'gaus' if particle in ['Lambda'] else 'gausexp'

    x_min = cfg['x_min_fit']
    x_max = cfg['x_max_fit']

    fit_results_df = None

    x_bin_min = h2_invariant_mass.GetXaxis().FindBin(x_min)
    x_bin_max = h2_invariant_mass.GetXaxis().FindBin(x_max)
    for x_bin in range(x_bin_min, x_bin_max+1):

        ix = h2_invariant_mass.GetXaxis().GetBinCenter(x_bin)
        x_error = h2_invariant_mass.GetXaxis().GetBinWidth(x_bin) / 2.
        x_low_edge = h2_invariant_mass.GetXaxis().GetBinLowEdge(x_bin)
        x_high_edge = h2_invariant_mass.GetXaxis().GetBinLowEdge(x_bin+1)
        
        bkg_func = 'pol1' #if (particle in ['Lambda'] and np.abs(ix) > 1.5) else 'pol2'
        signal, signal_pars = init_signal_roofit(invmass, function=signal_func, particle=particle)
        bkg, bkg_pars = init_background_roofit(invmass, function=bkg_func, particle=particle)
        if particle == 'Omega' and 'bkg_p0' in bkg_pars and 'bkg_p1' in bkg_pars:
            bkg_pars['bkg_p0'].setRange(0.1, 1e5)
            bkg_pars['bkg_p1'].setRange(-1000, 1)
        if np.abs(ix) < 1.5 and particle == 'Lambda' and 'erf' in bkg_func:
            bkg_pars['mean'].setVal(1.115)
            bkg_pars['mean'].setRange(1.10, 1.13)
        
        h_invmass = h2_invariant_mass.ProjectionY(f'invmass_{ix:.2f}', x_bin, x_bin, 'e')
        for ibin in range(1, h_invmass.GetNbinsX()):
            xvalue_bin = h_invmass.GetBinCenter(ibin)
            if xvalue_bin < cfg['y_min'] or xvalue_bin > cfg['y_max']:
                h_invmass.SetBinContent(ibin, 0)
        if h_invmass.GetEntries() <= 0:
            print(f'No entries for particle {particle}, p = {ix:.2f}, skipping...')
            continue

        model = None
        was_template_used = False
        
        if h_invmass.GetEntries() > 30:
            #if not was_template_used:
            #    print(f'{tc.BLUE}Fitting with sidebands for particle {particle}, p = {ix:.2f}{tc.RESET}')
            #    fit_with_sidebands(h_invmass,  invmass, bkg, bkg_pars, 
            #                    (cfg['invariant_mass_peak'] - cfg['invariant_mass_cut'],
            #                     cfg['invariant_mass_peak'] + cfg['invariant_mass_cut']))
            #    for par in bkg_pars.values():
            #        par.setConstant(True)

            means, sigmas = initialize_means_and_covariances(h_invmass, 1, method='kmeans')
            signal_pars['mean'].setVal(means[0])
            signal_pars['mean'].setRange(means[0]-0.1*cfg['invariant_mass_cut'], means[0]+0.1*cfg['invariant_mass_cut'])
            signal_pars['sigma'].setVal(np.sqrt(sigmas[0]))

            if particle == 'Lambda':
                init_signal_from_template(particle, x_bin, invmass, signal, signal_pars, outdir_prefits, [x_low_edge, x_high_edge])

            if 'exp' in str(bkg):  # Check if using exponential
                slope_est, offset_est = initialize_exponential_from_data(h_invmass, invmass, particle)
                bkg_pars['slope'].setVal(slope_est)
                bkg_pars['offset'].setVal(offset_est)

        sig_frac = RooRealVar('sig_frac', 'sig_frac', 0.5, 0., 1.)
        model = RooAddPdf('model', 'signal + bkg', [signal, bkg], [sig_frac])

        # Template background
        if particle == 'Lambda' and np.abs(ix) < 1.5 and False:
            bkg, h_bkg, dh_bkg = get_template_background(particle, x_bin, invmass, mode='mixing', outfile=outfile)
            model = RooAddPdf('model', f'signal + bkg_{particle}_{x_bin}', [signal, bkg], [sig_frac])
            was_template_used = True

        frame, fit_results = calibration_fit_slice(model, h_invmass, invmass, signal_pars, x_low_edge, x_high_edge,
                                                   range=(cfg['y_min'], cfg['y_max']), draw_param_on=(particle!='Lambda'))
        frame.SetTitle(f'{particle} invariant mass, #it{{p}} = {ix:.2f} GeV/#it{{c}};{invmass.GetTitle()};Counts')

        invmass.setRange('signal_range', cfg['invariant_mass_peak'] - cfg['invariant_mass_cut'], cfg['invariant_mass_peak'] + cfg['invariant_mass_cut'])
        sig_integral = signal.createIntegral([invmass], Range='signal_range', NormSet=invmass)
        background_integral = bkg.createIntegral([invmass], Range='signal_range', NormSet=invmass)
        
        fit_results['x'] = np.abs(ix)
        fit_results['x_error'] = x_error
        fit_results['purity'] = sig_frac.getVal() * sig_integral.getVal() / (sig_frac.getVal() * sig_integral.getVal() + (1 - sig_frac.getVal()) * background_integral.getVal())
        fit_results['purity_err'] = 0
        if fit_results_df is None:
            fit_results_df = pd.DataFrame.from_dict([fit_results])
        else:
            fit_results_df = pd.concat([fit_results_df, pd.DataFrame.from_dict([fit_results])], ignore_index=True)

        canvas = TCanvas(f'cInvMass_{ix:.2f}', f'cInvMass_{ix:.2f}', 800, 600)

        vertical_lines = [TLine(cfg['invariant_mass_peak'] - cfg['invariant_mass_cut'], 10, cfg['invariant_mass_peak'] - cfg['invariant_mass_cut'], h_invmass.GetMaximum()),
                          TLine(cfg['invariant_mass_peak'] + cfg['invariant_mass_cut'], 10, cfg['invariant_mass_peak'] + cfg['invariant_mass_cut'], h_invmass.GetMaximum())]
        canvas.SetLogy()
        frame.SetMinimum(10 if particle=='Lambda' else 0.1)
        frame.Draw()
        
        text = TPaveText(0.7, 0.7, .85, 0.85, 'ndc')
        text.SetFillColor(0)
        text.SetBorderSize(0)
        text.AddText(f'#chi^{{2}} / ndf = {fit_results["chi2_ndf"]:.2f}')
        text.AddText(f'Purity = {fit_results["purity"]:.3f}')
        text.Draw('same')
        
        legend = TLegend(0.2, 0.7, 0.35, 0.85)
        legend.SetFillColor(0)
        legend.SetBorderSize(0)
        legend.AddEntry(frame.findObject('model'), 'signal+bkg', 'l')
        legend.AddEntry(frame.findObject('signal'), 'signal', 'l')
        legend.AddEntry(frame.findObject('bkg'), 'bkg', 'l')
        legend.AddEntry(vertical_lines[0], 'ROI', 'l')
        legend.Draw('same')
        
        for line in vertical_lines:
            set_root_object(line, line_style=2, line_width=2, line_color=797)
            line.Draw('same')
        outdir_fits.cd()
        canvas.Write()

    if fit_results_df is None:
        print(f'No fit results for particle {particle}, skipping...')
        return
    visualize_fit_results(fit_results_df, particle, outfile)

    print(fit_results_df)

    outfile.cd()
    h2_invariant_mass.Write()

    del h2_invariant_mass, invmass, signal, signal_pars, bkg, bkg_pars

def lazy_fit_routine(h2_invariant_mass, outfile: TFile, particle: str): 

    cfg = CONF[particle]

    bkg_func = TF1('bkg_func', 'expo', cfg['y_min'], cfg['y_max'])
    bkg_func.SetLineColor(4)

    x_min = cfg['x_min_fit']
    x_max = cfg['x_max_fit']

    fit_results_df = pd.DataFrame({'x': [], 'x_error': [], 'purity': [], 'purity_err': [], 'chi2_ndf': []})

    x_bin_min = h2_invariant_mass.GetXaxis().FindBin(x_min)
    x_bin_max = h2_invariant_mass.GetXaxis().FindBin(x_max)
    for x_bin in range(x_bin_min, x_bin_max+1):
        
        ix = h2_invariant_mass.GetXaxis().GetBinCenter(x_bin)
        x_error = h2_invariant_mass.GetXaxis().GetBinWidth(x_bin) / 2.
        x_low_edge = h2_invariant_mass.GetXaxis().GetBinLowEdge(x_bin)
        x_high_edge = h2_invariant_mass.GetXaxis().GetBinLowEdge(x_bin+1)
        
        h_invmass = h2_invariant_mass.ProjectionY(f'invmass_{ix:.2f}', x_bin, x_bin, 'e')
        if h_invmass.GetEntries() <= 0:
            print(f'No entries for particle {particle}, p = {ix:.2f}, skipping...')
            continue
        h_invmass_copy = h_invmass.Clone(f'invmass_copy_{ix:.2f}')
        for ibin in range(h_invmass_copy.FindBin(1.109), h_invmass_copy.FindBin(1.123)+1):
                h_invmass_copy.SetBinContent(ibin, 0)
        h_invmass_copy.Fit(bkg_func, 'RQM+', '', cfg['y_min_lazy'], cfg['y_max_lazy'])

        total_counts = h_invmass.Integral(h_invmass.FindBin(cfg['invariant_mass_peak'] - cfg['invariant_mass_cut']),
                                            h_invmass.FindBin(cfg['invariant_mass_peak'] + cfg['invariant_mass_cut']))
        bkg_counts = bkg_func.Integral(cfg['invariant_mass_peak'] - cfg['invariant_mass_cut'],
                                        cfg['invariant_mass_peak'] + cfg['invariant_mass_cut']) / h_invmass.GetBinWidth(1)
        purity = (total_counts - bkg_counts) / total_counts if total_counts > 0 else 0
        purity_err = np.sqrt(purity * (1 - purity) / total_counts) if total_counts > 0 else 0

        print('lines:', cfg['invariant_mass_peak'] - cfg['invariant_mass_cut'], cfg['invariant_mass_peak'] + cfg['invariant_mass_cut'])

        fit_results = {
            'x': np.abs(ix),
            'x_error': x_error,
            'purity': (total_counts - bkg_counts) / total_counts if total_counts > 0 else 0,
            'purity_err': np.sqrt(purity * (1 - purity) / total_counts) if total_counts > 0 else 0,
            'chi2_ndf': bkg_func.GetChisquare() / bkg_func.GetNDF() if bkg_func.GetNDF() > 0 else -1,
        }

        if fit_results_df is None:
            fit_results_df = pd.DataFrame.from_dict([fit_results])
        else:
            fit_results_df = pd.concat([fit_results_df, pd.DataFrame.from_dict([fit_results])], ignore_index=True)

        lines = [TLine(cfg['invariant_mass_peak'] - cfg['invariant_mass_cut'], 0, cfg['invariant_mass_peak'] - cfg['invariant_mass_cut'], h_invmass.GetMaximum()),
                TLine(cfg['invariant_mass_peak'] + cfg['invariant_mass_cut'], 0, cfg['invariant_mass_peak'] + cfg['invariant_mass_cut'], h_invmass.GetMaximum())]
        for line in lines:
            line.SetLineColor(2)
            line.SetLineStyle(kDashed)

        canvas = TCanvas(f'cInvMass_{ix:.2f}', f'cInvMass_{ix:.2f}', 800, 600)
        h_invmass.SetMarkerStyle(20)
        h_invmass.Draw('hist pe')
        bkg_func.Draw('same')
        for line in lines:
            line.Draw('same')
        canvas.SetLogy()
        outfile.cd()
        canvas.Write()

    if fit_results_df is None:
        print(f'No fit results for particle {particle}, skipping...')
        return
    visualize_fit_results(fit_results_df, particle, outfile)

    print(fit_results_df)

    outfile.cd()
    h2_invariant_mass.Write()

    del h2_invariant_mass

def main_routine(infile_path: str):

    infile = TFile.Open(infile_path, 'READ')

    #outfile_path = f'output/purity_v0_cascade_lazy.root'
    outfile_path = f'output/purity_v0_cascade.root'
    outfile = TFile.Open(outfile_path, 'RECREATE')
    
    particles = ['Lambda', 'Omega']
    for particle in particles:
        
        h2_invariant_mass = infile.Get(f'lf-tree-creator-cluster-studies/LFTreeCreator/mass{particle}')
        particle_dir = outfile.mkdir(particle)
        fit_routine(h2_invariant_mass, particle_dir, particle)
        #lazy_fit_routine(h2_invariant_mass, particle_dir, particle)

    outfile.Close()


if __name__ == '__main__':

    #infile_path = '/Users/glucia/Projects/ALICE/data/its_pid/LHC22o_pass7_minBias_small.root'
    #infile_path = '/Users/glucia/Projects/ALICE/data/its_pid/LHC24f3c.root'
    #infile_path = '/Users/glucia/Projects/ALICE/data/its_pid/LHC25a3.root'
    infile_path = '/data/galucia/its_pid/LHC24_pass1_skimmed/analysis_results_04_08_2025.root'

    main_routine(infile_path)