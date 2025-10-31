'''
    Code to run calibration of ITS and TPC parametrisations
'''

from typing import List
import numpy as np
import pandas as pd
from ROOT import TFile, TCanvas, TF1, TMath, TLine, kDashed, TDirectory, TPaveText, TLegend, TH2F
from ROOT import RooRealVar, RooCrystalBall, RooGaussian, RooAddPdf, RooGenericPdf, RooArgusBG, RooBukinPdf,\
    RooChebychev, RooHistPdf, RooDataHist, RooFit, RooArgSet, RooExtendPdf

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
        'y_min': 1.102,
        'y_max': 1.136,
        'y_min_lazy': 1.102,
        'y_max_lazy': 1.132,
        'invariant_mass_peak': 1.1157,
        'invariant_mass_cut': 0.06,
    },
    'Omega': {
        'x_min_fit': 1.,
        'x_max_fit': 5,
        'x_nbins': 8,
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

def init_signal_roofit(x: RooRealVar, function: str = 'crystalball', particle: str = 'Lambda'):

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
        signal = RooCrystalBall('signal', 'signal', x, signal_pars['mean'], signal_pars['sigma'],
                                signal_pars['aL'], signal_pars['nL'], #doubleSided=True)
                                signal_pars['aR'], signal_pars['nR'])

        return signal, signal_pars
    
    elif function == 'bukin':
        signal_pars = {
            'mean': RooRealVar('mean', '#mu', 1.115, 1.10, 1.13),
            'sigma': RooRealVar('sigma', '#sigma', 0.003, 0.001, 0.02),
            'xi': RooRealVar('xi', '#xi', -0.2),
            'rho1': RooRealVar('rho1', '#rho_{1}', -0.5),
            'rho2': RooRealVar('rho2', '#rho_{2}', 0.1),
        }
        if particle == 'Omega':
            signal_pars['mean'].setVal(1.672)
            signal_pars['mean'].setRange(1.66, 1.685)
        signal = RooBukinPdf('signal', 'signal', x, signal_pars['mean'], signal_pars['sigma'],
                                signal_pars['xi'], signal_pars['rho1'], signal_pars['rho2'])
        return signal, signal_pars
    
    elif function == 'gausexp':
        signal_pars = {
            'mean': RooRealVar('mean', 'mean', 0., 2, ''),
            'sigma': RooRealVar('sigma', 'sigma', 0.0001, 10, ''),
            'rlife': RooRealVar('rlife', 'rlife', 2., 0., 10.),
        }
        signal = RooGausExp('signal', 'signal', x, *signal_pars.values())
        return signal, signal_pars
    
    else:
        raise ValueError(f'Unknown function: {function}. Supported functions are "crystalball" and "gausexp".')

def init_background_roofit(x: RooRealVar, function: str = 'gaus', particle: str = 'Lambda', suffix:str=''):
    """
    Initialize background model in RooFit.
    Parameters
    ----------
    x : RooRealVar
        Variable for the background model (e.g., invariant mass)
    function : str
        Background function type. Supported: 'gausexp', 'gaus', 'exp', 'pol1', 'pol2', 'argus', 'cheb2', 'cheb1', 'cheb3', 'erf'
    """

    if function == 'exp':
        bkg_pars = {
            'offset': RooRealVar(f'bkg_offset{suffix}', f'bkg_offset{suffix}', 5., 0., 1.e5),
            'slope': RooRealVar(f'bkg_slope{suffix}', f'bkg_slope{suffix}', -5., -1.e5, 0.),
        }

        bkg = RooGenericPdf(f'bkg{suffix}', 'bkg', f'exp(bkg_offset{suffix} + bkg_slope{suffix} * {x.GetName()})', [x, *bkg_pars.values()])
        return bkg, bkg_pars
    
    elif function == 'pol1':
        bkg_pars = {
            'p0': RooRealVar(f'bkg_p0{suffix}', 'bkg_p0', 100, 0., 1e5),
            'p1': RooRealVar(f'bkg_p1{suffix}', 'bkg_p1', 0.1, -1.e5, 1.e5),
        }
        bkg = RooGenericPdf(f'bkg{suffix}', 'bkg', f'bkg_p0{suffix} + bkg_p1{suffix} * {x.GetName()}', [x, *bkg_pars.values()])
        return bkg, bkg_pars
    
    elif function == 'pol2':
        bkg_pars = {
            'p0': RooRealVar(f'bkg_p0{suffix}', 'bkg_p0', 0.1, -1.e3, 1.e3),
            'p1': RooRealVar(f'bkg_p1{suffix}', 'bkg_p1', 0.1, -1.e3, 1.e3),
            'p2': RooRealVar(f'bkg_p2{suffix}', 'bkg_p2', -0.1, -1.e3, 1.e3),
        }
        bkg = RooGenericPdf(f'bkg{suffix}', 'bkg', f'bkg_p0{suffix} + bkg_p1{suffix} * {x.GetName()} + bkg_p2{suffix} * {x.GetName()}^2', [x, *bkg_pars.values()])
        return bkg, bkg_pars
    
    elif function == 'argus':
        bkg_pars = {
            'm0': RooRealVar(f'bkg_m0{suffix}', 'bkg_m0', 1.1, 0., 2.),
            'chi': RooRealVar(f'bkg_chi{suffix}', 'bkg_chi', 0.5, 0., 6.),
            'p': RooRealVar(f'bkg_p{suffix}', 'bkg_p', 0.5),
        }
        bkg_pars['p'].setConstant(True)
        if particle == 'Omega':
            bkg_pars['m0'] = RooRealVar(f'bkg_m0{suffix}', 'bkg_m0', 2., 1.7, 3.)
        bkg = RooArgusBG(f'bkg{suffix}', 'bkg', x, *bkg_pars.values())
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

def init_signal_from_template(particle: str, ilow_bin: int, ihigh_bin: int, xvar: RooRealVar, signal_pdf, signal_pars, outfile, pt_edges: List[float]): 

    if H2_SIG_TEMPLATE[particle] is None:
        load_template_signal(particle)
    
    h2_sig = H2_SIG_TEMPLATE[particle]
    h_sig = h2_sig.ProjectionY(f'h_sig_{particle}_{ilow_bin}', ilow_bin, ihigh_bin, 'e')
    dh_sig = RooDataHist(f'data_sig_{particle}_{ilow_bin}', f'data_sig_{particle}_{ilow_bin}', [xvar], Import=h_sig)
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

    canvas = TCanvas(f'cInvMassSignal_{ilow_bin}', f'cInvMassSignal_{ilow_bin}', 800, 600)
    frame.Draw()
    outfile.cd()
    canvas.Write()

    

# BACKGROUND INITIALISATION

H2_BKG_TEMPLATE = {
    'Lambda': None,
    'Omega': None
}

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



def fit_sidebands(h_data, invmass, bkg, bkg_pars, cfg, sig_window, x, outdir):
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

    invmass.setRange(f'{bkg.GetName()}_lower_region', cfg['y_min'], sig_window[0])
    invmass.setRange(f'{bkg.GetName()}_upper_region', sig_window[1], cfg['y_max'])
    invmass.setRange('full', cfg['y_min'], cfg['y_max'])

    bkg_normalisation = RooRealVar('bkg_normalisation', '#it{N}_{bkg}', 1., 0., 1e6)
    extended_bkg = RooExtendPdf('extended_bkg', 'extended_bkg', bkg, bkg_normalisation)
    extended_bkg.fitTo(data, RooFit.Range(f'{bkg.GetName()}_lower_region,{bkg.GetName()}_upper_region'), RooFit.Save(), Extended=True, PrintLevel=-1)

    frame = invmass.frame()
    data.plotOn(frame)
    extended_bkg.plotOn(frame)
    extended_bkg.paramOn(frame)
    chi2 = frame.chiSquare()
    
    text = TPaveText(0.7, 0.5, .85, 0.55, 'ndc')
    text.SetFillColor(0)
    text.SetBorderSize(0)
    text.AddText(f'#chi^{{2}} / ndf = {chi2:.2f}')
    text.Draw('same')

    canvas = TCanvas(f'sidebands_{bkg.GetName()}_fit_{x:.2f}', '')
    canvas.SetLogy(True)
    frame.SetMinimum(1)
    frame.Draw()
    text.Draw('same')

    outdir.cd()
    canvas.Write()

    # Fix background shape parameters
    for par in bkg_pars.values():
        par.setConstant(True)
    
    # Calculate integrals for normalization scaling
    integral_lower = bkg.createIntegral([invmass], RooFit.Range(f'{bkg.GetName()}_lower_region')).getVal()
    integral_upper = bkg.createIntegral([invmass], RooFit.Range(f'{bkg.GetName()}_upper_region')).getVal()
    integral_sidebands = integral_lower + integral_upper
    integral_full = bkg.createIntegral([invmass], RooFit.Range('full')).getVal()
    
    # Scale the normalization from sidebands to full range
    N_bkg_sidebands = bkg_normalisation.getVal()
    N_bkg_full = N_bkg_sidebands * (integral_full / integral_sidebands)
    
    print(f'{tc.GREEN}Sideband fit at p={x:.2f}: N_bkg(sidebands)={N_bkg_sidebands:.1f}, scale={integral_full/integral_sidebands:.3f}, N_bkg(full)={N_bkg_full:.1f}{tc.RESET}')
    
    bkg_normalisation.setVal(N_bkg_full)
    bkg_normalisation.setConstant(False)  # Allow it to float in final fit, but with good initial value

    return bkg_normalisation, chi2

def fit_sidebands_root(h_data, bkg, bkg_pars, cfg, sig_window, x, outdir):
    """
    Perform a two-step fit:
      1. Fit sidebands with TF1 to get normalization
      2. Transfer parameters to RooFit background model
    
    Parameters
    ----------
    h_data : TH1
        Data histogram
    invmass : RooRealVar
        Mass variable
    bkg : RooAbsPdf
        Background model
    bkg_pars : dict
        Dictionary of background parameters (RooRealVar)
    sig_window : tuple
        Signal window (min, max) to exclude from sideband fit
    """
    
    bkg_name = bkg.GetName()
    if 'exp' in bkg_name:
        tf1_bkg = TF1(f'tf1_{bkg_name}', '[0]*exp([1] + [2]*x)', cfg['y_min'], cfg['y_max'])
        tf1_bkg.SetParameters(100, 5, -10)  # Initial guesses
        tf1_bkg.SetParLimits(0, 0, 1e6)
    elif 'pol1' in bkg_name:
        tf1_bkg = TF1(f'tf1_{bkg_name}', '[0] + [1]*x', cfg['y_min'], cfg['y_max'])
        tf1_bkg.SetParameters(100, 0)
        tf1_bkg.SetParLimits(0, 0, 1e6)
    elif 'pol2' in bkg_name:
        tf1_bkg = TF1(f'tf1_{bkg_name}', '[0] + [1]*x + [2]*x^2', cfg['y_min'], cfg['y_max'])
        tf1_bkg.SetParameters(100, 0, 0)
        tf1_bkg.SetParLimits(0, 0, 1e6)
    else:
        raise ValueError(f'Unsupported background function for TF1 fit: {bkg_name}')
    
    h_sidebands = h_data.Clone(f'{h_data.GetName()}_sidebands')
    x_low_max = h_sidebands.GetXaxis().FindBin(sig_window[0])
    x_high_min = h_sidebands.GetXaxis().FindBin(sig_window[1])
    for ibin in range(x_low_max, x_high_min + 1):
        h_sidebands.SetBinContent(ibin, 0.)
        h_sidebands.SetBinError(ibin, 0.)
    
    h_sidebands.Fit(tf1_bkg, 'RQN')  # R=range, Q=quiet, N=no draw
    
    if 'exp' in bkg_name:
        suffix = bkg_name.replace('bkg', '')
        bkg_pars[f'offset'].setVal(tf1_bkg.GetParameter(1))
        bkg_pars[f'slope'].setVal(tf1_bkg.GetParameter(2))
    elif 'pol1' in bkg_name:
        bkg_pars['p0'].setVal(tf1_bkg.GetParameter(0))
        bkg_pars['p1'].setVal(tf1_bkg.GetParameter(1))
    elif 'pol2' in bkg_name:
        bkg_pars['p0'].setVal(tf1_bkg.GetParameter(0))
        bkg_pars['p1'].setVal(tf1_bkg.GetParameter(1))
        bkg_pars['p2'].setVal(tf1_bkg.GetParameter(2))
    
    for par in bkg_pars.values():
        par.setConstant(True)
    
    N_bkg_full = tf1_bkg.Integral(cfg['y_min'], cfg['y_max']) / h_data.GetBinWidth(1)
    print(f'{tc.GREEN}Sideband TF1 fit at p={x:.2f}: N_bkg(full)={N_bkg_full:.1f}, chi2/ndf={tf1_bkg.GetChisquare()/tf1_bkg.GetNDF():.2f}{tc.RESET}')
    
    bkg_normalisation = RooRealVar('bkg_normalisation', '#it{N}_{bkg}', N_bkg_full, N_bkg_full*0.5, N_bkg_full*1.5)
    bkg_normalisation.setConstant(True)
    
    canvas = TCanvas(f'sidebands_{bkg_name}_fit_{x:.2f}', '')
    canvas.SetLogy(True)
    h_sidebands.SetMinimum(0.1)
    h_sidebands.Draw('E')
    tf1_bkg.Draw('same')
    
    text = TPaveText(0.7, 0.5, .85, 0.55, 'ndc')
    text.SetFillColor(0)
    text.SetBorderSize(0)
    text.AddText(f'#chi^{{2}} / ndf = {tf1_bkg.GetChisquare()/tf1_bkg.GetNDF():.2f}')
    text.AddText(f'#it{{N}}_{{bkg}} = {N_bkg_full}')
    text.Draw('same')
    
    outdir.cd()
    canvas.Write()
    
    chi2 = tf1_bkg.GetChisquare() / tf1_bkg.GetNDF()
    
    return bkg_normalisation, chi2

def fit_core(h_data, invmass, signal, signal_pars, sig_window, x, outdir):
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

    invmass.setRange('core_region', sig_window[0], sig_window[1])

    signal.fitTo(data, RooFit.Range('core_region'), RooFit.Save())

    frame = invmass.frame()
    data.plotOn(frame)
    signal.plotOn(frame)
    signal.paramOn(frame)
    chi2 = frame.chiSquare()

    text = TPaveText(0.7, 0.5, .85, 0.55, 'ndc')
    text.SetFillColor(0)
    text.SetBorderSize(0)
    text.AddText(f'#chi^{{2}} / ndf = {chi2:.2f}')
    text.Draw('same')
    
    canvas = TCanvas(f'core_signal_fit_{x:.2f}', '')
    canvas.SetLogy(True)
    frame.SetMinimum(1)
    frame.Draw()
    text.Draw('same')

    outdir.cd()
    canvas.Write()

    for par_name in ['mean', 'sigma']:
        signal_pars[par_name].setConstant(True)

def draw_final_fit(h_invmass, bkg, frame, particle, ix, invmass, cfg, fit_results, outdir_fits):

    frame.SetTitle(f'#{particle}, #it{{p}} = {ix:.2f} GeV/#it{{c}};{invmass.GetTitle()};Counts')

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
    legend.AddEntry(frame.findObject(bkg.GetName()), 'bkg', 'l')
    legend.AddEntry(vertical_lines[0], 'ROI', 'l')
    legend.Draw('same')
    
    for line in vertical_lines:
        set_root_object(line, line_style=2, line_width=2, line_color=797)
        line.Draw('same')
    outdir_fits.cd()
    canvas.Write()


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

def _fit_bin(h2_invariant_mass:TH2F, x_low_bin:int, x_high_bin:int, invmass:RooRealVar,
             signal_func, cfg:dict, particle:str, 
             outdir_prefits:TDirectory, outdir_fits:TDirectory, outdir_bkg_fits:TDirectory, outdir_core_fits:TDirectory):

    ix = (h2_invariant_mass.GetXaxis().GetBinCenter(x_high_bin) + h2_invariant_mass.GetXaxis().GetBinCenter(x_low_bin)) / 2.
    x_error = (h2_invariant_mass.GetXaxis().GetBinCenter(x_high_bin) - h2_invariant_mass.GetXaxis().GetBinCenter(x_low_bin)) / 2.
    x_low_edge = h2_invariant_mass.GetXaxis().GetBinLowEdge(x_low_bin)
    x_high_edge = h2_invariant_mass.GetXaxis().GetBinLowEdge(x_high_bin+1)
    
    h_invmass = h2_invariant_mass.ProjectionY(f'invmass_{ix:.2f}', x_low_bin, x_high_bin, 'e')
    if h_invmass.GetEntries() < 30:
        print(f'Too few entries for particle {particle}, p = {ix:.2f}, skipping...')
        return None

    signal, signal_pars = init_signal_roofit(invmass, function=signal_func, particle=particle)
    particle_mass = Particle.from_name('Lambda').mass/1_000 if particle == 'Lambda' else Particle.from_name('Omega-').mass/1_000
    signal_window = (particle_mass - 0.01, particle_mass + 0.01)
    signal_window_bin = (h_invmass.FindBin(signal_window[0]), h_invmass.FindBin(signal_window[1]))
    signal_window = (h_invmass.GetBinLowEdge(signal_window_bin[0]+1), h_invmass.GetBinLowEdge(signal_window_bin[1]))
    signal_pars['mean'].setVal(particle_mass)
    signal_pars['mean'].setRange(signal_window[0], signal_window[1])
    
    if particle == 'Lambda':
        init_signal_from_template(particle, x_low_bin, x_high_bin, invmass, signal, signal_pars, outdir_prefits, [x_low_edge, x_high_edge])
    fit_core(h_invmass, invmass, signal, signal_pars, signal_window, ix, outdir_core_fits)

    bkg_funcs = ['pol1', 'exp']
    bkgs, bkgs_pars, bkgs_normalisation, bkgs_chi2 = [], [], [], []
    for bkg_func in bkg_funcs:
        bkg, bkg_pars = init_background_roofit(invmass, function=bkg_func, particle=particle, suffix=f'_{bkg_func}')
        #bkg_normalisation, chi2 = fit_sidebands(h_invmass, invmass, bkg, bkg_pars, cfg, signal_window, ix, outdir_bkg_fits)
        bkg_normalisation, chi2 = fit_sidebands_root(h_invmass, bkg, bkg_pars, cfg, signal_window, ix, outdir_bkg_fits)
        
        if chi2 < 0:
            continue
        bkgs.append(bkg)
        bkgs_pars.append(bkg_pars)
        bkgs_chi2.append(chi2)
        bkgs_normalisation.append(bkg_normalisation)

    index_min = min(range(len(bkgs_chi2)), key=bkgs_chi2.__getitem__)
    bkg = bkgs[index_min]
    bkg_pars = bkgs_pars[index_min]
    bkg_normalisation = bkgs_normalisation[index_min] if len(bkgs_normalisation) > 0 else RooRealVar('bkg_normalisation', 'bkg_normalisation', 1., 0., 1e6)

    #sig_frac = RooRealVar('sig_frac', 'sig_frac', 0.5, 0., 1.)
    #model = RooAddPdf('model', 'signal + bkg', [signal, bkg], [sig_frac])
    sig_normalisation = RooRealVar('sig_normalisation', '#it{N}_{sig}', 1., 0., 1.e10)
    model = RooAddPdf('model', 'signal + bkg', [signal, bkg], [sig_normalisation, bkg_normalisation])

    # Template background
    #if particle == 'Lambda' and np.abs(ix) < 1.5 and False:
    #    bkg, h_bkg, dh_bkg = get_template_background(particle, x_bin, invmass, mode='mixing', outfile=outfile)
    #    model = RooAddPdf('model', f'signal + bkg_{particle}_{x_bin}', [signal, bkg], [sig_frac])
    #    was_template_used = True

    frame, fit_results = calibration_fit_slice(model, h_invmass, invmass, signal_pars, x_low_edge, x_high_edge,
                                               range=(cfg['y_min'], cfg['y_max']), draw_param_on=(particle=='Lambda'), 
                                               extended=True, show_constant_params=False)

    invmass.setRange('signal_range', cfg['invariant_mass_peak'] - cfg['invariant_mass_cut'], cfg['invariant_mass_peak'] + cfg['invariant_mass_cut'])
    sig_integral = signal.createIntegral([invmass], Range='signal_range', NormSet=invmass)
    background_integral = bkg.createIntegral([invmass], Range='signal_range', NormSet=invmass)
    
    fit_results['x'] = np.abs(ix)
    fit_results['x_error'] = x_error
    #fit_results['purity'] = sig_frac.getVal() * sig_integral.getVal() / (sig_frac.getVal() * sig_integral.getVal() + (1 - sig_frac.getVal()) * background_integral.getVal())
    fit_results['purity'] = sig_normalisation.getVal() * sig_integral.getVal() / (sig_normalisation.getVal() * sig_integral.getVal() + bkg_normalisation.getVal() * background_integral.getVal())
    fit_results['purity_err'] = 0

    draw_final_fit(h_invmass, bkg, frame, particle, ix, invmass, cfg, fit_results, outdir_fits)

    return fit_results

def fit_routine(h2_invariant_mass, outfile: TFile, particle: str): 

    cfg = CONF[particle]
    outdir_bkg_fits = outfile.mkdir('bkg_fits')
    outdir_fits = outfile.mkdir('fits')
    outdir_prefits = outfile.mkdir('prefits')
    outdir_core_fits = outfile.mkdir('core_fits')

    invmass = RooRealVar('fInvariantMass', '#it{m}_{#Lambda}' if particle == 'Lambda' else '#it{m}_{#Omega}', cfg['y_min'], cfg['y_max'], 'GeV/c^{2}')
    signal_func = 'crystalball'# if particle in ['Lambda'] else 'bukin'

    fit_results_df = None

    bin_edges = np.linspace(cfg['x_min_fit'], cfg['x_max_fit'], cfg['x_nbins'])
    bin_index_edges = [h2_invariant_mass.GetXaxis().FindBin(bin_edge) for bin_edge in bin_edges]

    for idx in range(len(bin_index_edges)-1):

        fit_results = _fit_bin(h2_invariant_mass, bin_index_edges[idx], bin_index_edges[idx+1], invmass, signal_func, cfg, particle,
                               outdir_prefits, outdir_fits, outdir_bkg_fits, outdir_core_fits)
        if fit_results is None:
            continue
        if fit_results_df is None:
            fit_results_df = pd.DataFrame.from_dict([fit_results])
        else:
            fit_results_df = pd.concat([fit_results_df, pd.DataFrame.from_dict([fit_results])], ignore_index=True)

    if fit_results_df is None:
        print(f'No fit results for particle {particle}, skipping...')
        return
    visualize_fit_results(fit_results_df, particle, outfile)

    print(fit_results_df)

    outfile.cd()
    h2_invariant_mass.Write()

    del h2_invariant_mass, invmass

def main_routine(infile_path: str):

    infile = TFile.Open(infile_path, 'READ')

    outfile_path = f'output/purity_v0_cascade_new.root'
    #outfile_path = f'output/purity_v0_cascade.root'
    outfile = TFile.Open(outfile_path, 'RECREATE')
    
    particles = ['Lambda', 'Omega']
    for particle in particles:
        
        h2_invariant_mass = infile.Get(f'lf-tree-creator-cluster-studies/LFTreeCreator/mass{particle}')
        particle_dir = outfile.mkdir(particle)
        fit_routine(h2_invariant_mass, particle_dir, particle)

    outfile.Close()


if __name__ == '__main__':

    infile_path = '/data/galucia/its_pid/LHC24_pass1_skimmed/analysis_results_04_08_2025.root'

    main_routine(infile_path)