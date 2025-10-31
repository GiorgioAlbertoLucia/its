'''
    Script devoted to the visualisation of the average cluster size per species
'''

from scipy.special import erf
import numpy as np
from particle import Particle

from ROOT import TFile, TCanvas, TF1, TLegend, TPaveText, TGraphErrors, TLatex
from ROOT import kOrange, kBlue, kGreen, kRed, kMagenta, kYellow, kBlack, kAzure,kViolet

import sys
sys.path.append('..')
from utils.pid_routine import PARTICLE_ID, PDG_CODE, LATEX_PARTICLE
from utils.utils import set_root_object

TLATEX_PARTICLE_NAMES = {
    'Pi': '#pi^{#pm}',
    'Ka': 'K^{#pm}',
    'Pr': 'p +#kern[0.6]{#bar{p}}',
    'Xi': '#Xi +#kern[0.6]{#bar{#Xi}}',
    'Omega': '#Omega +#kern[0.6]{#bar{#Omega}}',
    'De': '#bar{d}',
    'He': '^{3}He +#kern[0.2]{^{3}#bar{He}}',
}

TCOLOR_PARTICLE = {
    'Pi': kGreen,
    'Ka': 4,  # 
    'Pr': kRed,
    'Xi': kOrange, # Blue
    'Omega': kViolet+6,
    'De': kAzure+1,
    'He': 9, # Ocean Blue
}

TMARKER_PARTICLE = {
    'Pi': 22,  # Triangle
    'Ka': 20,  # Circle
    'Pr': 23,  # Upside-down triangle
    'Xi': 21,  # Square
    'Omega': 33,  # Star
    'De': 20,  # Circle
    'He': 24,  # Empty circle
}

def get_alice_watermark(x_min, y_min, x_max, y_max) -> TPaveText:

    watermark = TPaveText(x_min, y_min, x_max, y_max, 'NDC')
    watermark.SetBorderSize(0)
    watermark.SetFillColor(0)
    watermark.SetTextAlign(12)
    watermark.SetTextSize(0.04)
    watermark.AddText('#bf{ALICE Performance}')
    watermark.AddText('#bf{Run 3}')
    watermark.AddText('#bf{pp #sqrt{#it{s}} = 13.6 TeV}')
    #watermark.AddText('#bf{LHC24 pass1}')

    return watermark

def init_legend(x_min, y_min, x_max, y_max) -> TLegend:
    legend = TLegend(x_min, y_min, x_max, y_max)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.04)
    legend.SetFillColor(0)
    legend.SetNColumns(2)
    return legend

def prepare_graph(graph, legend, particle, legend_option: str = 'p'):
    graph.SetMarkerColor(TCOLOR_PARTICLE[particle])  
    graph.SetMarkerStyle(TMARKER_PARTICLE[particle])
    graph.SetMarkerSize(.8)
    graph.SetLineColor(TCOLOR_PARTICLE[particle])
    legend.AddEntry(graph, TLATEX_PARTICLE_NAMES[particle], legend_option)
    return graph

def hist_to_graph(hist, name: str = ''):

    if name == '':
        name = hist.GetName() + '_graph'
    n_bins = hist.GetNbinsX()
    x_values = np.array([hist.GetBinCenter(i + 1) for i in range(n_bins)], dtype=float)
    y_values = np.array([hist.GetBinContent(i + 1) for i in range(n_bins)], dtype=float)
    x_errors = np.array([hist.GetBinWidth(i + 1) / 2 for i in range(n_bins)], dtype=float)
    y_errors = np.array([hist.GetBinError(i + 1) for i in range(n_bins)], dtype=float)

    graph = TGraphErrors(n_bins, x_values, y_values, x_errors, y_errors)
    graph.SetName(name)
    graph.SetTitle(hist.GetTitle())
    
    return graph    

def convert_momentum_graph_to_rigidity(graph: TGraphErrors, charge:float) -> TGraphErrors:
    n_points = graph.GetN()
    x_values = np.array([graph.GetX()[i] for i in range(n_points)], dtype=float)
    y_values = np.array([graph.GetY()[i] for i in range(n_points)], dtype=float)
    x_errors = np.array([graph.GetEX()[i] for i in range(n_points)], dtype=float)
    y_errors = np.array([graph.GetEY()[i] for i in range(n_points)], dtype=float)

    rigidity_values = x_values / charge
    rigidity_errors = x_errors / charge
    rigidity_graph = TGraphErrors(n_points, rigidity_values, y_values, rigidity_errors, y_errors)
    rigidity_graph.SetName(graph.GetName() + '_rigidity')
    return rigidity_graph

def get_resolution_band(graph: TGraphErrors, particle:str, n_sigma: float = 1.0) -> TF1:

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

    mass = Particle.from_pdgid(PDG_CODE[particle]).mass / 1_000  # GeV/c2

    for ipoint in range(graph.GetN()):
        p = graph.GetPointX(ipoint)
        p_err = graph.GetErrorX(ipoint)

        if particle in ['He']:
            charge = 2
            exp_params = FIT_PARAMS['Z=2']['mean']
            res_params = FIT_PARAMS['Z=2']['resolution']
        else:
            charge = 1
            exp_params = FIT_PARAMS['Z=1']['mean']
            res_params = FIT_PARAMS['Z=1']['resolution']

        beta_gamma = p * charge / (mass)
        if beta_gamma <= 0:
            continue
        expected = exp_params[2] + exp_params[0] / (beta_gamma**exp_params[1])
        sigma = expected * (res_params[0] + res_params[1]*beta_gamma + res_params[2]*beta_gamma**2) \
            if particle in ['He'] else res_params[0] * erf((beta_gamma - res_params[1]) / res_params[2])
        graph.SetPointError(ipoint, p_err, sigma * 2 * n_sigma)

        if particle in ['He']:
            print(f'{particle}: {p * charge=:.2f}, {beta_gamma=:.2f}, {expected=:.2f}, {sigma=:.2f}')

    return graph

def get_sigma_band(graph: TGraphErrors, sigma_graph: TGraphErrors, n_sigma: float = 1.0) -> TF1:

    for ipoint in range(graph.GetN()):
        p_err = graph.GetErrorX(ipoint)
        y_err = sigma_graph.GetPointY(ipoint)

        graph.SetPointError(ipoint, p_err, y_err * n_sigma)
    return graph

    
def get_frame_init(mode:str, y:str='mean'):
    
    if y == 'mean':
        FRAME_INIT = {
                        'beta_gamma': (0, 1, 5.5, 9.5, f'{mode};#beta#gamma;#LT ITS cluster size#kern[1]{{#GT}} #times #LT cos#lambda#kern[1]{{#GT}}'),
                        'p': (0.15, 1, 7, 13, f'{mode};#it{{p}}/|#it{{Z}}| (GeV/#it{{c}});#LT ITS cluster size#kern[1]{{#GT}} #times #LT cos#lambda#kern[1]{{#GT}}')
                     }
    elif y == 'resolution':
        FRAME_INIT = {
                        'beta_gamma': (0, 0.07, 5.5, 0.28, f'{mode};#beta#gamma;#frac{{#sigma}}{{#LT ITS cluster size#kern[1]{{#GT}} #times #LT cos#lambda#kern[1]{{#GT}}}}'),
                        'p': (-0.5, 0.09, 7, 0.28, f'{mode};#it{{p}}/|#it{{Z}}| (GeV/#it{{c}});#frac{{#sigma}}{{#LT ITS cluster size#kern[1]{{#GT}} #times #LT cos#lambda#kern[1]{{#GT}}}}')
                     }
    else:
        raise ValueError(f'Unsupported y value: {y}. Supported values are "mean" and "resolution".')
    return FRAME_INIT

def superimpose_average_cluster_size(output_canvas_path, x: str = 'beta_gamma', mode: str = 'mean'):

    output_canvas = TCanvas('cMeanClusterSize', '', 800, 600)
    
    g_means = []
    input_file = TFile.Open(f'../output/data/LHC24_pass1_skimmed_calibration_{x}_{mode}.root', 'READ')

    legend = init_legend(0.5, 0.65, 0.85, 0.85)

    for particle in ['Pi', 'Ka', 'Pr', 'De', 'He']:

        print(f'Processing {particle=}, {TLATEX_PARTICLE_NAMES[particle]=}')
        c_mean = input_file.Get(f'{particle}/c_mean')
        if not (c_mean and hasattr(c_mean, 'InheritsFrom') and c_mean.InheritsFrom('TCanvas')):
            print(f'Skipping {particle=}, no mean cluster size found.')
            continue

        list_mean = c_mean.GetListOfPrimitives()
        g_mean = list_mean.FindObject('g_mean')
        g_mean.GetListOfFunctions().Clear()
        if x == 'p' and particle in ['He']:
            g_mean = convert_momentum_graph_to_rigidity(g_mean, charge=2)

        g_mean = prepare_graph(g_mean, legend, particle)
        g_means.append(g_mean)

    input_file.Close()

    watermark = get_alice_watermark(0.55, 0.35, 0.8, 0.5)

    mode_suffix = '_trunc' if mode == 'truncated' else ''
    x_suffix = 'momentum' if x == 'p' else 'betagamma'
    for particle_small, particle in zip(['xi', 'omega'], ['Xi', 'Omega']):
        input_file = TFile.Open(f'../output/caliva/cluster_size_{particle_small}_vs_{x_suffix}{mode_suffix}.root', 'READ')
        hist = input_file.Get(f'mean_{particle_small}_pos')
        graph = hist_to_graph(hist, f'g_mean_{particle}')
        
        graph = prepare_graph(graph, legend, particle)
        g_means.append(graph)

    output_canvas.cd()
    
    hframe = output_canvas.DrawFrame(*get_frame_init(mode, y='mean')[x])

    for g_mean in g_means:
        g_mean.Draw('P SAME')
    legend.Draw()
    watermark.Draw()

    output_canvas.Print(output_canvas_path)

def superimpose_average_cluster_size_with_bands(output_canvas_path, x: str = 'beta_gamma', mode: str = 'mean', option: str = 'parametrisation'):

    output_canvas = TCanvas('cMeanClusterSize', '', 800, 600)
    output_canvas.SetLeftMargin(0.11)
    output_canvas.SetRightMargin(0.05)
    output_canvas.SetBottomMargin(0.15)
    
    g_means = []
    input_file = TFile.Open(f'../output/data/LHC24_pass1_skimmed_calibration_{x}_{mode}.root', 'READ')

    legend = init_legend(0.6, 0.47, 0.85, 0.67)
    text = TLatex(0.625, 0.42, '#bf{1#sigma band contour}')
    if x == 'p':
        legend = init_legend(0.15, 0.65, 0.4, 0.85)
        text = TLatex(0.175, 0.61, '#bf{1#sigma band contour}')
    text.SetNDC()
    text.SetTextSize(0.04)

    for particle in ['Pi', 'Ka', 'Pr', 'De', 'He']:

        print(f'Processing {particle=}, {TLATEX_PARTICLE_NAMES[particle]=}')
        c_mean = input_file.Get(f'{particle}/c_mean')
        if not (c_mean and hasattr(c_mean, 'InheritsFrom') and c_mean.InheritsFrom('TCanvas')):
            print(f'Skipping {particle=}, no mean cluster size found.')
            continue

        list_mean = c_mean.GetListOfPrimitives()
        g_mean = list_mean.FindObject('g_mean')
        g_mean.GetListOfFunctions().Clear()
        if x == 'p' and particle in ['He']:
            g_mean = convert_momentum_graph_to_rigidity(g_mean, charge=2)
        
        if option == 'parametrisation':
            g_mean = get_resolution_band(g_mean, particle, n_sigma=1.0)
        if option == 'sigma':
            g_sigma = input_file.Get(f'{particle}/g_sigma')
            g_mean = get_sigma_band(g_mean, g_sigma, n_sigma=1.0) 

        g_mean = prepare_graph(g_mean, legend, particle, legend_option='fp')
        set_root_object(g_mean, fill_style=3004, fill_color=TCOLOR_PARTICLE[particle])
        g_means.append(g_mean)

    input_file.Close()

    watermark = get_alice_watermark(0.6, 0.7, 0.85, 0.85)

    mode_suffix = '_trunc' if mode == 'truncated' else ''
    x_suffix = 'momentum' if x == 'p' else 'betagamma'
    if x == 'beta_gamma':
        for particle_small, particle in zip(['xi', 'omega'], ['Xi', 'Omega']):
            input_file = TFile.Open(f'../output/caliva/cluster_size_{particle_small}_vs_{x_suffix}{mode_suffix}.root', 'READ')
            hist_pos = input_file.Get(f'mean_{particle_small}_pos')
            hist_neg = input_file.Get(f'mean_{particle_small}_neg')
            hist = hist_pos.Clone(f'mean_{particle_small}')
            for ibin in range(1, hist.GetNbinsX() + 1):
                content_pos = hist_pos.GetBinContent(ibin)
                content_neg = hist_neg.GetBinContent(ibin)
                error_pos = hist_pos.GetBinError(ibin)
                error_neg = hist_neg.GetBinError(ibin)
                if content_pos > 0 and content_neg > 0:
                    new_content = 0.5 * (content_pos + content_neg)
                    new_error = 0.5 * np.sqrt(error_pos**2 + error_neg**2)
                    hist.SetBinContent(ibin, new_content)
                    hist.SetBinError(ibin, new_error)
                else:
                    hist.SetBinContent(ibin, 0)
                    hist.SetBinError(ibin, 0)

            graph = hist_to_graph(hist, f'g_mean_{particle}')
            graph = get_resolution_band(graph, particle, n_sigma=1.0)

            graph = prepare_graph(graph, legend, particle, legend_option='fp')
            set_root_object(graph, fill_style=3004, fill_color=TCOLOR_PARTICLE[particle])
            g_means.append(graph)

    output_canvas.cd()
    if x == 'p':
        output_canvas.SetLogx()
    
    hframe = output_canvas.DrawFrame(*get_frame_init(mode, y='mean')[x])
    #hframe.SetTitle(option)
    hframe.SetTitle('')
    hframe.GetXaxis().SetTitleSize(0.05)
    hframe.GetYaxis().SetTitleSize(0.05)

    for g_mean in g_means:
        g_mean.Draw('P3 SAME')
    legend.Draw()
    text.Draw()
    watermark.Draw()

    output_canvas.Print(output_canvas_path)

def superimpose_resolution(output_canvas_path, x: str = 'beta_gamma', mode: str = 'mean'):

    output_canvas = TCanvas('cResolution', '', 800, 600)
    output_canvas.SetLeftMargin(0.15)
    
    g_resolutions = []
    input_file = TFile.Open(f'../output/data/LHC24_pass1_skimmed_calibration_{x}_{mode}.root', 'READ')

    legend = init_legend(0.5, 0.65, 0.85, 0.85)

    for particle in ['Pi', 'Ka', 'Pr', 'De', 'He']:

        print(f'Processing {particle=}, {TLATEX_PARTICLE_NAMES[particle]=}')
        c_mean = input_file.Get(f'{particle}/c_resolution')
        if not (c_mean and hasattr(c_mean, 'InheritsFrom') and c_mean.InheritsFrom('TCanvas')):
            print(f'Skipping {particle=}, no mean cluster size found.')
            continue

        list_mean = c_mean.GetListOfPrimitives()
        g_resolution = list_mean.FindObject('g_resolution')
        g_resolution.GetListOfFunctions().Clear()

        g_resolution = prepare_graph(g_resolution, legend, particle)

        g_resolutions.append(g_resolution)

    input_file.Close()

    watermark = get_alice_watermark(0.6, 0.25, 0.85, 0.45)

    mode_suffix = '_trunc' if mode == 'truncated' else ''
    x_suffix = 'momentum' if x == 'p' else 'betagamma'
    for particle_small, particle in zip(['xi', 'omega'], ['Xi', 'Omega']):
        input_file = TFile.Open(f'../output/caliva/cluster_size_{particle_small}_vs_{x_suffix}{mode_suffix}.root', 'READ')
        hist = input_file.Get(f'res_{particle_small}_pos')
        graph = hist_to_graph(hist, f'g_resolution_{particle}')
        
        graph = prepare_graph(graph, legend, particle)

        g_resolutions.append(graph)

    output_canvas.cd()
    
    hframe = output_canvas.DrawFrame(*get_frame_init(mode, y='resolution')[x])

    for g_mean in g_resolutions:
        g_mean.Draw('P SAME')
    legend.Draw()
    watermark.Draw()

    output_canvas.Print(output_canvas_path)
    
if __name__ == '__main__':
    
    output_canvas_path = 'output/cluster_size_per_species.pdf'
    _canvas = TCanvas('', '', 800, 600)
    _canvas.Print(f'{output_canvas_path}(')

    for x in ['beta_gamma', 'p']:
        for mode in ['mean', 'truncated']:
            superimpose_average_cluster_size(output_canvas_path, x=x, mode=mode)
            superimpose_resolution(output_canvas_path, x=x, mode=mode)
    
    for option in ['sigma']:#'parametrisation', 'sigma']:
        for x in ['beta_gamma', 'p']:
            superimpose_average_cluster_size_with_bands(output_canvas_path, x=x, mode='mean', option=option)

    _canvas.Print(f'{output_canvas_path})')
