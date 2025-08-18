'''
    Script devoted to the visualisation of the average cluster size per species
'''

import numpy as np
from ROOT import TFile, TCanvas, TF1, TLegend, TPaveText, TGraphErrors
from ROOT import kOrange, kBlue, kGreen, kRed, kMagenta, kYellow, kBlack, kAzure,kViolet

TLATEX_PARTICLE_NAMES = {
    'Pi': '#pi^{#pm}',
    'Ka': 'K^{#pm}',
    'Pr': '(anti)p',
    'Xi': '#Xi^{+}',
    'Omega': '#Omega^{+}',
    'De': 'anti d',
    'He': '(anti)^{3}He',
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

def get_alice_watermark():

    watermark = TPaveText(0.6, 0.25, 0.85, 0.45, 'NDC')
    watermark.SetBorderSize(0)
    watermark.SetFillColor(0)
    watermark.SetTextAlign(12)
    watermark.SetTextSize(0.04)
    watermark.AddText('ALICE')
    watermark.AddText('#bf{pp #sqrt{s} = 13.6 TeV}')
    watermark.AddText('#bf{LHC24 pass1}')

    return watermark

def init_legend() -> TLegend:
    legend = TLegend(0.5, 0.65, 0.85, 0.85)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.04)
    legend.SetFillColor(0)
    legend.SetNColumns(2)
    return legend

def prepare_graph(graph, legend, particle):
    graph.SetMarkerColor(TCOLOR_PARTICLE[particle])  
    graph.SetMarkerStyle(TMARKER_PARTICLE[particle])
    graph.SetMarkerSize(.8)
    graph.SetLineColor(TCOLOR_PARTICLE[particle])
    legend.AddEntry(graph, TLATEX_PARTICLE_NAMES[particle], 'p')
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

def get_frame_init(mode:str, y:str='mean'):
    
    if y == 'mean':
        FRAME_INIT = {
                        'beta_gamma': (0, 1, 5.5, 9.5, f'{mode};#beta#gamma;#LT ITS cluster size#GT #times #LT cos#lambda#GT'),
                        'p': (-0.5, 1, 9, 9.5, f'{mode};#it{{p}} (GeV/#it{{c}});#LT ITS cluster size#GT #times #LT cos#lambda#GT')
                     }
    elif y == 'resolution':
        FRAME_INIT = {
                        'beta_gamma': (0, 0.09, 5.5, 0.28, f'{mode};#beta#gamma;#frac{{#sigma}}{{#LT ITS cluster size#GT #times #LT cos#lambda#GT}}'),
                        'p': (-0.5, 0.09, 9, 0.28, f'{mode};#it{{p}} (GeV/#it{{c}});#frac{{#sigma}}{{#LT ITS cluster size#GT #times #LT cos#lambda#GT}}')
                     }
    else:
        raise ValueError(f'Unsupported y value: {y}. Supported values are "mean" and "resolution".')
    return FRAME_INIT

def superimpose_average_cluster_size(output_canvas_path, x: str = 'beta_gamma', mode: str = 'mean'):

    output_canvas = TCanvas('cMeanClusterSize', '', 800, 600)
    
    g_means = []
    input_file = TFile.Open(f'../output/data/LHC24_pass1_skimmed_calibration_{x}_{mode}.root', 'READ')

    legend = init_legend()

    for particle in ['Pi', 'Ka', 'Pr', 'De', 'He']:

        print(f'Processing {particle=}, {TLATEX_PARTICLE_NAMES[particle]=}')
        c_mean = input_file.Get(f'{particle}/c_mean')
        if not (c_mean and hasattr(c_mean, 'InheritsFrom') and c_mean.InheritsFrom('TCanvas')):
            print(f'Skipping {particle=}, no mean cluster size found.')
            continue

        list_mean = c_mean.GetListOfPrimitives()
        g_mean = list_mean.FindObject('g_mean')
        g_mean.GetListOfFunctions().Clear()

        g_mean = prepare_graph(g_mean, legend, particle)
        g_means.append(g_mean)

    input_file.Close()

    watermark = get_alice_watermark()

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

def superimpose_resolution(output_canvas_path, x: str = 'beta_gamma', mode: str = 'mean'):

    output_canvas = TCanvas('cResolution', '', 800, 600)
    output_canvas.SetLeftMargin(0.15)
    
    g_resolutions = []
    input_file = TFile.Open(f'../output/data/LHC24_pass1_skimmed_calibration_{x}_{mode}.root', 'READ')

    legend = init_legend()

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

    watermark = get_alice_watermark()

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

    _canvas.Print(f'{output_canvas_path})')
