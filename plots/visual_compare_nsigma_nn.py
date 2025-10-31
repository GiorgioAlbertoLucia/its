from typing import Dict
from ROOT import TDirectory, TFile, TCanvas
from ROOT import kBlack, kBlue, kMagenta, kRed, kGreen, kOrange, kViolet, kAzure

from torchic.utils.terminal_colors import TerminalColors as tc

import sys
sys.path.append('..')
from utils.pid_routine import PDG_CODE, LATEX_PARTICLE
from utils.utils import set_root_object
from utils.plot_utils import get_alice_watermark, init_legend

import re

def replace_nth(string, sub, wanted, n):
    where = [m.start() for m in re.finditer(sub, string)][n-1]
    before = string[:where]
    after = string[where:]
    after = after.replace(sub, wanted, 1)
    new_string = before + after
    return new_string


def curve_comparison(files: Dict[str, TDirectory], pdf_output_file: str, particle: str):
    
    momentum_bins = [0.1, 0.3, 0.5, 0.7, 0.9, 1.1, 1.3, 1.5, 2.0, 3.0, 5.0, 10]
    canvas = TCanvas(f'canvas_{particle}', f'canvas_{particle}', 800, 600)
    canvas.SetLeftMargin(0.15)
    particle_latex = LATEX_PARTICLE.get(particle, particle)

    colors = [kOrange-3, kViolet+2, kAzure+1, kGreen+2, kRed+1, kMagenta+2, kBlack, kBlue, kGreen+3, kRed+3]
    
    for imomentum in range(len(momentum_bins)-1):
        
        rocs = {}
        for file_name, file in files.items():
            roc = file.Get(f'EfficiencyPurity/{particle}/efficiency_purity_curve_bin_{imomentum}')

            if not roc or (hasattr(roc, 'ClassName') and 'TGraph' not in roc.ClassName()):
                continue

            rocs[file_name] = roc
            rocs[file_name].SetName(file_name)
        if len(rocs) == 0:
            continue

        legend = init_legend(0.18, 0.18, 0.62, 0.34)
        pmin = momentum_bins[imomentum]
        pmax = momentum_bins[imomentum+1]
        for iroc, (roc_name, roc) in enumerate(rocs.items()):
            if iroc == 0:
                roc.SetTitle(f'{particle_latex}, {pmin} < #it{{p}} < {pmax} GeV/#it{{c}}; Efficiency; Purity')
            set_root_object(roc, marker_style=20+iroc, marker_color=colors[iroc], line_color=colors[iroc])
            legend.AddEntry(roc, roc_name, 'lp')

        watermark = get_alice_watermark(0.25, 0.34, 0.55, 0.54)
        
        canvas.cd()
        for iroc, roc in enumerate(rocs.values()):
            if iroc == 0:
                roc.DrawClone('APL')
            else:
                roc.DrawClone('PL SAME')
        legend.Draw()
        watermark.Draw()

        canvas.Print(f'{pdf_output_file}')
        canvas.Clear()

        del legend, watermark, rocs

def compare_nsigma_ml():

    files = {
        'n#sigma_{ITS}': TFile.Open('/home/galucia/its/plots/output/LHC24_pass1_skimmed_roc_curve_nsigmaITS_p.root', 'READ'),
        'NN': TFile.Open('/home/galucia/its/output/nn/results.root', 'READ'),
        'NN (no L0,1)': TFile.Open('/home/galucia/its/output/nn/results_no_L01.root', 'READ'),
        'BDT (momentum-aware)': TFile.Open('/home/galucia/its/output/bdt/results_momentum_aware.root', 'READ'),
        'BDT (momentum-aware-optuna)': TFile.Open('/home/galucia/its/output/bdt/results_momentum_aware_optuna.root', 'READ'),
        'BDT (ensemble)': TFile.Open('/home/galucia/its/output/bdt/results_bdt_ensemble.root', 'READ'),
    }
    
    output_file_pdf = 'output/nsigma_nn_comparison.pdf'
    blank_canvas = TCanvas('blank', 'blank', 800, 600)
    blank_canvas.Print(f'{output_file_pdf}(')
    
    particles = ['Pi', 'Ka', 'Pr', 'De', 'He']
    for particle in particles:
        curve_comparison(files, output_file_pdf, particle)
    
    for file in files.values():
        file.Close()
    
    blank_canvas.Print(f'{output_file_pdf})')

def compare_nn_configurations():

    files = {
        'NN trial 1': TFile.Open('/home/galucia/its/output/nn/results_trial1.root', 'READ'),
        'NN trial 2': TFile.Open('/home/galucia/its/output/nn/results_trial2.root', 'READ'),
        'NN trial 3': TFile.Open('/home/galucia/its/output/nn/results_trial3.root', 'READ'),
    }
    
    output_file_pdf = 'output/nn_trials_comparison.pdf'
    blank_canvas = TCanvas('blank', 'blank', 800, 600)
    blank_canvas.Print(f'{output_file_pdf}(')
    
    particles = ['Pi', 'Ka', 'Pr', 'De', 'He']
    for particle in particles:
        curve_comparison(files, output_file_pdf, particle)

    for file in files.values():
        file.Close()
    
    blank_canvas.Print(f'{output_file_pdf})')

if __name__ == '__main__':

    compare_nsigma_ml()
    #compare_nn_configurations()
