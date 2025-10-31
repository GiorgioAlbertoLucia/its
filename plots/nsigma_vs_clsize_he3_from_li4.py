import numpy as np
from ROOT import TCanvas, gStyle, \
    kOrange, kGreen, kRed, kAzure, kViolet, TLatex

from torchic import Dataset, AxisSpec
from torchic.physics.ITS import average_cluster_size

import sys
sys.path.append('..')
from utils.plot_utils import get_alice_watermark

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

    dataset['fItsClusterSize'] = np.array(dataset['fItsClusterSizeHe3'], np.uint64)
    dataset['fAvgClSizeCosLam'], dataset['fNHitsIts'] = np.zeros(dataset.shape[0], dtype=float), np.zeros(dataset.shape[0], dtype=int)
    dataset['fAvgClusterSize'], dataset['fNHitsIts'] = average_cluster_size(dataset['fItsClusterSize'], do_truncated=False)

    dataset.query('fNHitsIts > 5', inplace=True)
    dataset.query('abs(fNSigmaTPCHe3) < 2', inplace=True)
    dataset.query('0.5 < fChi2TPCHe3 < 4', inplace=True)
    dataset['fAvgClSizeCosLam'] = dataset['fAvgClusterSize'] / np.cosh(dataset['fEtaHe3'])

PARTICLE_LATEX = {
    'Pi': '#pi',
    'Ka': 'K',
    'Pr': 'p',
    'De': 'd',
    'He': '^{3}He'
}

def visualise_cluster_size_he3(dataset: Dataset, pdf_file_path:str):

    axis_spec_nsigma = AxisSpec(80, -2., 2., 'nsigma', 
                                f';#it{{p}}/#it{{Z}} (GeV/#it{{c}});n#sigma_{{TPC}} (^{{3}}He);')
    axis_spec_clsize = AxisSpec(222, 0, 17.5, 'AvgClSizeCosLam', ';n#sigma_{{TPC}} (^{{3}}He);#LT ITS Cluster size#kern[1]{#GT} #times #LT cos#lambda#kern[1]{#GT};')

    h2_nsigma = dataset.build_th2('fNSigmaTPCHe3', f'fAvgClSizeCosLam', axis_spec_nsigma, axis_spec_clsize,
                                    title=';n#sigma_{TPC} (^{3}He);#LT ITS Cluster size#kern[1]{#GT} #times #LT cos#lambda#kern[1]{#GT};')
    
    canvas = TCanvas(f'c_he3', '', 800, 600)
    canvas.SetLeftMargin(0.15)
    canvas.SetRightMargin(0.15)
    canvas.SetBottomMargin(0.15)
    canvas.SetTopMargin(0.15)
    canvas.SetLogz()

    h2_nsigma.GetXaxis().SetTitleSize(0.05)
    h2_nsigma.GetYaxis().SetTitleSize(0.05)

    he3_text = TLatex(0.5, 0.5, '#bf{^{3}He}')
    he3_text.SetNDC()
    he3_text.SetTextSize(0.05)
    he3_text.SetTextColor(0)

    h3_text = TLatex(0.7, 0.23, '#bf{^{3}H}')
    h3_text.SetNDC()
    h3_text.SetTextSize(0.05)
    h3_text.SetTextColor(0)

    h2_nsigma.Draw('col')
    h2_nsigma.SetTitle(';n#sigma_{TPC} (^{3}He);#LT ITS cluster size#kern[1]{#GT} #times #LT cos#lambda#kern[1]{#GT};')
    watermark = get_alice_watermark(0.17, 0.71, 0.44, 0.83)
    watermark.Draw('same')
    he3_text.Draw('same')
    h3_text.Draw('same')
    canvas.Print(pdf_file_path)
    del h2_nsigma, canvas
    
            



def main_routine(dataset: Dataset):

    gStyle.SetOptStat(0)

    pdf_file_path = f'/home/galucia/its/plots/output/LHC24_pass1_skimmed_cluster_size_he3.pdf'
    
    blank_canvas = TCanvas('c_blank', '', 800, 600)
    blank_canvas.Print(pdf_file_path + '(')

    prepare_dataset(dataset)
    
    visualise_cluster_size_he3(dataset, pdf_file_path)
    
    blank_canvas.Print(pdf_file_path + ')')
    del dataset
    del blank_canvas

if __name__ == '__main__':
    input_files = [ '/data/galucia/lithium_local/same/LHC24af_pass1_skimmed_same.root',
                    '/data/galucia/lithium_local/same/LHC24ag_pass1_skimmed_same.root',
                    '/data/galucia/lithium_local/same/LHC24aj_pass1_skimmed_same.root',
                    '/data/galucia/lithium_local/same/LHC24al_pass1_skimmed_same.root',
                    '/data/galucia/lithium_local/same/LHC24am_pass1_skimmed_same.root',
                    '/data/galucia/lithium_local/same/LHC24an_pass1_skimmed_same.root',]
    folder_name = 'DF*'
    tree_name = 'O2lithium4table'
    
    dataset = Dataset.from_root(input_files, tree_name, folder_name)
    
    main_routine(dataset)
