'''
    Deuteron preliminary analysis script.
'''

import numpy as np
from ROOT import TFile, TLegend, TCanvas
from particle import Particle
from torchic import Dataset, AxisSpec

import sys
sys.path.append('..')
from utils.pid_routine import define_variables, visualize_dataset, PARTICLE_ID
from scripts.calibration import calibration_routine

DATASET_COLUMN_NAMES = {
    'P':'fP',
    'Pt':'fPt',
    'Eta':'fEta',
    'TpcSignal':'fSignalTPC',
    'Chi2TPC':'fChi2TPC',
    'ItsClusterSize':'fItsClusterSize',
}

def visualize_dataset_de(dataset, directory):

    visualize_dataset(dataset, directory)

    axis_spec_pt = AxisSpec(100, -10., 10., '', ';#it{p}_{T} (GeV/#it{c});#LT ITS cluster size #GT #LT cos#lambda #GT')
    axis_spec_nsigma_tof = AxisSpec(100, -10., 10., 'nsigma_tof_v2', ';#it{p}_{T} (GeV/#it{c});TOF #sigma')

    h2_pt_tofnsigma = dataset.build_th2('fPt', 'fTofNSigma_v2', axis_spec_pt, axis_spec_nsigma_tof)
    directory.cd()
    h2_pt_tofnsigma.Write('h2PtTofNSigma_v2')

def compare_core_and_sidebands(infile:TFile, canvas_output_path: str):

    c_mean_core = infile.Get('TighterSelections/c_mean')
    if not (c_mean_core and hasattr(c_mean_core, 'InheritsFrom') and c_mean_core.InheritsFrom('TCanvas')):
        print(f'Skipping core, no mean cluster size found.')
    list_mean_core = c_mean_core.GetListOfPrimitives()
    g_mean_core = list_mean_core.FindObject('g_mean')
    g_mean_core.GetListOfFunctions().Clear()
    g_mean_core.SetMarkerColor(2)  # Red
    g_mean_core.SetMarkerStyle(22)  # Circle
    g_mean_core.SetMarkerSize(1.5)
    g_mean_core.SetName('g_mean_core')

    c_mean_sidebands = infile.Get('Sidebands/c_mean')
    if not (c_mean_sidebands and hasattr(c_mean_sidebands, 'InheritsFrom') and c_mean_sidebands.InheritsFrom('TCanvas')):
        print(f'Skipping sidebands, no mean cluster size found.')
    list_mean_sidebands = c_mean_sidebands.GetListOfPrimitives()
    g_mean_sidebands = list_mean_sidebands.FindObject('g_mean')
    g_mean_sidebands.GetListOfFunctions().Clear()
    g_mean_sidebands.SetMarkerColor(4)  # Blue
    g_mean_sidebands.SetMarkerStyle(23)  # Square
    g_mean_sidebands.SetMarkerSize(1.5)
    g_mean_sidebands.SetName('g_mean_sidebands')

    legend = TLegend(0.6, 0.7, 0.85, 0.85)
    legend.SetBorderSize(0)
    legend.SetTextSize(0.04)
    legend.SetFillColor(0)
    legend.AddEntry(g_mean_core, 'Core', 'p')
    legend.AddEntry(g_mean_sidebands, 'Sidebands', 'p')
    
    output_canvas = TCanvas('cMeanClusterSizeComparison', '', 800, 600)
    output_canvas.cd()
    hframe = output_canvas.DrawFrame(0, 1, 2.5, 6, 'Deuterons;#beta#gamma;#LT ITS cluster size #GT #times #LT cos#lambda #GT') 
    g_mean_core.Draw('P SAME')
    g_mean_sidebands.Draw('P SAME')
    legend.Draw()

    output_canvas.SaveAs(canvas_output_path)
    


if __name__ == '__main__':

    #infile_path = '/data/galucia/its_pid/MC_LHC24f3/MC_LHC24f3_small.root'
    infile_path = '/data/galucia/its_pid/LHC24_pass1_skimmed/data_04_08_2025.root'
    tree_names = ['O2clsttable', 'O2clsttableextra']
    folder_name = 'DF*'
    datasets = []
    for tree_name in tree_names:
        datasets.append(Dataset.from_root(infile_path, tree_name=tree_name, folder_name=folder_name,))
    dataset = datasets[0].concat(datasets[1], axis=1)

    dataset.query(f'fPartID == {PARTICLE_ID["De"]}')  # Filter for De particles

    define_variables(dataset, DATASET_COLUMN_NAMES=DATASET_COLUMN_NAMES)
    mass_de = Particle.from_name('D2').mass / 1_000 # GeV/c^2
    dataset['fBetaGamma'] = np.abs(dataset['fP']) / mass_de
    dataset['fTofMass'] = dataset['fTofMass'] / 2 # fix a bug in the original code
    dataset['fTofNSigma_v2'] = (dataset['fTofMass'] - mass_de) / (mass_de * 0.025)  # Assuming a resolution of 1% for TOF mass

    outfile_path = 'output/de.root'
    outfile = TFile.Open(outfile_path, 'RECREATE')

    dir_tot = outfile.mkdir('BasicSelections')
    visualize_dataset_de(dataset, dir_tot)
    print(f'{dataset["fPIDinTrk"].unique()}')
    
    dataset.query(f'abs(fTofNSigma) < 1', inplace=True)  # tighter selections to tof
    dataset.query(f'fPIDinTrk == 5', inplace=True)

    dataset_sidebands = dataset.query(f'fTpcNSigma > 0.8 or fTpcNSigma < -1.2', inplace=False)
    dir_sidebands = outfile.mkdir('Sidebands')
    visualize_dataset_de(dataset_sidebands, dir_sidebands)
    calibration_pars_sidebands_file_path = 'output/de_calibration_pars_sidebands.csv'
    calibration_routine(dataset_sidebands, outfile=dir_sidebands, \
                        params_file_path=calibration_pars_sidebands_file_path, particle='De', x='beta_gamma')

    dir_sel = outfile.mkdir('TighterSelections')
    #dataset.query(f'0.5 < fChi2TPC < 4', inplace=True)
    dataset.query(f'-1.2 < fTpcNSigma < 0.8', inplace=True)
    visualize_dataset_de(dataset, dir_sel)
    calibration_pars_file_path = 'output/de_calibration_pars_core.csv'
    calibration_routine(dataset, outfile=dir_sel, params_file_path=calibration_pars_file_path, \
                        particle='De', x='beta_gamma')

    outfile.Close()
    infile = TFile.Open(outfile_path, 'READ')
    compare_core_and_sidebands(infile, 'output/de_mean_cluster_size_comparison.pdf')

    #dir_pdg = outfile.mkdir('PDGSelections')
    #dataset.query(f'abs(fPartIDMc) == {int(Particle.from_name("D2").pdgid)}')  # Filter for He3 particles in MC
    #visualize_dataset(dataset, dir_pdg)
