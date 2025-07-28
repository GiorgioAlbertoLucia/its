from ROOT import TFile
from particle import Particle
from torchic import Dataset

import sys
sys.path.append('..')
from utils.pid_routine import define_variables, visualize_dataset, PARTICLE_ID

if __name__ == '__main__':

    infile_path = '/data/galucia/its_pid/MC_LHC24f3/MC_LHC24f3_small.root'
    tree_name = 'O2clsttablemcext'
    folder_name = 'DF*'
    dataset = Dataset.from_root(infile_path, tree_name=tree_name, folder_name=folder_name,)
    print(f'{dataset.columns=}')

    dataset.query(f'fPartID == {PARTICLE_ID["He"]}')  # Filter for He3 particles

    dataset['fP'] = dataset['fP'] * 2  # rigidity is stored 
    define_variables(dataset)
    mass_he = Particle.from_name('He3').mass / 1_000 # GeV/c^2
    dataset['fBetaGamma'] = dataset['fP'] / mass_he

    outfile = TFile.Open('output/he.root', 'RECREATE')

    dir_tot = outfile.mkdir('BasicSelections')
    visualize_dataset(dataset, dir_tot)

    dir_pdg = outfile.mkdir('PDGSelections')
    dataset.query(f'abs(fPartIDMc) == {int(Particle.from_name("He3").pdgid)}')  # Filter for He3 particles in MC
    visualize_dataset(dataset, dir_pdg)
