import numpy as np
import uproot
from torchic import Dataset, AxisSpec
from torchic.physics.ITS import average_cluster_size, expected_cluster_size, sigma_its

import sys
sys.path.append('..')
from utils.pid_routine import PARTICLE_ID

if __name__ == "__main__":
    
    infile = '/home/galucia/its/task/LHC24an_pass1.root'
    folder_name = 'DF*'
    tree_names = ['O2clsttable', 'O2clsttableextra']

    datasets = []
    for tree_name in tree_names:
        dataset = Dataset.from_root(infile, tree_name, folder_name)
        datasets.append(dataset)

    dataset = datasets[0] 
    dataset.concat(datasets[1], axis=1)
    dataset['fAvgClSize'], __ = average_cluster_size(dataset['fItsClusterSize'])
    dataset['fClSizeCosLam'] = dataset['fAvgClSize'] / np.cosh(dataset['fEta'])
    dataset['fPt'] = dataset['fP'] / np.cosh(dataset['fEta'])
    print(f'{dataset.columns=}')
    print(f'{dataset["fPartID"].unique()=}')
    
    axis_spec_pt = AxisSpec(100, 0., 10., 'pt', ';#it{p}_{T} (GeV/#it{c});#LT ITS cluster size #GT #LT cos#lambda #GT')
    axis_spec_cl = AxisSpec(90, 0., 15., 'cl', ';#it{cl} (cm);#LT ITS cluster size #GT #LT cos#lambda #GT')

    outfile = uproot.recreate('inspection.root')
    for part in ['Pi', 'Pr', 'De', 'He']:
        dataset.add_subset(part, dataset['fPartID'] == PARTICLE_ID[part])
        hist = dataset.build_boost2d('fPt', 'fClSizeCosLam', axis_spec_pt, axis_spec_cl, subset=part)
        outfile[f'h2PtClSizeCosLam_{part}'] = hist

    outfile.close()