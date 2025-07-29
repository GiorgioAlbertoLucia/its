import numpy as np
from torchic import Dataset, AxisSpec
from torchic.physics.ITS import average_cluster_size, N_ITS_LAYERS

PARTICLE_ID = {
    'none': 0,
    'El': 1,
    'Pi': 2,
    'Ka': 3,
    'Pr': 4,
    'De': 5,
    'He': 6,
}

PDG_CODE = {
    'Pi': 211,
    'Ka': 321,
    'Pr': 2212,
    'De': 1000010020,
    'He': 1000020030,
}

PID_ROUTINE_DATASET_COLUMN_NAMES = {
    'P':'fInnerParamTPC',
    'Pt':'fPt',
    'Eta':'fEta',
    'TpcSignal':'fSignalTPC',
    'Chi2TPC':'fChi2TPC',
    'ItsClusterSize':'fItsClusterSize',
}

def average_cluster_size_with_mean(cluster_sizes):
    '''
        Compute the average cluster size with the normal mean
    '''
    
    np_cluster_sizes = cluster_sizes.to_numpy()
    avg_cluster_size = np.zeros(len(np_cluster_sizes))

    n_hits = np.zeros(len(np_cluster_sizes))
    for ilayer in range(N_ITS_LAYERS):
        cluster_size_layer = np.right_shift(np_cluster_sizes, 4*ilayer) & 0b1111
        avg_cluster_size += cluster_size_layer
        n_hits += (cluster_size_layer > 0).astype(int)
    
    avg_cluster_size /= n_hits

    return avg_cluster_size

def define_variables(dataset: Dataset, **kwargs):

    cols = kwargs.get('DATASET_COLUMN_NAMES', PID_ROUTINE_DATASET_COLUMN_NAMES)

    dataset['fAvgClSize'], dataset['fNHitsIts'] = average_cluster_size(dataset[cols['ItsClusterSize']])
    dataset['fClSizeCosLam'] = dataset['fAvgClSize'] / np.cosh(dataset[cols['Eta']])
    
    dataset['fAvgClSizeMean'] = average_cluster_size_with_mean(dataset[cols['ItsClusterSize']])
    dataset['fAvgClSizeCosLamMean'] = dataset['fAvgClSizeMean'] / np.cosh(dataset[cols['Eta']])

    dataset[cols['Pt']] = dataset[cols['P']] / np.cosh(dataset[cols['Eta']])

def visualize_dataset(dataset: Dataset, outfile):

    axis_spec_pt = AxisSpec(100, 0., 10., '', ';#it{p}_{T} (GeV/#it{c});#LT ITS cluster size #GT #LT cos#lambda #GT')
    axis_spec_cl = AxisSpec(90, 0., 15., '', ';#it{p}_{T} (GeV/#it{c});#LT ITS cluster size #GT #LT cos#lambda #GT')
    axis_spec_nsigma_tpc = AxisSpec(100, -10., 10., 'nsigma_tpc', ';#it{p}_{T} (GeV/#it{c});TPC #sigma')
    axis_spec_nsigma_tof = AxisSpec(100, -10., 10., 'nsigma_tof', ';#it{p}_{T} (GeV/#it{c});TOF #sigma')
    axis_spec_tof_mass = AxisSpec(100, 0., 10., 'tof_mass', ';#it{p}_{T} (GeV/#it{c});TOF mass (GeV/#it{c}^{2})')

    h2_pt_clsize = dataset.build_th2('fPt', 'fClSizeCosLam', axis_spec_pt, axis_spec_cl)
    h2_pt_clsize_nsigma_tpc = dataset.build_th2('fPt', 'fTpcNSigma', axis_spec_pt, axis_spec_nsigma_tpc)
    h2_pt_clsize_nsigma_tof = dataset.build_th2('fPt', 'fTofNSigma', axis_spec_pt, axis_spec_nsigma_tof)
    h2_pt_clsize_tof_mass = dataset.build_th2('fPt', 'fTofMass', axis_spec_pt, axis_spec_tof_mass)
    
    outfile.cd()
    h2_pt_clsize.Write('h2PtClSizeCosLam')
    h2_pt_clsize_nsigma_tpc.Write('h2PtClSizeCosLamNSigmaTPC')
    h2_pt_clsize_nsigma_tof.Write('h2PtClSizeCosLamNSigmaTOF')
    h2_pt_clsize_tof_mass.Write('h2PtClSizeCosLamTOFMass')

def standard_selections_de(dataset: Dataset, **kwargs):

    cols = kwargs.get('DATASET_COLUMN_NAMES', PID_ROUTINE_DATASET_COLUMN_NAMES)

    dataset.query(f'0.5 < {cols["Chi2TPC"]} < 4', inplace=True)

def standard_selections_he(dataset: Dataset, **kwargs):

    cols = kwargs.get('DATASET_COLUMN_NAMES', PID_ROUTINE_DATASET_COLUMN_NAMES)

    dataset.query(f'0.5 < {cols["Chi2TPC"]} < 4', inplace=True)
    dataset.query('fClSizeCosLam > 4', inplace=True)

SELECTION_FUNCTIONS_DICT = {
    'De': standard_selections_de,
    'He': standard_selections_he,
}

def standard_selections(dataset: Dataset, particle: str, **kwargs):

    func = SELECTION_FUNCTIONS_DICT.get(particle, None)
    if func is None:
        raise ValueError(f'No standard selection function defined for particle: {particle}')
    func(dataset, **kwargs)