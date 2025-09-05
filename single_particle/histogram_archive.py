'''
    Set of histograms commoly used in the annalysis
'''

from histogram_registry import HistogramRegistry, RegistryEntry
from copy import deepcopy

PT_NBINS, PT_MIN, PT_MAX = 200, -10, 10

QA_HISTOGRAMS = {
    "h2PNSigmaTPC": RegistryEntry("h2PNSigmaTPC", ";#it{p} / Z (GeV/#it{c});n#sigma_{TPC}", "fP", PT_NBINS, PT_MIN, PT_MAX, "fTpcNSigma", 100, -4, 4, 'True', ''),
    "h2NSigmaTPCClusterSize": RegistryEntry("h2NSigmaTPCClusterSize", ";n#sigma_{TPC};#LT ITS Cluster Size #GT #LT cos#lambda #GT;", "fTpcNSigma", 100, -4, 4, "fClSizeCosL", 90, 0, 15, 'True', ''),
    "h2PClusterSize": RegistryEntry("h2PClusterSize", ";#it{p} / Z (GeV/#it{c});#LT ITS Cluster Size #GT #LT cos#lambda #GT", "fP", PT_NBINS, PT_MIN, PT_MAX, "fClSizeCosL", 90, 0, 15, 'True', ''),
    "h2PNSigmaITS": RegistryEntry("h2PNSigmaITS", ";#it{p} / Z (GeV/#it{c});n#sigma_{ITS}", "fP", PT_NBINS, PT_MIN, PT_MAX, "fNSigmaIts", 100, -4, 4, 'True', ''),
    "h2PClusterSizeL0": RegistryEntry("h2PClusterSizeL0", ";#it{p} / Z (GeV/#it{c});Cluster size L0", "fP", PT_NBINS, PT_MIN, PT_MAX, "fItsClusterSizeL0", 15, 0, 15, 'True', ''),
    "h2PClusterSizeL1": RegistryEntry("h2PClusterSizeL1", ";#it{p} / Z (GeV/#it{c});Cluster size L1", "fP", PT_NBINS, PT_MIN, PT_MAX, "fItsClusterSizeL1", 15, 0, 15, 'True', ''),
    "h2PClusterSizeL2": RegistryEntry("h2PClusterSizeL2", ";#it{p} / Z (GeV/#it{c});Cluster size L2", "fP", PT_NBINS, PT_MIN, PT_MAX, "fItsClusterSizeL2", 15, 0, 15, 'True', ''),
    "h2PClusterSizeL3": RegistryEntry("h2PClusterSizeL3", ";#it{p} / Z (GeV/#it{c});Cluster size L3", "fP", PT_NBINS, PT_MIN, PT_MAX, "fItsClusterSizeL3", 15, 0, 15, 'True', ''),
    "h2PClusterSizeL4": RegistryEntry("h2PClusterSizeL4", ";#it{p} / Z (GeV/#it{c});Cluster size L4", "fP", PT_NBINS, PT_MIN, PT_MAX, "fItsClusterSizeL4", 15, 0, 15, 'True', ''),
    "h2PClusterSizeL5": RegistryEntry("h2PClusterSizeL5", ";#it{p} / Z (GeV/#it{c});Cluster size L5", "fP", PT_NBINS, PT_MIN, PT_MAX, "fItsClusterSizeL5", 15, 0, 15, 'True', ''),
    "h2PClusterSizeL6": RegistryEntry("h2PClusterSizeL6", ";#it{p} / Z (GeV/#it{c});Cluster size L6", "fP", PT_NBINS, PT_MIN, PT_MAX, "fItsClusterSizeL6", 15, 0, 15, 'True', ''),
    "h2PClusterSizeStd": RegistryEntry("h2PClusterSizeStd", ";#it{p} / Z (GeV/#it{c});Cluster size std", "fP", PT_NBINS, PT_MIN, PT_MAX, "fClusterSizeStd", 90, 0, 15, 'True', ''),
    "h2PClusterSizeSkew": RegistryEntry("h2PClusterSizeSkew", ";#it{p} / Z (GeV/#it{c});Cluster size skew", "fP", PT_NBINS, PT_MIN, PT_MAX, "fClusterSizeSkew", 90, 0, 15, 'True', ''),
    "h2PClusterSizeRange": RegistryEntry("h2PClusterSizeRange", ";#it{p} / Z (GeV/#it{c});Cluster size range", "fP", PT_NBINS, PT_MIN, PT_MAX, "fClusterSizeRange", 15, 0, 15, 'True', ''),
    "h2ClusterSizeClusterSizeRange": RegistryEntry("h2ClusterSizeClusterSizeRange", ";#LT ITS Cluster Size #GT #LT cos#lambda #GT;Cluster size range", "fClSizeCosL", 90, 0, 15, "fClusterSizeRange", 15, 0, 15, 'True', ''),
    "h2PEarlyLateRatio": RegistryEntry("h2PEarlyLateRatio", ";#it{p} / Z (GeV/#it{c});Early/Late cluster size ratio", "fP", PT_NBINS, PT_MIN, PT_MAX, "fEarlyLateRatio", 90, 0, 3, 'True', ''),
}

def register_qa_histograms(registry: HistogramRegistry):
    """
        Register the QA histograms in the registry.
    """
    for name, entry in QA_HISTOGRAMS.items():
        if 'NSigmaITS' in name:
            for particle in ['Pi', 'Ka', 'Pr', 'De', 'He']:
                tmp_entry = deepcopy(entry)
                tmp_entry.name += particle
                tmp_entry.title += particle
                tmp_entry.yvar += particle
                registry.register(tmp_entry)
        else:
            registry.register(entry)
