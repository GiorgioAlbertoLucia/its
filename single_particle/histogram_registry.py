from functools import singledispatchmethod
from dataclasses import dataclass
import pandas as pd
from ROOT import RDataFrame, TFile

from torchic.utils.terminal_colors import TerminalColors as tc
from torchic.core.histogram import build_TH1, build_TH2, AxisSpec

@dataclass
class RegistryEntry:

    name: str
    title: str
    xvar: str
    nbinsx: int
    xmin: float
    xmax: float
    yvar: str = ''
    nbinsy: int = 0
    ymin: float = 0.0
    ymax: float = 0.0
    condition: str = 'True'
    save_directory: str = ''

class HistogramRegistry:
    
    def __init__(self):
        self._registry = {}
        self._registry_entries = {}

    def register(self, entry: RegistryEntry):
        if entry.name in self._registry_entries:
            raise ValueError(f"Histogram '{entry.name}' is already registered.")
        self._registry_entries[entry.name] = entry

    @singledispatchmethod
    def draw_histogram(self, arg, *args, **kwargs):
        raise NotImplementedError(f"Unsupported type: {type(arg)}")

    @draw_histogram.register
    def _(self, rdf: RDataFrame):
        
        for entry in self._registry_entries.values():

            if entry.nbinsy > 0:
                hist = rdf.Filter(entry.condition).Histo2D((entry.name, entry.title, 
                                                            entry.nbinsx, entry.xmin, entry.xmax, 
                                                            entry.nbinsy, entry.ymin, entry.ymax), 
                                                            entry.xvar, entry.yvar)
            else:
                hist = rdf.Filter(entry.condition).Histo1D((entry.name, entry.title, 
                                                            entry.nbinsx, entry.xmin, entry.xmax), 
                                                            entry.xvar)
            self._registry[entry.name] = hist

    @draw_histogram.register
    def _(self, df: pd.DataFrame):
        
        for entry in self._registry_entries.values():

            df_filtered = df.query(entry.condition) if entry.condition and entry.condition != 'True' else df

            if entry.nbinsy > 0:
                hist = build_TH2(df_filtered[entry.xvar], df_filtered[entry.yvar],
                                 AxisSpec(entry.nbinsx, entry.xmin, entry.xmax, entry.name, entry.title),
                                    AxisSpec(entry.nbinsy, entry.ymin, entry.ymax, entry.name, entry.title))
            else:
                hist = build_TH1(df_filtered[entry.xvar],
                                 AxisSpec(entry.nbinsx, entry.xmin, entry.xmax, entry.name, entry.title))
            self._registry[entry.name] = hist

    def prepare_directories(self, output_file: TFile):
        if not self._registry_entries:
            raise ValueError("No registry entries to prepare directories for.")
        
        seen_directories = set()
        for entry in self._registry_entries.values():
            if entry.save_directory and entry.save_directory not in seen_directories and entry.save_directory != '':
                output_file.mkdir(entry.save_directory)
                seen_directories.add(entry.save_directory)
                print(f"Created directory: {entry.save_directory}")

    def save_histograms(self, output_file: TFile):

        print(f"\n{tc.BOLD}Saving histograms to output file{tc.RESET}")
        if not self._registry:
            raise ValueError("No histograms to save.")
        
        entry_names_per_dir = {}
        for entry in self._registry_entries.values():
            idir = entry.save_directory or ''
            entry_names_per_dir.setdefault(idir, []).append(entry)

        for idir, entries in entry_names_per_dir.items():
            if idir:
                output_file.cd(idir)
            else:
                output_file.cd()

            for entry in entries:
                hist = self._registry.get(entry.name)
                if hist:
                    hist.Write(entry.name)
                    print(f"\tSaved histogram: {tc.CYAN+tc.UNDERLINE}{idir}:{entry.name}{tc.RESET}")
                else:
                    print(f"\t{tc.RED}WARNING{tc.RESET}Histogram {tc.CYAN+tc.UNDERLINE}{entry.name}{tc.RESET} not found in registry.")

